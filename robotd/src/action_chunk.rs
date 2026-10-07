//! A bounded, robot-clock timeline owned by the existing control loop.
//!
//! IPC receives an acknowledgement only after the loop has accepted the command.
//! Session generations keep a queued/late inference from undoing an operator stop.
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use arc_swap::ArcSwap;
use duck_control::NUM_JOINTS;
use duck_control::safety::{ACTUATOR_MAX, ACTUATOR_MIN};
use duck_ipc_proto::{self as proto, ActionPhase, ActionStatus};
use tokio::sync::{mpsc, oneshot};

const CAPACITY: usize = 8;
const HORIZON_NS: u64 = proto::ACTION_STEP_NS * proto::MAX_ACTION_STEPS as u64;
type Answer = Result<serde_json::Value, proto::Error>;

pub struct Bridge {
    pub available: AtomicBool,
    pub status: ArcSwap<ActionStatus>,
    generation: AtomicU64,
    tx: mpsc::Sender<Pending>,
    rx: Mutex<Option<mpsc::Receiver<Pending>>>,
}

struct Pending {
    generation: u64,
    call: proto::Call,
    answer: oneshot::Sender<Answer>,
}

fn refused(message: impl Into<String>) -> proto::Error {
    proto::Error::new(proto::code::BUSY, message)
}

fn invalid(message: impl Into<String>) -> proto::Error {
    proto::Error::new(proto::code::INVALID_PARAMS, message)
}

impl Bridge {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::channel(CAPACITY);
        Self {
            available: AtomicBool::new(false),
            status: ArcSwap::from_pointee(ActionStatus::default()),
            generation: AtomicU64::new(0),
            tx,
            rx: Mutex::new(Some(rx)),
        }
    }

    pub fn executor(&self) -> Executor {
        Executor::new(
            self.rx
                .lock()
                .unwrap()
                .take()
                .expect("one control-loop owner"),
        )
    }

    pub fn owns_body(&self) -> bool {
        matches!(
            self.status.load().phase,
            ActionPhase::Waiting | ActionPhase::Running
        )
    }

    pub fn interrupt(&self) {
        self.generation.fetch_add(1, Ordering::AcqRel);
    }

    /// Stop/power commands preempt; other movement writers must wait for release.
    pub fn check_call(&self, call: &proto::Call) -> Result<(), proto::Error> {
        use proto::Call::*;
        if matches!(
            call,
            RobotStop | RobotInit | RobotRelax | RobotRebootMotors(_) | RobotShutdown
        ) || matches!(call, RobotEnable(p) if !p.on && !p.toggle)
        {
            self.interrupt();
        } else if self.owns_body()
            && matches!(
                call,
                RobotMove(_)
                    | RobotHead(_)
                    | RobotLook(_)
                    | RobotPose(_)
                    | RobotMouth(_)
                    | RobotDo(_)
                    | RobotEnable(_)
                    | RobotSetMode(_)
                    | RobotLoadPolicy(_)
                    | RobotReloadPolicies
                    | RobotTheremin(_)
                    | RobotChorale(_)
            )
        {
            return Err(refused("action session owns the joints; end it first"));
        }
        Ok(())
    }

    pub async fn request(&self, call: proto::Call) -> Answer {
        if !self.available.load(Ordering::Acquire) {
            return Err(refused("action chunks require --fake or --sim at 50 Hz"));
        }
        if let proto::Call::RobotActionsSubmit(p) = &call {
            validate(p)?;
        }
        let (answer, reply) = oneshot::channel();
        self.tx
            .try_send(Pending {
                generation: self.generation.load(Ordering::Acquire),
                call,
                answer,
            })
            .map_err(|_| refused("action command queue is full or stopped"))?;
        // A wedged motor loop must not wedge the IPC service. Closed replies are
        // skipped by the owner, so a timed-out request cannot execute later.
        tokio::time::timeout(Duration::from_millis(250), reply)
            .await
            .map_err(|_| refused("control loop did not accept the action request in time"))?
            .map_err(|_| refused("action executor stopped"))?
    }
}

fn validate(p: &proto::ActionChunkParams) -> Result<(), proto::Error> {
    if p.step_ns != proto::ACTION_STEP_NS
        || p.positions.is_empty()
        || p.positions.len() > proto::MAX_ACTION_STEPS
    {
        return Err(invalid("expected 1..100 actions with step_ns=20000000"));
    }
    if p.positions
        .iter()
        .flatten()
        .any(|v| !v.is_finite() || !(ACTUATOR_MIN..=ACTUATOR_MAX).contains(v))
    {
        return Err(invalid(
            "positions must be finite absolute radians within actuator travel",
        ));
    }
    if p.start_t_ns
        .checked_add(p.step_ns * p.positions.len() as u64)
        .is_none()
    {
        return Err(invalid("action timeline overflows"));
    }
    Ok(())
}

struct Frame {
    at: u64,
    sequence: u64,
    index: usize,
    target: [f64; NUM_JOINTS],
}

pub struct Executor {
    rx: mpsc::Receiver<Pending>,
    timeline: VecDeque<Frame>,
    status: ActionStatus,
    generation: u64,
    serial: u64,
    first_chunk_deadline: u64,
}

pub struct Tick {
    pub owned: bool,
    pub began: bool,
    pub ended: bool,
    pub target: Option<[f64; NUM_JOINTS]>,
}

impl Executor {
    fn new(rx: mpsc::Receiver<Pending>) -> Self {
        Self {
            rx,
            timeline: VecDeque::with_capacity(proto::MAX_ACTION_STEPS),
            status: ActionStatus::default(),
            generation: 0,
            serial: 0,
            first_chunk_deadline: 0,
        }
    }

    fn owns(&self) -> bool {
        matches!(
            self.status.phase,
            ActionPhase::Waiting | ActionPhase::Running
        )
    }

    fn finish(&mut self, reason: &str) {
        if self.owns() {
            tracing::info!(session = ?self.status.session_id, reason, "action session ended");
            self.timeline.clear();
            self.status.phase = ActionPhase::Ended;
            self.status.reason = Some(reason.to_owned());
            self.status.remaining = 0;
        }
    }

    /// Called only by the motor loop; fresh state and power/mode gates remain authoritative.
    pub fn tick(
        &mut self,
        bridge: &Bridge,
        now: u64,
        ready: bool,
        positions: [f64; NUM_JOINTS],
    ) -> Tick {
        let owned_before = self.owns();
        let generation = bridge.generation.load(Ordering::Acquire);
        if generation != self.generation {
            self.finish("operator_preempted");
            self.generation = generation;
        }
        if !ready {
            self.finish("body_not_ready");
        }
        self.status.t_ns = now;
        let mut began = false;
        // Bound work per tick even if an IPC client keeps filling the mailbox.
        for _ in 0..CAPACITY {
            let Ok(pending) = self.rx.try_recv() else {
                break;
            };
            if pending.answer.is_closed() {
                continue;
            }
            let answer = if pending.generation != self.generation {
                Err(refused("request was superseded by an operator command"))
            } else {
                self.accept(pending.call, now, ready, positions, &mut began)
            };
            self.publish(bridge);
            let _ = pending.answer.send(answer);
        }
        // Drop expired intervals; do not burst missed samples or stretch old actions.
        while self
            .timeline
            .front()
            .is_some_and(|f| f.at + proto::ACTION_STEP_NS <= now)
        {
            self.timeline.pop_front();
        }
        let target = self.timeline.iter().rev().find(|f| f.at <= now).map(|f| {
            self.status.selected_sequence = Some(f.sequence);
            self.status.selected_index = Some(f.index);
            f.target
        });
        if self.owns() {
            if self.timeline.is_empty()
                && (self.status.sequence.is_some() || now >= self.first_chunk_deadline)
            {
                self.finish("buffer_exhausted");
            } else {
                self.status.phase = if target.is_some() {
                    ActionPhase::Running
                } else {
                    ActionPhase::Waiting
                };
            }
        }
        self.status.remaining = self.timeline.len();
        self.publish(bridge);
        Tick {
            owned: self.owns(),
            began,
            ended: owned_before && !self.owns(),
            target,
        }
    }

    fn accept(
        &mut self,
        call: proto::Call,
        now: u64,
        ready: bool,
        positions: [f64; NUM_JOINTS],
        began: &mut bool,
    ) -> Answer {
        match call {
            proto::Call::RobotActionsBegin => {
                if !ready || self.owns() {
                    return Err(refused(
                        "body is not ready or another action session is active",
                    ));
                }
                self.serial += 1;
                let id = format!("{now:x}-{:x}", self.serial);
                self.status = ActionStatus {
                    session_id: Some(id.clone()),
                    phase: ActionPhase::Waiting,
                    t_ns: now,
                    ..Default::default()
                };
                self.timeline.clear();
                self.first_chunk_deadline = now + HORIZON_NS;
                *began = true;
                tracing::info!(session = id, "action session acquired joints");
                Ok(serde_json::to_value(proto::ActionSession {
                    session_id: id,
                    t_ns: now,
                    step_ns: proto::ACTION_STEP_NS,
                    positions,
                    joint_names: proto::JOINT_NAMES.iter().map(|s| (*s).to_owned()).collect(),
                })
                .unwrap())
            }
            proto::Call::RobotActionsSubmit(p) => {
                if !self.owns() || self.status.session_id.as_deref() != Some(&p.session_id) {
                    return Err(refused("action session is no longer active"));
                }
                validate(&p)?;
                if self.status.sequence.is_some_and(|s| p.sequence <= s) {
                    return Err(invalid("sequence must increase within the session"));
                }
                let end = p.start_t_ns + p.step_ns * p.positions.len() as u64;
                if p.observation_t_ns > now
                    || p.observation_t_ns > p.start_t_ns
                    || now.saturating_sub(p.observation_t_ns) > HORIZON_NS
                    || end <= now
                    || end - now > HORIZON_NS
                {
                    return Err(invalid(
                        "expired observation/chunk or timeline outside the two-second horizon",
                    ));
                }
                if self
                    .timeline
                    .back()
                    .is_some_and(|f| p.start_t_ns > f.at + p.step_ns)
                {
                    return Err(invalid(
                        "replacement must overlap or continue the buffered timeline",
                    ));
                }
                let prefix = self
                    .timeline
                    .iter()
                    .filter(|f| f.at < p.start_t_ns && f.at + p.step_ns > now)
                    .count();
                let incoming = (0..p.positions.len())
                    .filter(|i| p.start_t_ns + (*i as u64 + 1) * p.step_ns > now)
                    .count();
                if prefix + incoming > proto::MAX_ACTION_STEPS {
                    return Err(invalid("combined action buffer exceeds 100 intervals"));
                }
                // Retain the old prefix until takeover, discard its entire obsolete tail.
                self.timeline
                    .retain(|f| f.at < p.start_t_ns && f.at + proto::ACTION_STEP_NS > now);
                for (index, target) in p.positions.into_iter().enumerate() {
                    let at = p.start_t_ns + index as u64 * p.step_ns;
                    if at + p.step_ns > now {
                        self.timeline.push_back(Frame {
                            at,
                            sequence: p.sequence,
                            index,
                            target,
                        });
                    }
                }
                self.status.sequence = Some(p.sequence);
                tracing::debug!(
                    sequence = p.sequence,
                    remaining = self.timeline.len(),
                    "action chunk accepted"
                );
                Ok(
                    serde_json::json!({"accepted": true, "sequence": p.sequence, "remaining": self.timeline.len()}),
                )
            }
            proto::Call::RobotActionsEnd(p) => {
                if !self.owns() || self.status.session_id.as_deref() != Some(&p.session_id) {
                    return Err(refused("action session is no longer active"));
                }
                self.finish("cancelled");
                Ok(serde_json::json!({"accepted": true}))
            }
            _ => Err(invalid("not an action request")),
        }
    }

    fn publish(&self, bridge: &Bridge) {
        bridge.status.store(Arc::new(self.status.clone()));
    }

    pub fn written(&mut self, bridge: &Bridge, ok: bool) {
        if self.owns() {
            self.status.last_write_ok = Some(ok);
            if !ok {
                self.finish("bus_write_failed");
            }
            self.publish(bridge);
        }
    }
}

#[cfg(test)]
#[path = "action_chunk_tests.rs"]
mod tests;
