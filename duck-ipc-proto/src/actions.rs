//! Robot-clock action chunks. Model normalization belongs to the caller.
use serde::{Deserialize, Serialize};

pub const ACTION_STEP_NS: u64 = 20_000_000;
pub const MAX_ACTION_STEPS: usize = 100;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActionChunkParams {
    pub session_id: String,
    pub sequence: u64,
    /// Timestamp of the observation used for this inference, on robot.state's clock.
    pub observation_t_ns: u64,
    pub start_t_ns: u64,
    pub step_ns: u64,
    /// Absolute radians in JOINT_NAMES order, including the mouth.
    pub positions: Vec<[f64; 15]>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActionEndParams {
    pub session_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActionSession {
    pub session_id: String,
    pub t_ns: u64,
    pub step_ns: u64,
    pub joint_names: Vec<String>,
    pub positions: [f64; 15],
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionPhase {
    #[default]
    Idle,
    Waiting,
    Running,
    Ended,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ActionStatus {
    pub session_id: Option<String>,
    pub phase: ActionPhase,
    pub sequence: Option<u64>,
    /// Sequence/index selected for the most recent write attempt; not measured completion.
    pub selected_sequence: Option<u64>,
    pub selected_index: Option<usize>,
    pub remaining: usize,
    pub t_ns: u64,
    pub reason: Option<String>,
    pub last_write_ok: Option<bool>,
}
