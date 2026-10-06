use super::*;

fn setup() -> (Bridge, Executor) {
    let bridge = Bridge::new();
    bridge.available.store(true, Ordering::Release);
    let mut executor = bridge.executor();
    executor
        .accept(
            proto::Call::RobotActionsBegin,
            1_000_000_000,
            true,
            [0.0; 15],
            &mut false,
        )
        .unwrap();
    (bridge, executor)
}

fn chunk(executor: &Executor, sequence: u64, start: u64) -> proto::Call {
    proto::Call::RobotActionsSubmit(proto::ActionChunkParams {
        session_id: executor.status.session_id.clone().unwrap(),
        sequence,
        observation_t_ns: 1_000_000_000,
        start_t_ns: start,
        step_ns: proto::ACTION_STEP_NS,
        positions: (1..=5).map(|i| [i as f64 / 10.0; 15]).collect(),
    })
}

fn accept(executor: &mut Executor, call: proto::Call, now: u64) -> Answer {
    executor.accept(call, now, true, [0.0; 15], &mut false)
}

#[test]
fn late_inference_skips_expired_prefix_and_does_not_burst() {
    let (bridge, mut executor) = setup();
    let request = chunk(&executor, 1, 1_000_000_000);
    accept(&mut executor, request, 1_040_000_000).unwrap();
    assert_eq!(
        executor
            .tick(&bridge, 1_040_000_000, true, [0.0; 15])
            .target,
        Some([0.3; 15])
    );
    assert_eq!(
        executor
            .tick(&bridge, 1_080_000_000, true, [0.0; 15])
            .target,
        Some([0.5; 15])
    );
    let exhausted = executor.tick(&bridge, 1_100_000_000, true, [0.5; 15]);
    assert!(exhausted.ended);
    assert!(!exhausted.owned);
    assert_eq!(executor.status.reason.as_deref(), Some("buffer_exhausted"));
}

#[test]
fn future_replacement_preserves_prefix_and_discards_old_tail() {
    let (bridge, mut executor) = setup();
    let old = chunk(&executor, 1, 1_000_000_000);
    accept(&mut executor, old, 1_000_000_000).unwrap();
    let next = chunk(&executor, 2, 1_040_000_000);
    accept(&mut executor, next, 1_000_000_000).unwrap();
    assert_eq!(
        executor
            .tick(&bridge, 1_020_000_000, true, [0.0; 15])
            .target,
        Some([0.2; 15])
    );
    assert_eq!(
        executor
            .tick(&bridge, 1_040_000_000, true, [0.0; 15])
            .target,
        Some([0.1; 15])
    );
    assert_eq!(executor.status.selected_sequence, Some(2));
}

#[test]
fn expired_and_out_of_order_responses_leave_current_timeline_untouched() {
    let (bridge, mut executor) = setup();
    let good = chunk(&executor, 2, 1_020_000_000);
    accept(&mut executor, good, 1_000_000_000).unwrap();
    for (sequence, at, now) in [
        (1, 1_020_000_000, 1_020_000_000),
        (3, 1_000_000_000, 1_100_000_000),
    ] {
        let invalid = chunk(&executor, sequence, at);
        assert!(accept(&mut executor, invalid, now).is_err());
    }
    assert_eq!(
        executor
            .tick(&bridge, 1_100_000_000, true, [0.0; 15])
            .target,
        Some([0.5; 15])
    );
}

#[test]
fn cancellation_and_body_failure_invalidate_session() {
    for operator_stop in [false, true] {
        let (bridge, mut executor) = setup();
        let late = chunk(&executor, 1, 1_000_000_000);
        if operator_stop {
            bridge.interrupt();
        }
        let tick = executor.tick(&bridge, 1_000_000_000, operator_stop, [0.0; 15]);
        assert!(tick.ended);
        assert!(accept(&mut executor, late, 1_020_000_000).is_err());
    }
}

#[test]
fn end_and_new_begin_cannot_resurrect_a_previous_session() {
    let (_, mut executor) = setup();
    let late = chunk(&executor, 1, 1_000_000_000);
    let end = proto::Call::RobotActionsEnd(proto::ActionEndParams {
        session_id: executor.status.session_id.clone().unwrap(),
    });
    accept(&mut executor, end, 1_000_000_000).unwrap();
    accept(&mut executor, proto::Call::RobotActionsBegin, 1_020_000_000).unwrap();
    assert!(accept(&mut executor, late, 1_020_000_000).is_err());
}

#[test]
fn rejects_wrong_frequency_nonfinite_out_of_range_and_overflow() {
    let (_, executor) = setup();
    let proto::Call::RobotActionsSubmit(template) = chunk(&executor, 1, 1_000_000_000) else {
        unreachable!()
    };
    for value in [f64::NAN, f64::INFINITY, 4.0] {
        let mut p = template.clone();
        p.positions[0][0] = value;
        assert!(validate(&p).is_err());
    }
    let mut p = template.clone();
    p.step_ns = 1;
    assert!(validate(&p).is_err());
    let mut p = template;
    p.start_t_ns = u64::MAX;
    assert!(validate(&p).is_err());
}

#[tokio::test]
async fn acknowledgement_waits_for_loop_and_stop_invalidates_pending_requests() {
    let bridge = Arc::new(Bridge::new());
    bridge.available.store(true, Ordering::Release);
    let mut executor = bridge.executor();
    let b = bridge.clone();
    let request = tokio::spawn(async move { b.request(proto::Call::RobotActionsBegin).await });
    tokio::task::yield_now().await;
    assert!(!request.is_finished());
    bridge.interrupt();
    executor.tick(&bridge, 1_000_000_000, true, [0.0; 15]);
    assert!(request.await.unwrap().is_err());
    assert!(!executor.owns());
}

#[test]
fn no_initial_chunk_times_out_and_write_failure_releases_control() {
    let (bridge, mut executor) = setup();
    assert!(executor.tick(&bridge, 3_000_000_000, true, [0.0; 15]).ended);
    accept(&mut executor, proto::Call::RobotActionsBegin, 3_000_000_000).unwrap();
    executor.written(&bridge, false);
    assert!(!executor.owns());
    assert_eq!(executor.status.last_write_ok, Some(false));
    assert_eq!(executor.status.reason.as_deref(), Some("bus_write_failed"));
}

#[tokio::test]
async fn hardware_path_is_unavailable() {
    assert!(
        Bridge::new()
            .request(proto::Call::RobotActionsBegin)
            .await
            .is_err()
    );
}

#[test]
fn rejected_gap_or_overfull_replacement_preserves_accepted_timeline() {
    let (bridge, mut executor) = setup();
    let old = chunk(&executor, 1, 1_000_000_000);
    accept(&mut executor, old, 1_000_000_000).unwrap();
    let gap = chunk(&executor, 2, 1_120_000_000);
    assert!(accept(&mut executor, gap, 1_000_000_000).is_err());
    // A half-step offset can fit inside the time horizon yet exceed 100 entries.
    let proto::Call::RobotActionsSubmit(mut full) = chunk(&executor, 2, 1_010_000_000) else {
        unreachable!()
    };
    full.positions = vec![[0.7; 15]; 100];
    assert!(
        accept(
            &mut executor,
            proto::Call::RobotActionsSubmit(full),
            1_010_000_000
        )
        .is_err()
    );
    assert_eq!(executor.status.sequence, Some(1));
    assert_eq!(
        executor
            .tick(&bridge, 1_020_000_000, true, [0.0; 15])
            .target,
        Some([0.2; 15])
    );
}

#[test]
fn abandoned_admission_cannot_acquire_joints_later() {
    let bridge = Bridge::new();
    let mut executor = bridge.executor();
    let (answer, reply) = oneshot::channel();
    bridge
        .tx
        .try_send(Pending {
            generation: 0,
            call: proto::Call::RobotActionsBegin,
            answer,
        })
        .unwrap();
    drop(reply);
    executor.tick(&bridge, 1_000_000_000, true, [0.0; 15]);
    assert!(!executor.owns());
}
