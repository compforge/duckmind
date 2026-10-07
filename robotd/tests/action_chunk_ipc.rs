//! The real daemon, IPC, control loop and FakeIo — never a replacement executor.
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use duck_ipc_proto as proto;
use serde_json::{Value, json};

struct Daemon {
    child: Child,
    socket: PathBuf,
    _dir: tempfile::TempDir,
}

impl Drop for Daemon {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Daemon {
    fn spawn() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let socket = dir.path().join("robot.sock");
        let params = dir.path().join("robotd.toml");
        std::fs::write(&params, "[audio]\nenabled=false\n[chorale]\naccept=false\n").unwrap();
        let child = Command::new(env!("CARGO_BIN_EXE_robotd"))
            .args(["--fake", "--no-policy", "--socket"])
            .arg(&socket)
            .arg("--params")
            .arg(params)
            .env("RUST_LOG", "error")
            .stdout(Stdio::null())
            .spawn()
            .unwrap();
        let daemon = Self {
            child,
            socket,
            _dir: dir,
        };
        let deadline = Instant::now() + Duration::from_secs(5);
        while !daemon.socket.exists() {
            assert!(Instant::now() < deadline, "robotd did not open its socket");
            std::thread::sleep(Duration::from_millis(10));
        }
        daemon
    }

    fn call(&self, method: &str, params: Value) -> Value {
        let mut stream = UnixStream::connect(&self.socket).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        writeln!(
            stream,
            "{}",
            json!({"jsonrpc":"2.0","id":1,"method":method,"params":params})
        )
        .unwrap();
        let mut line = String::new();
        BufReader::new(stream).read_line(&mut line).unwrap();
        serde_json::from_str(&line).unwrap()
    }

    fn subscribe(&self) -> BufReader<UnixStream> {
        let mut stream = UnixStream::connect(&self.socket).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        writeln!(
            stream,
            "{}",
            json!({"jsonrpc":"2.0","id":2,"method":"robot.subscribe","params":{"hz":50}})
        )
        .unwrap();
        BufReader::new(stream)
    }
}

fn state(
    reader: &mut BufReader<UnixStream>,
    predicate: impl Fn(&proto::RobotState) -> bool,
) -> proto::RobotState {
    let deadline = Instant::now() + Duration::from_secs(4);
    loop {
        assert!(Instant::now() < deadline, "no matching execution feedback");
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        let value: Value = serde_json::from_str(&line).unwrap();
        if value["method"] == "robot.state" {
            let frame: proto::RobotState = serde_json::from_value(value["params"].clone()).unwrap();
            if predicate(&frame) {
                return frame;
            }
        }
    }
}

#[test]
fn chunk_drives_joints_and_stop_cancels_the_tail_and_late_response() {
    let daemon = Daemon::spawn();
    assert!(daemon.call("robot.init", json!({}))["error"].is_null());
    let deadline = Instant::now() + Duration::from_secs(6);
    let session = loop {
        let response = daemon.call("robot.actions.begin", json!({}));
        if response["error"].is_null() {
            break response["result"].clone();
        }
        assert!(Instant::now() < deadline, "begin refused: {response}");
        std::thread::sleep(Duration::from_millis(20));
    };
    assert_eq!(session["joint_names"].as_array().unwrap().len(), 15);
    let mut stream = daemon.subscribe();
    let initial = state(&mut stream, |s| {
        s.actions
            .as_ref()
            .is_some_and(|a| a.phase == proto::ActionPhase::Waiting)
    });
    let base: [f64; 15] = serde_json::from_value(session["positions"].clone()).unwrap();
    let mut first = base;
    first[7] += 0.15;
    let mut tail = base;
    tail[7] -= 0.15;
    let positions: Vec<_> = (0..75).map(|i| if i < 30 { first } else { tail }).collect();
    let chunk = json!({
        "session_id":session["session_id"], "sequence":1,
        "observation_t_ns":initial.t_ns,
        "start_t_ns":initial.actions.as_ref().unwrap().t_ns + 100_000_000,
        "step_ns":20_000_000, "positions":positions,
    });
    let accepted = daemon.call("robot.actions.submit", chunk.clone());
    assert_eq!(accepted["result"]["accepted"], true, "{accepted}");
    let reached = state(&mut stream, |s| (s.joints[7] - first[7]).abs() < 1e-6);
    assert_eq!(reached.policy, "action_chunk");
    assert_eq!(reached.actions.as_ref().unwrap().last_write_ok, Some(true));
    assert_eq!(
        daemon.call(
            "robot.head",
            json!({"neck_pitch":0.0,"head_pitch":0.0,"head_yaw":-1.0,"head_roll":0.0})
        )["error"]["code"],
        proto::code::BUSY
    );
    assert!(daemon.call("robot.stop", json!({}))["error"].is_null());
    let stopped = state(&mut stream, |s| {
        s.actions
            .as_ref()
            .is_some_and(|a| a.phase == proto::ActionPhase::Ended)
    });
    assert_eq!(
        stopped.actions.as_ref().unwrap().reason.as_deref(),
        Some("operator_preempted")
    );
    let mut late = chunk;
    late["sequence"] = json!(2);
    assert!(!daemon.call("robot.actions.submit", late)["error"].is_null());
    let after_tail = state(&mut stream, |s| s.t_ns > stopped.t_ns + 800_000_000);
    assert!(
        (after_tail.joints[7] - first[7]).abs() < 1e-6,
        "cancelled tail still moved the joint"
    );
    assert_eq!(after_tail.policy, "held");
}
