//! Thin action-session client; the daemon owns timing and admission.
use std::path::PathBuf;

use clap::Subcommand;
use duck_ipc_proto as proto;

use crate::{Client, Failure, compact, exit, result_of};

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Acquire all joints on a ready fake/sim body; prints session and robot clock.
    Begin,
    /// Submit an ActionChunkParams JSON file with robot-clock timestamps.
    Submit { file: PathBuf },
    /// Cancel queued actions and release this session.
    End { session_id: String },
}

pub fn run(client: &mut Client, command: &Command) -> Result<(), Failure> {
    let call = match command {
        Command::Begin => proto::Call::RobotActionsBegin,
        Command::Submit { file } => {
            let bytes = std::fs::read(file)
                .map_err(|e| Failure::new(exit::USAGE, format!("{}: {e}", file.display())))?;
            let params = serde_json::from_slice(&bytes)
                .map_err(|e| Failure::new(exit::USAGE, format!("invalid action chunk: {e}")))?;
            proto::Call::RobotActionsSubmit(params)
        }
        Command::End { session_id } => proto::Call::RobotActionsEnd(proto::ActionEndParams {
            session_id: session_id.clone(),
        }),
    };
    let result = result_of(client.call(&call)?)?;
    println!("{}", compact(&result));
    Ok(())
}
