//! ag-api-matrix — Matrix bot for AgentGraph.
//!
//! Connects to a Matrix homeserver via the Client-Server API,
//! joins rooms, and routes messages through the AgentGraph
//! session tree.
//!
//! Configured via the `api-matrix` section in config.yaml.
//!
//! Status: **skeleton** — IPC connectivity and config parsing work.
//! The Matrix sync loop needs authentication + message polling.

use ag_ipc::{Command, IpcResponse, SessionChatResponse, SessionStep};
use ag_utils::find_leader_socket;
use clap::Parser;
use serde::Deserialize;
use std::collections::HashMap;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;

#[derive(Parser)]
struct Cli {
    #[arg(long)]
    socket: String,
    #[arg(long)]
    section: String,
}

#[derive(Debug, Deserialize)]
struct MatrixConfig {
    #[serde(default)]
    enabled: bool,
    /// Homeserver URL (e.g. https://matrix.org)
    homeserver: String,
    /// User ID (e.g. @bot:matrix.org)
    user_id: String,
    /// Access token from login or Element settings
    access_token: String,
    #[serde(default = "default_agent")]
    default_agent: String,
    #[serde(default)]
    allowed_users: Vec<String>,  // Matrix user IDs
}

fn default_agent() -> String { "api".to_string() }

async fn ipc_send(socket: &str, cmd: &Command) -> Result<IpcResponse, String> {
    let mut stream = UnixStream::connect(socket).await.map_err(|e| format!("connect: {e}"))?;
    let payload = serde_json::to_vec(cmd).map_err(|e| format!("serialize: {e}"))?;
    stream.write_all(&payload).await.map_err(|e| format!("write: {e}"))?;
    stream.flush().await.map_err(|e| format!("flush: {e}"))?;
    stream.shutdown().await.map_err(|e| format!("shutdown: {e}"))?;
    let mut buf = String::new();
    stream.read_to_string(&mut buf).await.map_err(|e| format!("read: {e}"))?;
    serde_json::from_str::<IpcResponse>(&buf).map_err(|e| format!("deserialize: {e} (raw: {buf})"))
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    let cfg: MatrixConfig = serde_yaml::from_str(&cli.section).unwrap_or_else(|e| {
        eprintln!("ag-api-matrix: failed to parse config: {e}");
        std::process::exit(1);
    });
    if !cfg.enabled && cfg.homeserver.is_empty() {
        return;
    }

    let socket = match find_leader_socket().await {
        Some(s) => s.to_string_lossy().to_string(),
        None => {
            eprintln!("ag-api-matrix: no leader socket found, exiting");
            std::process::exit(1);
        }
    };

    println!("ag-api-matrix: starting (homeserver: {})", cfg.homeserver);

    // TODO: Matrix sync via
    //   GET {homeserver}/_matrix/client/v3/sync?access_token={token}
    //   → parse rooms.join.{room_id}.timeline.events
    //   → filter for m.room.message with msgtype:m.text, filter sender
    //   → check allowed_users (if configured)

    // TODO: On message:
    //   1. Build SessionStep with role="user", content=msg.body
    //   2. SessionChat IPC to leader with session_id="mx-{room_id}"
    //   3. Send reply via
    //      PUT {homeserver}/_matrix/client/v3/rooms/{roomId}/send/m.room.message/{txnId}

    eprintln!("ag-api-matrix: TODO — Matrix sync loop not yet implemented");
    eprintln!("ag-api-matrix: session_id would be mx-{{room_id}}");
    eprintln!("ag-api-matrix: model would be {}", cfg.default_agent);
}
