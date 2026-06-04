//! ag-api-signal — Signal Messenger bot for AgentGraph.
//!
//! Connects to Signal via the signald or libsignal-service-rs
//! bridge, routes messages through the AgentGraph session tree.
//!
//! Configured via the `api-signal` section in config.yaml.
//!
//! Status: **skeleton** — IPC connectivity and config parsing work.
//! The Signal protocol requires a native library bridge (libsignal)
//! or a running signald REST API instance.

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
struct SignalConfig {
    #[serde(default)]
    enabled: bool,
    /// signald REST API endpoint (e.g. http://localhost:8080)
    /// or path to libsignal native bridge
    signald_url: String,
    /// Signal account phone number in E.164 format
    phone_number: String,
    #[serde(default = "default_agent")]
    default_agent: String,
    #[serde(default)]
    allowed_numbers: Vec<String>,
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
    let cfg: SignalConfig = serde_yaml::from_str(&cli.section).unwrap_or_else(|e| {
        eprintln!("ag-api-signal: failed to parse config: {e}");
        std::process::exit(1);
    });
    if !cfg.enabled && cfg.signald_url.is_empty() {
        return;
    }

    let socket = match find_leader_socket().await {
        Some(s) => s.to_string_lossy().to_string(),
        None => {
            eprintln!("ag-api-signal: no leader socket found, exiting");
            std::process::exit(1);
        }
    };

    println!("ag-api-signal: starting (signald: {})", cfg.signald_url);

    // TODO: Register with signald via
    //   POST {signald_url}/v1/register
    //   → link device

    // TODO: Subscribe to receive messages via
    //   GET {signald_url}/v1/receive/{account}
    //   → stream incoming Signal messages

    // TODO: On message:
    //   1. Build SessionStep with role="user", content=msg.dataMessage.message
    //   2. SessionChat IPC to leader with session_id="sg-{sender_number}"
    //   3. Send reply via
    //      POST {signald_url}/v1/send
    //      with { username, recipientAddress, messageBody }

    eprintln!("ag-api-signal: TODO — Signal receive loop not yet implemented");
    eprintln!("ag-api-signal: session_id would be sg-{{sender_number}}");
    eprintln!("ag-api-signal: model would be {}", cfg.default_agent);
}
