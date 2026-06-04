//! ag-api-whatsapp — WhatsApp bot for AgentGraph.
//!
//! Bridges to WhatsApp via a Node.js subprocess running baileys
//! (WhatsApp Web API library).  Communicates over stdin/stdout JSON
//! to avoid needing a REST intermediary.
//!
//! Configured via the `api-whatsapp` section in config.yaml.
//!
//! Status: **skeleton** — IPC connectivity and config parsing work.
//! Needs a companion `ag-whatsapp-bridge.js` script.

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
struct WhatsAppConfig {
    #[serde(default)]
    enabled: bool,
    /// Path to the Node.js bridge script
    bridge_script: String,
    /// Path to baileys auth state directory
    auth_dir: String,
    #[serde(default = "default_agent")]
    default_agent: String,
    #[serde(default)]
    allowed_jids: Vec<String>,
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
    let cfg: WhatsAppConfig = serde_yaml::from_str(&cli.section).unwrap_or_else(|e| {
        eprintln!("ag-api-whatsapp: failed to parse config: {e}");
        std::process::exit(1);
    });
    if !cfg.enabled && cfg.bridge_script.is_empty() {
        return;
    }

    let socket = match find_leader_socket().await {
        Some(s) => s.to_string_lossy().to_string(),
        None => {
            eprintln!("ag-api-whatsapp: no leader socket found, exiting");
            std::process::exit(1);
        }
    };

    println!("ag-api-whatsapp: starting (bridge: {})", cfg.bridge_script);

    // TODO: Spawn Node.js bridge process:
    //   node {bridge_script} --auth-dir {auth_dir}
    //   → read JSON messages from stdout (baileys connection.updates)
    //   → on messages.upsert: extract conversation, push to history
    //   → send replies by writing JSON to stdin

    // TODO: On incoming message:
    //   1. Build SessionStep with role="user", content=msg.message.conversation
    //   2. SessionChat IPC to leader with session_id="wa-{jid}"
    //   3. Send reply via writing JSON to bridge stdin

    eprintln!("ag-api-whatsapp: TODO — baileys bridge loop not yet implemented");
    eprintln!("ag-api-whatsapp: session_id would be wa-{{jid}}");
    eprintln!("ag-api-whatsapp: model would be {}", cfg.default_agent);
}
