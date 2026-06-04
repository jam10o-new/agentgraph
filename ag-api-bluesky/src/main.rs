//! ag-api-bluesky — Bluesky / AT Protocol bot for AgentGraph.
//!
//! Polls the Bluesky firehose or account notifications via the
//! AT Protocol XRPC API, routes mentions and DMs through the
//! AgentGraph session tree, and posts replies.
//!
//! Configured via the `api-bluesky` section in config.yaml.
//!
//! Status: **skeleton** — IPC connectivity and config parsing work.
//! The AT Protocol message loop needs the app password + DID setup.

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
struct BlueskyConfig {
    #[serde(default)]
    enabled: bool,
    /// AT Protocol service endpoint (e.g. https://bsky.social)
    service: String,
    /// Account handle (e.g. user.bsky.social)
    handle: String,
    /// App password from Settings > App Passwords
    app_password: String,
    #[serde(default = "default_agent")]
    default_agent: String,
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
    let cfg: BlueskyConfig = serde_yaml::from_str(&cli.section).unwrap_or_else(|e| {
        eprintln!("ag-api-bluesky: failed to parse config: {e}");
        std::process::exit(1);
    });
    if !cfg.enabled && cfg.service.is_empty() {
        return; // disabled
    }

    let socket = match find_leader_socket().await {
        Some(s) => s.to_string_lossy().to_string(),
        None => {
            eprintln!("ag-api-bluesky: no leader socket found, exiting");
            std::process::exit(1);
        }
    };

    println!("ag-api-bluesky: starting (service: {})", cfg.service);
    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(10))
        .timeout(std::time::Duration::from_secs(60))
        .build()
        .expect("failed to build HTTP client");

    // TODO: AT Protocol auth — create session via
    //   POST {service}/xrpc/com.atproto.server.createSession
    //   → receive accessJwt + did

    // TODO: Poll notifications via
    //   GET {service}/xrpc/app.bsky.notification.listNotifications
    //   → iterate mentions, extract text

    // TODO: On mention:
    //   1. Build SessionStep with role="user", content=text
    //   2. SessionChat IPC to leader
    //   3. Post reply via
    //      POST {service}/xrpc/com.atproto.repo.createRecord
    //      with collection: app.bsky.feed.post, reply: parent

    eprintln!("ag-api-bluesky: TODO — AT Protocol message loop not yet implemented");
    eprintln!("ag-api-bluesky: session_id would be bsky-{{author_did}}");
    eprintln!("ag-api-bluesky: model would be {}", cfg.default_agent);
    drop(client);
}
