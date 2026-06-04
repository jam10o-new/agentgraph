//! ag-api-debug-ui — Browser-based debug UI for AgentGraph sessions.
//!
//! Connects to the leader via Unix socket and exposes a web interface at
//! http://localhost:PORT.  No external dependencies beyond axum; no JS
//! framework — all rendering is server-side HTML with minimal inline CSS.

use ag_ipc::{Command, IpcResponse, SessionChatResponse, SessionStep};
use ag_utils::find_leader_socket;
use axum::{
    Form, Router,
    extract::{Path, State},
    response::{Html, IntoResponse, Redirect},
    routing::get,
};
use clap::Parser;
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;
use tokio::sync::Mutex;

// ── CLI ──────────────────────────────────────────────────────────────

#[derive(Parser)]
struct Cli {
    /// Ignored — accepted for leader auto-spawn compatibility
    #[arg(long, default_value = "")]
    config: String,
    /// Ignored — accepted for leader auto-spawn compatibility
    #[arg(long, default_value = "")]
    section: String,
    #[arg(long)]
    socket: String,
    #[arg(long, default_value = "9090")]
    port: u16,
}

// ── App state ────────────────────────────────────────────────────────

#[derive(Clone)]
struct AppState {
    socket_path: String,
    /// In-memory tracked sessions so we don't need to create sessions
    /// that don't exist.
    session_ids: Arc<Mutex<Vec<String>>>,
}

// ── IPC helpers ──────────────────────────────────────────────────────

async fn ipc_send(socket: &str, cmd: &Command) -> Result<IpcResponse, String> {
    let mut stream = UnixStream::connect(socket)
        .await
        .map_err(|e| format!("connect: {e}"))?;
    let payload = serde_json::to_vec(cmd).map_err(|e| format!("serialize: {e}"))?;
    stream.write_all(&payload).await.map_err(|e| format!("write: {e}"))?;
    stream.flush().await.map_err(|e| format!("flush: {e}"))?;
    stream.shutdown().await.map_err(|e| format!("shutdown: {e}"))?;
    let mut buf = String::new();
    stream.read_to_string(&mut buf).await.map_err(|e| format!("read: {e}"))?;
    serde_json::from_str::<IpcResponse>(&buf)
        .map_err(|e| format!("deserialize: {e} (raw: {buf})"))
}

fn session_id_from(s: &str) -> String {
    if s.starts_with("debug-") {
        s.to_string()
    } else {
        format!("debug-{s}")
    }
}

// ── HTML helpers ─────────────────────────────────────────────────────

const STYLE: &str = r#"
<style>
  * { box-sizing:border-box; margin:0; padding:0; }
  body { font-family:system-ui, sans-serif; background:#0d1117; color:#c9d1d9; max-width:900px; margin:0 auto; padding:20px; }
  h1 { color:#58a6ff; margin-bottom:16px; }
  h2 { color:#f0883e; margin:20px 0 10px; }
  a { color:#58a6ff; text-decoration:none; }
  a:hover { text-decoration:underline; }
  .nav { margin-bottom:20px; }
  .nav a { margin-right:16px; }
  table { width:100%; border-collapse:collapse; margin:10px 0; }
  th, td { padding:8px 12px; text-align:left; border-bottom:1px solid #21262d; }
  th { color:#8b949e; font-weight:600; }
  tr:hover { background:#161b22; }
  .msg { padding:8px 12px; margin:6px 0; border-radius:6px; }
  .msg-user { background:#0d419d; }
  .msg-assistant { background:#1a3142; }
  .msg-system { background:#2a1a3d; }
  .role { font-size:0.75em; color:#8b949e; margin-bottom:2px; display:block; }
  form { margin:16px 0; display:flex; gap:8px; }
  input[type=text], textarea { flex:1; padding:10px; background:#0d1117; border:1px solid #30363d; border-radius:6px; color:#c9d1d9; font-size:14px; }
  button, input[type=submit] { padding:10px 20px; background:#238636; color:#fff; border:none; border-radius:6px; cursor:pointer; font-size:14px; }
  button:hover { background:#2ea043; }
  .danger { background:#da3633; }
  .danger:hover { background:#f85149; }
  .hash { font-family:monospace; font-size:0.8em; color:#8b949e; }
  .section { background:#161b22; border:1px solid #21262d; border-radius:8px; padding:16px; margin:16px 0; }
  .branch { padding:4px 0; border-bottom:1px solid #21262d; }
  .branch:last-child { border-bottom:none; }
  .empty { color:#484f58; font-style:italic; padding:16px; }
</style>"#;

fn page(title: &str, body: String) -> Html<String> {
    Html(format!(
        r#"<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>{STYLE}</head><body>{body}</body></html>"#
    ))
}

// ── Routes ───────────────────────────────────────────────────────────

/// Dashboard: list sessions + create new.
async fn dashboard(State(state): State<AppState>) -> impl IntoResponse {
    let sessions = state.session_ids.lock().await.clone();
    let mut rows = String::new();
    for id in &sessions {
        rows.push_str(&format!(
            r#"<tr><td><a href="/sessions/{id}">{id}</a></td><td><a href="/sessions/{id}" class="hash">view</a></td></tr>"#
        ));
    }
    if rows.is_empty() {
        rows = r#"<tr><td colspan="2" class="empty">No sessions yet — create one below.</td></tr>"#.into();
    }
    page(
        "AgentGraph Debug UI",
        format!(
            r#"<h1>AgentGraph Debug UI</h1>
<div class="nav"><a href="/">Dashboard</a></div>
<div class="section">
<h2>Sessions</h2>
<table>{rows}</table>
</div>
<div class="section">
<h2>New session</h2>
<form action="/sessions" method="post">
  <input type="text" name="id" placeholder="Session name (e.g. my-chat)" required />
  <input type="submit" value="Create" />
</form>
</div>"#
        ),
    )
}

/// Create a new session.
async fn create_session(
    State(state): State<AppState>,
    Form(form): Form<HashMap<String, String>>,
) -> impl IntoResponse {
    let raw_id = form.get("id").cloned().unwrap_or_default();
    let sid = session_id_from(&raw_id);
    let _ = ipc_send(
        &state.socket_path,
        &Command::SessionCreate { session_id: sid.clone() },
    )
    .await;
    let mut ids = state.session_ids.lock().await;
    if !ids.contains(&sid) {
        ids.push(sid.clone());
    }
    Redirect::to(&format!("/sessions/{sid}"))
}

/// View a single session: tree, message form, branches.
async fn session_view(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let sid = session_id_from(&id);

    // Ensure session exists
    let _ = ipc_send(
        &state.socket_path,
        &Command::SessionCreate { session_id: sid.clone() },
    )
    .await;

    let mut ids = state.session_ids.lock().await;
    if !ids.contains(&sid) {
        ids.push(sid.clone());
    }
    drop(ids);

    // Get branches
    let branch_html = match ipc_send(
        &state.socket_path,
        &Command::SessionListChildren {
            session_id: sid.clone(),
            hash: String::new(),
        },
    )
    .await
    {
        Ok(resp) if resp.ok => {
            if let Some(data) = &resp.data {
                match serde_json::from_str::<Vec<Vec<String>>>(data) {
                    Ok(children) if !children.is_empty() => {
                        let mut b = String::new();
                        for c in &children {
                            let hash = c.first().map(|s| s.as_str()).unwrap_or("?");
                            let role = c.get(1).map(|s| s.as_str()).unwrap_or("?");
                            b.push_str(&format!(
                                r#"<div class="branch"><span class="hash">{} {}</span> <a href="/sessions/{sid}/branches?hash={hash}">explore</a></div>"#,
                                &hash[..8.min(hash.len())],
                                role
                            ));
                        }
                        b
                    }
                    _ => r#"<div class="empty">No branches yet — send a message.</div>"#.into(),
                }
            } else {
                r#"<div class="empty">No data.</div>"#.into()
            }
        }
        _ => r#"<div class="empty">Leader not available.</div>"#.into(),
    };

    page(
        &format!("Session {sid}"),
        format!(
            r#"<h1>Session <code>{sid}</code></h1>
<div class="nav"><a href="/">← Dashboard</a></div>
<div class="section">
<h2>Send Message</h2>
<form action="/sessions/{sid}/send" method="post">
  <input type="text" name="content" placeholder="Type a message..." autofocus required />
  <input type="submit" value="Send" />
</form>
</div>
<div class="section">
<h2>Branches</h2>
{branch_html}
</div>
<div class="section">
<h2>Actions</h2>
<form action="/sessions/{sid}/delete" method="post" style="display:inline">
  <input type="submit" value="Delete Session" class="danger" onclick="return confirm('Delete this session?')" />
</form>
</div>"#
        ),
    )
}

/// Send a message to a session via SessionChat IPC.
#[derive(Deserialize)]
struct ChatForm {
    content: String,
}

async fn session_chat(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Form(form): Form<ChatForm>,
) -> impl IntoResponse {
    let sid = session_id_from(&id);
    let steps = vec![SessionStep {
        role: "user".to_string(),
        content: form.content.clone(),
        media: Vec::new(),
    }];
    let response_text = match ipc_send(
        &state.socket_path,
        &Command::SessionChat {
            session_id: sid.clone(),
            steps,
            model: "api".to_string(),
            stream: false,
        },
    )
    .await
    {
        Ok(resp) if resp.ok => {
            if let Some(data) = &resp.data {
                match serde_json::from_str::<SessionChatResponse>(data) {
                    Ok(sc) => sc.content.unwrap_or_else(|| "_(empty)_".into()),
                    Err(_) => data.clone(),
                }
            } else {
                "_(empty response)_".into()
            }
        }
        Ok(resp) => resp.error.unwrap_or_else(|| "error".into()),
        Err(e) => format!("IPC error: {e}"),
    };

    page(
        &format!("Response - {sid}"),
        format!(
            r#"<h1>Session <code>{sid}</code></h1>
<div class="nav"><a href="/sessions/{sid}">← Back</a></div>
<div class="section">
<h2>You</h2>
<div class="msg msg-user"><span class="role">user</span>{}</div>
<h2>Response</h2>
<div class="msg msg-assistant"><span class="role">assistant</span>{response_text}</div>
</div>"#,
            form.content.replace('<', "&lt;")
        ),
    )
}

/// Delete a session.
async fn session_delete(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let sid = session_id_from(&id);
    let _ = ipc_send(
        &state.socket_path,
        &Command::SessionDelete { session_id: sid.clone() },
    )
    .await;
    let mut ids = state.session_ids.lock().await;
    ids.retain(|i| i != &sid);
    Redirect::to("/")
}

/// View branches starting from a specific hash.
async fn branch_view(
    State(state): State<AppState>,
    Path(id): Path<String>,
    axum::extract::Query(params): axum::extract::Query<HashMap<String, String>>,
) -> impl IntoResponse {
    let sid = session_id_from(&id);
    let hash = params.get("hash").cloned().unwrap_or_default();
    let children = match ipc_send(
        &state.socket_path,
        &Command::SessionListChildren {
            session_id: sid.clone(),
            hash: hash.clone(),
        },
    )
    .await
    {
        Ok(resp) if resp.ok => {
            if let Some(data) = &resp.data {
                match serde_json::from_str::<Vec<Vec<String>>>(data) {
                    Ok(c) => c,
                    _ => Vec::new(),
                }
            } else {
                Vec::new()
            }
        }
        _ => Vec::new(),
    };

    let mut items = String::new();
    for c in &children {
        let c_hash = c.first().map(|s| s.as_str()).unwrap_or("?");
        let role = c.get(1).map(|s| s.as_str()).unwrap_or("?");
        let preview = c.get(2).map(|s| s.as_str()).unwrap_or("");
        items.push_str(&format!(
            r#"<div class="branch"><a href="/sessions/{sid}/branches?hash={c_hash}"><span class="role">{role}</span></a> <span class="hash">{}</span> {preview}</div>"#,
            &c_hash[..8.min(c_hash.len())]
        ));
    }
    if items.is_empty() {
        items = r#"<div class="empty">No children at this hash.</div>"#.into();
    }

    page(
        &format!("Branch {hash} - {sid}"),
        format!(
            r#"<h1>Branch <code>{}</code></h1>
<div class="nav"><a href="/sessions/{sid}">← Session</a> | <a href="/sessions/{sid}/branches">← Root</a></div>
<div class="section">{items}</div>"#,
            &hash[..8.min(hash.len())]
        ),
    )
}

// ── Main ─────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    let socket = match find_leader_socket().await {
        Some(s) => s.to_string_lossy().to_string(),
        None => cli.socket.clone(),
    };

    let state = AppState {
        socket_path: socket.clone(),
        session_ids: Arc::new(Mutex::new(Vec::new())),
    };

    let app = Router::new()
        .route("/", get(dashboard))
        .route("/sessions", axum::routing::post(create_session))
        .route("/sessions/{id}", get(session_view))
        .route("/sessions/{id}/send", axum::routing::post(session_chat))
        .route("/sessions/{id}/delete", axum::routing::post(session_delete))
        .route("/sessions/{id}/branches", get(branch_view))
        .with_state(state);

    let addr = format!("0.0.0.0:{}", cli.port);
    println!("ag-api-debug-ui: listening on http://{addr}");
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
