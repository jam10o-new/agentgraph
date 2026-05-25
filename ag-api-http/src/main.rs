//! ag-api-http — OpenAI-compatible HTTP API frontend for AgentGraph.
//!
//! Connects to the leader's Unix socket for session-tree operations and
//! agent inference.  All session state lives in the leader process; this
//! binary is a thin wire-format adapter: JSON-over-IPC in, SSE/JSON out.

use ag_ipc::{Command, IpcResponse, SessionStep};
use ag_utils::find_leader_socket;
use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::SystemTime;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;
use tower_http::cors::CorsLayer;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
struct Cli {
    /// Path to the shared config YAML.
    #[arg(long)]
    config: String,

    /// Leader Unix socket path.
    #[arg(long)]
    socket: String,

    /// JSON-serialized api-http section from the config.
    #[arg(long)]
    section: String,
}

#[derive(Debug, Deserialize)]
struct HttpConfig {
    #[serde(default = "default_true")]
    enabled: bool,
    #[serde(default = "default_bind")]
    bind_address: String,
    #[serde(default = "default_port")]
    port: u16,
}

fn default_true() -> bool {
    true
}
fn default_bind() -> String {
    "127.0.0.1".into()
}
fn default_port() -> u16 {
    3000
}

// ---------------------------------------------------------------------------
// Application state
// ---------------------------------------------------------------------------

struct AppState {
    /// Path to the shared config file (read again for model listing).
    config_path: String,
    /// Leader socket path.
    socket: String,
}

// ---------------------------------------------------------------------------
// OpenAI types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    stream: bool,
}

#[derive(Debug, Deserialize, Clone)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<Choice>,
}

#[derive(Debug, Serialize)]
struct Choice {
    index: usize,
    message: MessageResponse,
    finish_reason: String,
}

#[derive(Debug, Serialize)]
struct MessageResponse {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct ModelsListResponse {
    object: String,
    data: Vec<ModelObject>,
}

#[derive(Debug, Serialize)]
struct ModelObject {
    id: String,
    object: String,
    created: u64,
    owned_by: String,
}

// ---------------------------------------------------------------------------
// IPC helper
// ---------------------------------------------------------------------------

/// Send a command to the leader and receive the raw response bytes.
async fn ipc_raw(socket: &str, cmd: &Command) -> Result<String, String> {
    let mut stream = UnixStream::connect(socket)
        .await
        .map_err(|e| format!("connect: {e}"))?;

    let payload = serde_json::to_vec(cmd).map_err(|e| format!("serialize: {e}"))?;
    stream
        .write_all(&payload)
        .await
        .map_err(|e| format!("write: {e}"))?;
    stream.flush().await.map_err(|e| format!("flush: {e}"))?;
    stream
        .shutdown()
        .await
        .map_err(|e| format!("shutdown: {e}"))?;

    let mut buf = String::new();
    stream
        .read_to_string(&mut buf)
        .await
        .map_err(|e| format!("read: {e}"))?;
    Ok(buf)
}

/// Send a command and parse the response as IpcResponse.
async fn ipc(socket: &str, cmd: &Command) -> Result<IpcResponse, String> {
    let raw = ipc_raw(socket, cmd).await?;
    serde_json::from_str::<IpcResponse>(&raw)
        .map_err(|e| format!("deserialize IpcResponse: {e} (raw: {raw})"))
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

async fn list_models(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ModelsListResponse>, StatusCode> {
    let config = ag_config::Config::load(&state.config_path)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let created = SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let data: Vec<ModelObject> = config
        .agents
        .keys()
        .map(|name| ModelObject {
            id: name.clone(),
            object: "model".into(),
            created,
            owned_by: "agentgraph".into(),
        })
        .collect();

    Ok(Json(ModelsListResponse {
        object: "list".into(),
        data,
    }))
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<axum::response::Response, StatusCode> {
    let start_time = SystemTime::now();
    let id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let created = start_time
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let steps: Vec<SessionStep> = req
        .messages
        .iter()
        .map(|m| SessionStep {
            role: m.role.clone(),
            content: m.content.clone(),
        })
        .collect();

    // Check if leader socket is available
    let leader_ready = find_leader_socket().await.is_some();

    if leader_ready {
        let cmd = Command::SessionChat {
            session_id: format!("http-{}", req.model),
            steps,
            model: req.model.clone(),
            stream: req.stream,
        };

        match ipc(&state.socket, &cmd).await {
            Ok(resp) => {
                if resp.ok {
                    let content = resp.data.unwrap_or_default();

                    Ok(Json(ChatCompletionResponse {
                        id,
                        object: "chat.completion".into(),
                        created,
                        model: req.model,
                        choices: vec![Choice {
                            index: 0,
                            message: MessageResponse {
                                role: "assistant".into(),
                                content,
                            },
                            finish_reason: "stop".into(),
                        }],
                    })
                    .into_response())
                } else {
                    let _err = resp.error.unwrap_or_else(|| "unknown error".into());
                    Err(StatusCode::INTERNAL_SERVER_ERROR)
                }
            }
            Err(e) => {
                eprintln!("IPC error: {e}");
                Err(StatusCode::SERVICE_UNAVAILABLE)
            }
        }
    } else {
        Err(StatusCode::SERVICE_UNAVAILABLE)
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    let _http_cfg: HttpConfig = serde_json::from_str(&cli.section).unwrap_or_else(|e| {
        eprintln!("ag-api-http: failed to parse section config: {e}");
        std::process::exit(1);
    });

    if !_http_cfg.enabled {
        eprintln!("ag-api-http: enabled=false, exiting");
        return;
    }

    let state = Arc::new(AppState {
        config_path: cli.config,
        socket: cli.socket,
    });

    let bind_addr = format!("{}:{}", _http_cfg.bind_address, _http_cfg.port);
    let app = router(state);

    println!("ag-api-http listening on {bind_addr}");

    match tokio::net::TcpListener::bind(&bind_addr).await {
        Ok(listener) => {
            if let Err(e) = axum::serve(listener, app).await {
                eprintln!("ag-api-http: server error: {e}");
            }
        }
        Err(e) => {
            eprintln!("ag-api-http: failed to bind {bind_addr}: {e}");
            std::process::exit(1);
        }
    }
}
