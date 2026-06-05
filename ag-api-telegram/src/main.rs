//! ag-api-telegram — Telegram bot frontend for AgentGraph.
//!
//! Reads its own `api-telegram` config section, connects to the leader's
//! Unix socket for session operations, and exposes AgentGraph agents
//! through a persistent Telegram bot with /commands.

use ag_ipc::{Command, IpcResponse, SessionChatResponse, SessionStep};
use ag_utils::find_leader_socket;
use clap::Parser;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;
use tokio::sync::Mutex;

// ── CLI ────────────────────────────────────────────────────────────────────

#[derive(Parser)]
struct Cli {
    #[arg(long)]
    config: String,
    #[arg(long)]
    socket: String,
    #[arg(long)]
    section: String,
}

// ── Telegram config (read from --section JSON) ─────────────────────────────

#[derive(Debug, Deserialize)]
struct TelegramConfig {
    #[serde(default)]
    enabled: bool,
    bot_token: String,
    #[serde(default = "default_agent")]
    default_agent: String,
    #[serde(default)]
    user_agents: HashMap<String, String>,
    #[serde(default)]
    group_agents: HashMap<String, String>,
    #[serde(default)]
    channel_agents: HashMap<String, String>,
    #[serde(default)]
    allowed_users: Vec<i64>,
    /// Users allowed to use privileged commands (/config, /system write, etc.).
    /// If empty, defaults to `allowed_users`. If both are empty, all users
    /// may use privileged commands (insecure default for open deployments).
    #[serde(default)]
    privileged_users: Vec<i64>,
    /// How often (in milliseconds) to poll the stream file and call
    /// editMessageText for progressive output.  Higher values reduce
    /// Telegram API calls but make updates feel less responsive.
    /// Default: 2000 (2 seconds).  Set to 0 to disable streaming entirely.
    #[serde(default = "default_stream_poll")]
    stream_poll_interval_ms: u64,
}

fn default_stream_poll() -> u64 {
    2000
}

fn default_agent() -> String {
    "api".to_string()
}

// ── Per-chat state ─────────────────────────────────────────────────────────

struct ChatState {
    history: Vec<SessionStep>,
    agent: String,
}

struct BotState {
    socket_path: String,
    tg: TelegramConfig,
    chats: Mutex<HashMap<i64, ChatState>>,
}

// ── IPC helper ─────────────────────────────────────────────────────────────

async fn ipc_send(socket: &str, cmd: &Command) -> Result<IpcResponse, String> {
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
    serde_json::from_str::<IpcResponse>(&buf).map_err(|e| format!("deserialize: {e} (raw: {buf})"))
}

/// Send a raw IPC command that does not produce JSON (Status, RunAgent, etc.).
/// Returns the raw text response from the leader.
async fn ipc_send_raw(socket: &str, cmd: &Command) -> Result<String, String> {
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
    Ok(buf.trim().to_string())
}

// ── Telegram API helpers ────────────────────────────────────────────────────

/// Always-on formatted message helper.  Code fences pass through intact.
/// The rest is sent as MarkdownV2 (preserving bold/italic/etc.).
/// On parse error: wraps the plain text inside a single ``` code block
/// so that raw formatting tags never leak.
async fn send_message_telegram(
    client: &reqwest::Client,
    token: &str,
    chat_id: i64,
    text: &str,
) {
    if send_message_inner(client, token, chat_id, text, "MarkdownV2").await {
        return;
    }
    // Parse error → wrap entire message in a code block
    let escaped_text = format!("```\n{text}\n```");
    if send_message_inner(client, token, chat_id, &escaped_text, "MarkdownV2").await {
        eprintln!("ag-api-telegram: sendMessage delivered via code-block escape (chat {chat_id})");
        return;
    }
    // Ultimate fallback: plain text
    let _ = send_message_raw(client, token, chat_id, text, None).await;
    eprintln!("ag-api-telegram: sendMessage reverted to plain text (chat {chat_id})");
}

/// Try to send a message with the given parse_mode.  Returns true on success.
async fn send_message_inner(
    client: &reqwest::Client,
    token: &str,
    chat_id: i64,
    text: &str,
    parse_mode: &str,
) -> bool {
    let body = serde_json::json!({
        "chat_id": chat_id,
        "text": text,
        "parse_mode": parse_mode,
    });
    let resp = match client
        .post(format!("https://api.telegram.org/bot{token}/sendMessage"))
        .json(&body)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            eprintln!("ag-api-telegram: sendMessage transport failed (chat {chat_id}): {e}");
            return false;
        }
    };
    resp.status().is_success()
}

/// Send a message without specifying parse_mode (Telegram sends plain text).
async fn send_message_raw(
    client: &reqwest::Client,
    token: &str,
    chat_id: i64,
    text: &str,
    parse_mode: Option<&str>,
) -> bool {
    let mut body = serde_json::json!({
        "chat_id": chat_id,
        "text": text,
    });
    if let Some(pm) = parse_mode {
        body["parse_mode"] = serde_json::Value::String(pm.to_string());
    }
    let Ok(resp) = client
        .post(format!("https://api.telegram.org/bot{token}/sendMessage"))
        .json(&body)
        .send()
        .await
    else {
        return false;
    };
    resp.status().is_success()
}

/// Send a message and return the message_id for subsequent edits.
async fn send_initial_message(
    client: &reqwest::Client,
    token: &str,
    chat_id: i64,
    text: &str,
) -> Option<i64> {
    let body = serde_json::json!({
        "chat_id": chat_id,
        "text": text,
    });
    match client
        .post(format!("https://api.telegram.org/bot{token}/sendMessage"))
        .json(&body)
        .send()
        .await
    {
        Ok(resp) => {
            if let Ok(v) = resp.json::<serde_json::Value>().await {
                return v["result"]["message_id"].as_i64();
            }
            None
        }
        Err(e) => {
            eprintln!("ag-api-telegram: sendMessage failed (chat {chat_id}): {e}");
            None
        }
    }
}

/// Edit an existing message (for progressive streaming updates).
/// Same strategy as send_message: try MarkdownV2 first, fall back to code-block wrapping.
async fn edit_message_telegram(
    client: &reqwest::Client,
    token: &str,
    chat_id: i64,
    message_id: i64,
    text: &str,
) {
    if text.is_empty() {
        return;
    }
    if edit_message_inner(client, token, chat_id, message_id, text, "MarkdownV2").await {
        return;
    }
    // Parse error → wrap in code block
    let escaped_text = format!("```\n{text}\n```");
    if edit_message_inner(client, token, chat_id, message_id, &escaped_text, "MarkdownV2").await {
        eprintln!("ag-api-telegram: editMessageText delivered via code-block escape (chat {chat_id})");
        return;
    }
    // Ultimate fallback: plain text
    let _ = edit_message_raw(client, token, chat_id, message_id, text, None).await;
    eprintln!("ag-api-telegram: editMessageText reverted to plain text (chat {chat_id})");
}

/// Try to edit a message with the given parse_mode.  Returns true on success.
async fn edit_message_inner(
    client: &reqwest::Client,
    token: &str,
    chat_id: i64,
    message_id: i64,
    text: &str,
    parse_mode: &str,
) -> bool {
    let body = serde_json::json!({
        "chat_id": chat_id,
        "message_id": message_id,
        "text": text,
        "parse_mode": parse_mode,
    });
    let resp = match client
        .post(format!(
            "https://api.telegram.org/bot{token}/editMessageText"
        ))
        .json(&body)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            eprintln!(
                "ag-api-telegram: editMessageText transport failed (chat {chat_id}, msg {message_id}): {e}"
            );
            return false;
        }
    };
    let body_text = resp.text().await.unwrap_or_default();
    match serde_json::from_str::<serde_json::Value>(&body_text) {
        Ok(v) if v["ok"].as_bool().unwrap_or(false) => true,
        Ok(v) => {
            let desc = v["description"].as_str().unwrap_or("unknown error");
            let is_modified = desc.contains("message is not modified");
            if is_modified {
                // identical - not a failure
                return true;
            }
            let is_parse = desc.contains("can't parse entities");
            if !is_parse {
                eprintln!(
                    "ag-api-telegram: editMessageText failed (chat {chat_id}, msg {message_id}): {desc}"
                );
                return false;
            }
            false // parse error → try next tier
        }
        Err(e) => {
            eprintln!(
                "ag-api-telegram: editMessageText unparseable response (chat {chat_id}): {e}"
            );
            false
        }
    }
}

/// Edit a message without specifying parse_mode (plain text).
async fn edit_message_raw(
    client: &reqwest::Client,
    token: &str,
    chat_id: i64,
    message_id: i64,
    text: &str,
    parse_mode: Option<&str>,
) -> bool {
    let mut body = serde_json::json!({
        "chat_id": chat_id,
        "message_id": message_id,
        "text": text,
    });
    if let Some(pm) = parse_mode {
        body["parse_mode"] = serde_json::Value::String(pm.to_string());
    }
    let Ok(resp) = client
        .post(format!(
            "https://api.telegram.org/bot{token}/editMessageText"
        ))
        .json(&body)
        .send()
        .await
    else {
        return false;
    };
    let body_text = resp.text().await.unwrap_or_default();
    serde_json::from_str::<serde_json::Value>(&body_text)
        .map(|v| v["ok"].as_bool().unwrap_or(false))
        .unwrap_or(false)
}

/// Read the latest content from the stream output file inside `stream_dir`.
/// Returns None if no stream file has appeared yet.
async fn read_stream_content(stream_dir: &str) -> Option<String> {
    let mut dir = tokio::fs::read_dir(stream_dir).await.ok()?;
    let mut files = Vec::new();
    loop {
        match dir.next_entry().await {
            Ok(Some(entry)) => {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with("out-") {
                    files.push(entry.path());
                }
            }
            Ok(None) => break,
            Err(_) => break,
        }
    }
    files.sort();
    let path = files.last()?;
    tokio::fs::read_to_string(path).await.ok()
}

/// Extract media from a Telegram message JSON, returning (file_id, kind, mime_or_ext).
/// kind is one of: "photo", "video", "voice", "audio", "document", "animation"
fn extract_media(msg: &serde_json::Value) -> Vec<(String, String, String)> {
    let mut media = Vec::new();
    // photo — array ofPhotoSize, last is largest
    if let Some(photos) = msg["photo"].as_array() {
        if let Some(largest) = photos.last() {
            if let Some(fid) = largest["file_id"].as_str() {
                media.push((fid.to_string(), "photo".into(), "jpg".into()));
            }
        }
    }
    // video
    if let Some(fid) = msg["video"]["file_id"].as_str() {
        let ext = msg["video"]["mime_type"].as_str()
            .and_then(|m| ext_from_mime(m))
            .unwrap_or("mp4");
        media.push((fid.to_string(), "video".into(), ext.into()));
    }
    // voice / audio (voice is OGG, audio can be any)
    if let Some(fid) = msg["voice"]["file_id"].as_str() {
        media.push((fid.to_string(), "voice".into(), "ogg".into()));
    }
    if let Some(fid) = msg["audio"]["file_id"].as_str() {
        let ext = msg["audio"]["mime_type"].as_str()
            .and_then(|m| ext_from_mime(m))
            .unwrap_or("mp3");
        media.push((fid.to_string(), "audio".into(), ext.into()));
    }
    // document (any file)
    if let Some(fid) = msg["document"]["file_id"].as_str() {
        let ext = msg["document"]["file_name"].as_str()
            .and_then(|n| std::path::Path::new(n).extension()?.to_str())
            .or_else(|| msg["document"]["mime_type"].as_str().and_then(|m| ext_from_mime(m)))
            .unwrap_or("bin");
        media.push((fid.to_string(), "document".into(), ext.into()));
    }
    // animation (GIF)
    if let Some(fid) = msg["animation"]["file_id"].as_str() {
        media.push((fid.to_string(), "animation".into(), "mp4".into()));
    }
    // sticker
    if let Some(fid) = msg["sticker"]["file_id"].as_str() {
        let ext = if msg["sticker"]["is_animated"].as_bool().unwrap_or(false) || msg["sticker"]["is_video"].as_bool().unwrap_or(false) {
            "webm"
        } else {
            "webp"
        };
        media.push((fid.to_string(), "sticker".into(), ext.into()));
    }
    media
}

fn ext_from_mime(mime: &str) -> Option<&str> {
    match mime {
        "image/jpeg" => Some("jpg"),
        "image/png" => Some("png"),
        "image/webp" => Some("webp"),
        "image/gif" => Some("gif"),
        "video/mp4" => Some("mp4"),
        "video/webm" => Some("webm"),
        "video/quicktime" => Some("mov"),
        "video/x-msvideo" => Some("avi"),
        "audio/wav" | "audio/wave" | "audio/x-wav" => Some("wav"),
        "audio/mpeg" | "audio/mp3" => Some("mp3"),
        "audio/ogg" => Some("ogg"),
        "audio/opus" => Some("opus"),
        "audio/x-flac" | "audio/flac" => Some("flac"),
        "audio/aac" | "audio/x-aac" => Some("aac"),
        "audio/mp4" | "audio/m4a" => Some("m4a"),
        _ => None,
    }
}

/// Download a Telegram file by file_id. Returns a temp file path with the given extension.
async fn download_telegram_media(
    client: &reqwest::Client,
    token: &str,
    file_id: &str,
    ext: &str,
) -> Option<PathBuf> {
    // Step 1: getFile
    let gfr: serde_json::Value = client
        .get(format!("https://api.telegram.org/bot{token}/getFile"))
        .query(&[("file_id", file_id)])
        .send()
        .await
        .ok()?
        .json()
        .await
        .ok()?;
    let file_path = gfr["result"]["file_path"].as_str()?;
    // Step 2: download
    let bytes = client
        .get(format!("https://api.telegram.org/file/bot{token}/{file_path}"))
        .send()
        .await
        .ok()?
        .bytes()
        .await
        .ok()?;
    // Step 3: save to temp
    let tmp = tempfile::Builder::new()
        .suffix(&format!(".{}", ext))
        .tempfile()
        .ok()?;
    let (mut file, path) = tmp.keep().ok()?;
    use std::io::Write;
    file.write_all(&bytes).ok()?;
    file.flush().ok()?;
    Some(path)
}

async fn send_chat_action(client: &reqwest::Client, token: &str, chat_id: i64, action: &str) {
    let result = client
        .post(format!(
            "https://api.telegram.org/bot{token}/sendChatAction"
        ))
        .json(&serde_json::json!({ "chat_id": chat_id, "action": action }))
        .send()
        .await;
    if let Err(e) = result {
        eprintln!("ag-api-telegram: sendChatAction failed (chat {chat_id}): {e}");
    }
}

async fn write_access_request(user_id: i64, chat_id: i64, chat_type: &str, agent: &str, message: &str) {
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp"));
    let dir = home
        .join(".agentgraph")
        .join("requests");
    let _ = tokio::fs::create_dir_all(&dir).await;
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let path = dir.join(format!("{ts}-{user_id}.json"));
    let payload = serde_json::json!({
        "user_id": user_id,
        "chat_id": chat_id,
        "chat_type": chat_type,
        "agent": agent,
        "message": message,
        "timestamp_ms": ts,
    });
    if let Ok(json) = serde_json::to_string_pretty(&payload) {
        let _ = tokio::fs::write(&path, &json).await;
        eprintln!(
            "ag-api-telegram: wrote access request for user {user_id} to {}",
            path.display()
        );
    }
}

// ── Command handlers ───────────────────────────────────────────────────────

async fn handle_command(
    state: &BotState,
    client: &reqwest::Client,
    chat_id: i64,
    chat_type: &str,
    user_id: i64,
    cmd: &str,
) {
    let token = &state.tg.bot_token;
    let first_word = cmd.split_whitespace().next().unwrap_or("");

    let response = match first_word {
        "/start" => format!(
            "🤖 AgentGraph bot ready.\nAgent: `{}`\nType /help for commands.",
            state.agent_for(chat_id, chat_type).await
        ),
        "/help" => concat!(
            "**Commands**\n",
            "/tree — Show conversation tree\n",
            "/back — Roll back one turn\n",
            "/retry — Regenerate last response\n",
            "/branches — Show diverging paths\n",
            "/system — Read system directory\n",
            "/be <text> — Add to system dir\n",
            "/reset — Reset active tree (keeps persisted archive)\n",
            "/persist — Save active branch to disk (privileged)\n",
            "/delete — Wipe active tree + persisted archive (privileged)\n",
            "/status — Show leader status (privileged)\n",
            "/run <agent> [msg] — Trigger an agent turn (privileged)\n",
            "/stop <agent> — Stop a running agent (privileged)\n",
            "/reload — Reload leader config (privileged)\n",
            "/config — Show agent config\n",
            "/help — This help"
        )
        .to_string(),
        "/tree" => state.show_tree(chat_id).await,
        "/back" => {
            let mut chats = state.chats.lock().await;
            if let Some(chat) = chats.get_mut(&chat_id) {
                // Pop the last assistant+user pair (if assistant was appended)
                let len = chat.history.len();
                if len >= 2 {
                    chat.history.truncate(len - 2);
                    format!(
                        "Rolled back to turn {}. Send a new message to continue.",
                        chat.history.len() / 2
                    )
                } else if len == 1 {
                    chat.history.clear();
                    "Rolled back to start. Send a new message.".into()
                } else {
                    "Nothing to roll back — history is empty.".into()
                }
            } else {
                "No conversation to roll back.".into()
            }
        }
        "/retry" => {
            let mut chats = state.chats.lock().await;
            if let Some(chat) = chats.get_mut(&chat_id) {
                // Pop just the last assistant response so the last user
                // message triggers a fresh assistant generation.
                if chat.history.last().map(|s| s.role.as_str()) == Some("assistant") {
                    chat.history.pop();
                }
                "Regenerating — send any message to trigger a new reply.".into()
            } else {
                "No conversation to retry.".into()
            }
        }
        "/branches" => {
            // Query the session tree for branches from root.
            // Future: could use current position hash from a prior SessionBuild.
            let session_id = format!("tg-{chat_id}");
            match ipc_send(
                &state.socket_path,
                &Command::SessionListChildren {
                    session_id,
                    hash: String::new(), // query root branches
                },
            )
            .await
            {
                Ok(resp) if resp.ok => {
                    if let Some(data) = &resp.data {
                        match serde_json::from_str::<Vec<Vec<String>>>(data) {
                            Ok(children) if !children.is_empty() => {
                                let mut out = String::from("**Branches**\n\n");
                                for c in &children {
                                    let label = if c.len() >= 2 {
                                        format!(
                                            "`{}` {}: …",
                                            c[0].chars().take(8).collect::<String>(),
                                            c[1]
                                        )
                                    } else {
                                        format!("`{}`", c[0].chars().take(8).collect::<String>())
                                    };
                                    out.push_str(&format!("- {label}\n"));
                                }
                                out
                            }
                            _ => "No branches in the tree yet.".into(),
                        }
                    } else {
                        "No branch data.".into()
                    }
                }
                _ => "No branches found.".into(),
            }
        }
        "/reset" => {
            let mut chats = state.chats.lock().await;
            chats.remove(&chat_id);
            let _ = ipc_send(
                &state.socket_path,
                &Command::SessionReset {
                    session_id: format!("tg-{chat_id}"),
                },
            )
            .await;
            "Conversation reset to fresh state. Persisted history (if any) is still accessible via /branches.".into()
        }
        "/persist" => {
            let request_msg = cmd.strip_prefix("/persist").unwrap_or("").trim();
            if state.is_privileged(user_id) {
                let agent = state.agent_for(chat_id, chat_type).await;
                let hist = state
                    .chats
                    .lock()
                    .await
                    .get(&chat_id)
                    .map(|c| c.history.clone())
                    .unwrap_or_default();
                let build_resp = ipc_send(
                    &state.socket_path,
                    &Command::SessionBuild {
                        session_id: format!("tg-{chat_id}"),
                        steps: hist,
                        agent_name: None,
                    },
                )
                .await;
                match build_resp {
                    Ok(resp) if resp.ok => {
                        if let Some(data) = &resp.data {
                            if let Ok(v) = serde_json::from_str::<serde_json::Value>(data) {
                                let current_hash = v["current_hash"]
                                    .as_str()
                                    .unwrap_or("")
                                    .to_string();
                                if current_hash.is_empty() {
                                    "Nothing to persist — no conversation yet.".into()
                                } else {
                                    match ipc_send(
                                        &state.socket_path,
                                        &Command::SessionPersist {
                                            session_id: format!("tg-{chat_id}"),
                                            agent,
                                            current_hash,
                                        },
                                    )
                                    .await
                                    {
                                        Ok(r) if r.ok => "Session persisted. Future messages continue from this point; persisted history survives leader restart.".into(),
                                        Ok(r) => format!("Persist failed: {}", r.error.unwrap_or_else(|| "unknown error".into())),
                                        Err(e) => format!("IPC error: {e}"),
                                    }
                                }
                            } else {
                                "Failed to parse session state.".into()
                            }
                        } else {
                            "No session data.".into()
                        }
                    }
                    Ok(_) => "Failed to build conversation state.".into(),
                    Err(e) => format!("IPC error: {e}"),
                }
            } else if !request_msg.is_empty() {
                // Non-privileged user with a message — write a request
                // file so the admin can review and optionally add them.
                write_access_request(user_id, chat_id, chat_type, &state.tg.default_agent, request_msg).await;
                "Your access request has been saved. An admin will review it.".into()
            } else {
                "Access denied — persist is a privileged command. Use /persist <message> to request access.".into()
            }
        }
"/delete" => {
            if !state.is_privileged(user_id) {
                "Access denied — delete is a privileged command.".into()
            } else {
                let mut chats = state.chats.lock().await;
                chats.remove(&chat_id);
                let _ = ipc_send(
                    &state.socket_path,
                    &Command::SessionDeletePersisted {
                        session_id: format!("tg-{chat_id}"),
                    },
                )
                .await;
                "Session fully deleted. Persisted history removed.".into()
            }
        }
        "/status" => {
            if !state.is_privileged(user_id) {
                "Access denied — status is a privileged command.".into()
            } else {
                match ipc_send_raw(&state.socket_path, &Command::Status).await {
                    Ok(text) => text,
                    Err(e) => format!("IPC error: {e}"),
                }
            }
        }
        "/reload" => {
            if !state.is_privileged(user_id) {
                "Access denied — reload is a privileged command.".into()
            } else {
                match ipc_send(&state.socket_path, &Command::ReloadConfig).await {
                    Ok(resp) if resp.ok => {
                        resp.data.unwrap_or_else(|| "Config reloaded.".into())
                    }
                    Ok(resp) => resp.error.unwrap_or_else(|| "error".into()),
                    Err(e) => format!("IPC error: {e}"),
                }
            }
        }
        "/config" => {
            let agent = state.agent_for(chat_id, chat_type).await;
            format!(
                "**Session config**\nChat ID: `{chat_id}`\nType: `{chat_type}`\nAgent: `{agent}`\nPrivileged: `{}`",
                state.is_privileged(user_id)
            )
        }
        "/run" => {
            if !state.is_privileged(user_id) {
                "Access denied — run is a privileged command.".into()
            } else {
                let args = cmd.strip_prefix("/run").unwrap_or("").trim();
                if args.is_empty() {
                    "Usage: /run <agent_name> [message]".into()
                } else {
                    let (agent_name, message) = args.split_once(' ')
                        .map(|(a, m)| (a.to_string(), Some(m.to_string())))
                        .unwrap_or_else(|| (args.to_string(), None));
                    match ipc_send_raw(
                        &state.socket_path,
                        &Command::RunAgent(agent_name.clone(), message, false),
                    )
                    .await
                    {
                        Ok(text) => {
                            if text.is_empty() {
                                format!("Agent `{agent_name}` triggered.")
                            } else {
                                text
                            }
                        }
                        Err(e) => format!("IPC error: {e}"),
                    }
                }
            }
        }
        "/stop" => {
            if !state.is_privileged(user_id) {
                "Access denied — stop is a privileged command.".into()
            } else {
                let name = cmd.strip_prefix("/stop").unwrap_or("").trim();
                if name.is_empty() {
                    "Usage: /stop <agent_name>".into()
                } else {
                    match ipc_send_raw(
                        &state.socket_path,
                        &Command::StopAgent(name.to_string()),
                    )
                    .await
                    {
                        Ok(text) => text,
                        Err(e) => format!("IPC error: {e}"),
                    }
                }
            }
        }
        "/be" => {
            if !state.is_privileged(user_id) {
                "Access denied — system write is a privileged command.".into()
            } else {
                let content = cmd.strip_prefix("/be").unwrap_or("");
                if content.is_empty() {
                    "Usage: /be <text>\nAdds a system prompt to this chat's history.".into()
                } else {
                    let agent = state.agent_for(chat_id, chat_type).await;
                    let mut chats = state.chats.lock().await;
                    let chat = chats.entry(chat_id).or_insert_with(|| ChatState {
                        history: Vec::new(),
                        agent: agent.clone(),
                    });
                    chat.history.push(SessionStep {
                        role: "system".to_string(),
                        content: content.to_string(),
                        media: Vec::new(),
                    });
                    "System message added.".into()
                }
            }
        }
        _ => "Unknown command. Type /help for available commands.".into(),
    };

    send_message_telegram(client, token, chat_id, &response).await;
}

// ── Tree display helper ──────────────────────────────────────────────────

impl BotState {
    async fn show_tree(&self, chat_id: i64) -> String {
        let hist = self
            .chats
            .lock()
            .await
            .get(&chat_id)
            .map(|c| c.history.clone())
            .unwrap_or_default();
        if hist.is_empty() {
            return "No conversation history yet. Send a message to start.".into();
        }
        let mut out = String::from("**Conversation tree**\n\n");
        for (i, step) in hist.iter().enumerate() {
            let icon = match step.role.as_str() {
                "user" => "👤",
                "assistant" => "🤖",
                "system" => "⚙️",
                _ => "❓",
            };
            let snippet: String = step.content.chars().take(80).collect();
            out.push_str(&format!("{i}. {icon} {snippet}\n"));
        }
        out.push_str(&format!(
            "\nUse /back to roll back, /branches to see forks, /retry to regenerate."
        ));
        out
    }
}

impl BotState {
    /// Whether the given user_id is allowed to run privileged commands.
    fn is_privileged(&self, user_id: i64) -> bool {
        if self.tg.privileged_users.is_empty() {
            if self.tg.allowed_users.is_empty() {
                true
            } else {
                self.tg.allowed_users.contains(&user_id)
            }
        } else {
            self.tg.privileged_users.contains(&user_id)
        }
    }

    /// Resolve which agent to use for a chat, respecting per-user, per-group,
    /// and per-channel overrides (in that priority order). Falls back to
    /// `default_agent`.
    async fn agent_for(&self, chat_id: i64, chat_type: &str) -> String {
        let id_str = chat_id.to_string();
        // Check user override
        if let Some(agent) = self.tg.user_agents.get(&id_str) {
            return agent.clone();
        }
        // Check group/channel override based on chat type
        match chat_type {
            "group" | "supergroup" => {
                if let Some(agent) = self.tg.group_agents.get(&id_str) {
                    return agent.clone();
                }
            }
            "channel" => {
                if let Some(agent) = self.tg.channel_agents.get(&id_str) {
                    return agent.clone();
                }
            }
            _ => {}
        }
        self.tg.default_agent.clone()
    }
}

// ── Main bot loop ──────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    let tg: TelegramConfig = serde_yaml::from_str(&cli.section).unwrap_or_else(|e| {
        eprintln!("ag-api-telegram: failed to parse config: {e}");
        std::process::exit(1);
    });

    if !tg.enabled {
        eprintln!("ag-api-telegram: enabled=false, exiting");
        return;
    }

    if find_leader_socket().await.is_none() {
        eprintln!("ag-api-telegram: no leader socket found, exiting");
        std::process::exit(1);
    }

    let state = Arc::new(BotState {
        socket_path: cli.socket.clone(),
        tg,
        chats: Mutex::new(HashMap::new()),
    });

    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(10))
        .timeout(std::time::Duration::from_secs(60))
        .build()
        .expect("failed to build HTTP client");
    let token = &state.tg.bot_token.clone();
    let base_url = format!("https://api.telegram.org/bot{token}");
    let mut offset: i64 = 0;

    println!(
        "ag-api-telegram: bot starting (agent: {})",
        state.tg.default_agent
    );

    loop {
        let resp = match client
            .get(format!("{base_url}/getUpdates"))
            .query(&[
                ("offset", offset.to_string()),
                ("timeout", "30".to_string()),
                ("allowed_updates", r#"["message"]"#.to_string()),
            ])
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => {
                eprintln!("ag-api-telegram: getUpdates error: {e}");
                tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                continue;
            }
        };

        let body: serde_json::Value = match resp.json().await {
            Ok(b) => b,
            Err(e) => {
                eprintln!("ag-api-telegram: JSON parse error: {e}");
                continue;
            }
        };

        if let Some(results) = body.get("result").and_then(|r| r.as_array()) {
            for update in results {
                offset = update["update_id"].as_i64().unwrap_or(offset) + 1;

                let msg = match update.get("message") {
                    Some(m) => m,
                    None => continue,
                };

                let chat_id = msg["chat"]["id"].as_i64().unwrap_or(0);
                let chat_type = msg["chat"]["type"].as_str().unwrap_or("private");
                let text = msg["text"].as_str().or_else(|| msg["caption"].as_str()).unwrap_or("");
                let user_id = msg["from"]["id"].as_i64().unwrap_or(0);

                // Download any media attached to this message
                let media_items = extract_media(msg);
                let mut media_paths: Vec<String> = Vec::new();
                for (file_id, _kind, ext) in &media_items {
                    if let Some(path) =
                        download_telegram_media(&client, token, file_id, ext).await
                    {
                        media_paths.push(path.to_string_lossy().to_string());
                    }
                }

                // If we have media but no text, synthesize a fallback caption
                let effective_text = if text.is_empty() && !media_paths.is_empty() {
                    format!(
                        "[{} media file{} attached: {}]",
                        media_paths.len(),
                        if media_paths.len() == 1 { "" } else { "s" },
                        media_paths
                            .iter()
                            .map(|p| {
                                let p = std::path::Path::new(p);
                                p.file_name().unwrap_or_default().to_string_lossy()
                            })
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                } else {
                    text.to_string()
                };

                // Access control
                if !state.tg.allowed_users.is_empty() && !state.tg.allowed_users.contains(&user_id)
                {
                    send_message_telegram(&client, token, chat_id, "Access denied.").await;
                    continue;
                }

                // Commands
                if text.starts_with('/') {
                    handle_command(&state, &client, chat_id, chat_type, user_id, text).await;
                    continue;
                }

                // Chat message → session inference
                let agent = state.agent_for(chat_id, chat_type).await;

                // Update local history
                let steps = {
                    let mut chats = state.chats.lock().await;
                    let chat = chats.entry(chat_id).or_insert_with(|| ChatState {
                        history: Vec::new(),
                        agent: agent.clone(),
                    });
                    chat.agent = agent.clone();
                    chat.history.push(SessionStep {
                        role: "user".to_string(),
                        content: effective_text.clone(),
                        media: media_paths
                            .iter()
                            .map(|p| p.to_string())
                            .collect(),
                    });
                    chat.history.clone()
                };

                send_chat_action(&client, token, chat_id, "typing").await;

                let poll_ms = state.tg.stream_poll_interval_ms;
                let use_stream = poll_ms > 0;

                match ipc_send(
                    &cli.socket,
                    &Command::SessionChat {
                        session_id: format!("tg-{chat_id}"),
                        steps: steps.clone(),
                        model: agent,
                        stream: use_stream,
                    },
                )
                .await
                {
                    Ok(resp) => {
                        if !resp.ok {
                            let err = resp.error.unwrap_or_else(|| "unknown error".into());
                            send_message_telegram(&client, token, chat_id, &format!("Error: {err}")).await;
                            continue;
                        }
                        // Parse the SessionChatResponse from data
                        let data = match &resp.data {
                            Some(d) => d,
                            None => {
                                send_message_telegram(&client, token, chat_id, "_(empty response)_").await;
                                continue;
                            }
                        };
                        let sc_resp: SessionChatResponse =
                            match serde_json::from_str(data) {
                                Ok(r) => r,
                                Err(e) => {
send_message_telegram(
                                        &client,
                                        token,
                                        chat_id,
                                        &format!("Parse error: {e}"),
                                    )
                                    .await;
                                    continue;
                                }
                            };

                        if let Some(stream_dir) = &sc_resp.stream_path {
                            // ── Streaming path ──────────────────────────
                            let message_id = send_initial_message(
                                &client, token, chat_id, "⏳",
                            )
                            .await
                            .unwrap_or(0);
                            if message_id == 0 {
                                eprintln!("ag-api-telegram: failed to get message_id for streaming (chat {chat_id})");
                                continue;
                            }
                            eprintln!(
                                "ag-api-telegram: streaming started (chat {chat_id}, msg {message_id}, dir {stream_dir})"
                            );
                            let mut last_content = String::new();
                            let stream_dir = stream_dir.clone();
                            let mut reads_without_content: u32 = 0;
                            loop {
                                tokio::time::sleep(Duration::from_millis(poll_ms)).await;

                                // Check for .done marker
                                let done_path = PathBuf::from(&stream_dir).join(".done");
                                let is_done =
                                    tokio::fs::metadata(&done_path).await.is_ok();

                                // Read latest stream content
                                match read_stream_content(&stream_dir).await {
                                    Some(content) => {
                                        if content != last_content {
                                            eprintln!(
                                                "ag-api-telegram: streaming update (chat {chat_id}, {} chars)",
                                                content.len(),
                                            );
                                            edit_message_telegram(
                                                &client,
                                                token,
                                                chat_id,
                                                message_id,
                                                &content,
                                            )
                                            .await;
                                            last_content = content;
                                        }
                                        reads_without_content = 0;
                                    }
                                    None => {
                                        reads_without_content += 1;
                                        if reads_without_content == 1 {
                                            eprintln!(
                                                "ag-api-telegram: stream file not yet visible (chat {chat_id})"
                                            );
                                        }
                                    }
                                }

                                if is_done {
                                    eprintln!(
                                        "ag-api-telegram: streaming done (chat {chat_id}, final {} chars)",
                                        last_content.len(),
                                    );
                                    break;
                                }
                            }
                            // Store in history
                            let mut chats = state.chats.lock().await;
                            if let Some(chat) = chats.get_mut(&chat_id) {
                                if !last_content.is_empty() {
                                    chat.history.push(SessionStep {
                                        role: "assistant".to_string(),
                                        content: last_content,
                                        media: Vec::new(),
                                    });
                                }
                            }
                        } else if let Some(reply) = sc_resp.content {
                            // ── Blocking path ───────────────────────────
                            send_message_telegram(&client, token, chat_id, &reply).await;
                            let mut chats = state.chats.lock().await;
                            if let Some(chat) = chats.get_mut(&chat_id) {
                                chat.history.push(SessionStep {
                                    role: "assistant".to_string(),
                                    content: reply,
                                    media: Vec::new(),
                                });
                            }
                        } else {
                            send_message_telegram(&client, token, chat_id, "_(empty response)_").await;
                        }
                    }
                    Err(e) => {
                        send_message_telegram(&client, token, chat_id, &format!("IPC error: {e}")).await;
                    }
                }
            }
        }
    }
}
