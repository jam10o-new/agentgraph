//! ag-api-telegram — Telegram bot frontend for AgentGraph.
//!
//! Reads its own `api-telegram` config section, connects to the leader's
//! Unix socket for session operations, and exposes AgentGraph agents
//! through a persistent Telegram bot with /commands.

use agentgraph::ipc::{Command, IpcResponse, SessionStep};
use agentgraph::find_leader_socket;
use clap::Parser;
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::Arc;
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
    allowed_users: Vec<i64>,
}

fn default_agent() -> String { "api".to_string() }

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
        .await.map_err(|e| format!("connect: {e}"))?;
    let payload = serde_json::to_vec(cmd).map_err(|e| format!("serialize: {e}"))?;
    stream.write_all(&payload).await.map_err(|e| format!("write: {e}"))?;
    stream.flush().await.map_err(|e| format!("flush: {e}"))?;
    stream.shutdown().await.map_err(|e| format!("shutdown: {e}"))?;
    let mut buf = String::new();
    stream.read_to_string(&mut buf).await.map_err(|e| format!("read: {e}"))?;
    serde_json::from_str::<IpcResponse>(&buf)
        .map_err(|e| format!("deserialize: {e} (raw: {buf})"))
}

// ── Telegram API helpers ────────────────────────────────────────────────────

async fn send_message(client: &reqwest::Client, token: &str, chat_id: i64, text: &str) {
    let _ = client
        .post(format!("https://api.telegram.org/bot{token}/sendMessage"))
        .json(&serde_json::json!({
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "Markdown",
        }))
        .send().await;
}

async fn send_chat_action(client: &reqwest::Client, token: &str, chat_id: i64, action: &str) {
    let _ = client
        .post(format!("https://api.telegram.org/bot{token}/sendChatAction"))
        .json(&serde_json::json!({ "chat_id": chat_id, "action": action }))
        .send().await;
}

// ── Command handlers ───────────────────────────────────────────────────────

async fn handle_command(
    state: &BotState, client: &reqwest::Client,
    chat_id: i64, cmd: &str,
) {
    let token = &state.tg.bot_token;

    let response = match cmd.split_whitespace().next().unwrap_or("") {
        "/start" => format!(
            "🤖 AgentGraph bot ready.\nAgent: `{}`\nType /help for commands.",
            state.agent_for(chat_id).await
        ),
        "/help" => concat!(
            "**Commands**\n",
            "/tree — Show conversation tree\n",
            "/system — Read system directory\n",
            "/reset — Reset conversation\n",
            "/config — Show agent config\n",
            "/help — This help",
        ).to_string(),
        "/tree" => {
            let hist = state.chats.lock().await
                .get(&chat_id)
                .map(|c| c.history.clone())
                .unwrap_or_default();
            if hist.is_empty() {
                "No conversation history yet. Send a message to start.".into()
            } else {
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
                out
            }
        },
        "/reset" => {
            let mut chats = state.chats.lock().await;
            chats.remove(&chat_id);
            // Also tell leader to delete the session
            let _ = ipc_send(&state.socket_path, &Command::SessionDelete {
                session_id: format!("tg-{chat_id}"),
            }).await;
            "Conversation reset. Start fresh!".into()
        },
        "/config" => {
            let agent = state.agent_for(chat_id).await;
            format!("**Session config**\nChat ID: `{chat_id}`\nAgent: `{agent}`")
        },
        "/system" => {
            let hist = state.chats.lock().await
                .get(&chat_id)
                .map(|c| c.history.clone())
                .unwrap_or_default();
            if let Ok(resp) = ipc_send(&state.socket_path, &Command::SessionBuild {
                session_id: format!("tg-{chat_id}"),
                steps: hist,
            }).await {
                if resp.ok {
                    if let Some(data) = &resp.data {
                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(data) {
                            let sys_msgs = v["system_msgs"].as_array()
                                .map(|a| a.iter().filter_map(|s| s.as_str()).collect::<Vec<_>>())
                                .unwrap_or_default();
                            if sys_msgs.is_empty() {
                                "**System dir** — empty (no system messages)".into()
                            } else {
                                let mut out = String::from("**System dir**\n\n");
                                for (i, msg) in sys_msgs.iter().enumerate() {
                                    let snippet: String = msg.chars().take(200).collect();
                                    out.push_str(&format!("`{i}`: {snippet}\n\n"));
                                }
                                out
                            }
                        } else {
                            "Failed to parse session state.".into()
                        }
                    } else {
                        "No session data.".into()
                    }
                } else {
                    resp.error.unwrap_or_else(|| "session error".into())
                }
            } else {
                "Failed to reach leader.".into()
            }
        },
        _ => "Unknown command. Type /help for available commands.".into(),
    };

    send_message(client, token, chat_id, &response).await;
}

impl BotState {
    async fn agent_for(&self, chat_id: i64) -> String {
        let user_id = chat_id.to_string();
        self.tg.user_agents.get(&user_id)
            .cloned()
            .unwrap_or_else(|| self.tg.default_agent.clone())
    }
}

// ── Main bot loop ──────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    let tg: TelegramConfig = serde_json::from_str(&cli.section)
        .unwrap_or_else(|e| {
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

    let client = reqwest::Client::new();
    let token = &state.tg.bot_token.clone();
    let base_url = format!("https://api.telegram.org/bot{token}");
    let mut offset: i64 = 0;

    println!("ag-api-telegram: bot starting (agent: {})", state.tg.default_agent);

    loop {
        let resp = match client
            .get(format!("{base_url}/getUpdates"))
            .query(&[
                ("offset", offset.to_string()),
                ("timeout", "30".to_string()),
                ("allowed_updates", r#"["message"]"#.to_string()),
            ])
            .send().await
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
                let text = msg["text"].as_str().unwrap_or("");
                let user_id = msg["from"]["id"].as_i64().unwrap_or(0);

                // Access control
                if !state.tg.allowed_users.is_empty() && !state.tg.allowed_users.contains(&user_id) {
                    send_message(&client, token, chat_id, "Access denied.").await;
                    continue;
                }

                // Commands
                if text.starts_with('/') {
                    handle_command(&state, &client, chat_id, text).await;
                    continue;
                }

                // Chat message → session inference
                let agent = state.agent_for(chat_id).await;

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
                        content: text.to_string(),
                    });
                    chat.history.clone()
                };

                send_chat_action(&client, token, chat_id, "typing").await;

                match ipc_send(&cli.socket, &Command::SessionChat {
                    session_id: format!("tg-{chat_id}"),
                    steps: steps.clone(),
                    model: agent,
                    stream: false,
                }).await {
                    Ok(resp) => {
                        if resp.ok {
                            let reply = resp.data.unwrap_or_else(|| "_(empty response)_".into());
                            send_message(&client, token, chat_id, &reply).await;
                            // Store assistant response in local history
                            let mut chats = state.chats.lock().await;
                            if let Some(chat) = chats.get_mut(&chat_id) {
                                chat.history.push(SessionStep {
                                    role: "assistant".to_string(),
                                    content: reply,
                                });
                            }
                        } else {
                            let err = resp.error.unwrap_or_else(|| "unknown error".into());
                            send_message(&client, token, chat_id, &format!("Error: {err}")).await;
                        }
                    }
                    Err(e) => {
                        send_message(&client, token, chat_id, &format!("IPC error: {e}")).await;
                    }
                }
            }
        }
    }
}
