//! Shared utilities for AgentGraph inference provider plugin binaries.
//!
//! ## Wire protocol (stdin → JSON, stdout → streaming lines)
//!
//! The provider binary is spawned per-request.  A single JSON request is
//! written to its stdin (then closed), and the response is streamed back
//! as type-prefixed lines on stdout.
//!
//! ### Request (stdin, one JSON object)
//!
//! ```json
//! {"type":"chat","request_id":0,"messages":[...],"tools":null,"enable_thinking":false}
//! ```
//!
//! Multi-modal data is referenced by **file path** — no inline base64.
//!
//! ### Response (stdout, line-delimited events)
//!
//! | Prefix | Payload          | Meaning                            |
//! |--------|------------------|------------------------------------|
//! | `T:`   | text             | Token chunk (streaming chat)       |
//! | `R:`   | text             | Reasoning / thinking token chunk   |
//! | `C:`   | JSON             | Tool call `{id,name,arguments}`    |
//! | `D`    |                  | Stream complete (normal)           |
//! | `X:`   | error message    | Fatal error                        |
//! | `E:`   | JSON             | Embedding result `{embedding:[..]}`|
//! | `K:`   | integer          | Token count                        |
//! | `H:`   | `true` or `false`| Health check result                |
//! | `I:`   | JSON             | Provider info `{name,model,...}`   |
//!
//! Convention: binary name `ag-provider-<name>` → config key `provider-<name>`.

use serde::{Deserialize, Serialize};
use std::io::{BufRead, Write};

// ── Wire protocol types ───────────────────────────────────────────────

/// Full request envelope written to stdin (single JSON object).
#[derive(Debug, Serialize, Deserialize)]
pub struct Request {
    #[serde(flatten)]
    pub kind: RequestKind,
    #[serde(default)]
    pub request_id: u64,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RequestKind {
    /// Streaming chat completion.
    Chat(ChatRequest),
    /// Embedding.
    Embed(EmbedRequest),
    /// Token count.
    CountTokens(CountTokensRequest),
    /// Health check (no payload).
    Health,
    /// Provider metadata (no payload).
    Info,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatRequest {
    pub messages: Vec<ChatMessage>,
    pub tools: Option<Vec<ToolDef>>,
    pub tool_choice: Option<serde_json::Value>,
    pub constraint: Option<Constraint>,
    pub enable_thinking: bool,
    pub model: Option<String>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
}

/// A single message in the conversation.
///
/// Modality data (images, audio, video) is referenced by **file path**
/// so the provider can read it directly — no inline base64 encoding.
#[derive(Debug, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    /// Text fragments and modality references.
    #[serde(default)]
    pub content: Vec<ContentPart>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
    Image { file_path: String },
    Audio { file_path: String },
    Video { file_path: String },
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Constraint {
    pub schema: Option<serde_json::Value>,
    pub regex: Option<String>,
    pub grammar: Option<String>,
}

// ── Sub-request types ─────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize)]
pub struct EmbedRequest {
    pub text: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CountTokensRequest {
    pub text: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct EmbedResponse {
    embedding: Vec<f32>,
}

// ── Response reading (for the agentgraph side) ────────────────────────

/// Events that the agentgraph side can receive from a provider plugin.
#[derive(Debug)]
pub enum ProviderEvent {
    /// Streaming text token.
    Chunk(String),
    /// Streaming reasoning/thinking token.
    Reasoning(String),
    /// Complete tool call.
    ToolCall(ToolCall),
    /// Stream finished normally.
    Done,
    /// Embedding result.
    Embedding(Vec<f32>),
    /// Token count.
    TokenCount(usize),
    /// Health check.
    Healthy(bool),
    /// Provider info.
    Info(InfoResponse),
    /// Fatal error.
    Error(String),
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InfoResponse {
    pub name: String,
    pub model: Option<String>,
    pub max_seq_len: Option<usize>,
    pub supports_tools: bool,
    pub supports_embeddings: bool,
    pub supports_constraints: bool,
    pub supports_modalities: Vec<String>,
}

/// Parse a single response line from the provider's stdout into an event.
/// Returns `None` for blank lines (which are ignored).
///
/// # Format
/// Each line is `<prefix><payload>` where prefix is one of `T:`, `R:`,
/// `C:`, `D`, `E:`, `K:`, `H:`, `I:`, `X:`.
pub fn parse_event_line(line: &str) -> Result<Option<ProviderEvent>, String> {
    let line = line.trim();
    if line.is_empty() {
        return Ok(None);
    }

    let (prefix, payload) = match line.split_once(':') {
        Some((p, rest)) => (p, rest),
        None => (line, ""),
    };

    match prefix {
        "T" => Ok(Some(ProviderEvent::Chunk(payload.to_string()))),
        "R" => Ok(Some(ProviderEvent::Reasoning(payload.to_string()))),
        "C" => {
            let tc: ToolCall =
                serde_json::from_str(payload).map_err(|e| format!("bad tool call: {e}"))?;
            Ok(Some(ProviderEvent::ToolCall(tc)))
        }
        "D" => Ok(Some(ProviderEvent::Done)),
        "E" => {
            let resp: EmbedResponse =
                serde_json::from_str(payload).map_err(|e| format!("bad embed: {e}"))?;
            Ok(Some(ProviderEvent::Embedding(resp.embedding)))
        }
        "K" => {
            let count: usize = payload
                .parse()
                .map_err(|e| format!("bad token count: {e}"))?;
            Ok(Some(ProviderEvent::TokenCount(count)))
        }
        "H" => match payload {
            "true" => Ok(Some(ProviderEvent::Healthy(true))),
            "false" => Ok(Some(ProviderEvent::Healthy(false))),
            other => Err(format!("bad health value: {other}")),
        },
        "I" => {
            let info: InfoResponse =
                serde_json::from_str(payload).map_err(|e| format!("bad info: {e}"))?;
            Ok(Some(ProviderEvent::Info(info)))
        }
        "X" => Ok(Some(ProviderEvent::Error(payload.to_string()))),
        other => Err(format!("unknown event prefix: {other}")),
    }
}

/// Read all events from the provider's stdout into a Vec.
/// Blocks until EOF or an `X:` (error) or `D` (done) event.
pub fn read_events(reader: impl BufRead) -> Result<Vec<ProviderEvent>, String> {
    let mut events = Vec::new();
    for line in reader.lines() {
        let line = line.map_err(|e| format!("read error: {e}"))?;
        if let Some(ev) = parse_event_line(&line)? {
            match &ev {
                ProviderEvent::Done | ProviderEvent::Error(_) => {
                    events.push(ev);
                    return Ok(events);
                }
                _ => events.push(ev),
            }
        }
    }
    Ok(events)
}

// ── Request writing (for the agentgraph side) ─────────────────────────

/// Write a JSON request to a writer (typically the provider's stdin).
pub fn write_request(request: &Request, writer: &mut impl Write) -> Result<(), String> {
    let json = serde_json::to_string(request).map_err(|e| format!("serialize: {e}"))?;
    writer
        .write_all(json.as_bytes())
        .map_err(|e| format!("write: {e}"))?;
    writer
        .flush()
        .map_err(|e| format!("flush: {e}"))
}

// ── Response writing (for the provider binary side) ───────────────────

/// Write a single event line to stdout.  Panics on write error
/// (provider-side convenience).
pub fn emit_event(event: &ProviderEvent) {
    let line = match event {
        ProviderEvent::Chunk(t) => format!("T:{t}\n"),
        ProviderEvent::Reasoning(t) => format!("R:{t}\n"),
        ProviderEvent::ToolCall(tc) => {
            let json = serde_json::to_string(tc).expect("tool call json");
            format!("C:{json}\n")
        }
        ProviderEvent::Done => "D\n".to_string(),
        ProviderEvent::Embedding(emb) => {
            let json = serde_json::to_string(&EmbedResponse {
                embedding: emb.clone(),
            })
            .expect("embed json");
            format!("E:{json}\n")
        }
        ProviderEvent::TokenCount(n) => format!("K:{n}\n"),
        ProviderEvent::Healthy(b) => format!("H:{b}\n"),
        ProviderEvent::Info(info) => {
            let json = serde_json::to_string(info).expect("info json");
            format!("I:{json}\n")
        }
        ProviderEvent::Error(msg) => {
            // Strip newlines from error messages to keep the protocol clean
            let clean = msg.replace('\n', " ");
            format!("X:{clean}\n")
        }
    };
    print!("{line}");
    std::io::stdout().flush().ok();
}

// ── CLI helpers ───────────────────────────────────────────────────────

/// Check whether a flag is present in the process arguments.
pub fn has_flag(flag: &str) -> bool {
    std::env::args().any(|a| a == flag)
}

/// Print a provider description JSON to stdout (for `--describe`).
pub fn describe(info: &InfoResponse) {
    println!(
        "{}",
        serde_json::to_string(info).expect("serialize provider info")
    );
}

/// Print free-text help text to stdout (for `--help`).
pub fn guidance(text: &str) {
    println!("{}", text);
}

/// Read a JSON [`Request`] from stdin.  Blocks until EOF.
pub fn read_request() -> Result<Request, String> {
    let mut input = String::new();
    std::io::Read::read_to_string(&mut std::io::stdin(), &mut input)
        .map_err(|e| format!("Failed to read stdin: {e}"))?;
    serde_json::from_str(&input).map_err(|e| format!("Failed to parse request: {e}"))
}
