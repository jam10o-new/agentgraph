use ag_tool_common::{describe, guidance, has_flag};
use ag_ipc::Command;
use ag_config::AgentConfig;
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;

#[derive(Debug)]
struct Args {
    name: String,
    inputs: Vec<String>,
    output: Vec<String>,
    stream_output: Option<String>,
    tool_output: Option<String>,
    system: Vec<String>,
    model: String,
    history_limit: Option<usize>,
    realtime_audio: bool,
    prompt: Option<String>,
    tools_enabled: bool,
    excluded_from_summary: Vec<String>,
    context_checkpoint_limit: Option<usize>,
}

fn parse_args() -> Result<Args, String> {
    let raw: serde_json::Value = ag_tool_common::read_args()?;
    let output = match raw.get("output") {
        Some(serde_json::Value::String(s)) => vec![s.clone()],
        Some(serde_json::Value::Array(arr)) => arr
            .iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect(),
        _ => vec![],
    };

    fn arr(raw: &serde_json::Value, key: &str) -> Vec<String> {
        raw.get(key)
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default()
    }

    fn str_opt(raw: &serde_json::Value, key: &str) -> Option<String> {
        raw.get(key).and_then(|v| v.as_str()).map(String::from)
    }

    fn int_opt(raw: &serde_json::Value, key: &str) -> Option<usize> {
        raw.get(key).and_then(|v| v.as_u64()).map(|u| u as usize)
    }

    fn bool_def(raw: &serde_json::Value, key: &str, def: bool) -> bool {
        raw.get(key).and_then(|v| v.as_bool()).unwrap_or(def)
    }

    Ok(Args {
        name: str_opt(&raw, "name").unwrap_or_default(),
        inputs: arr(&raw, "inputs"),
        output,
        stream_output: str_opt(&raw, "stream_output"),
        tool_output: str_opt(&raw, "tool_output"),
        system: arr(&raw, "system"),
        model: str_opt(&raw, "model").unwrap_or_else(|| "primary".into()),
        history_limit: int_opt(&raw, "history_limit"),
        realtime_audio: bool_def(&raw, "realtime_audio", true),
        prompt: str_opt(&raw, "prompt"),
        tools_enabled: bool_def(&raw, "tools_enabled", true),
        excluded_from_summary: arr(&raw, "excluded_from_summary"),
        context_checkpoint_limit: int_opt(&raw, "context_checkpoint_limit"),
    })
}

#[tokio::main]
async fn main() {
    if has_flag("--describe") {
        describe(
            "spawn_new_agent",
            "Dynamically spawn a new agent with the given configuration.",
            json!({
                "type": "object",
                "properties": {
                    "name": { "type": "string", "description": "Name for the new agent." },
                    "inputs": { "type": "array", "items": { "type": "string" }, "description": "Input directories." },
                    "output": { "type": "array", "items": { "type": "string" }, "description": "Output directories (legacy single string also accepted)." },
                    "stream_output": { "type": "string", "nullable": true, "description": "Streaming output directory." },
                    "tool_output": { "type": "string", "nullable": true, "description": "Tool output directory." },
                    "system": { "type": "array", "items": { "type": "string" }, "description": "System prompt directories." },
                    "model": { "type": "string", "description": "Model name (defaults to 'primary')." },
                    "history_limit": { "type": "integer", "nullable": true },
                    "realtime_audio": { "type": "boolean" },
                    "prompt": { "type": "string", "nullable": true, "description": "Optional system prompt suffix." },
                    "tools_enabled": { "type": "boolean", "description": "Whether the new agent can use tools." },
                    "excluded_from_summary": { "type": "array", "items": { "type": "string" } },
                    "context_checkpoint_limit": { "type": "integer", "nullable": true }
                }
            }),
        );
        return;
    }

    if has_flag("--help") {
        guidance(
            "When using spawn_new_agent:\n\
             - Creates a new agent that runs in parallel with the current one.\n\
             - The new agent watches its own input directories and runs independently.\n\
             - Use output directories to wire agents together (colony pattern).\n\
             - Provide a unique name to avoid conflicts with existing agents.\n\
             - The new agent inherits the global model configuration.",
        );
        return;
    }

    // ── Execute ──
    let Args {
        name, inputs, output, stream_output, tool_output, system, model,
        history_limit, realtime_audio, prompt, tools_enabled,
        excluded_from_summary, context_checkpoint_limit,
    } = parse_args().unwrap_or_else(|e| {
        eprintln!("Error: {e}");
        std::process::exit(1);
    });

    let config = AgentConfig {
        inputs,
        output,
        stream_output,
        tool_output,
        system,
        model,
        history_limit,
        realtime_audio,
        allowed_extensions: vec![],
        prompt,
        sampling: Default::default(),
        compression: Default::default(),
        context_checkpoint_limit,
        compression_db_path: None,
        excluded_from_summary,
        tools_enabled,
        consume_tool_calls: false,
        enable_thinking: false,
        inference_retries: 3,
        enable_oom_recovery: true,
        inference_retry_delay_ms: 500,
    };

    let cmd = Command::SpawnAgent { name, config };

    // Find leader socket and send IPC command
    let socket_path = match ag_utils::find_leader_socket().await {
        Some(p) => p,
        None => {
            println!("Error: leader process not found");
            return;
        }
    };

    let mut stream = match UnixStream::connect(&socket_path).await {
        Ok(s) => s,
        Err(e) => {
            println!("Error: failed to connect to leader: {e}");
            return;
        }
    };

    match serde_json::to_vec(&cmd) {
        Ok(payload) => {
            if let Err(e) = stream.write_all(&payload).await {
                println!("Error: failed to send command: {e}");
                return;
            }
        }
        Err(e) => {
            println!("Error: failed to serialize command: {e}");
            return;
        }
    }

    let _ = stream.flush().await;
    let _ = stream.shutdown().await;

    let mut resp = String::new();
    let _ = stream.read_to_string(&mut resp).await;

    match serde_json::from_str::<ag_ipc::IpcResponse>(&resp) {
        Ok(response) => {
            if response.ok {
                if let Some(data) = response.data {
                    println!("{data}");
                } else {
                    println!("Agent spawned successfully.");
                }
            } else {
                println!(
                    "Error: {}",
                    response.error.unwrap_or_else(|| "unknown error".into())
                );
            }
        }
        Err(e) => {
            println!("Error: {e}\nRaw response: {resp}");
        }
    }
}