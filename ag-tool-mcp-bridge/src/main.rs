use ag_tool_common::{describe, guidance, has_flag, read_args};
use serde::Deserialize;
use serde_json::{json, Value};
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;

#[derive(Deserialize)]
struct Args {
    command: String,
    #[serde(default)]
    args: Vec<String>,
    method: String,
    #[serde(default)]
    params: Option<Value>,
}

/// Read a single MCP message from `reader` according to the stdio framing.
/// Returns the JSON body as a String.
async fn read_mcp_message<R: tokio::io::AsyncRead + Unpin>(
    reader: &mut BufReader<R>,
) -> Result<String, String> {
    // Read Content-Length header line: "Content-Length: <number>\r\n"
    let mut header = String::new();
    reader
        .read_line(&mut header)
        .await
        .map_err(|e| format!("Failed to read Content-Length: {e}"))?;

    if !header.starts_with("Content-Length:") {
        return Err("Expected Content-Length header".into());
    }
    let len_str = header["Content-Length:".len()..].trim();
    let len: usize = len_str
        .parse()
        .map_err(|e| format!("Invalid Content-Length value: {e}"))?;

    // Read the empty line separating header and body (\r\n)
    let mut empty_line = String::new();
    reader
        .read_line(&mut empty_line)
        .await
        .map_err(|e| format!("Failed to read separator: {e}"))?;

    // Read exactly `len` bytes of the JSON body
    let mut body = vec![0u8; len];
    reader
        .read_exact(&mut body)
        .await
        .map_err(|e| format!("Failed to read body: {e}"))?;

    String::from_utf8(body).map_err(|e| format!("Invalid UTF-8 in body: {e}"))
}

/// Write an MCP message with Content-Length framing.
async fn write_mcp_message<W: tokio::io::AsyncWrite + Unpin>(
    writer: &mut W,
    json: &str,
) -> Result<(), String> {
    let content_length = json.len();
    let header = format!("Content-Length: {content_length}\r\n\r\n");
    writer
        .write_all(header.as_bytes())
        .await
        .map_err(|e| format!("Write error: {e}"))?;
    writer
        .write_all(json.as_bytes())
        .await
        .map_err(|e| format!("Write error: {e}"))?;
    writer
        .flush()
        .await
        .map_err(|e| format!("Flush error: {e}"))?;
    Ok(())
}

#[tokio::main]
async fn main() {
    if has_flag("--describe") {
        describe(
            "mcp_bridge",
            "Connect to an MCP server over stdio, send a JSON-RPC request, and return the response.",
            json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command to launch the MCP server."
                    },
                    "args": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Optional arguments for the server command."
                    },
                    "method": {
                        "type": "string",
                        "description": "MCP method to call, e.g. 'tools/list' or 'tools/call'."
                    },
                    "params": {
                        "type": "object",
                        "description": "Optional JSON object of parameters for the method."
                    }
                },
                "required": ["command", "method"]
            }),
        );
        return;
    }

    if has_flag("--help") {
        guidance(
            "When using mcp_bridge:\n\
             - Launches the given MCP server command as a child process.\n\
             - Performs the mandatory MCP initialize handshake.\n\
             - Sends the specified JSON-RPC method (with optional params).\n\
             - Prints the JSON result to stdout; errors are shown on stderr.\n\
             - The connection is closed immediately after the single request.\n\
             - Use to integrate external MCP‑based tools into your agent.\n\
             - Only stdio transport is supported; for SSE add a separate bridge.",
        );
        return;
    }

    let args: Args = read_args().unwrap_or_else(|e| {
        eprintln!("Error parsing arguments: {e}");
        std::process::exit(1);
    });

    // Spawn the MCP server process
    let mut child = Command::new(&args.command)
        .args(&args.args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .kill_on_drop(true)
        .spawn()
        .unwrap_or_else(|e| {
            eprintln!("Failed to spawn MCP server: {e}");
            std::process::exit(1);
        });

    let stdin = child.stdin.take().expect("failed to open stdin");
    let stdout = child.stdout.take().expect("failed to open stdout");

    let mut writer = tokio::io::BufWriter::new(stdin);
    let mut reader = BufReader::new(stdout);

    // ---- MCP Initialize Handshake ----
    let initialize_request = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},           // no special client capabilities
            "clientInfo": {
                "name": "ag_tool_mcp_bridge",
                "version": "0.1.0"
            }
        }
    });

    if let Err(e) = write_mcp_message(&mut writer, &initialize_request.to_string()).await {
        eprintln!("Failed to send initialize request: {e}");
        std::process::exit(1);
    }

    let init_response = read_mcp_message(&mut reader).await.unwrap_or_else(|e| {
        eprintln!("Failed to read initialize response: {e}");
        std::process::exit(1);
    });

    // Very brief check: we should have received a result (ignoring actual content)
    if let Ok(resp) = serde_json::from_str::<Value>(&init_response) {
        if resp.get("error").is_some() {
            eprintln!("Initialize error: {resp}");
            std::process::exit(1);
        }
    } else {
        eprintln!("Invalid JSON in initialize response");
        std::process::exit(1);
    }

    // Send the 'initialized' notification (no id)
    let initialized_notification = json!({
        "jsonrpc": "2.0",
        "method": "notifications/initialized"
    });
    if let Err(e) = write_mcp_message(&mut writer, &initialized_notification.to_string()).await {
        eprintln!("Failed to send initialized notification: {e}");
        std::process::exit(1);
    }

    // ---- Send the user's request ----
    let request_id = 2; // unique id (we only send one request)
    let request = json!({
        "jsonrpc": "2.0",
        "id": request_id,
        "method": args.method,
        "params": args.params.unwrap_or(Value::Null)
    });

    if let Err(e) = write_mcp_message(&mut writer, &request.to_string()).await {
        eprintln!("Failed to send request: {e}");
        std::process::exit(1);
    }

    // Wait for the response
    let response = read_mcp_message(&mut reader).await.unwrap_or_else(|e| {
        eprintln!("Failed to read response: {e}");
        std::process::exit(1);
    });

    // Print the raw JSON response (could be prettified)
    println!("{response}");

    // The process will be killed when `child` is dropped.
}
