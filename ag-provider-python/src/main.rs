//! Inference provider plugin using llama-cpp-python (or compatible Python
//! inference library) as the backend.
//!
//! Protocol (stdin → JSON, stdout → streaming type-prefixed lines):
//!   See [`ag_provider_common`] for the full wire format.
//!
//! The Python script path comes from `--section` JSON (`{"script": "..."}`)
//! or the `AG_PROVIDER_PYTHON_SCRIPT` environment variable.

use ag_provider_common::*;
use anyhow::{Context, Result, anyhow};
use std::io::BufRead;
use std::process::{Command, Stdio};

fn resolve_script() -> Result<String> {
    // 1. Check --section CLI flag
    let mut args = std::env::args().peekable();
    while let Some(arg) = args.next() {
        if arg == "--section" {
            if let Some(json_str) = args.next() {
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(&json_str) {
                    if let Some(s) = val.get("script").and_then(|v| v.as_str()) {
                        return Ok(s.to_string());
                    }
                }
            }
        }
    }

    // 2. Check env var
    if let Ok(s) = std::env::var("AG_PROVIDER_PYTHON_SCRIPT") {
        if !s.is_empty() {
            return Ok(s);
        }
    }

    // 3. Check for bundled script next to binary
    let bundled = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join("python_backend.py")));
    if let Some(ref path) = bundled {
        if path.exists() {
            return Ok(path.to_string_lossy().to_string());
        }
    }

    Err(anyhow!(
        "No Python script configured. Set AG_PROVIDER_PYTHON_SCRIPT env var \
         or provide a `script` path via --section."
    ))
}

fn main() -> Result<()> {
    if has_flag("--describe") {
        describe(&InfoResponse {
            name: "python".into(),
            model: None,
            max_seq_len: None,
            supports_tools: true,
            supports_embeddings: true,
            supports_constraints: true,
            supports_modalities: vec!["text".into(), "image".into()],
        });
        return Ok(());
    }

    if has_flag("--help") {
        guidance(
            "Inference provider using llama-cpp-python or a compatible Python backend.\n\
             Supports GGUF models, multimodal (text + image), tool calling, grammar \
             constraints, and embeddings.",
        );
        return Ok(());
    }

    let script = resolve_script()?;

    // Read the JSON request from stdin
    let request = read_request().map_err(|e| anyhow!("{}", e))?;

    // Delegate to Python and stream results back
    match &request.kind {
        RequestKind::Chat(chat_req) => {
            let input = serde_json::to_string(chat_req)
                .with_context(|| "serialize chat request")?;

            let mut child = Command::new("python3")
                .arg(&script)
                .arg("--mode")
                .arg("chat")
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::inherit())
                .spawn()
                .with_context(|| format!("spawn python3 with {}", script))?;

            // Write request to python's stdin
            if let Some(mut stdin) = child.stdin.take() {
                use std::io::Write;
                stdin.write_all(input.as_bytes()).ok();
                drop(stdin);
            }

            // Stream stdout lines back as provider events
            let stdout = child.stdout.take().expect("piped stdout");
            for line in std::io::BufReader::new(stdout).lines() {
                let line = line.with_context(|| "read python stdout")?;
                // Forward the line verbatim — the Python script emits the
                // same T:/R:/C:/D:/X: format.
                println!("{line}");
                if line.starts_with('D') || line.starts_with('X') {
                    break;
                }
            }

            let status = child.wait().with_context(|| "wait for python")?;
            if !status.success() {
                // Error already streamed via X:, but exit code confirms it
            }
        }
        RequestKind::Embed(embed_req) => {
            let input = serde_json::to_string(embed_req)
                .with_context(|| "serialize embed request")?;

            let output = Command::new("python3")
                .arg(&script)
                .arg("--mode")
                .arg("embed")
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::inherit())
                .output()
                .with_context(|| format!("spawn python3 with {}", script))?;

            if !output.status.success() {
                let err = String::from_utf8_lossy(&output.stderr);
                emit_event(&ProviderEvent::Error(format!(
                    "python3 exited {}: {}",
                    output.status,
                    err.trim()
                )));
            } else {
                let stdout = String::from_utf8_lossy(&output.stdout);
                print!("{}", stdout);
            }
        }
        RequestKind::CountTokens(ct_req) => {
            // Rough estimate; Python script can do better if available
            let count = (ct_req.text.len() + 3) / 4;
            emit_event(&ProviderEvent::TokenCount(count));
            emit_event(&ProviderEvent::Done);
        }
        RequestKind::Health => {
            emit_event(&ProviderEvent::Healthy(true));
            emit_event(&ProviderEvent::Done);
        }
        RequestKind::Info => {
            emit_event(&ProviderEvent::Info(InfoResponse {
                name: "python".into(),
                model: None,
                max_seq_len: None,
                supports_tools: true,
                supports_embeddings: true,
                supports_constraints: true,
                supports_modalities: vec!["text".into(), "image".into()],
            }));
            emit_event(&ProviderEvent::Done);
        }
    }

    // Flush stdout
    use std::io::Write;
    std::io::stdout().flush().ok();
    Ok(())
}
