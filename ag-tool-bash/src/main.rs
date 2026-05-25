use ag_tool_common::{describe, guidance, has_flag, read_args};
use serde::Deserialize;
use serde_json::json;

#[derive(Deserialize)]
struct Args {
    command: String,
    #[serde(default)]
    args: Vec<String>,
}

#[tokio::main]
async fn main() {
    if has_flag("--describe") {
        describe(
            "execute_command",
            "Execute a shell command on the host system and return stdout and stderr.",
            json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command to execute."
                    },
                    "args": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Optional arguments to pass to the command."
                    }
                },
                "required": ["command"]
            }),
        );
        return;
    }

    if has_flag("--help") {
        guidance(
            "When using execute_command:\n\
             - Use for system commands, file operations, package management, etc.\n\
             - The 'command' field is the program to run; use 'args' for arguments.\n\
             - Do NOT use shell syntax like &&, |, or >. These won't work.\n\
             - stdout and stderr are both captured and returned.\n\
             - Commands run with the agent's working directory and environment.\n\
             - Be careful with destructive commands — they actually execute.",
        );
        return;
    }

    // ── Execute ──
    let args: Args = read_args().unwrap_or_else(|e| {
        eprintln!("Error: {e}");
        std::process::exit(1);
    });

    match tokio::process::Command::new(&args.command)
        .args(&args.args)
        .output()
        .await
    {
        Ok(output) => {
            print!(
                "Stdout: {}\nStderr: {}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
        Err(e) => {
            println!("Error executing command: {e}");
        }
    }
}