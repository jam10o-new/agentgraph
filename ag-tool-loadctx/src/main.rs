use ag_tool_common::{describe, guidance, has_flag, read_args};
use serde::Deserialize;
use serde_json::json;
use std::path::PathBuf;

#[derive(Deserialize)]
struct Args {
    files: Vec<String>,
}

fn main() {
    if has_flag("--describe") {
        describe(
            "load_into_context",
            "Load files into the agent's volatile context for the next turn (not persisted).",
            json!({
                "type": "object",
                "properties": {
                    "files": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Paths of files to load into context."
                    }
                },
                "required": ["files"]
            }),
        );
        return;
    }

    if has_flag("--help") {
        guidance(
            "When using load_into_context:\n\
             - Loads file contents into volatile context visible on the NEXT turn.\n\
             - Unlike read_file, results are not immediately returned — they appear as system messages in following turns.\n\
             - Useful for preloading reference material, configs, or documentation.\n\
             - Files are not persisted across agent restarts.",
        );
        return;
    }

    // ── Execute ──
    let args: Args = read_args().unwrap_or_else(|e| {
        eprintln!("Error: {e}");
        std::process::exit(1);
    });

    if args.files.is_empty() {
        println!("No files requested.");
        return;
    }

    let mut loaded = Vec::new();
    for p in &args.files {
        let path = PathBuf::from(p);
        match std::fs::read_to_string(&path) {
            Ok(content) => {
                loaded.push(format!("File {p}:\n{content}"));
            }
            Err(e) => {
                loaded.push(format!("Error reading file {p}: {e}"));
            }
        }
    }

    // Signal to the caller which files were loaded so the agent loop knows
    // to inject these into volatile_context.
    print!("{}", loaded.join("\n---\n"));
}