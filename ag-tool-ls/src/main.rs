use ag_tool_common::{describe, guidance, has_flag, read_args};
use serde::Deserialize;
use serde_json::json;
use std::path::PathBuf;

#[derive(Deserialize)]
struct Args {
    path: String,
}

fn main() {
    if has_flag("--describe") {
        describe(
            "list_directory",
            "List files and directories at a given path, showing type, name, and size.",
            json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to list."
                    }
                },
                "required": ["path"]
            }),
        );
        return;
    }

    if has_flag("--help") {
        guidance(
            "When using list_directory:\n\
             - Use absolute paths whenever possible.\n\
             - The output shows [file] or [dir] prefix, full path, and size in bytes.\n\
             - Useful for exploring directory structure before reading files.\n\
             - An empty or inaccessible directory returns an appropriate message.",
        );
        return;
    }

    // ── Execute ──
    let args: Args = read_args().unwrap_or_else(|e| {
        eprintln!("Error: {e}");
        std::process::exit(1);
    });

    let p = PathBuf::from(&args.path);
    let mut out = String::new();

    match std::fs::read_dir(&p) {
        Ok(entries) => {
            for entry in entries.flatten() {
                let file_type = if entry.path().is_dir() { "dir" } else { "file" };
                let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
                out.push_str(&format!(
                    "[{file_type}] {} ({} bytes)\n",
                    entry.path().display(),
                    size
                ));
            }
        }
        Err(e) => {
            out.push_str(&format!("Error reading directory: {e}"));
        }
    }

    if out.is_empty() {
        println!("Directory is empty or path not found.");
    } else {
        print!("{out}");
    }
}