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
            "read_file",
            "Read the contents of one or more files and return them immediately.",
            json!({
                "type": "object",
                "properties": {
                    "files": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Absolute or relative paths of files to read."
                    }
                },
                "required": ["files"]
            }),
        );
        return;
    }

    if has_flag("--help") {
        guidance(
            "When using read_file:\n\
             - Specify absolute file paths whenever possible.\n\
             - You can read multiple files in a single call.\n\
             - File contents are returned with filename headers.\n\
             - If a file cannot be read, an error message is returned for that file.\n\
             - Use read_file to inspect code, config files, or any text content.",
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

    let mut out = String::new();
    for p in &args.files {
        let path = PathBuf::from(p);
        match std::fs::read_to_string(&path) {
            Ok(content) => {
                out.push_str(&format!("--- {} ---\n{content}\n", path.display()));
            }
            Err(e) => {
                out.push_str(&format!("--- {} ---\nError: {e}\n", path.display()));
            }
        }
    }

    print!("{out}");
}