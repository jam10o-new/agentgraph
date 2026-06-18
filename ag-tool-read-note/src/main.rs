use ag_tool_common::{describe, guidance, has_flag, read_args};
use serde::Deserialize;
use serde_json::json;
use std::path::PathBuf;

#[derive(Deserialize)]
struct Args {
    #[serde(default)]
    filename: Option<String>,
}

#[tokio::main]
async fn main() {
    if has_flag("--describe") {
        describe(
            "read_note",
            "Read a markdown note from ~/notes/, or list all notes if no filename is given.",
            json!({
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Optional. Name of the note (without .md extension). If omitted, lists all available notes."
                    }
                },
                "required": []
            }),
        );
        return;
    }

    if has_flag("--help") {
        guidance(
            "When using read_note:\n\
             - Provide a filename to read the contents of ~/notes/<filename>.md.\n\
             - Omit the argument to list all .md files in ~/notes/.\n\
             - Use to retrieve previously saved notes or get an overview of existing ones.",
        );
        return;
    }

    // Parse arguments
    let args: Args = read_args().unwrap_or_else(|e| {
        eprintln!("Error parsing arguments: {e}");
        std::process::exit(1);
    });

    // Determine home directory
    let home = match std::env::var("HOME") {
        Ok(h) => h,
        Err(_) => std::env::var("USERPROFILE").expect("Could not find home directory"),
    };
    let notes_dir = PathBuf::from(&home).join("notes");

    // Ensure notes directory exists
    if !notes_dir.exists() {
        println!("No notes directory found at {}. Nothing to read.", notes_dir.display());
        return;
    }

    match args.filename {
        Some(name) => {
            let mut note_path = notes_dir.join(&name);
            if note_path.extension().is_none() {
                note_path.set_extension("md");
            }
            match tokio::fs::read_to_string(&note_path).await {
                Ok(content) => println!("{}", content),
                Err(e) => {
                    eprintln!("Failed to read note {}: {e}", note_path.display());
                    std::process::exit(1);
                }
            }
        }
        None => {
            // List all .md files
            let mut entries = match tokio::fs::read_dir(&notes_dir).await {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("Failed to read directory {}: {e}", notes_dir.display());
                    std::process::exit(1);
                }
            };

            let mut found = false;
            println!("Notes in {}:", notes_dir.display());
            while let Some(entry) = entries.next_entry().await.unwrap_or(None) {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("md") {
                    if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                        println!("  - {}", stem);
                        found = true;
                    }
                }
            }
            if !found {
                println!("  (no .md files found)");
            }
        }
    }
}
