use ag_tool_common::{describe, guidance, has_flag, read_args};
use serde::Deserialize;
use serde_json::json;
use std::path::PathBuf;

#[derive(Deserialize)]
struct Args {
    filename: String,
    content: String,
}

#[tokio::main]
async fn main() {
    if has_flag("--describe") {
        describe(
            "write_note",
            "Create or overwrite a markdown note in ~/notes/.",
            json!({
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Name of the note (without .md extension)."
                    },
                    "content": {
                        "type": "string",
                        "description": "Markdown content to write."
                    }
                },
                "required": ["filename", "content"]
            }),
        );
        return;
    }

    if has_flag("--help") {
        guidance(
            "When using write_note:\n\
             - Creates the ~/notes/ directory if it does not exist.\n\
             - Writes a file named <filename>.md inside ~/notes/.\n\
             - Overwrites any existing file with the same name.\n\
             - Content should be valid markdown.\n\
             - Use to persist important information or agent scratchpads.",
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

    // Create notes directory
    if let Err(e) = tokio::fs::create_dir_all(&notes_dir).await {
        eprintln!("Failed to create notes directory {}: {e}", notes_dir.display());
        std::process::exit(1);
    }

    // Build file path, adding .md extension
    let mut note_path = notes_dir.join(&args.filename);
    if note_path.extension().is_none() {
        note_path.set_extension("md");
    }

    // Write content
    match tokio::fs::write(&note_path, &args.content).await {
        Ok(_) => println!("Note written to {}", note_path.display()),
        Err(e) => {
            eprintln!("Failed to write note {}: {e}", note_path.display());
            std::process::exit(1);
        }
    }
}
