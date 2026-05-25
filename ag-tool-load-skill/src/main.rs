use ag_tool_common::{describe, guidance, has_flag, read_args};
use serde::Deserialize;
use serde_json::json;
use std::fs;
use std::path::PathBuf;

#[derive(Deserialize)]
struct Args {
    skill_path: String,
    #[serde(default)]
    system_dir: Option<String>,
}

fn copy_dir_recursive(src: &PathBuf, dest: &PathBuf) -> Result<(), String> {
    fs::create_dir_all(dest).map_err(|e| format!("mkdir: {e}"))?;
    for entry in fs::read_dir(src).map_err(|e| format!("read_dir: {e}"))?.flatten() {
        let entry_path = entry.path();
        let dest_path = dest.join(entry.file_name());
        if entry_path.is_dir() {
            copy_dir_recursive(&entry_path, &dest_path)?;
        } else {
            fs::copy(&entry_path, &dest_path).map_err(|e| format!("copy file: {e}"))?;
        }
    }
    Ok(())
}

fn main() {
    if has_flag("--describe") {
        describe(
            "load_skill",
            "Copy a skill directory into a system prompt directory so its contents become part of the agent's system context.",
            json!({
                "type": "object",
                "properties": {
                    "skill_path": {
                        "type": "string",
                        "description": "Path to the skill directory containing SKILL.md."
                    },
                    "system_dir": {
                        "type": "string",
                        "description": "Target system directory. Falls back to caller-configured default."
                    }
                },
                "required": ["skill_path"]
            }),
        );
        return;
    }

    if has_flag("--help") {
        guidance(
            "When using load_skill:\n\
             - First use list_skills to discover available skills.\n\
             - skill_path must point to a directory containing a SKILL.md file.\n\
             - The skill directory is copied into the target system directory.\n\
             - The skill's SKILL.md and any reference files become part of the system prompt.\n\
             - If system_dir is not provided, the tool uses the agent's first system directory.",
        );
        return;
    }

    // ── Execute ──
    let args: Args = read_args().unwrap_or_else(|e| {
        eprintln!("Error: {e}");
        std::process::exit(1);
    });

    if args.skill_path.is_empty() {
        println!("Error: skill_path is required");
        return;
    }

    let src = PathBuf::from(&args.skill_path);
    let skill_md = src.join("SKILL.md");
    if !skill_md.exists() {
        println!("Error: No SKILL.md found at {}", skill_md.display());
        return;
    }

    let system_dir = args
        .system_dir
        .unwrap_or_else(|| ".".to_string());
    let dest = PathBuf::from(&system_dir).join(src.file_name().unwrap_or_default());

    match copy_dir_recursive(&src, &dest) {
        Ok(_) => println!(
            "Skill loaded from {} into {}",
            src.display(),
            dest.display()
        ),
        Err(e) => println!("Error copying skill: {e}"),
    }
}