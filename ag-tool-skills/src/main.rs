use ag_tool_common::{describe, guidance, has_flag, read_args};
use serde::Deserialize;
use serde_json::json;
use std::path::PathBuf;

#[derive(Deserialize)]
struct Args {
    #[serde(default)]
    search_paths: Vec<String>,
}

fn main() {
    if has_flag("--describe") {
        describe(
            "list_skills",
            "Discover available SKILL.md files by searching directories recursively.",
            json!({
                "type": "object",
                "properties": {
                    "search_paths": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Optional paths to search. Defaults to ~/.config and cwd."
                    }
                }
            }),
        );
        return;
    }

    if has_flag("--help") {
        guidance(
            "When using list_skills:\n\
             - Searches recursively for SKILL.md files in the given search paths.\n\
             - Defaults to ~/.config and current working directory if no paths provided.\n\
             - Returns skill name, path, and description (from YAML frontmatter) for each found skill.\n\
             - Use this before load_skill to discover what skills are available.",
        );
        return;
    }

    // ── Execute ──
    let Args {
        mut search_paths,
    } = read_args().unwrap_or_else(|e| {
        eprintln!("Error: {e}");
        std::process::exit(1);
    });

    if search_paths.is_empty() {
        if let Ok(home) = std::env::var("HOME") {
            search_paths.push(format!("{home}/.config"));
        }
        if let Ok(cwd) = std::env::current_dir() {
            search_paths.push(cwd.to_string_lossy().to_string());
        }
    }

    let mut skills = Vec::new();
    for root in &search_paths {
        let root_path = PathBuf::from(root);
        if !root_path.exists() {
            continue;
        }
        let mut stack = vec![root_path];
        while let Some(dir) = stack.pop() {
            let skill_md = dir.join("SKILL.md");
            if skill_md.exists() && skill_md.is_file() {
                let name = dir
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_default();
                let mut description = String::new();
                if let Ok(content) = std::fs::read_to_string(&skill_md) {
                    if content.starts_with("---") {
                        if let Some(end) = content[3..].find("---") {
                            let frontmatter = &content[3..end + 3];
                            for line in frontmatter.lines() {
                                if line.starts_with("description:") {
                                    description =
                                        line["description:".len()..].trim().to_string();
                                }
                            }
                        }
                    }
                    if description.is_empty() && content.len() > 100 {
                        description = content
                            .lines()
                            .skip(1)
                            .find(|l| !l.trim().is_empty() && !l.starts_with("---"))
                            .unwrap_or("")
                            .to_string();
                    }
                }
                skills.push(format!(
                    "- {name} ({}): {description}",
                    dir.display()
                ));
                continue; // don't recurse into skill dirs
            }
            if let Ok(entries) = std::fs::read_dir(&dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_dir() {
                        stack.push(path);
                    }
                }
            }
        }
    }

    if skills.is_empty() {
        println!("No skills found. Skills are directories containing a SKILL.md file.");
    } else {
        print!("Found {} skills:\n{}", skills.len(), skills.join("\n"));
    }
}