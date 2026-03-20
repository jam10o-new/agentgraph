//! Command execution for agentgraph.
//!
//! Executes commands parsed from model output (EXEC, KILL, READ, WRIT).

use crate::Args;
use crate::messages::{extract_frontmatter, load_full_skill_content};
use crate::types::{CommandIO, CommandType};
use anyhow::Result;
use std::path::PathBuf;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::Command;
use tokio::sync::mpsc::{Sender, channel};
use tokio::sync::oneshot;

/// Generic helper to pump data from an async reader to a channel.
async fn pump_stream<R>(mut reader: R, tx: Sender<Vec<u8>>)
where
    R: AsyncReadExt + Unpin,
{
    let mut buf = [0u8; 4096];
    loop {
        match reader.read(&mut buf).await {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                if tx.send(buf[..n].to_vec()).await.is_err() {
                    break;
                }
            }
        }
    }
}

/// Spawn a command with stdin/stdout/stderr pipes
pub async fn spawn_command_io<S, I>(program: S, args: I) -> Result<CommandIO>
where
    S: AsRef<str>,
    I: IntoIterator,
    I::Item: AsRef<str>,
{
    let mut child = Command::new(program.as_ref())
        .args(args.into_iter().map(|s| s.as_ref().to_owned()))
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true)
        .spawn()?;

    let mut child_stdin = child.stdin.take().expect("stdin missing");
    let child_stdout = child.stdout.take().expect("stdout missing");
    let child_stderr = child.stderr.take().expect("stderr missing");

    let (stdin_tx, mut stdin_rx) = channel::<Vec<u8>>(32);
    let (stdout_tx, stdout_rx) = channel::<Vec<u8>>(32);
    let (stderr_tx, stderr_rx) = channel::<Vec<u8>>(32);
    let (kill_tx, kill_rx) = oneshot::channel::<()>();
    let (exit_tx, exit_rx) = oneshot::channel();

    // stdin bridge
    tokio::spawn(async move {
        while let Some(buf) = stdin_rx.recv().await {
            if child_stdin.write_all(&buf).await.is_err() {
                break;
            }
        }
    });

    // stdout/stderr bridges
    tokio::spawn(pump_stream(child_stdout, stdout_tx));
    tokio::spawn(pump_stream(child_stderr, stderr_tx));

    // Child reaper
    tokio::spawn(async move {
        tokio::select! {
            _ = child.wait() => {
                let _ = exit_tx.send(());
            }
            _ = kill_rx => {
                let _ = child.kill().await;
                let _ = child.wait().await;
            }
        }
    });

    Ok(CommandIO::new(
        stdin_tx,
        stdout_rx,
        stderr_rx,
        Some(kill_tx),
        Some(exit_rx),
    ))
}

/// Execute a command and return the output string
pub async fn execute_command(
    cmd: CommandType,
    subprocesses: &mut Vec<CommandIO>,
    tool_context: &mut String,
    daemon_args: &Args,
) -> String {
    // ReadSkill command is always available (not gated by --tools flag)
    if let CommandType::ReadSkill(skill_name) = cmd {
        return execute_read_skill(skill_name, tool_context, daemon_args).await;
    }

    // All other commands require --tools flag
    if !daemon_args.tools {
        return "[TOOL USE FAILED: TOOLS DISABLED]".to_string();
    }

    match cmd {
        CommandType::Exec(command_str) => {
            let parts: Vec<&str> = command_str.split_whitespace().collect();
            if parts.is_empty() {
                return String::new();
            }

            let program = parts[0];
            let args = &parts[1..];

            match spawn_command_io(program, args.iter().map(|s| *s)).await {
                Ok(cmd_io) => {
                    let idx = subprocesses.len();
                    subprocesses.push(cmd_io);
                    if daemon_args.verbose {
                        eprintln!("[EXEC {}: spawned '{}']", idx, command_str);
                    }
                    String::new()
                }
                Err(e) => {
                    if daemon_args.verbose {
                        eprintln!("[EXEC failed: {}]", e);
                    }
                    String::new()
                }
            }
        }

        CommandType::Kill(idx) => {
            if idx < subprocesses.len() {
                let mut cmd_io = std::mem::replace(
                    &mut subprocesses[idx],
                    CommandIO::new(channel(1).0, channel(1).1, channel(1).1, None, None),
                );
                cmd_io.kill();
                subprocesses.remove(idx);
                if daemon_args.verbose {
                    eprintln!("[KILL {}: terminated]", idx);
                }
                format!("[KILL {}: terminated]\n", idx)
            } else {
                format!("[KILL {}: invalid index]\n", idx)
            }
        }

        CommandType::Read(idx) => {
            if idx >= subprocesses.len() {
                return format!("[READ {}: invalid index]\n", idx);
            }

            let cmd_io = &mut subprocesses[idx];
            let mut output = String::new();

            while let Ok(chunk) = cmd_io.stdout_rx.try_recv() {
                output.push_str(&String::from_utf8_lossy(&chunk));
            }
            while let Ok(chunk) = cmd_io.stderr_rx.try_recv() {
                output.push_str(&String::from_utf8_lossy(&chunk));
            }

            if output.is_empty() {
                if daemon_args.verbose {
                    eprintln!("[READ {}: (no new output)]", idx);
                }
                format!("[READ {}: (no new output)]\n", idx)
            } else {
                let formatted = format!(
                    "=== Command {} Output ===\n{}\n=== End Output ===\n",
                    idx, output
                );
                tool_context.push_str(&formatted);
                if daemon_args.verbose {
                    eprintln!("[READ {}: {} bytes]", idx, output.len());
                }
                format!("[READ {}: {} bytes captured]\n", idx, output.len())
            }
        }

        CommandType::Writ(idx, input) => {
            if idx >= subprocesses.len() {
                return format!("[WRIT {}: invalid index]\n", idx);
            }

            let cmd_io = &subprocesses[idx];
            match cmd_io.stdin_tx.send(input.clone().into_bytes()).await {
                Ok(_) => {
                    if daemon_args.verbose {
                        eprintln!("[WRIT {}: {} bytes sent]", idx, input.len());
                    }
                    format!("[WRIT {}: {} bytes sent]\n", idx, input.len())
                }
                Err(_) => {
                    format!("[WRIT {}: failed - channel closed]\n", idx)
                }
            }
        }

        CommandType::ReadSkill(_) => unreachable!("ReadSkill handled before tools check"),
    }
}

/// Execute ReadSkill command - search context directories for skill by frontmatter name
async fn execute_read_skill(
    skill_name: String,
    tool_context: &mut String,
    daemon_args: &Args,
) -> String {
    // Search for skill by scanning frontmatter in ALL context directories
    // This includes system, input, and assistant directories
    let search_dirs: Vec<&PathBuf> = daemon_args
        .system_cat
        .iter()
        .chain(daemon_args.system_final.iter())
        .chain(daemon_args.input_cat.iter())
        .chain(daemon_args.input_final.iter())
        .chain(daemon_args.assistant_cat.iter())
        .chain(daemon_args.assistant_final.iter())
        .collect();

    for dir in search_dirs {
        if !dir.exists() {
            continue;
        }

        // Walk all files in the directory
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if !path.is_file() {
                    continue;
                }

                // Read file content and check for matching frontmatter
                if let Ok(content) = std::fs::read_to_string(&path) {
                    // Quick check: does file start with YAML frontmatter?
                    if let Some((frontmatter, _)) = extract_frontmatter(&content) {
                        // Check if skill name matches
                        if frontmatter.name == skill_name {
                            // Load full content and return
                            match load_full_skill_content(&path).await {
                                Ok(full_content) => {
                                    let formatted = format!(
                                        "=== Skill: {} ===\n{}\n=== End Skill ===\n",
                                        skill_name, full_content
                                    );
                                    tool_context.push_str(&formatted);
                                    if daemon_args.verbose {
                                        eprintln!(
                                            "[RRDS {}: loaded from {}]",
                                            skill_name,
                                            path.display()
                                        );
                                    }
                                    return format!(
                                        "[RRDS {}: {} bytes loaded]\n",
                                        skill_name,
                                        full_content.len()
                                    );
                                }
                                Err(e) => {
                                    if daemon_args.verbose {
                                        eprintln!(
                                            "[RRDS {}: read error: {}]",
                                            skill_name, e
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    format!(
        "[RRDS {}: skill not found in any context directory]\n",
        skill_name
    )
}
