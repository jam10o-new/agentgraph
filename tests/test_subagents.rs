//! Integration test for agentgraph subagent spawning.
//!
//! Graph topology:
//!
//!     [ SUPERVISOR AGENT ]
//!           |
//!           +-- EXEC --> [ watcher_a ]  (watch mode, monitors shared_screenshots)
//!           |              watches: watcher_shared/input
//!           |              outputs: watcher_a/output/
//!           |
//!           +-- EXEC --> [ watcher_b ]  (watch mode, monitors shared_screenshots)
//!                          watches: watcher_shared/input
//!                          outputs: watcher_b/output/
//!
//! Key concept demonstrated:
//!   - Autonomous subagent spawning via EXEC command system
//!   - Supervisor coordinates multiple subagent instances
//!   - Mix of watch-mode and oneshot inference subagents
//!   - Subagents are additional agentgraph daemon processes
//!
//! Pass conditions:
//!   - Supervisor spawns at least 2 subagents successfully
//!   - Subagent outputs contain expected structured fields
//!   - Supervisor produces final [COORDINATION_SUMMARY]

mod common;

use agentgraph::{CMD_CLOSE_EXEC, CMD_OPEN_EXEC};
use anyhow::Result;
use common::{AgentConfig, TestResult, TestRunner, create_dirs, get_binary_path};
use std::fs;
use std::thread;
use std::time::Duration;

#[test]
fn test_subagents_spawning() -> Result<()> {
    let project_root = common::get_project_root()?;
    let base_dir = project_root.join("test_subagents");
    let binary = get_binary_path()?;

    println!("=== Subagents Spawning Test ===");
    println!("Base directory: {:?}", base_dir);
    println!("Binary: {:?}", binary);

    // Create directory structure
    create_dirs(
        &base_dir,
        &[
            "supervisor/system",
            "supervisor/input",
            "supervisor/output",
            "watcher_a/system",
            "watcher_a/output",
            "watcher_b/system",
            "watcher_b/output",
        ],
    )?;

    // Initialize test runner
    let mut runner = TestRunner::new(&base_dir, &binary)?;
    runner.verbose = std::env::var("VERBOSE").map(|v| v == "1").unwrap_or(false);
    runner.max_runtime_secs = 400; // Increased for subagents

    // Set success/fail patterns
    runner.add_fail_pattern("^EXEC failed");
    runner.add_fail_pattern("^subagent failed");
    runner.add_fail_pattern("^failed to spawn");
    runner.add_fail_pattern("^out of memory");
    runner.add_fail_pattern("^CUDA_ERROR");
    runner.add_fail_pattern("[COORDINATION FAILED]");
    runner.add_fail_pattern("^thread '.*' panicked");

    // Custom check: supervisor output file length (detects hallucination vs actual command execution)
    runner.set_max_output_length_check("supervisor/output", 2000,
        "Supervisor output exceeds length limit - likely hallucinating instead of executing commands");

    runner.add_success_pattern("[SUBAGENT_SPAWNED]:");
    runner.add_success_pattern("[COORDINATION_SUMMARY]:");
    runner.add_success_pattern("SILVER");
    runner.add_success_pattern("BLUE");

    // Pre-run cleanup
    runner.pre_run_cleanup(&[
        "supervisor/input",
        "supervisor/output",
        "supervisor/system",
        "watcher_a/input",
        "watcher_a/output",
        "watcher_b/input",
        "watcher_b/output",
        "logs",
    ])?;

    // Write hardcoded system prompt for supervisor
    let bin_str = binary.to_string_lossy();
    let system_prompt = format!(
        r#"You are a supervisor agent coordinating multiple subagents.

## Command Syntax
Spawn processes: {open_exec}command args{close_exec}

## Your Task
First, spawn these 2 subagents using these exact responses, seperated by a newline:

1. watcher_a (watch mode daemon):
{open_exec}bash -c "setsid {bin} -W --verbose -S {base}/watcher_a/system -I {base}/watcher_a/input -O {base}/watcher_a/output &"{close_exec}

2. watcher_b (watch mode daemon):
{open_exec}bash -c "setsid {bin} -W --verbose -S {base}/watcher_b/system -I {base}/watcher_b/input -O {base}/watcher_b/output &"{close_exec}

Once you have spawned a subagent, respond with "[SUBAGENT_SPAWNED]: <subagent name>".
If you have not yet seen responses from all of your subagents, respond only "[AWAITING RESPONSE]" or "...".
If a subagent errors, panics, or fails, respond with only "[COORDINATION FAILED]"

One of your subagents will inform you of your final task in their response.

Be brief and do not include any preamble or summary other than what is explicitly necessary.
"#,
        open_exec = CMD_OPEN_EXEC,
        close_exec = CMD_CLOSE_EXEC,
        bin = bin_str,
        base = base_dir.to_string_lossy(),
    );

    fs::write(
        base_dir.join("supervisor/system/prompt.txt"),
        &system_prompt,
    )?;
    println!("    System prompt written to supervisor/system/prompt.txt");

    // Supervisor agent - orchestrates subagent spawning
    let supervisor_config = AgentConfig {
        label: "SUPERVISOR".to_string(),
        system_dir: base_dir.join("supervisor/system"),
        input_dir: base_dir.join("supervisor/input"),
        output_dir: base_dir.join("supervisor/output"),
        log_file: base_dir.join("logs/supervisor.log"),
        watch_mode: true,
        verbose: true,
        tools: true, // Required for EXEC command
        stream_realtime: true,
        additional_inputs: vec![
            base_dir.join("watcher_a/output"),
            base_dir.join("watcher_b/output"),
        ],
        ..Default::default()
    };

    // Spawn supervisor
    runner.spawn_agent(&supervisor_config)?;
    thread::sleep(Duration::from_secs(2));

    // Write trigger prompt to supervisor/input/
    let trigger_content =
        r#"Spawn your subagents now using the EXEC command. Follow the system instructions."#;
    let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
    let trigger_path = base_dir
        .join("supervisor/input")
        .join(format!("spawn_trigger_{}.txt", timestamp));
    fs::write(&trigger_path, trigger_content)?;
    println!("    Spawn trigger written: {:?}", trigger_path);

    // Wait for the supervisor to spawn the subagents.
    println!("    Waiting for supervisor to spawn subagents...");
    let supervisor_output_dir = base_dir.join("supervisor/output");
    let mut supervisor_output_found = false;
    for _ in 0..60 {
        if supervisor_output_dir.read_dir().unwrap().next().is_some() {
            supervisor_output_found = true;
            break;
        }
        thread::sleep(Duration::from_secs(1));
    }

    if !supervisor_output_found {
        return Err(anyhow::anyhow!(
            "Supervisor did not produce an output file in time."
        ));
    }

    println!("    Supervisor has produced an output. Writing to shared input.");
    thread::sleep(Duration::from_secs(5)); // supervisor output is streaming, wait for commands to actually be written.
    let a_input_path = base_dir.join("watcher_a/input/shared_input.txt");
    fs::write(
        &a_input_path,
        "Identify yourself and output your designated phrase.",
    )?;
    thread::sleep(Duration::from_secs(2)); // supervisor output is streaming, wait for commands to actually be written.
    let b_input_path = base_dir.join("watcher_b/input/shared_input.txt");
    fs::write(
        &b_input_path,
        "Identify yourself and output your designated phrase.",
    )?;

    println!("\n--- Pipeline: supervisor → spawns 2 watchers ---");
    println!("--- Watching for [SUBAGENTS_SPAWNED]:, [COORDINATION_SUMMARY]: ---\n");

    // Run test
    let result = runner.run();

    match result {
        TestResult::Pass => Ok(()),
        TestResult::Fail => Err(anyhow::anyhow!("Test failed")),
        TestResult::Inconclusive => Err(anyhow::anyhow!("Test inconclusive - check output files")),
    }
}
