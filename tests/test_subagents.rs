//! Integration test for agentgraph subagent spawning.
//!
//! Graph topology:
//!
//!     [ SUPERVISOR AGENT ]
//!           |
//!           +-- EXEC --> [ watcher_a ]  (watch mode, monitors shared_screenshots)
//!           |              watches: shared_screenshots/
//!           |              outputs: watcher_a/output/
//!           |
//!           +-- EXEC --> [ watcher_b ]  (watch mode, monitors shared_screenshots)
//!           |              watches: shared_screenshots/
//!           |              outputs: watcher_b/output/
//!           |
//!           +-- EXEC --> [ oneshot_c ]  (oneshot inference, system state analysis)
//!           |              input: supervisor/input/system_state.txt
//!           |              outputs: oneshot_c/output/
//!           |
//!           +-- EXEC --> [ oneshot_d ]  (oneshot inference, recommendation generation)
//!                          input: watcher outputs (synthesized)
//!                          outputs: oneshot_d/output/
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

use anyhow::Result;
use common::{
    AgentConfig, TestRunner, TestResult,
    create_dirs, get_binary_path,
};
use std::fs;
use agentgraph::{CMD_OPEN_EXEC, CMD_CLOSE_EXEC};

#[test]
#[ignore] // Requires significant runtime and model loading
fn test_subagents_spawning() -> Result<()> {
    let project_root = common::get_project_root()?;
    let base_dir = project_root.join("test_subagents");
    let binary = get_binary_path()?;

    println!("=== Subagents Spawning Test ===");
    println!("Base directory: {:?}", base_dir);
    println!("Binary: {:?}", binary);

    // Create directory structure
    create_dirs(&base_dir, &[
        "supervisor/system",
        "supervisor/input",
        "supervisor/output",
        "watcher_a/system",
        "watcher_a/input",
        "watcher_a/output",
        "watcher_b/system",
        "watcher_b/input",
        "watcher_b/output",
        "oneshot_c/system",
        "oneshot_c/input",
        "oneshot_c/output",
        "oneshot_d/system",
        "oneshot_d/input",
        "oneshot_d/output",
        "shared_screenshots",
    ])?;

    // Initialize test runner
    let mut runner = TestRunner::new(&base_dir, &binary)?;
    runner.verbose = std::env::var("VERBOSE").map(|v| v == "1").unwrap_or(false);
    runner.max_runtime_secs = 300;

    // Set success/fail patterns
    runner.add_fail_pattern("^EXEC failed");
    runner.add_fail_pattern("^subagent failed");
    runner.add_fail_pattern("^failed to spawn");
    runner.add_fail_pattern("^out of memory");
    runner.add_fail_pattern("^CUDA_ERROR");
    runner.add_fail_pattern("^thread '.*' panicked");

    runner.add_success_pattern("[SUBAGENTS_SPAWNED]:");
    runner.add_success_pattern("[COORDINATION_SUMMARY]:");

    // Pre-run cleanup
    runner.pre_run_cleanup(&[
        "supervisor/input",
        "supervisor/output",
        "supervisor/system",
        "watcher_a/input",
        "watcher_a/output",
        "watcher_b/input",
        "watcher_b/output",
        "oneshot_c/input",
        "oneshot_c/output",
        "oneshot_d/input",
        "oneshot_d/output",
        "shared_screenshots",
        "logs",
    ])?;

    // Write hardcoded system prompt for supervisor
    // Simple prompt - just spawn the 4 subagents
    let system_prompt = format!(
        r#"You are a supervisor agent coordinating multiple subagents.

## Command Syntax
Spawn processes: {open_exec}command args{close_exec}

## Your Task
Spawn these 4 subagents:

1. watcher_a (watch mode daemon):
{open_exec}setsid cargo run --release -- -W --verbose -S /home/jam/agentgraph/test_subagents/watcher_a/system -I /home/jam/agentgraph/test_subagents/watcher_a/input -O /home/jam/agentgraph/test_subagents/watcher_a/output{close_exec}

2. watcher_b (watch mode daemon):
{open_exec}setsid cargo run --release -- -W --verbose -S /home/jam/agentgraph/test_subagents/watcher_b/system -I /home/jam/agentgraph/test_subagents/watcher_b/input -O /home/jam/agentgraph/test_subagents/watcher_b/output{close_exec}

3. oneshot_c (oneshot inference):
{open_exec}setsid cargo run --release -- --verbose -S /home/jam/agentgraph/test_subagents/oneshot_c/system -I /home/jam/agentgraph/test_subagents/oneshot_c/input -O /home/jam/agentgraph/test_subagents/oneshot_c/output{close_exec}

4. oneshot_d (oneshot inference):
{open_exec}setsid cargo run --release -- --verbose -S /home/jam/agentgraph/test_subagents/oneshot_d/system -I /home/jam/agentgraph/test_subagents/oneshot_d/input -O /home/jam/agentgraph/test_subagents/oneshot_d/output{close_exec}

Execute all 4 commands above."#,
        open_exec = CMD_OPEN_EXEC,
        close_exec = CMD_CLOSE_EXEC,
    );
    
    fs::write(base_dir.join("supervisor/system/prompt.txt"), &system_prompt)?;
    println!("    System prompt written to supervisor/system/prompt.txt");

    // Supervisor agent - orchestrates subagent spawning
    // Wire up subagent output directories as additional inputs for filesystem-based IPC
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
        model: None,
        secondary_model: None,
        realtime_listener: None,
        audio_chunk_min_secs: None,
        audio_chunk_max_secs: None,
        input_from_output: None,
        additional_inputs: vec![
            base_dir.join("watcher_a/output"),
            base_dir.join("watcher_b/output"),
            base_dir.join("oneshot_c/output"),
            base_dir.join("oneshot_d/output"),
        ],
    };

    // Spawn supervisor
    runner.spawn_agent(&supervisor_config)?;
    std::thread::sleep(std::time::Duration::from_secs(2));

    // Write trigger prompt to supervisor/input/
    let trigger_content = r#"Spawn your subagents now using the EXEC command. Follow the system instructions."#;

    let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
    let trigger_path = base_dir.join("supervisor/input")
        .join(format!("spawn_trigger_{}.txt", timestamp));
    fs::write(&trigger_path, trigger_content)?;
    println!("    Spawn trigger written: {:?}", trigger_path);

    // Wait for supervisor to spawn subagents (they take ~60s to load models)
    // Then write trigger files to start their inference
    println!("    Waiting 90s for subagents to load models...");
    std::thread::sleep(std::time::Duration::from_secs(90));
    
    // Check if subagent processes are running
    let subagent_count = std::process::Command::new("pgrep")
        .arg("-f")
        .arg("agentgraph.*test_subagents")
        .output()
        .ok()
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map(|s| s.lines().count())
        .unwrap_or(0);
    println!("    Found {} running subagent processes", subagent_count);
    
    println!("    Writing trigger files to subagent inputs...");
    let subagent_trigger = "Process this input and provide your analysis.";
    let trigger_ts = chrono::Utc::now().format("%Y%m%d_%H%M%S");
    let _ = fs::write(base_dir.join("watcher_a/input").join(format!("trigger_{}.txt", trigger_ts)), subagent_trigger);
    let _ = fs::write(base_dir.join("watcher_b/input").join(format!("trigger_{}.txt", trigger_ts)), subagent_trigger);
    let _ = fs::write(base_dir.join("oneshot_c/input").join(format!("trigger_{}.txt", trigger_ts)), subagent_trigger);
    let _ = fs::write(base_dir.join("oneshot_d/input").join(format!("trigger_{}.txt", trigger_ts)), subagent_trigger);
    println!("    Subagent triggers written");

    println!("\n--- Pipeline: supervisor → spawns 4 subagents (2 watchers, 2 oneshot) ---");
    println!("--- Watching for [SUBAGENTS_SPAWNED]:, [COORDINATION_SUMMARY]: ---\n");

    // Run test
    let result = runner.run();

    match result {
        TestResult::Pass => Ok(()),
        TestResult::Fail => Err(anyhow::anyhow!("Test failed")),
        TestResult::Inconclusive => {
            Err(anyhow::anyhow!("Test inconclusive - check output files"))
        }
    }
}
