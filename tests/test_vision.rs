//! Integration test for agentgraph vision pipeline.
//!
//! Graph topology:
//!
//!   [grim screenshot loop]
//!          | (writes PNG to agent_a/input/ every SCREENSHOT_INTERVAL_SECS)
//!          |
//!     [ Agent A ]  -- screen observer / vision describer (Qwen3-VL primary)
//!                     watches agent_a/input/ for new PNGs
//!                     writes structured reports to agent_a/output/
//!          |
//!     [ Agent B ]  -- downstream screen-activity summariser
//!                     watches agent_a/output/ as user input (-I)
//!                     writes rolling summary to agent_b/output/ (stream-realtime)
//!
//! Agent A: no --stream-realtime (agent-to-agent node)
//!          triggers on Create/Close(Write) of each new screenshot PNG
//!
//! Agent B: --stream-realtime (human-facing final node)
//!          triggers on Close(Write) of each completed Agent A report
//!
//! Pass condition:
//!   - Agent B produces output containing [CURRENT]:
//!   - Agent A produces output containing [FOCUS]:

mod common;

use anyhow::Result;
use common::{AgentConfig, TestResult, TestRunner, create_dirs, get_binary_path};
use std::process::Command;
use std::time::Duration;

#[test]
fn test_vision_pipeline() -> Result<()> {
    let project_root = common::get_project_root()?;
    let base_dir = project_root.join("test_vision");
    let binary = get_binary_path()?;

    println!("=== Vision Pipeline Test ===");
    println!("Base directory: {:?}", base_dir);
    println!("Binary: {:?}", binary);

    // Create directory structure
    create_dirs(
        &base_dir,
        &[
            "agent_a/system",
            "agent_a/input",
            "agent_a/output",
            "agent_b/system",
            "agent_b/input",
            "agent_b/output",
        ],
    )?;

    // Initialize test runner
    let mut runner = TestRunner::new(&base_dir, &binary)?;
    runner.verbose = std::env::var("VERBOSE").map(|v| v == "1").unwrap_or(false);
    runner.max_runtime_secs = 300;

    // Set success/fail patterns
    runner.add_fail_pattern("^VISION_PARSE_FAILED");
    runner.add_fail_pattern("^Failed to create AudioInput from bytes");
    runner.add_fail_pattern("^out of memory");
    runner.add_fail_pattern("^CUDA_ERROR");
    runner.add_fail_pattern("^thread '.*' panicked");

    runner.add_success_pattern("[CURRENT]:");
    runner.add_success_pattern("[FOCUS]:");

    // Pre-run cleanup
    runner.pre_run_cleanup(&["agent_a/input", "agent_a/output", "agent_b/output", "logs"])?;

    // Agent A - vision screen observer
    let agent_a_config = AgentConfig {
        label: "AGENT_A".to_string(),
        system_dir: base_dir.join("agent_a/system"),
        input_dir: base_dir.join("agent_a/input"),
        output_dir: base_dir.join("agent_a/output"),
        log_file: base_dir.join("logs/agent_a.log"),
        watch_mode: true,
        verbose: true,
        tools: false,
        stream_realtime: false,
        model: None,
        secondary_model: None,
        realtime_listener: None,
        audio_chunk_min_secs: None,
        audio_chunk_max_secs: None,
        input_from_output: None,
        additional_inputs: Vec::new(),
    };

    // Agent B - screen activity summariser
    let agent_b_config = AgentConfig {
        label: "AGENT_B".to_string(),
        system_dir: base_dir.join("agent_b/system"),
        input_dir: base_dir.join("agent_a/output"), // Watches agent A output
        output_dir: base_dir.join("agent_b/output"),
        log_file: base_dir.join("logs/agent_b.log"),
        watch_mode: true,
        verbose: true,
        tools: false,
        stream_realtime: true,
        model: None,
        secondary_model: None,
        realtime_listener: None,
        audio_chunk_min_secs: None,
        audio_chunk_max_secs: None,
        input_from_output: None,
        additional_inputs: Vec::new(),
    };

    // Spawn agents
    runner.spawn_agent(&agent_a_config)?;
    std::thread::sleep(Duration::from_secs(1));

    runner.spawn_agent(&agent_b_config)?;

    // Spawn screenshot loop in background
    let screenshot_handle = {
        let input_dir = base_dir.join("agent_a/input");
        let screenshot_count = 2;
        let screenshot_wait_secs = 60;

        std::thread::spawn(move || {
            for i in 1..=screenshot_count {
                let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
                let outfile = input_dir.join(format!("screenshot_{}.png", timestamp));

                let result = Command::new("grim")
                    .arg("-s")
                    .arg("0.5")
                    .arg(&outfile)
                    .output();

                match result {
                    Ok(output) if output.status.success() => {
                        println!(
                            "[screenshot {}/{}] Captured: {:?}",
                            i, screenshot_count, outfile
                        );
                    }
                    _ => {
                        println!(
                            "[screenshot {}/{}] grim failed — is WAYLAND_DISPLAY set?",
                            i, screenshot_count
                        );
                    }
                }

                // Wait between shots, but not after the last one
                if i < screenshot_count {
                    println!(
                        "[screenshot] Waiting {}s before next shot...",
                        screenshot_wait_secs
                    );
                    std::thread::sleep(Duration::from_secs(screenshot_wait_secs));
                }
            }
            println!("[screenshot] Sequence complete.");
        })
    };

    println!("\n--- Shot 1 fires immediately to load models (~60s to first inference). ---");
    println!("--- Shot 2 fires after 60s to test re-inference. ---");
    println!("--- Watching for [FOCUS]: (Agent A) and [CURRENT]: (Agent B) ---\n");

    // Run test
    let result = runner.run();

    // Wait for screenshot thread to finish
    let _ = screenshot_handle.join();

    match result {
        TestResult::Pass => Ok(()),
        TestResult::Fail => Err(anyhow::anyhow!("Test failed")),
        TestResult::Inconclusive => Err(anyhow::anyhow!("Test inconclusive - check output files")),
    }
}
