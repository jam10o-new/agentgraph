//! Integration test for agentgraph tools pipeline.
//!
//! Graph topology:
//!
//!     [ Agent SCREENSHOT ]  -- tool-calling agent, uses EXEC to run grim
//!                              triggered by an initial prompt in its input dir
//!                              writes "SCREENSHOT_READY: <path>" to output
//!              |
//!     [ Agent VISION ]      -- vision describer (Qwen3-VL primary)
//!                              watches agent_vision/input/ for new PNGs
//!                              (the screenshot agent writes directly there)
//!                              writes structured [LAYOUT]/[FOCUS]/etc to output
//!              |
//!     [ Agent SUMMARY ]     -- rolling screen summariser (stream-realtime)
//!                              watches agent_vision/output/ as user input
//!                              writes [CURRENT]/[TIMELINE]/etc to output
//!
//! Pass conditions:
//!   - agent_screenshot output contains "SCREENSHOT_READY:"
//!   - agent_vision output contains "[FOCUS]:"
//!   - agent_summary output contains "[CURRENT]:"

mod common;

use anyhow::Result;
use common::{AgentConfig, TestResult, TestRunner, create_dirs, get_binary_path};
use std::fs;

#[test]
fn test_tools_pipeline() -> Result<()> {
    let project_root = common::get_project_root()?;
    let base_dir = project_root.join("test_tools");
    let binary = get_binary_path()?;

    println!("=== Tools Pipeline Test ===");
    println!("Base directory: {:?}", base_dir);
    println!("Binary: {:?}", binary);

    // Create directory structure
    create_dirs(
        &base_dir,
        &[
            "agent_screenshot/system",
            "agent_screenshot/input",
            "agent_screenshot/output",
            "agent_vision/system",
            "agent_vision/input",
            "agent_vision/output",
            "agent_summary/system",
            "agent_summary/input",
            "agent_summary/output",
        ],
    )?;

    // Initialize test runner
    let mut runner = TestRunner::new(&base_dir, &binary)?;
    runner.verbose = std::env::var("VERBOSE").map(|v| v == "1").unwrap_or(false);
    runner.max_runtime_secs = 300;

    // Set success/fail patterns
    runner.add_fail_pattern("^SCREENSHOT_FAILED:");
    runner.add_fail_pattern("^VISION_PARSE_FAILED");
    runner.add_fail_pattern("^TOOL USE FAILED");
    runner.add_fail_pattern("^EXEC failed");
    runner.add_fail_pattern("^out of memory");
    runner.add_fail_pattern("^CUDA_ERROR");
    runner.add_fail_pattern("^thread '.*' panicked");

    runner.add_success_pattern("SCREENSHOT_READY:");
    runner.add_success_pattern("[FOCUS]:");
    runner.add_success_pattern("[CURRENT]:");

    // Pre-run cleanup
    runner.pre_run_cleanup(&[
        "agent_screenshot/input",
        "agent_screenshot/output",
        "agent_vision/input",
        "agent_vision/output",
        "agent_summary/output",
        "logs",
    ])?;

    // Agent SCREENSHOT - tool-calling agent that runs grim
    let screenshot_config = AgentConfig {
        label: "SCREENSHOT".to_string(),
        system_dir: base_dir.join("agent_screenshot/system"),
        input_dir: base_dir.join("agent_screenshot/input"),
        output_dir: base_dir.join("agent_screenshot/output"),
        log_file: base_dir.join("logs/agent_screenshot.log"),
        watch_mode: true,
        verbose: true,
        tools: true,
        stream_realtime: false,
        model: None,
        secondary_model: None,
        realtime_listener: None,
        audio_chunk_min_secs: None,
        audio_chunk_max_secs: None,
        input_from_output: None,
        additional_inputs: Vec::new(),
    };

    // Agent VISION - vision describer
    let vision_config = AgentConfig {
        label: "VISION".to_string(),
        system_dir: base_dir.join("agent_vision/system"),
        input_dir: base_dir.join("agent_vision/input"),
        output_dir: base_dir.join("agent_vision/output"),
        log_file: base_dir.join("logs/agent_vision.log"),
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

    // Agent SUMMARY - stream-realtime summariser
    let summary_config = AgentConfig {
        label: "SUMMARY".to_string(),
        system_dir: base_dir.join("agent_summary/system"),
        input_dir: base_dir.join("agent_vision/output"), // Watches vision output
        output_dir: base_dir.join("agent_summary/output"),
        log_file: base_dir.join("logs/agent_summary.log"),
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
    runner.spawn_agent(&screenshot_config)?;
    std::thread::sleep(std::time::Duration::from_secs(1));

    runner.spawn_agent(&vision_config)?;
    std::thread::sleep(std::time::Duration::from_secs(1));

    runner.spawn_agent(&summary_config)?;

    // Write trigger prompt to screenshot agent
    let trigger_content = r#"Take a screenshot of the current desktop now. Use grim with the -s 0.5 scale flag and save it to the designated screenshots directory. Report the full path using the SCREENSHOT_READY: prefix as instructed."#;

    let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
    let trigger_path = base_dir
        .join("agent_screenshot/input")
        .join(format!("trigger_{}.txt", timestamp));
    fs::write(&trigger_path, trigger_content)?;
    println!("    Trigger written: {:?}", trigger_path);

    println!("\n--- Pipeline: screenshot agent → grim → vision agent → summary agent ---");
    println!("--- Watching for SCREENSHOT_READY:, [FOCUS]:, [CURRENT]: ---\n");

    // Run test
    let result = runner.run();

    match result {
        TestResult::Pass => Ok(()),
        TestResult::Fail => Err(anyhow::anyhow!("Test failed")),
        TestResult::Inconclusive => Err(anyhow::anyhow!("Test inconclusive - check output files")),
    }
}
