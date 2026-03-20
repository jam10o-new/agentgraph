//! Integration test for agentgraph audio streaming pipeline.
//!
//! Graph topology:
//!
//!   [PipeWire mic/loopback]
//!          |
//!     [ Agent A ]  -- realtime audio listener / transcriber (Voxtral secondary model)
//!          |  output written to agent_a/output/ (new file per inference)
//!          |
//!     [ Agent B ]  -- downstream summariser, watches agent_a/output as user input
//!                     writes running summary to agent_b/output/
//!
//! Agent A: --realtime-listener pipewire  --stream-realtime
//!          triggers on audio speech events; uses secondary (Voxtral) model for
//!          detect_speech + audio transcription, primary (Qwen3-VL) for synthesis.
//!
//! Agent B: -W watch mode, -I agent_a/output  (latest output of A is user input)
//!          pure text path, primary model only.
//!
//! Pass conditions:
//!   - Audio interrupt triggered
//!   - Agent A produces [TYPE]: output
//!   - Agent B produces [SITUATION]: output

mod common;

use anyhow::Result;
use common::{AgentConfig, TestResult, TestRunner, create_dirs, get_binary_path};
use std::fs;

#[test]
fn test_audio_pipeline() -> Result<()> {
    let project_root = common::get_project_root()?;
    let base_dir = project_root.join("test_audio");
    let binary = get_binary_path()?;

    println!("=== Audio Pipeline Test ===");
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
    // Note: These patterns use contains() matching (not anchored)
    runner.add_fail_pattern("Speech detection error");
    runner.add_fail_pattern("Failed to create AudioInput from bytes");
    runner.add_fail_pattern("Speech detection request failed");
    runner.add_fail_pattern("Audio stream error");
    runner.add_fail_pattern("Failed to build input stream");
    runner.add_fail_pattern("Failed to start audio stream");
    runner.add_fail_pattern("No default input device found");
    runner.add_fail_pattern("No supported 48kHz mono F32 input config");
    runner.add_fail_pattern("Failed to check secondary model load state");
    runner.add_fail_pattern("Failed to reload secondary model");
    runner.add_fail_pattern("Full error from mistralrs");
    runner.add_fail_pattern("AUDIO_PARSE_FAILED");
    runner.add_fail_pattern("[AUDIO_STATUS]: NO_AUDIO_DETECTED");

    runner.add_success_pattern("Audio interrupt");
    runner.add_success_pattern("[TYPE]:");
    runner.add_success_pattern("[SITUATION]:");

    // Pre-run cleanup
    runner.pre_run_cleanup(&["agent_a/input", "agent_a/output", "agent_b/output", "logs"])?;

    // Agent A - realtime audio listener / transcriber
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
        model: Some("Qwen/Qwen3-VL-8B-Instruct".to_string()),
        secondary_model: Some("mistralai/Voxtral-Mini-4B-Realtime-2602".to_string()),
        realtime_listener: Some("pipewire".to_string()),
        audio_chunk_min_secs: Some(3.0),
        audio_chunk_max_secs: Some(8.0),
        input_from_output: None,
        additional_inputs: Vec::new(),
        ..Default::default()
    };

    // Agent B - downstream summariser
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
        ..Default::default()
    };

    // Spawn agents
    runner.spawn_agent(&agent_a_config)?;
    std::thread::sleep(std::time::Duration::from_secs(2));

    runner.spawn_agent(&agent_b_config)?;

    // Write initial prompt to Agent A input
    let trigger_content = r#"Begin monitoring the ambient audio environment. Each time you detect and process an audio chunk, output your structured report.

IMPORTANT: If you do not detect any audio within a reasonable time, or if the audio chunks you receive contain no speech or meaningful sound, you MUST report this by outputting:
[AUDIO_STATUS]: NO_AUDIO_DETECTED

Stay in watch mode — you will receive new audio chunks as the environment changes."#;

    let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
    let trigger_path = base_dir
        .join("agent_a/input")
        .join(format!("prompt_{}.txt", timestamp));
    fs::write(&trigger_path, trigger_content)?;
    println!("    Prompt written: {:?}", trigger_path);

    println!("\n--- Watching speech from speakers should trigger audio detection ---");
    println!("--- (podcast / voice audio should be detected as speech) ---\n");

    // Run test
    let result = runner.run();

    match result {
        TestResult::Pass => Ok(()),
        TestResult::Fail => Err(anyhow::anyhow!("Test failed")),
        TestResult::Inconclusive => Err(anyhow::anyhow!(
            "Test inconclusive - no audio interrupt triggered"
        )),
    }
}
