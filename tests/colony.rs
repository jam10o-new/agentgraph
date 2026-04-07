use std::fs;
use std::process::Stdio;
use std::time::Duration;
use tempfile::tempdir;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::time::timeout;

async fn cleanup_sockets() {
    let dir = "/tmp/agentgraph";
    if let Ok(mut entries) = tokio::fs::read_dir(dir).await {
        while let Ok(Some(entry)) = entries.next_entry().await {
            let _ = tokio::fs::remove_file(entry.path()).await;
        }
    }
}

#[tokio::test]
async fn test_real_agent_colony() {
    // 0. Cleanup
    cleanup_sockets().await;

    let tmp = tempdir().unwrap();
    let root = tmp.path();

    let researcher_input = root.join("researcher/input");
    let researcher_output = root.join("researcher/output");
    let researcher_system = root.join("researcher/system");
    let coder_input = root.join("coder/input");
    let coder_output = root.join("coder/output");
    let coder_system = root.join("coder/system");

    fs::create_dir_all(&researcher_input).unwrap();
    fs::create_dir_all(&researcher_output).unwrap();
    fs::create_dir_all(&researcher_system).unwrap();
    fs::create_dir_all(&coder_input).unwrap();
    fs::create_dir_all(&coder_output).unwrap();
    fs::create_dir_all(&coder_system).unwrap();

    // 1. Create Config with ISQ match
    let config_content = format!(
        r#"
models:
  primary:
    id: "gemma-4-E4B"
    builder: "vision"
    path: "google/gemma-4-E4B"
    isq: "4"
sampling:
  temperature: 0.1
  top_p: 0.9
  top_k: 40
  max_len: 128
agents:
  researcher:
    inputs: ["{}"]
    output: "{}"
    system: ["{}"]
    model: "primary"
    stream: false
  coder:
    inputs: ["{}", "{}"]
    output: "{}"
    system: ["{}"]
    model: "primary"
    stream: false
"#,
        researcher_input.display(),
        researcher_output.display(),
        researcher_system.display(),
        coder_input.display(),
        researcher_output.display(),
        coder_output.display(),
        coder_system.display()
    );

    let config_path = root.join("colony_config.yaml");
    fs::write(&config_path, config_content).unwrap();

    // 2. Start Leader
    println!("Colony Test: Starting leader with cargo run...");
    let mut leader = Command::new("cargo")
        .args([
            "run",
            "--bin",
            "ag",
            "--",
            "leader",
            "--config",
            config_path.to_str().unwrap(),
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn leader");

    let stdout = leader.stdout.take().unwrap();
    let stderr = leader.stderr.take().unwrap();

    let mut stdout_reader = BufReader::new(stdout).lines();
    let mut stderr_reader = BufReader::new(stderr).lines();

    // Watch stdout for "listening on"
    println!("Colony Test: Waiting for leader to be ready ...");
    let wait_ready = async {
        while let Ok(Some(line)) = stdout_reader.next_line().await {
            println!("[LEADER STDOUT] {}", line);
            if line.contains("listening on") {
                return true;
            }
        }
        false
    };

    // Increased timeout to 10 minutes for the large model
    let ready = timeout(Duration::from_secs(1200), wait_ready)
        .await
        .unwrap_or(false);
    assert!(
        ready,
        "Leader failed to become ready (listening) within timeout"
    );

    // Start background tasks to keep draining and printing
    tokio::spawn(async move {
        while let Ok(Some(line)) = stdout_reader.next_line().await {
            println!("[LEADER STDOUT] {}", line);
        }
    });
    tokio::spawn(async move {
        while let Ok(Some(line)) = stderr_reader.next_line().await {
            eprintln!("[LEADER STDERR] {}", line);
        }
    });

    // 3. Trigger Researcher
    println!("Colony Test: Triggering researcher...");
    fs::write(
        researcher_input.join("query.txt"),
        "Research this: What is the capital of France? Only output the city name.",
    )
    .unwrap();

    // 4. Wait for Researcher Output
    let mut found_researcher_output = false;
    for i in 0..180 {
        tokio::time::sleep(Duration::from_secs(1)).await;
        if let Ok(entries) = fs::read_dir(&researcher_output) {
            let files: Vec<_> = entries.filter_map(|e| e.ok()).collect();
            if !files.is_empty() {
                println!("Colony Test: Researcher produced output after {}s", i);
                found_researcher_output = true;
                break;
            }
        }
    }
    assert!(
        found_researcher_output,
        "Researcher failed to produce output"
    );

    // 5. Wait for Coder Output (triggered by researcher output)
    println!("Colony Test: Waiting for coder...");
    let mut found_coder_output = false;
    for i in 0..180 {
        tokio::time::sleep(Duration::from_secs(1)).await;
        if let Ok(entries) = fs::read_dir(&coder_output) {
            let files: Vec<_> = entries.filter_map(|e| e.ok()).collect();
            if !files.is_empty() {
                println!("Colony Test: Coder produced output after {}s", i);
                found_coder_output = true;
                break;
            }
        }
    }

    // 6. Shutdown
    println!("Colony Test: Shutting down...");
    let _ = Command::new("cargo")
        .args(["run", "--bin", "ag", "--", "shutdown"])
        .output()
        .await;
    let _ = leader.kill().await;
    cleanup_sockets().await;

    assert!(
        found_coder_output,
        "Coder failed to trigger from researcher output"
    );
}
