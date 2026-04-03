use tokio::process::Command;
use std::time::Duration;
use std::fs;
use tempfile::tempdir;
use std::process::Stdio;

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

    // 1. Create Config with Real Models from Cache
    let config_content = format!(r#"
models:
  primary:
    id: "gemma-4-E4B"
    builder: "vision"
    path: "google/gemma-4-E4B"
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
    researcher_input.display(), researcher_output.display(), researcher_system.display(),
    coder_input.display(), researcher_output.display(), coder_output.display(), coder_system.display());

    let config_path = root.join("colony_config.yaml");
    fs::write(&config_path, config_content).unwrap();

    // 2. Start Leader
    let mut leader = Command::new("cargo")
        .args(["run", "--bin", "ag", "--", "leader", "--config", config_path.to_str().unwrap()])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn leader");

    // Wait for leader to be ready (load models takes time)
    tokio::time::sleep(Duration::from_secs(30)).await;

    // 3. Trigger Researcher
    fs::write(researcher_input.join("query.txt"), "Research this: What is the capital of France? Only output the city name.").unwrap();

    // Wait for researcher to finish and write to researcher_output
    // This should automatically trigger the Coder because coder watches researcher_output.
    let mut found_researcher_output = false;
    for _ in 0..60 {
        tokio::time::sleep(Duration::from_secs(1)).await;
        if let Ok(entries) = fs::read_dir(&researcher_output) {
            if entries.count() > 0 {
                found_researcher_output = true;
                break;
            }
        }
    }
    assert!(found_researcher_output, "Researcher failed to produce output");

    // 4. Check if Coder triggered and produced output
    // Coder's system prompt could be "You are a coder who translates research into code comments."
    fs::write(coder_system.join("prompt.txt"), "You are an assistant. When you see research, acknowledge it briefly.").unwrap();
    
    let mut found_coder_output = false;
    for _ in 0..60 {
        tokio::time::sleep(Duration::from_secs(1)).await;
        if let Ok(entries) = fs::read_dir(&coder_output) {
            if entries.count() > 0 {
                found_coder_output = true;
                break;
            }
        }
    }
    assert!(found_coder_output, "Coder failed to trigger from researcher output");

    // 5. Cleanup
    let _ = Command::new("cargo")
        .args(["run", "--bin", "ag", "--", "shutdown"])
        .output()
        .await;
    let _ = leader.kill().await;
    cleanup_sockets().await;
}
