use std::process::Stdio;
use tokio::process::Command;
use std::time::Duration;
use std::fs;

async fn cleanup_sockets() {
    let dir = "/tmp/agentgraph";
    if let Ok(mut entries) = tokio::fs::read_dir(dir).await {
        while let Ok(Some(entry)) = entries.next_entry().await {
            let _ = tokio::fs::remove_file(entry.path()).await;
        }
    }
}

#[tokio::test]
async fn test_leader_uniqueness_and_status() {
    // 0. Aggressive cleanup before start
    cleanup_sockets().await;

    // 1. Create a dummy config
    let config_content = r#"
models: {}
sampling:
  temperature: 0.7
  top_p: 0.9
  top_k: 40
  max_len: 1024
agents: {}
"#;
    let config_path = "test_config_uniqueness.yaml";
    fs::write(config_path, config_content).unwrap();

    // 2. Start the leader in the background
    let mut leader = Command::new("cargo")
        .args(["run", "--bin", "ag", "--", "leader", "--config", config_path])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn leader");

    // Wait for leader to start - retry status until it works or times out
    let mut success = false;
    for _ in 0..10 {
        tokio::time::sleep(Duration::from_millis(1000)).await;
        let status_output = Command::new("cargo")
            .args(["run", "--bin", "ag", "--", "status"])
            .output()
            .await;
        
        if let Ok(out) = status_output {
            if out.status.success() {
                success = true;
                break;
            }
        }
    }
    assert!(success, "Leader failed to start or respond to status");

    // 4. Try to start another leader (should fail)
    let second_leader = Command::new("cargo")
        .args(["run", "--bin", "ag", "--", "leader", "--config", config_path])
        .output()
        .await
        .expect("Failed to run second leader");
    
    let second_stderr = String::from_utf8_lossy(&second_leader.stderr);
    assert!(second_stderr.contains("Another leader is already running"), "Second leader did not detect existing one. Stderr: {}", second_stderr);

    // 5. Shutdown
    let shutdown_output = Command::new("cargo")
        .args(["run", "--bin", "ag", "--", "shutdown"])
        .output()
        .await
        .expect("Failed to run shutdown");
    
    assert!(String::from_utf8_lossy(&shutdown_output.stdout).contains("Shutting down"));

    // Final Cleanup
    let _ = leader.kill().await;
    let _ = fs::remove_file(config_path);
    cleanup_sockets().await;
}
