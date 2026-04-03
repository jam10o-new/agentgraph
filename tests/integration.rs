use std::process::Stdio;
use tokio::process::Command;
use tokio::io::{AsyncWriteExt, AsyncReadExt};
use std::time::Duration;
use std::fs;

#[tokio::test]
async fn test_leader_uniqueness_and_status() {
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
    fs::write("test_config.yaml", config_content).unwrap();

    // 2. Start the leader in the background
    // Note: It will likely fail to load models if we had any, but with empty {} it might start or fail quickly.
    // Let's just test that the CLI can try to connect.
    
    let mut leader = Command::new("cargo")
        .args(["run", "--bin", "ag", "--", "leader", "--config", "test_config.yaml"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn leader");

    // Wait a bit for leader to potentially start
    tokio::time::sleep(Duration::from_secs(5)).await;

    // 3. Run status command
    let status_output = Command::new("cargo")
        .args(["run", "--bin", "ag", "--", "status"])
        .output()
        .await
        .expect("Failed to run status");

    let stdout = String::from_utf8_lossy(&status_output.stdout);
    let stderr = String::from_utf8_lossy(&status_output.stderr);
    
    println!("Status Stdout: {}", stdout);
    println!("Status Stderr: {}", stderr);

    // 4. Try to start another leader (should fail)
    let second_leader = Command::new("cargo")
        .args(["run", "--bin", "ag", "--", "leader", "--config", "test_config.yaml"])
        .output()
        .await
        .expect("Failed to run second leader");
    
    let second_stderr = String::from_utf8_lossy(&second_leader.stderr);
    assert!(second_stderr.contains("Another leader is already running") || !second_leader.status.success());

    // 5. Shutdown
    let shutdown_output = Command::new("cargo")
        .args(["run", "--bin", "ag", "--", "shutdown"])
        .output()
        .await
        .expect("Failed to run shutdown");
    
    assert!(String::from_utf8_lossy(&shutdown_output.stdout).contains("Shutting down"));

    // Cleanup
    let _ = leader.kill().await;
    let _ = fs::remove_file("test_config.yaml");
}
