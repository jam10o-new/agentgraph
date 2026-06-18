use tokio::process::Command;
use std::time::Duration;
use std::fs;

/// Resolve the locally built `ag` binary relative to the test binary.
fn ag_binary() -> String {
    let mut path = std::env::current_exe().unwrap();
    path.pop(); // deps/
    path.pop(); // debug/
    path.push("ag");
    path.to_string_lossy().to_string()
}

/// Returns `true` if a leader appears to be running or in a degraded state
/// (e.g. from systemd) so the test will skip.
async fn leader_already_running(ag: &str) -> bool {
    let out = Command::new(ag).args(["status"]).output().await;
    match out {
        Ok(o) => {
            let s = String::from_utf8_lossy(&o.stdout);
            s.contains("Active Agents")
                || s.contains("leader process detected")
                || s.contains("Leader process")
        }
        Err(_) => false,
    }
}

/// Nuke leftover /tmp/agentgraph/ files (socket, PID, log) from a previous run.
async fn clean_runtime_dir() {
    let dir = std::path::Path::new("/tmp/agentgraph");
    if dir.exists() {
        if let Ok(mut entries) = tokio::fs::read_dir(dir).await {
            while let Ok(Some(entry)) = entries.next_entry().await {
                let _ = tokio::fs::remove_file(entry.path()).await;
            }
        }
    }
}

/// Helper: start a leader, wait for it to be Ready, run the closure, then
/// shut down and clean up.  Skips the test if a leader is already running.
async fn with_leader<F, Fut>(config_content: &str, config_path: &str, f: F)
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = ()>,
{
    let ag = ag_binary();
    if leader_already_running(&ag).await {
        eprintln!("SKIP: a leader is already running on this system (systemd?).");
        return;
    }

    clean_runtime_dir().await;
    fs::write(config_path, config_content).unwrap();

    let _ = Command::new(&ag)
        .args(["leader", "--config", config_path])
        .output()
        .await
        .expect("Failed to start leader");

    let mut ready = false;
    for _ in 0..60 {
        tokio::time::sleep(Duration::from_millis(500)).await;
        let out = Command::new(&ag).args(["status"]).output().await;
        if let Ok(out) = out {
            if out.status.success()
                && String::from_utf8_lossy(&out.stdout).contains("Active Agents")
            {
                ready = true;
                break;
            }
        }
    }
    assert!(ready, "Leader failed to start");

    f().await;

    let shutdown = Command::new(&ag)
        .args(["shutdown"])
        .output()
        .await
        .expect("Failed to run shutdown");
    assert!(
        String::from_utf8_lossy(&shutdown.stdout).contains("Shutting down")
    );

    let _ = fs::remove_file(config_path);
}

#[tokio::test]
async fn test_leader_uniqueness_and_status() {
    let ag = ag_binary();
    if leader_already_running(&ag).await {
        eprintln!("SKIP: a leader is already running (systemd?).");
        return;
    }

    clean_runtime_dir().await;

    let config_content = r#"
models: {}
agents: {}
"#;
    let config_path = "/tmp/test_config_uniqueness.yaml";
    fs::write(config_path, config_content).unwrap();

    // Start the leader
    let _ = Command::new(&ag)
        .args(["leader", "--config", config_path])
        .output()
        .await
        .expect("Failed to start leader");

    // Wait for leader to be Ready (up to 30 s)
    let mut success = false;
    for _ in 0..60 {
        tokio::time::sleep(Duration::from_millis(500)).await;
        let out = Command::new(&ag).args(["status"]).output().await;
        if let Ok(out) = out {
            if out.status.success()
                && String::from_utf8_lossy(&out.stdout).contains("Active Agents")
            {
                success = true;
                break;
            }
        }
    }
    assert!(success, "Leader failed to start or respond to status");

    // Second leader call should update config on the existing leader
    let second = Command::new(&ag)
        .args(["leader", "--config", config_path])
        .output()
        .await
        .expect("Failed to run second leader");

    let second_stdout = String::from_utf8_lossy(&second.stdout);
    assert!(
        second_stdout.contains("Config updated"),
        "Second leader call should have updated config. Stdout: {}",
        second_stdout
    );

    // Shutdown
    let shutdown = Command::new(&ag)
        .args(["shutdown"])
        .output()
        .await
        .expect("Failed to run shutdown");
    assert!(
        String::from_utf8_lossy(&shutdown.stdout).contains("Shutting down")
    );

    let _ = fs::remove_file(config_path);
}

#[tokio::test]
async fn test_model_idle_secs_does_not_crash_leader() {
    // Verify that `model_idle_secs` is accepted by the config parser
    // and that the background idle-checker task runs without crashing.
    //
    // With an empty models map there is nothing to unload, but the
    // idle watchdog loop still spins and exits gracefully each cycle.
    //
    // A full end-to-end test of actual unload/reload requires a real
    // GGUF model file and is more suited for manual / CI-with-artifacts
    // scenarios.  This test also skips when a systemd-managed leader
    // is already present.
    let config = r#"
models: {}
agents: {}
model_idle_secs: 2
"#;

    with_leader(config, "/tmp/test_idle_config.yaml", || async {
        // Wait for at least two idle-check cycles (10 s each)
        tokio::time::sleep(Duration::from_secs(25)).await;

        // The leader should still be alive and respond to status.
        let ag = ag_binary();
        let out = Command::new(&ag)
            .args(["status"])
            .output()
            .await
            .expect("status should work");

        assert!(
            out.status.success(),
            "Leader should still be responsive after idle cycles: {}",
            String::from_utf8_lossy(&out.stdout)
        );
    })
    .await;
}