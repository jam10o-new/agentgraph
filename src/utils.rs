use std::path::PathBuf;
use tokio::fs;

pub async fn find_leader_socket() -> Option<PathBuf> {
    let dir = "/tmp/agentgraph";
    let mut entries = fs::read_dir(dir).await.ok()?;
    let mut oldest: Option<(u32, PathBuf)> = None;

    while let Ok(Some(entry)) = entries.next_entry().await {
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with("ag-") && name.ends_with(".sock") {
            if let Some(pid_str) = name.strip_prefix("ag-").and_then(|s| s.strip_suffix(".sock")) {
                if let Ok(pid) = pid_str.parse::<u32>() {
                    // Check if process is still alive
                    if std::path::Path::new(&format!("/proc/{}", pid)).exists() {
                        if oldest.as_ref().map_or(true, |(old_pid, _)| pid < *old_pid) {
                            oldest = Some((pid, entry.path()));
                        }
                    } else {
                        // Cleanup dead socket
                        let _ = fs::remove_file(entry.path()).await;
                    }
                }
            }
        }
    }
    oldest.map(|(_, p)| p)
}
