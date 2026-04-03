use std::path::PathBuf;
use tokio::fs;

pub async fn find_leader_socket() -> Option<PathBuf> {
    let dir = "/tmp/agentgraph";
    if !std::path::Path::new(dir).exists() {
        return None;
    }
    let mut entries = fs::read_dir(dir).await.ok()?;
    let mut oldest: Option<(u32, PathBuf)> = None;

    while let Ok(Some(entry)) = entries.next_entry().await {
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with("ag-") && name.ends_with(".sock") {
            if let Some(pid_str) = name.strip_prefix("ag-").and_then(|s| s.strip_suffix(".sock")) {
                if let Ok(pid) = pid_str.parse::<u32>() {
                    // More robust process check: Use kill(pid, 0) via libc or just check /proc/PID/stat
                    // to ensure it's actually the same process and not a reused PID.
                    // For now, let's just be more aggressive about cleaning up.
                    let is_alive = std::path::Path::new(&format!("/proc/{}", pid)).exists();
                    
                    if is_alive {
                        if oldest.as_ref().map_or(true, |(old_pid, _)| pid < *old_pid) {
                            oldest = Some((pid, entry.path()));
                        }
                    } else {
                        // Aggressively cleanup dead socket
                        let _ = fs::remove_file(entry.path()).await;
                    }
                }
            }
        }
    }
    oldest.map(|(_, p)| p)
}
