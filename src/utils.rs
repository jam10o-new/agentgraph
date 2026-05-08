use std::path::PathBuf;
use tokio::fs;
use tokio::io::AsyncWriteExt;
use chrono::Local;

/// AgentGraph runtime directory under /tmp.
pub const AGENTGRAPH_DIR: &str = "/tmp/agentgraph";
/// PID file written by the leader on startup, removed on shutdown.
pub const LEADER_PID_FILE: &str = "/tmp/agentgraph/leader.pid";

/// Tier-1: socket-based detection (fastest, also returns the socket path).
pub async fn find_leader_socket() -> Option<PathBuf> {
    let dir = AGENTGRAPH_DIR;
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
                    let is_alive = pid_is_alive(pid);
                    if is_alive {
                        if oldest.as_ref().map_or(true, |(old_pid, _)| pid < *old_pid) {
                            oldest = Some((pid, entry.path()));
                        }
                    } else {
                        let _ = fs::remove_file(entry.path()).await;
                    }
                }
            }
        }
    }
    oldest.map(|(_, p)| p)
}

/// Tier-2: PID file check. Reads leader.pid and checks if that process is alive.
pub fn find_leader_by_pidfile() -> Option<PathBuf> {
    let pid_path = std::path::Path::new(LEADER_PID_FILE);
    if !pid_path.exists() {
        return None;
    }
    let content = std::fs::read_to_string(pid_path).ok()?;
    let pid: u32 = content.trim().parse().ok()?;
    if pid_is_alive(pid) {
        // Try socket-based detection with this specific PID
        let socket_path = std::path::Path::new(AGENTGRAPH_DIR)
            .join(format!("ag-{}.sock", pid));
        if socket_path.exists() {
            Some(socket_path)
        } else {
            // PID is alive but no socket — still report alive
            Some(socket_path)
        }
    } else {
        // Stale PID file — clean up
        let _ = std::fs::remove_file(pid_path);
        None
    }
}

/// Tier-3: /proc scan for any `ag` process (last resort, slow).
/// Only returns a result if the matching process has a socket file
/// in /tmp/agentgraph, confirming it's OUR leader not an unrelated
/// `ag` binary from another workspace.
pub fn find_leader_by_proc_scan() -> Option<PathBuf> {
    let proc_dir = std::path::Path::new("/proc");
    if !proc_dir.exists() {
        return None;
    }
    let entries = match std::fs::read_dir(proc_dir) {
        Ok(e) => e,
        Err(_) => return None,
    };

    let agentgraph_dir = std::path::Path::new(AGENTGRAPH_DIR);

    for entry in entries.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if !name_str.chars().all(|c| c.is_ascii_digit()) {
            continue;
        }
        let comm_path = entry.path().join("comm");
        if let Ok(comm) = std::fs::read_to_string(&comm_path) {
            if comm.trim() == "ag" {
                if let Ok(pid) = name_str.parse::<u32>() {
                    // Only count it if the socket file exists AND the
                    // process is alive — confirms this is our leader.
                    let socket_path = agentgraph_dir.join(format!("ag-{}.sock", pid));
                    if socket_path.exists() && pid_is_alive(pid) {
                        return Some(socket_path);
                    } else if socket_path.exists() {
                        // Stale socket — clean it up
                        let _ = std::fs::remove_file(&socket_path);
                    }
                }
            }
        }
    }
    None
}

/// Is the leader alive? Triages through socket → pidfile → /proc scan.
/// Returns `(true, Option<socket_path>)` — socket_path is set if a
/// usable socket was found (for IPC).
pub async fn is_leader_alive() -> (bool, Option<PathBuf>) {
    // Tier 1: socket-based (fastest, returns usable socket path)
    if let Some(socket) = find_leader_socket().await {
        return (true, Some(socket));
    }
    // Tier 2: PID file fallback
    if let Some(socket) = find_leader_by_pidfile() {
        return (true, Some(socket));
    }
    // Tier 3: /proc scan (slow but reliable under stress)
    if let Some(socket) = find_leader_by_proc_scan() {
        return (true, Some(socket));
    }
    (false, None)
}

/// Check if a process with the given PID is alive on this system.
fn pid_is_alive(pid: u32) -> bool {
    std::path::Path::new(&format!("/proc/{}", pid)).exists()
}

pub struct AgentLogger {
    pub name: String,
    pub log_dir: PathBuf,
    pub quiet: bool,
}

impl AgentLogger {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            log_dir: PathBuf::from("logs"),
            quiet: false,
        }
    }

    pub async fn log(&self, message: &str) {
        let _ = fs::create_dir_all(&self.log_dir).await;
        let log_file = self.log_dir.join(format!("{}.log", self.name));
        let timestamp = Local::now().format("%Y-%m-%d %H:%M:%S%.3f");
        let line = format!("[{}] {}\n", timestamp, message);
        
        if let Ok(mut file) = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_file)
            .await 
        {
            let _ = file.write_all(line.as_bytes()).await;
        }
        
        if !self.quiet {
            println!("Agent {}: {}", self.name, message);
        }
    }
}
