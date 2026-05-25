use chrono::Local;
use std::path::PathBuf;
use tokio::fs;
use tokio::io::AsyncWriteExt;

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
            if let Some(pid_str) = name
                .strip_prefix("ag-")
                .and_then(|s| s.strip_suffix(".sock"))
            {
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
        let socket_path = std::path::Path::new(AGENTGRAPH_DIR).join(format!("ag-{}.sock", pid));
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
/// Returns the PID of the first matching process, regardless of whether
/// it has a socket.  Callers check socket existence separately.
pub fn find_leader_by_proc_scan() -> Option<u32> {
    let proc_dir = std::path::Path::new("/proc");
    if !proc_dir.exists() {
        return None;
    }
    let entries = match std::fs::read_dir(proc_dir) {
        Ok(e) => e,
        Err(_) => return None,
    };

    let our_pid = std::process::id();
    let agentgraph_dir = std::path::Path::new(AGENTGRAPH_DIR);

    for entry in entries.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if !name_str.chars().all(|c| c.is_ascii_digit()) {
            continue;
        }
        // Skip our own process — `ag status` shouldn't detect itself
        if let Ok(pid) = name_str.parse::<u32>() {
            if pid == our_pid {
                continue;
            }
        }
        let comm_path = entry.path().join("comm");
        if let Ok(comm) = std::fs::read_to_string(&comm_path) {
            if comm.trim() == "ag" {
                if let Ok(pid) = name_str.parse::<u32>() {
                    if pid_is_alive(pid) {
                        return Some(pid);
                    } else {
                        // Dead process — clean up any leftover socket
                        let socket_path = agentgraph_dir.join(format!("ag-{}.sock", pid));
                        if socket_path.exists() {
                            let _ = std::fs::remove_file(&socket_path);
                        }
                    }
                }
            }
        }
    }
    None
}

/// Determine leader status via socket → pidfile → /proc scan.
pub async fn is_leader_alive() -> LeaderStatus {
    // Tier 1: socket-based (fastest, returns usable socket)
    if let Some(socket) = find_leader_socket().await {
        // Extract PID from socket filename
        let socket_name = socket.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if let Some(pid_str) = socket_name
            .strip_prefix("ag-")
            .and_then(|s| s.strip_suffix(".sock"))
        {
            if let Ok(pid) = pid_str.parse::<u32>() {
                return LeaderStatus::Ready { socket, pid };
            }
        }
        // If we can't parse the PID from the socket name, still usable
        return LeaderStatus::Ready { socket, pid: 0 };
    }

    // Tier 2: PID file fallback
    if let Some(socket) = find_leader_by_pidfile() {
        // Try to get PID from the file
        if let Ok(content) = std::fs::read_to_string(LEADER_PID_FILE) {
            if let Ok(pid) = content.trim().parse::<u32>() {
                if socket.exists() {
                    return LeaderStatus::Ready { socket, pid };
                }
                if pid_is_alive(pid) {
                    return LeaderStatus::Degraded { pid };
                }
            }
        }
    }

    // Tier 3: /proc scan (slow but reliable)
    if let Some(pid) = find_leader_by_proc_scan() {
        let socket_path = std::path::Path::new(AGENTGRAPH_DIR).join(format!("ag-{}.sock", pid));
        if socket_path.exists() {
            return LeaderStatus::Ready {
                socket: socket_path,
                pid,
            };
        }
        return LeaderStatus::Degraded { pid };
    }

    LeaderStatus::NotRunning
}

/// Result of leader detection, distinguishing ready vs degraded states.
#[derive(Debug)]
pub enum LeaderStatus {
    /// Leader is running and its IPC socket is available.
    Ready { socket: PathBuf, pid: u32 },
    /// A leader process was found via /proc scan but has no socket.
    /// The process may still be loading models, or the socket was lost.
    Degraded { pid: u32 },
    /// No leader process detected by any method.
    NotRunning,
}

/// Check if a process with the given PID is actually alive (not a zombie).
/// Reads `/proc/{pid}/status` to distinguish running/sleeping from zombie.
fn pid_is_alive(pid: u32) -> bool {
    let status_path = format!("/proc/{}/status", pid);
    if let Ok(content) = std::fs::read_to_string(&status_path) {
        for line in content.lines() {
            if line.starts_with("State:") {
                // State line format: "State:\tS (sleeping)" or "State:\tZ (zombie)"
                // A zombie has already exited; its parent hasn't reaped it.
                return !line.contains('Z') && !line.contains('z');
            }
        }
    }
    // Can't read status → assume dead
    false
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
