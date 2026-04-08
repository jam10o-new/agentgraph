use std::path::PathBuf;
use tokio::fs;
use tokio::io::AsyncWriteExt;
use chrono::Local;

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
                    let is_alive = std::path::Path::new(&format!("/proc/{}", pid)).exists();
                    
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

pub struct AgentLogger {
    pub name: String,
    pub log_dir: PathBuf,
}

impl AgentLogger {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            log_dir: PathBuf::from("logs"),
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
        
        // Also print to stdout for convenience in tests/CLI
        println!("Agent {}: {}", self.name, message);
    }
}
