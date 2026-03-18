//! IPC (Inter-Process Communication) for agentgraph.
//!
//! Unix domain socket communication and leader election via pipe files.

use crate::types::InferenceRequest;
use crate::{PIPE_DIR, PIPE_PREFIX};
use anyhow::Result;
use std::path::PathBuf;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};

/// Generate pipe path from PID
pub fn pipe_path(pid: u32) -> PathBuf {
    PathBuf::from(PIPE_DIR).join(format!("{}{}", PIPE_PREFIX, pid))
}

/// Find the oldest (leader) pipe
pub async fn find_oldest_pipe(my_pid: u32) -> Option<PathBuf> {
    let mut dir = tokio::fs::read_dir(PIPE_DIR).await.ok()?;
    let mut oldest: Option<(u32, PathBuf)> = None;

    while let Ok(Some(entry)) = dir.next_entry().await {
        let name = entry.file_name().to_str()?.to_owned();
        if let Some(pid_str) = name.strip_prefix(PIPE_PREFIX) {
            let pid = pid_str.parse::<u32>().ok()?;
            if pid < my_pid && is_process_alive(pid) {
                if oldest.clone().map_or(true, |(old_pid, _)| pid < old_pid) {
                    oldest = Some((pid, entry.path()));
                }
            } else if pid != my_pid && !is_process_alive(pid) {
                let _ = tokio::fs::remove_file(entry.path()).await;
            }
        }
    }
    oldest.map(|(_, p)| p)
}

/// Check if a process is alive
pub fn is_process_alive(pid: u32) -> bool {
    std::path::Path::new(&format!("/proc/{}", pid)).exists()
}

/// Cleanup pipe file on exit
pub async fn cleanup_my_pipe(pid: u32) {
    let _ = tokio::fs::remove_file(pipe_path(pid)).await;
}

/// Send inference request to leader
pub async fn send_request(pipe: &PathBuf, req: InferenceRequest) -> Result<()> {
    let mut stream = UnixStream::connect(pipe).await?;
    let data = serde_json::to_vec(&req)?;
    stream.write_all(&data).await?;
    stream.write_all(b"\n").await?;
    stream.flush().await?;
    Ok(())
}

/// Handle incoming request from follower
pub async fn handle_request(
    stream: UnixStream,
    model: std::sync::Arc<mistralrs::Model>,
    interrupt_tx: tokio::sync::broadcast::Sender<
        Result<notify::Event, std::sync::Arc<notify::Error>>,
    >,
    audio_channels: Option<(
        tokio::sync::broadcast::Receiver<()>,
        tokio::sync::watch::Receiver<Option<Vec<u8>>>,
    )>,
    pending_audio: Vec<Vec<u8>>,
) {
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    if let Ok(_) = reader.read_line(&mut line).await {
        if let Ok(req) = serde_json::from_str::<InferenceRequest>(&line) {
            println!("Handling request from PID {}", req.requesting_pid);
            let interrupt_rx = interrupt_tx.subscribe();
            let _ = crate::inference::run_once(
                &model,
                &req.args,
                Some(interrupt_rx),
                audio_channels,
                pending_audio,
            )
            .await;
        }
    }
}

/// Create and bind Unix listener
pub async fn create_listener(my_pid: u32) -> Result<(UnixListener, PathBuf)> {
    tokio::fs::create_dir_all(PIPE_DIR).await?;
    let my_pipe_path = pipe_path(my_pid);
    let _ = tokio::fs::remove_file(&my_pipe_path).await;
    let listener = UnixListener::bind(&my_pipe_path)?;
    Ok((listener, my_pipe_path))
}
