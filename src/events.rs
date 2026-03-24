//! Event handling for agentgraph.
//!
//! Filesystem event filtering and interrupt handling.

use crate::Args;
use anyhow::Result;
use std::path::PathBuf;
use tokio::fs::File;

/// Check if a filesystem event should trigger inference
pub fn is_interrupt_event(
    event: &notify::Event,
    last_modify: &mut Option<std::time::Instant>,
    args: &Args,
) -> bool {
    let watching: Vec<&PathBuf> = args
        .input_final
        .iter()
        .chain(args.input_cat.iter())
        .chain(args.system_final.iter())
        .chain(args.system_cat.iter())
        .collect();

    // Check if any event path is a child of (or equals) one of the watched directories.
    let is_watched = event.paths.iter().any(|event_path| {
        watching
            .iter()
            .any(|watch_dir| event_path.starts_with(watch_dir))
    });

    if !is_watched {
        return false;
    }

    match &event.kind {
        notify::EventKind::Access(notify::event::AccessKind::Close(
            notify::event::AccessMode::Write | notify::event::AccessMode::Any,
        ))
        | notify::EventKind::Modify(_) => {
            let now = std::time::Instant::now();
            let elapsed = last_modify
                .map(|t| now.duration_since(t).as_millis() as u64)
                .unwrap_or(u64::MAX);
            if elapsed >= args.sleep_ms {
                *last_modify = Some(now);
                true
            } else {
                false
            }
        }
        notify::EventKind::Create(_) => {
            let now = std::time::Instant::now();
            *last_modify = Some(now);
            true
        }
        _ => false,
    }
}

/// Handle interrupt event (create/wipe output file)
pub async fn handle_interrupt(
    event: notify::Event,
    current_file_path: &mut Option<PathBuf>,
    realtime_file: &mut Option<File>,
    args: &Args,
) -> Result<()> {
    match event.kind {
        notify::EventKind::Create(_)
        | notify::EventKind::Access(notify::event::AccessKind::Close(
            notify::event::AccessMode::Write | notify::event::AccessMode::Any,
        )) => {
            if args.output_new.is_some() {
                let new_timestamp = chrono::Utc::now().timestamp_millis();
                let new_path = args
                    .output_new
                    .as_ref()
                    .unwrap()
                    .join(format!("out-{}.txt", new_timestamp));

                if args.verbose {
                    eprintln!("Creating new file: {}", new_path.display());
                }

                if let Some(parent) = new_path.parent() {
                    let _ = tokio::fs::create_dir_all(parent).await;
                }

                *current_file_path = Some(new_path.clone());
                *realtime_file = Some(File::create(&new_path).await?);
            } else if args.output_overwrite.is_some() {
                if let Some(ref path) = args.output_overwrite {
                    *realtime_file = Some(File::create(path).await?);
                }
            }
        }
        notify::EventKind::Modify(_) => {
            if args.verbose {
                eprintln!("Wiping current output and restarting...");
            }

            if args.stream_realtime {
                if args.output_new.is_some() {
                    if let Some(ref path) = *current_file_path {
                        *realtime_file = Some(File::create(path).await?);
                    }
                } else if let Some(ref path) = args.output_overwrite {
                    *realtime_file = Some(File::create(path).await?);
                }
            }
        }
        _ => {}
    }
    Ok(())
}
