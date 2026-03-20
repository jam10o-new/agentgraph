//! AgentGraph daemon entry point.

use agentgraph::Args;
use agentgraph::{
    audio::spawn_realtime_listener,
    command_exec::spawn_command_io,
    inference::run_once,
    ipc::{cleanup_my_pipe, create_listener, find_oldest_pipe, handle_request, send_request},
    messages::collect_viewer_dirs,
    model::load_model,
    utils::{has_any_input, has_any_output},
};
use anyhow::Result;
use clap::Parser;
use notify::{RecursiveMode, Watcher};
use std::sync::Arc;
use tokio::sync::broadcast;
use tokio::sync::watch;
use tokio::sync::{Mutex, oneshot};

#[tokio::main]
async fn main() -> Result<()> {
    let my_pid = std::process::id();

    let (listener, my_pipe_path) = create_listener(my_pid).await?;
    let inference_lock = Arc::new(Mutex::new(()));

    println!("PID {} listening on {:?}", my_pid, my_pipe_path);

    let cleanup_pid = my_pid;
    tokio::spawn(async move {
        let _ = tokio::signal::ctrl_c().await;
        cleanup_my_pipe(cleanup_pid).await;
        std::process::exit(0);
    });

    let args = Args::parse();

    let has_input = has_any_input(&args);
    let has_output = has_any_output(&args);

    if !has_input || !has_output {
        eprintln!("No input or output specified.");
        std::process::exit(0);
    }

    // Filesystem event watcher
    let (fs_tx, mut fs_rx) = broadcast::channel(64);
    let tx_clone = fs_tx.clone();
    let mut watcher =
        notify::recommended_watcher(move |res: Result<notify::Event, notify::Error>| {
            let res = res.map_err(Arc::new);
            let _ = tx_clone.send(res);
        })?;

    let dirs = collect_viewer_dirs(&args).await?;
    let dir_args = dirs.iter().map(|p| p.to_string_lossy());

    for d in args
        .input_final
        .iter()
        .chain(args.input_cat.iter())
        .chain(args.system_final.iter())
        .chain(args.system_cat.iter())
    {
        tokio::fs::create_dir_all(d).await?;
        watcher.watch(d, RecursiveMode::NonRecursive)?;
        if args.verbose {
            eprintln!("Watching: {}", d.display());
        }
    }

    // Spawn psi-viewer process
    if !args.no_ui {
        let mut io = spawn_command_io("psi-viewer", dir_args).await?;
        let mut editor_exit = io.exited_rx.take();

        tokio::spawn(async move {
            loop {
                tokio::select! {
                    Some(chunk) = io.stdout_rx.recv() => {
                        print!("{}", String::from_utf8_lossy(&chunk));
                    }

                    Some(chunk) = io.stderr_rx.recv() => {
                        eprint!("{}", String::from_utf8_lossy(&chunk));
                    }

                    _ = async {
                        if let Some(rx) = &mut editor_exit {
                            let _ = rx.await;
                        }
                    }, if editor_exit.is_some() => {
                        eprintln!("Editor exited — shutting down daemon.");
                        std::process::exit(0);
                    }

                    else => break,
                }
            }
        });
    }

    let mut model: Option<Arc<mistralrs::Model>> = None;

    // Persistent realtime listener
    let mut persistent_listener: Option<(
        broadcast::Receiver<()>,
        watch::Receiver<Option<Vec<u8>>>,
    )> = None;
    let mut _listener_guard: Option<oneshot::Sender<()>> = None;
    let mut queued_audio: Vec<Vec<u8>> = Vec::new();
    let mut last_modify_interrupt: Option<std::time::Instant> = None;
    let mut upstream = find_oldest_pipe(my_pid).await;

    if let Some(upstream_pipe) = upstream {
        if !args.watch {
            let req = agentgraph::InferenceRequest {
                args: args.clone(),
                requesting_pid: my_pid,
            };
            if let Err(e) = send_request(&upstream_pipe, req).await {
                eprintln!("Failed to reach leader: {}", e);
            }
        }
    }

    loop {
        upstream = find_oldest_pipe(my_pid).await;

        tokio::select! {
            event = fs_rx.recv() => {
                match event {
                    Ok(Ok(event)) => {
                        if args.verbose {
                            eprintln!("Change detected: {:?}", event);
                        }

                        // Filter filesystem events
                        let is_interrupt = agentgraph::events::is_interrupt_event(
                            &event,
                            &mut last_modify_interrupt,
                            &args,
                        );

                        if !is_interrupt {
                            if args.verbose {
                                eprintln!("Ignoring non-interrupt event: {:?}", event);
                            }
                            continue;
                        }

                        // Skip if inference in progress (leader only)
                        let should_skip = upstream.is_none() && inference_lock.try_lock().is_err();

                        if should_skip {
                            if args.verbose {
                                eprintln!("Inference in progress; event delegated to running task.");
                            }
                            continue;
                        }

                        // Drain pending events
                        while let Ok(ev) = fs_rx.try_recv() {
                            if args.verbose {
                                eprintln!("Additional: {:?}", ev);
                            }
                        }

                        println!("Running inference after filesystem change");

                        if let Some(upstream_pipe) = upstream {
                            let req = agentgraph::InferenceRequest {
                                args: args.clone(),
                                requesting_pid: my_pid,
                            };
                            if let Err(e) = send_request(&upstream_pipe, req).await {
                                eprintln!("Failed to reach leader: {}", e);
                            }
                        } else {
                            // Leader path
                            if model.is_none() {
                                println!("Loading model for first inference...");
                                model = Some(Arc::new(load_model(&args).await?));
                                if args.verbose {
                                    eprintln!("Model initialized");
                                }

                                // Spawn persistent audio listener
                                if args.realtime_listener.is_some() && persistent_listener.is_none() {
                                    let source = args.realtime_listener.as_ref().unwrap();
                                    match spawn_realtime_listener(source, &args, model.as_ref().unwrap().clone()).await {
                                        Ok(rl) => {
                                            let (latest_audio_tx, latest_audio_rx) = watch::channel::<Option<Vec<u8>>>(None);
                                            let speech_rx = rl.speech_detected_rx.resubscribe();
                                            let mut chunk_rx = rl.chunk_rx;
                                            tokio::spawn(async move {
                                                while let Some(chunk) = chunk_rx.recv().await {
                                                    let _ = latest_audio_tx.send(Some(chunk));
                                                }
                                            });
                                            persistent_listener = Some((speech_rx, latest_audio_rx));
                                            _listener_guard = Some(rl.shutdown_tx);
                                            if args.verbose {
                                                eprintln!("Persistent audio listener spawned.");
                                            }
                                        }
                                        Err(e) => eprintln!("Failed to spawn realtime listener: {}", e),
                                    }
                                }
                            }

                            // Spawn inference task
                            let lock = inference_lock.clone();
                            let m = model.as_ref().unwrap().clone();
                            let interrupt_tx = fs_tx.clone();
                            let audio_chans = persistent_listener.as_ref().map(|(sdr, lar)| {
                                (sdr.resubscribe(), lar.clone())
                            });
                            let audio_queue = std::mem::take(&mut queued_audio);
                            let req_args = args.clone();

                            tokio::spawn(async move {
                                let _guard = lock.lock().await;
                                let interrupt_rx = interrupt_tx.subscribe();
                                let _ = run_once(&m, &req_args, Some(interrupt_rx), audio_chans, audio_queue).await;
                            });
                        }
                    }
                    Ok(Err(e)) => {
                        eprintln!("Watcher error: {}", e);
                        break;
                    }
                    Err(broadcast::error::RecvError::Closed) => {
                        break;
                    }
                    Err(broadcast::error::RecvError::Lagged(_)) => {
                        continue;
                    }
                }
            }

            result = listener.accept() => {
                if let Ok((stream, _)) = result {
                    if model.is_none() {
                        println!("Loading model for first request...");
                        model = Some(Arc::new(load_model(&args).await?));
                        if args.verbose {
                            eprintln!("Model initialized");
                        }

                        if args.realtime_listener.is_some() && persistent_listener.is_none() {
                            let source = args.realtime_listener.as_ref().unwrap();
                            match spawn_realtime_listener(source, &args, model.as_ref().unwrap().clone()).await {
                                Ok(rl) => {
                                    let (latest_audio_tx, latest_audio_rx) = watch::channel::<Option<Vec<u8>>>(None);
                                    let speech_rx = rl.speech_detected_rx.resubscribe();
                                    let mut chunk_rx = rl.chunk_rx;
                                    tokio::spawn(async move {
                                        while let Some(chunk) = chunk_rx.recv().await {
                                            let _ = latest_audio_tx.send(Some(chunk));
                                        }
                                    });
                                    persistent_listener = Some((speech_rx, latest_audio_rx));
                                    _listener_guard = Some(rl.shutdown_tx);
                                    if args.verbose {
                                        eprintln!("Persistent audio listener spawned.");
                                    }
                                }
                                Err(e) => eprintln!("Failed to spawn realtime listener: {}", e),
                            }
                        }
                    }

                    let m = model.as_ref().unwrap().clone();
                    let lock = inference_lock.clone();
                    let interrupt_tx = fs_tx.clone();
                    let audio_chans = persistent_listener.as_ref().map(|(sdr, lar)| {
                        (sdr.resubscribe(), lar.clone())
                    });
                    let audio_queue = std::mem::take(&mut queued_audio);

                    tokio::spawn(async move {
                        let _guard = lock.lock().await;
                        handle_request(stream, m, interrupt_tx, audio_chans, audio_queue).await;
                    });
                }
            }

            // Audio-triggered inference
            _ = async {
                if let Some((ref mut sdr, _)) = persistent_listener {
                    sdr.recv().await.ok().map(|_| ())
                } else {
                    std::future::pending::<Option<()>>().await
                }
            }, if inference_lock.try_lock().is_ok() => {
                if args.verbose {
                    eprintln!("Speech detected — triggering audio-first inference.");
                }
                if let Some((_, ref lar)) = persistent_listener {
                    if let Some(chunk) = lar.borrow().clone() {
                        queued_audio.push(chunk);
                    }
                }
                if model.is_none() {
                    println!("Loading model for audio-triggered inference...");
                    model = Some(Arc::new(load_model(&args).await?));
                }
                let req = agentgraph::InferenceRequest {
                    args: args.clone(),
                    requesting_pid: my_pid,
                };
                let _ = send_request(&my_pipe_path, req).await;
            }
        }

        if !args.watch {
            break;
        }
    }

    cleanup_my_pipe(my_pid).await;
    Ok(())
}
