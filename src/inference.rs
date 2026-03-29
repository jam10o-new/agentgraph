//! Inference system for agentgraph.
//!
//! Handles parallel inference, streaming, and the main inference loop.

use crate::Args;
use crate::command_exec::execute_command;
use crate::commands::CommandParser;
use crate::messages::build_messages;
use crate::types::{
    CoroutineResponse, InterruptKind, MODEL_SECONDARY, ModelSlot, ParallelInferenceParams,
    StreamOutcome,
};
use anyhow::{Context, Result};
use futures::StreamExt;
use futures::stream::FuturesUnordered;
use mistralrs::{
    ChatCompletionChunkResponse, ChunkChoice, Delta, RequestBuilder, RequestLike, ResponseOk,
    SamplingParams, TextMessageRole,
};
use std::sync::Arc;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;
use tokio::sync::broadcast;
use tokio::sync::mpsc::{UnboundedSender, unbounded_channel};
use tokio::sync::watch;

/// Extract content from mistralrs response chunk
pub fn extract_content(chunk: mistralrs::Response) -> anyhow::Result<String> {
    match chunk.as_result() {
        Ok(ResponseOk::Chunk(ChatCompletionChunkResponse { choices, .. })) => {
            if let Some(ChunkChoice {
                delta:
                    Delta {
                        content: Some(text),
                        ..
                    },
                ..
            }) = choices.first()
            {
                Ok(text.clone())
            } else {
                Ok(String::new())
            }
        }
        Ok(other) => {
            eprintln!("extract_content: Received non-chunk response: {:?}", other);
            Ok(format!("{:?}", other))
        }
        Err(e) => {
            eprintln!("extract_content: Received error: {:?}", e);
            Err(anyhow::anyhow!("{}", e))
        }
    }
}

/// Run a single inference coroutine
async fn run_inference_coroutine(
    model: Arc<mistralrs::Model>,
    params: ParallelInferenceParams,
    mut cancel_rx: watch::Receiver<bool>,
    token_tx: Option<UnboundedSender<String>>,
) -> CoroutineResponse {
    let sampling_params = SamplingParams {
        temperature: Some(0.9),        // Lowered from 1.5 for coherence
        top_k: Some(40),               // Standard "diverse but safe" pool
        top_p: Some(0.9),              // Typical "nucleus" sampling threshold
        min_p: Some(0.05),             // Filters out total noise
        repetition_penalty: Some(1.1), // Subtle; prevents loops without breaking code
        frequency_penalty: Some(0.02), // Tiny additive penalty to discourage overuse
        presence_penalty: Some(0.02),  // Tiny additive penalty to encourage new topics
        n_choices: 1,
        ..SamplingParams::deterministic()
    };

    eprintln!(
        "run_inference_coroutine: Calling stream_chat_request for slot {:?}",
        params.model_slot
    );
    let stream_result = match params.model_slot {
        ModelSlot::Primary => {
            let respondable: RequestBuilder = params.messages.into();
            eprintln!(
                "run_inference_coroutine: Primary model request created with {} messages",
                respondable.messages_ref().len()
            );
            model
                .stream_chat_request(respondable.set_sampling(sampling_params))
                .await
        }
        ModelSlot::Secondary => {
            let respondable: RequestBuilder = params.messages.into();
            eprintln!(
                "run_inference_coroutine: Secondary model request created with {} messages",
                respondable.messages_ref().len()
            );
            model
                .stream_chat_request_with_model(
                    respondable.set_sampling(sampling_params),
                    Some(MODEL_SECONDARY),
                )
                .await
        }
    };

    match &stream_result {
        Ok(_) => eprintln!("run_inference_coroutine: stream_chat_request returned Ok"),
        Err(e) => eprintln!(
            "run_inference_coroutine: stream_chat_request returned Err: {:?}",
            e
        ),
    }

    eprintln!("run_inference_coroutine: stream_chat_request returned");
    let mut stream = match stream_result {
        Ok(s) => {
            eprintln!("run_inference_coroutine: Stream created successfully");
            s
        }
        Err(e) => {
            eprintln!("run_inference_coroutine: Stream creation failed: {:?}", e);
            return CoroutineResponse::Error(e.into());
        }
    };

    let mut buffer = String::new();
    let mut empty_chunk_count = 0;
    const MAX_EMPTY_CHUNKS: usize = 500; // Force terminate if model stalls

    eprintln!("run_inference_coroutine: Entering select! loop");
    loop {
        tokio::select! {
            chunk_opt = stream.next() => {
                match chunk_opt {
                    Some(chunk) => {
                        // eprintln!("run_inference_coroutine: Received chunk from stream");
                        match extract_content(chunk) {
                            Ok(content) if !content.is_empty() => {
                                empty_chunk_count = 0; // Reset patience
                                if let Some(ref tx) = token_tx {
                                    let _ = tx.send(content.clone());
                                }
                                buffer.push_str(&content);
                            }
                            Ok(_) => {
                                empty_chunk_count += 1;
                                if empty_chunk_count > MAX_EMPTY_CHUNKS {
                                    eprintln!("run_inference_coroutine: Model stalled ({} empty chunks), force terminating stream.", empty_chunk_count);
                                    return CoroutineResponse::Complete(buffer);
                                }
                            }
                            Err(e) => {
                                eprintln!("run_inference_coroutine: extract_content error: {:?}", e);
                                return CoroutineResponse::Error(e);
                            }
                        }
                    }
                    None => {
                        eprintln!("run_inference_coroutine: Stream closed (Complete)");
                        return CoroutineResponse::Complete(buffer);
                    }
                }
            }
            result = cancel_rx.changed() => {
                if result.is_ok() && *cancel_rx.borrow() {
                    eprintln!("run_inference_coroutine: Cancelled");
                    return CoroutineResponse::Interrupted(buffer);
                }
            }
        }
    }
}

type CoroutineFut = std::pin::Pin<
    Box<
        dyn std::future::Future<Output = (usize, ParallelInferenceParams, CoroutineResponse)>
            + Send,
    >,
>;

/// Run parallel inference stages
pub async fn run_parallel_stage(
    model: &Arc<mistralrs::Model>,
    params_list: Vec<ParallelInferenceParams>,
    max_retries: u32,
) -> Vec<(usize, String)> {
    let (cancel_tx, cancel_rx) = watch::channel(false);
    let mut pool: FuturesUnordered<CoroutineFut> = FuturesUnordered::new();
    let mut retry_counts: std::collections::HashMap<usize, u32> = std::collections::HashMap::new();

    let total = params_list.len();
    for (idx, params) in params_list.into_iter().enumerate() {
        let model = model.clone();
        let rx = cancel_rx.clone();
        pool.push(Box::pin(async move {
            let resp = run_inference_coroutine(model, params.clone(), rx, None).await;
            (idx, params, resp)
        }));
    }

    let mut results: Vec<Option<String>> = vec![None; total];

    while let Some((idx, params, resp)) = pool.next().await {
        match resp {
            CoroutineResponse::Complete(s) => {
                let _ = cancel_tx.send(true);
                results[idx] = Some(s);
            }
            CoroutineResponse::Interrupted(s) => {
                results[idx] = Some(s);
            }
            CoroutineResponse::Error(e) => {
                let retries = retry_counts.entry(idx).or_insert(0);
                if *retries < max_retries {
                    *retries += 1;
                    eprintln!("Coroutine {} error (retry {}): {:?}", idx, retries, e);
                    let model = model.clone();
                    let rx = cancel_rx.clone();
                    pool.push(Box::pin(async move {
                        let resp = run_inference_coroutine(model, params.clone(), rx, None).await;
                        (idx, params, resp)
                    }));
                } else {
                    eprintln!(
                        "Coroutine {} failed after {} retries: {:?}",
                        idx, max_retries, e
                    );
                    results[idx] = Some(format!("[Specialist error: {}]", e));
                }
            }
        }
    }

    results
        .into_iter()
        .enumerate()
        .map(|(i, s)| (i, s.unwrap_or_default()))
        .collect()
}

/// Run streaming inference loop
pub async fn run_streaming_loop(
    model: Arc<mistralrs::Model>,
    params: ParallelInferenceParams,
    realtime_file: &mut Option<File>,
    stream_realtime: bool,
    speech_detected_rx: &mut Option<broadcast::Receiver<()>>,
    interrupt_rx: &mut Option<broadcast::Receiver<Result<notify::Event, Arc<notify::Error>>>>,
    latest_audio_rx: &watch::Receiver<Option<Vec<u8>>>,
    subprocesses: &mut Vec<crate::types::CommandIO>,
    args: &Args,
) -> Result<StreamOutcome> {
    use crate::events::is_interrupt_event;

    let (cancel_tx, cancel_rx) = watch::channel(false);
    let (token_tx, mut token_rx) = unbounded_channel::<String>();
    let mut cmd_parser = CommandParser::new();
    let mut tool_context = String::new();
    let mut response = String::new();
    let mut last_modify_interrupt: Option<std::time::Instant> = None;
    if args.verbose {
        eprintln!("run_streaming_loop: Starting inference coroutine...");
    }
    let mut inference_fut = std::pin::pin!(run_inference_coroutine(
        model,
        params,
        cancel_rx,
        Some(token_tx),
    ));

    loop {
        tokio::select! {
            result = &mut inference_fut => {
                if args.verbose {
                    match &result {
                        CoroutineResponse::Complete(_) => eprintln!("run_streaming_loop: inference_fut finished with Complete"),
                        CoroutineResponse::Interrupted(_) => eprintln!("run_streaming_loop: inference_fut finished with Interrupted"),
                        CoroutineResponse::Error(e) => eprintln!("run_streaming_loop: inference_fut finished with Error: {:?}", e),
                    }
                }
                while let Ok(tok) = token_rx.try_recv() {
                    let (out, cmd, _) = cmd_parser.process(&tok);
                    response.push_str(&out);
                    if stream_realtime {
                        if let Some(ref mut f) = *realtime_file {
                            let _ = f.write_all(out.as_bytes()).await;
                            let _ = f.flush().await;
                        }
                    }
                    if let Some(cmd) = cmd {
                        // Execute command but don't inject feedback into response
                        // This prevents the model from hallucinating command result continuations
                        let _ = execute_command(cmd, subprocesses, &mut tool_context, args).await;
                        return Ok(StreamOutcome::CommandExecuted(response));
                    }
                }
                let tail = cmd_parser.flush();
                if !tail.is_empty() {
                    response.push_str(&tail);
                    if stream_realtime {
                        if let Some(ref mut f) = *realtime_file {
                            let _ = f.write_all(tail.as_bytes()).await;
                            let _ = f.flush().await;
                        }
                    }
                }
                return match result {
                    CoroutineResponse::Complete(_) => {
                        // USER IDEA: If inference ends and we have a fresh audio chunk waiting, 
                        // treat it as an interrupt instead of completion.
                        if let Some(audio) = latest_audio_rx.borrow().clone() {
                            if args.verbose {
                                eprintln!("run_streaming_loop: Inference complete but audio pending, treating as interrupt.");
                            }
                            Ok(StreamOutcome::AudioInterrupted { response, audio: Some(audio) })
                        } else {
                            Ok(StreamOutcome::Complete(response))
                        }
                    }
                    CoroutineResponse::Interrupted(_) => Ok(StreamOutcome::FsInterrupted { response, event: None }),
                    CoroutineResponse::Error(e) => Err(e),
                };
            }
            Some(tok) = token_rx.recv() => {
                if args.verbose {
                    eprintln!("run_streaming_loop: Received token: {:?}", tok);
                }
                let (out, cmd, _) = cmd_parser.process(&tok);
                response.push_str(&out);
                if stream_realtime {
                    if let Some(ref mut f) = *realtime_file {
                        let _ = f.write_all(out.as_bytes()).await;
                        let _ = f.flush().await;
                    }
                }
                if let Some(cmd) = cmd {
                    // Execute command but don't inject feedback into response
                    // This prevents the model from hallucinating command result continuations
                    let _ = execute_command(cmd, subprocesses, &mut tool_context, args).await;
                    return Ok(StreamOutcome::CommandExecuted(response));
                }
            }
            _ = async {
                if let Some(ref mut rx) = *speech_detected_rx {
                    rx.recv().await.ok()
                } else {
                    std::future::pending::<Option<()>>().await
                }
            } => {
                let _ = cancel_tx.send(true);
                let audio = latest_audio_rx.borrow().clone();
                return Ok(StreamOutcome::AudioInterrupted { response, audio });
            }
            event = async {
                if let Some(ref mut rx) = *interrupt_rx {
                    rx.recv().await.ok()
                } else {
                    std::future::pending::<Option<Result<notify::Event, Arc<notify::Error>>>>().await
                }
            } => {
                if let Some(Ok(ev)) = event {
                    if is_interrupt_event(&ev, &mut last_modify_interrupt, &args)
                        && !matches!(&ev.kind, notify::EventKind::Modify(_))
                    {
                        let _ = cancel_tx.send(true);
                        return Ok(StreamOutcome::FsInterrupted { response, event: Some(ev) });
                    }
                }
            }
        }
    }
}

/// Run a single inference pass
pub async fn run_once(
    model: &Arc<mistralrs::Model>,
    args: &Args,
    mut interrupt_rx: Option<broadcast::Receiver<Result<notify::Event, Arc<notify::Error>>>>,
    audio_channels: Option<(broadcast::Receiver<()>, watch::Receiver<Option<Vec<u8>>>)>,
    pending_audio: Vec<Vec<u8>>,
) -> Result<()> {
    use crate::events::handle_interrupt;

    if args.verbose {
        eprintln!("Building messages..");
    }

    let timestamp = chrono::Utc::now().timestamp_millis();
    let new_file_path = args
        .output_new
        .as_ref()
        .map(|dir| dir.join(format!("out-{}.txt", timestamp)));

    if let Some(overwrite_path) = &args.output_overwrite {
        if args.verbose {
            eprintln!(
                "Pre-clearing file for overwrite: {}",
                overwrite_path.display()
            );
        }
        if let Some(parent) = overwrite_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        File::create(overwrite_path).await?;
    }
    if let Some(ref path) = new_file_path {
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
    }

    let stream_realtime = args.stream_realtime;
    let mut current_file_path: Option<std::path::PathBuf> = new_file_path.clone();

    let mut realtime_file: Option<File> = if stream_realtime {
        if let Some(ref path) = current_file_path {
            if args.verbose {
                eprintln!("File created: {}", path.display());
            }
            Some(File::create(path).await?)
        } else if let Some(ref path) = args.output_overwrite {
            Some(File::options().write(true).open(path).await?)
        } else {
            None
        }
    } else {
        None
    };

    let (mut speech_detected_rx, latest_audio_rx): (
        Option<broadcast::Receiver<()>>,
        watch::Receiver<Option<Vec<u8>>>,
    ) = match audio_channels {
        Some((sdr, lar)) => (Some(sdr), lar),
        None => {
            let (_, rx) = watch::channel::<Option<Vec<u8>>>(None);
            (None, rx)
        }
    };

    let mut output_buffer = String::new();
    let mut restart_count = 0;
    let mut audio_interrupt_pending = !pending_audio.is_empty();
    let mut pending_audio_queue: Vec<Vec<u8>> = pending_audio;
    let mut subprocesses: Vec<crate::types::CommandIO> = Vec::new();
    let mut pending_system_message: Option<String> = None;
    let tool_context: String = String::new();

    // Initialize context management if enabled
    let context_manager = if args.compress_context {
        let cache_dir = std::env::current_dir()?.join(".agentgraph_cache");
        Some(crate::context::ContextManager::new(cache_dir))
    } else {
        None
    };

    let compression_agent = if args.compress_context {
        Some(crate::context::CompressionAgent::new(model.clone()))
    } else {
        None
    };

    'restart_loop: loop {
        restart_count += 1;
        if args.verbose && restart_count > 1 {
            eprintln!("Restart iteration {}", restart_count);
        }

        let audio_first = std::mem::take(&mut audio_interrupt_pending);

        let (current_primary, current_secondary) =
            build_messages(args, context_manager.as_ref(), compression_agent.as_ref()).await?;
        let has_sec = current_secondary.is_some() && args.secondary_model != "none";

        let primary_slot = if audio_first && has_sec {
            ModelSlot::Secondary
        } else {
            ModelSlot::Primary
        };

        let synth_params = if has_sec {
            let mut audio_msgs = current_secondary.clone().unwrap();
            for audio in &pending_audio_queue {
                audio_msgs = audio_msgs.add_multimodal_message(
                    TextMessageRole::User,
                    "",
                    vec![],
                    vec![
                        mistralrs::AudioInput::from_bytes(audio)
                            .context("Failed to create AudioInput from bytes")?,
                    ],
                );
            }
            pending_audio_queue.clear();
            ParallelInferenceParams::full_pipeline(
                current_primary.clone(),
                audio_msgs,
                primary_slot,
            )
        } else {
            ParallelInferenceParams {
                messages: current_primary.clone(),
                system_addendum: String::new(),
                interrupted_by: InterruptKind::all(),
                context_inputs: Vec::new(),
                streaming: true,
                model_slot: ModelSlot::Primary,
            }
        };

        if args.verbose {
            eprintln!(
                "Checking context inputs (count: {}), has_sec: {}",
                synth_params.context_inputs.len(),
                has_sec
            );
        }

        if !synth_params.context_inputs.is_empty() {
            if args.verbose {
                eprintln!(
                    "Running {} parallel specialists...",
                    synth_params.context_inputs.len()
                );
            }
            let specialist_results =
                run_parallel_stage(model, synth_params.context_inputs.clone(), 2).await;
            if args.verbose {
                eprintln!("Parallel stage complete, building synthesis messages.");
            }

            let mut synth_msgs = synth_params.messages.clone();
            if !synth_params.system_addendum.is_empty() {
                synth_msgs = synth_msgs
                    .add_message(TextMessageRole::Tool, synth_params.system_addendum.clone());
            }
            // Add pending system message about subprocesses
            if let Some(ref sys_msg) = pending_system_message {
                synth_msgs = synth_msgs.add_message(TextMessageRole::Tool, sys_msg.clone());
                pending_system_message = None;
            }
            // Add tool context from previous command executions
            if !tool_context.is_empty() {
                synth_msgs = synth_msgs.add_message(
                    TextMessageRole::Tool,
                    format!("[PREVIOUS COMMAND OUTPUT]:\n{}", tool_context),
                );
            }
            for (idx, text) in &specialist_results {
                let label = match synth_params.context_inputs[*idx].model_slot {
                    ModelSlot::Primary => "VISION SPECIALIST",
                    ModelSlot::Secondary => "AUDIO SPECIALIST",
                };
                synth_msgs = synth_msgs.add_message(
                    TextMessageRole::Tool,
                    format!("[SPECIALIST RESPONSE ({label}): {text}]"),
                );
            }

            let streaming_params = ParallelInferenceParams {
                messages: synth_msgs,
                system_addendum: String::new(),
                interrupted_by: InterruptKind::all(),
                context_inputs: Vec::new(),
                streaming: true,
                model_slot: synth_params.model_slot,
            };

            match run_streaming_loop(
                model.clone(),
                streaming_params,
                &mut realtime_file,
                stream_realtime,
                &mut speech_detected_rx,
                &mut interrupt_rx,
                &latest_audio_rx,
                &mut subprocesses,
                args,
            )
            .await
            .context("Synthesis model failed")?
            {
                StreamOutcome::Complete(resp) => {
                    if args.verbose {
                        eprintln!("Synthesis complete.");
                    }

                    output_buffer.push_str(&resp);

                    // Check for running subprocesses and reprompt if any are still alive
                    let mut running_indices: Vec<usize> = Vec::new();
                    for (idx, subp) in subprocesses.iter_mut().enumerate() {
                        let is_alive = if let Some(ref mut exit_rx) = subp.exited_rx {
                            matches!(exit_rx.try_recv(), Err(_))
                        } else {
                            false
                        };
                        if is_alive {
                            running_indices.push(idx);
                        }
                    }

                    if !running_indices.is_empty() {
                        // Build system message with command syntax reminder
                        let mut subp_msg =
                            String::from("[SYSTEM: Active subprocesses. Use these commands:\n");
                        subp_msg.push_str(&format!(
                            "  - Read: {}<idx>{}\n",
                            crate::CMD_OPEN_READ,
                            crate::CMD_CLOSE_READ
                        ));
                        subp_msg.push_str(&format!(
                            "  - Kill: {}<idx>{}\n",
                            crate::CMD_OPEN_KILL,
                            crate::CMD_CLOSE_KILL
                        ));
                        subp_msg.push_str(&format!(
                            "  - Write: {}<idx> text{}\n",
                            crate::CMD_OPEN_WRIT,
                            crate::CMD_CLOSE_WRIT
                        ));
                        subp_msg.push_str("  Active: ");
                        for (i, idx) in running_indices.iter().enumerate() {
                            if i > 0 {
                                subp_msg.push_str(", ");
                            }
                            subp_msg.push_str(&format!("{}", idx));
                        }
                        subp_msg.push_str("]\n");

                        pending_system_message = Some(subp_msg);

                        if args.verbose {
                            eprintln!(
                                "Subprocesses still running: {:?}, reprompting",
                                running_indices
                            );
                        }
                        continue 'restart_loop;
                    }

                    break 'restart_loop;
                }
                StreamOutcome::CommandExecuted(resp) => {
                    output_buffer.push_str(&resp);
                    output_buffer.push_str("\n");
                    continue 'restart_loop;
                }
                StreamOutcome::AudioInterrupted { response, audio } => {
                    if args.verbose {
                        eprintln!("Audio interrupt during synthesis.");
                    }
                    output_buffer.push_str(&response);
                    audio_interrupt_pending = true;
                    if let Some(audio) = audio {
                        pending_audio_queue.push(audio);
                    }
                }
                StreamOutcome::FsInterrupted { response, event } => {
                    if args.verbose {
                        eprintln!("FS interrupt during synthesis.");
                    }
                    output_buffer.push_str(&response);
                    if let Some(ev) = event {
                        handle_interrupt(ev, &mut current_file_path, &mut realtime_file, args)
                            .await?;
                    }
                }
            }
        } else {
            if args.verbose {
                eprintln!("No context inputs, running direct streaming inference.");
            }
            let mut msgs = synth_params.messages.clone();
            if !synth_params.system_addendum.is_empty() {
                msgs =
                    msgs.add_message(TextMessageRole::Tool, synth_params.system_addendum.clone());
            }
            // Add pending system message about subprocesses
            if let Some(ref sys_msg) = pending_system_message {
                msgs = msgs.add_message(TextMessageRole::Tool, sys_msg.clone());
                pending_system_message = None;
            }
            if !output_buffer.is_empty() {
                msgs = msgs.add_message(TextMessageRole::Assistant, format!("{}", output_buffer));
            }

            let streaming_params = ParallelInferenceParams {
                messages: msgs,
                system_addendum: String::new(),
                interrupted_by: InterruptKind::all(),
                context_inputs: Vec::new(),
                streaming: true,
                model_slot: ModelSlot::Primary,
            };

            if args.verbose {
                eprintln!("Calling run_streaming_loop...");
            }
            match run_streaming_loop(
                model.clone(),
                streaming_params,
                &mut realtime_file,
                stream_realtime,
                &mut speech_detected_rx,
                &mut interrupt_rx,
                &latest_audio_rx,
                &mut subprocesses,
                args,
            )
            .await
            .context("Primary model failed")?
            {
                StreamOutcome::Complete(resp) => {
                    if args.verbose {
                        eprintln!("Primary response complete.");
                    }
                    output_buffer.push_str(&resp);

                    // Check for running subprocesses and reprompt if any are still alive
                    let mut running_indices: Vec<usize> = Vec::new();
                    for (idx, subp) in subprocesses.iter_mut().enumerate() {
                        let is_alive = if let Some(ref mut exit_rx) = subp.exited_rx {
                            matches!(exit_rx.try_recv(), Err(_))
                        } else {
                            false
                        };
                        if is_alive {
                            running_indices.push(idx);
                        }
                    }

                    if !running_indices.is_empty() {
                        // Build system message with command syntax reminder
                        let mut subp_msg =
                            String::from("[SYSTEM: Active subprocesses. Use these commands:\n");
                        subp_msg.push_str(&format!(
                            "  - Read: {}<idx>{}\n",
                            crate::CMD_OPEN_READ,
                            crate::CMD_CLOSE_READ
                        ));
                        subp_msg.push_str(&format!(
                            "  - Kill: {}<idx>{}\n",
                            crate::CMD_OPEN_KILL,
                            crate::CMD_CLOSE_KILL
                        ));
                        subp_msg.push_str(&format!(
                            "  - Write: {}<idx> text{}\n",
                            crate::CMD_OPEN_WRIT,
                            crate::CMD_CLOSE_WRIT
                        ));
                        subp_msg.push_str("  Active: ");
                        for (i, idx) in running_indices.iter().enumerate() {
                            if i > 0 {
                                subp_msg.push_str(", ");
                            }
                            subp_msg.push_str(&format!("{}", idx));
                        }
                        subp_msg.push_str("]\n");

                        pending_system_message = Some(subp_msg);

                        if args.verbose {
                            eprintln!(
                                "Subprocesses still running: {:?}, reprompting",
                                running_indices
                            );
                        }
                        continue 'restart_loop;
                    }

                    break 'restart_loop;
                }
                StreamOutcome::CommandExecuted(resp) => {
                    output_buffer.push_str(&resp);
                    output_buffer.push_str("\n");
                    continue 'restart_loop;
                }
                StreamOutcome::AudioInterrupted { response, audio } => {
                    if args.verbose {
                        eprintln!("Audio interrupt.");
                    }
                    output_buffer.push_str(&response);
                    audio_interrupt_pending = true;
                    if let Some(audio) = audio {
                        pending_audio_queue.push(audio);
                    }
                }
                StreamOutcome::FsInterrupted { response, event } => {
                    if args.verbose {
                        eprintln!("FS interrupt.");
                    }
                    output_buffer.push_str(&response);
                    if let Some(ev) = event {
                        handle_interrupt(ev, &mut current_file_path, &mut realtime_file, args)
                            .await?;
                    }
                }
            }
        }
    }

    if let Some(mut f) = realtime_file.take() {
        let _ = f.shutdown().await;
    }
    drop(subprocesses);

    if !stream_realtime {
        if let Some(ref path) = new_file_path {
            if args.verbose {
                eprintln!("Saving output to: {}", path.display());
            }
            tokio::fs::write(path, &output_buffer).await?;
        }
        if let Some(ref path) = args.output_overwrite {
            if args.verbose {
                eprintln!("Saving output (overwrite): {}", path.display());
            }
            tokio::fs::write(path, &output_buffer).await?;
        }
    } else if args.verbose {
        if let Some(ref path) = current_file_path {
            eprintln!("Realtime complete: {}", path.display());
        }
    }

    Ok(())
}
