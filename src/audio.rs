//! Audio handling for agentgraph.
//!
//! Real-time audio capture, speech detection, and chunking.

use crate::Args;
use crate::types::{MODEL_SECONDARY, RealtimeListener};
use anyhow::{Context, Result};
use mistralrs::{TextMessageRole, VisionMessages};
use rand::RngExt;
use std::sync::Arc;
use std::time::Duration;
/// Create WAV header for audio chunks
pub fn create_wav_header(data_len: usize) -> [u8; 44] {
    const SAMPLE_RATE: u32 = 48000;
    const BITS_PER_SAMPLE: u16 = 32;
    const CHANNELS: u16 = 1;
    const BYTES_PER_SECOND: u32 = SAMPLE_RATE * CHANNELS as u32 * (BITS_PER_SAMPLE as u32 / 8);
    const BLOCK_ALIGN: u16 = CHANNELS * (BITS_PER_SAMPLE / 8);
    const AUDIO_FORMAT: u16 = 3;

    let mut header = [0u8; 44];

    // RIFF chunk
    header[0..4].copy_from_slice(b"RIFF");
    header[4..8].copy_from_slice(&(36 + data_len as u32).to_le_bytes());
    header[8..12].copy_from_slice(b"WAVE");

    // fmt subchunk
    header[12..16].copy_from_slice(b"fmt ");
    header[16..20].copy_from_slice(&(16u32).to_le_bytes());
    header[20..22].copy_from_slice(&AUDIO_FORMAT.to_le_bytes());
    header[22..24].copy_from_slice(&CHANNELS.to_le_bytes());
    header[24..28].copy_from_slice(&SAMPLE_RATE.to_le_bytes());
    header[28..32].copy_from_slice(&BYTES_PER_SECOND.to_le_bytes());
    header[32..34].copy_from_slice(&BLOCK_ALIGN.to_le_bytes());
    header[34..36].copy_from_slice(&BITS_PER_SAMPLE.to_le_bytes());

    // data subchunk
    header[36..40].copy_from_slice(b"data");
    header[40..44].copy_from_slice(&(data_len as u32).to_le_bytes());

    header
}

/// Spawn realtime audio listener
pub async fn spawn_realtime_listener(
    source: &str,
    args: &Args,
    model: Arc<mistralrs::Model>,
) -> Result<RealtimeListener> {
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
    use flume;

    if source != "pipewire" {
        anyhow::bail!("Only 'pipewire' source is supported");
    }

    let (chunk_tx, chunk_rx) = tokio::sync::mpsc::channel::<Vec<u8>>(32);
    let (shutdown_tx, mut shutdown_rx) = tokio::sync::oneshot::channel();
    let (speech_detected_tx, speech_detected_rx) = tokio::sync::broadcast::channel(16);

    let min_duration = Duration::from_secs_f32(args.audio_chunk_min_secs);
    let max_duration = Duration::from_secs_f32(args.audio_chunk_max_secs);

    let model_clone = model.clone();

    tokio::spawn(async move {
        let (audio_tx, audio_rx) = flume::bounded::<Vec<u8>>(128);
        let (shutdown_tx_blocking, shutdown_rx_blocking) = std::sync::mpsc::channel();

        // Spawn blocking thread for cpal audio capture
        let audio_handle = tokio::task::spawn_blocking(move || {
            let host = cpal::default_host();
            let device = match host.default_input_device() {
                Some(dev) => dev,
                None => {
                    eprintln!("No default input device found");
                    return;
                }
            };

            let config = {
                let mut best = None;
                for cfg in device.supported_input_configs().unwrap() {
                    if cfg.channels() == 1 && cfg.sample_format() == cpal::SampleFormat::F32 {
                        let rate = 48000;
                        if cfg.min_sample_rate() <= rate && rate <= cfg.max_sample_rate() {
                            best = Some(cfg.with_sample_rate(rate));
                            break;
                        }
                    }
                }
                match best {
                    Some(cfg) => cfg,
                    None => {
                        eprintln!("No supported 48kHz mono F32 input config");
                        return;
                    }
                }
            };

            let tx = audio_tx.clone();
            let sample_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let last_log = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            
            let stream = device.build_input_stream(
                config.into(),
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    let mut non_zero = 0;
                    for &s in data {
                        if s.abs() > 0.00001 {
                            non_zero += 1;
                        }
                    }
                    let total = sample_count.fetch_add(non_zero, std::sync::atomic::Ordering::SeqCst) + non_zero;
                    let last = last_log.load(std::sync::atomic::Ordering::SeqCst);
                    if total > last + 48000 { // Log every ~1 second of audio
                        eprintln!("Audio callback: Captured {} non-zero samples total", total);
                        last_log.store(total, std::sync::atomic::Ordering::SeqCst);
                    }

                    let bytes: &[u8] = bytemuck::cast_slice(data);
                    let vec_bytes = bytes.to_vec();
                    if let Err(e) = tx.try_send(vec_bytes) {
                        // This might be noisy, but good for debugging if channel is full or disconnected
                        // eprintln!("Audio callback: try_send failed: {:?}", e);
                    }
                },
                |err| eprintln!("Audio stream error: {}", err),
                None,
            );

            let stream = match stream {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("Failed to build input stream: {}", e);
                    return;
                }
            };

            if let Err(e) = stream.play() {
                eprintln!("Failed to start audio stream: {}", e);
                return;
            }
            eprintln!("Audio callback: stream.play() called successfully");

            let _ = shutdown_rx_blocking.recv();
        });

        // Main async loop: receive audio bytes and perform chunking
        let mut chunk_start = std::time::Instant::now();
        let mut current_chunk_duration = rand::rng().random_range(min_duration..=max_duration);
        let mut accumulated_buffer = Vec::new();
        eprintln!("Audio listener: Main loop starting, target duration: {}ms", current_chunk_duration.as_millis());

        loop {
            tokio::select! {
                _ = &mut shutdown_rx => {
                    eprintln!("Audio listener: Shutdown received");
                    let _ = shutdown_tx_blocking.send(());
                    break;
                }
                result = audio_rx.recv_async() => {
                    match result {
                        Ok(bytes) => {
                            if accumulated_buffer.is_empty() {
                                // eprintln!("Audio listener: Receiving data from channel");
                            }
                            accumulated_buffer.extend_from_slice(&bytes);

                            if chunk_start.elapsed() >= current_chunk_duration {
                                eprintln!("Audio listener: Chunk duration reached ({}ms), buffer size: {} bytes", 
                                    chunk_start.elapsed().as_millis(), accumulated_buffer.len());
                                if !accumulated_buffer.is_empty() {
                                    let model_clone = model_clone.clone();
                                    let speech_detected_tx = speech_detected_tx.clone();
                                    let chunk = accumulated_buffer.clone();
                                    let tx_clone = chunk_tx.clone();
                                    eprintln!("Audio listener: Spawning detect_speech task");
                                    tokio::spawn(async move {
                                        let mut wav = create_wav_header(chunk.len()).to_vec();
                                        wav.append(&mut chunk.clone());
                                        match detect_speech(&wav, &model_clone, MODEL_SECONDARY).await {
                                            Ok(true) => {
                                                let _ = speech_detected_tx.send(());
                                                let _ = tx_clone.send(wav.to_vec()).await;
                                            }
                                            Ok(false) => {}
                                            Err(e) => {
                                                eprintln!("Speech detection error: {}", e);
                                            }
                                        }
                                    });

                                    accumulated_buffer.clear();
                                    chunk_start = std::time::Instant::now();
                                    current_chunk_duration = rand::rng()
                                        .random_range(min_duration..=max_duration);
                                }
                            }
                        }
                        Err(flume::RecvError::Disconnected) => {
                            break;
                        }
                    }
                }
            }
        }

        let _ = shutdown_tx_blocking.send(());
        let _ = audio_handle.await;
    });

    Ok(RealtimeListener::new(
        chunk_rx,
        shutdown_tx,
        speech_detected_rx,
    ))
}

/// Detect speech in audio chunk
pub async fn detect_speech(
    audio_chunk: &[u8],
    model: &mistralrs::Model,
    audio_model_id: &str,
) -> Result<bool> {
    eprintln!("detect_speech: Processing chunk of {} bytes", audio_chunk.len());
    
    if !model
        .is_model_loaded(audio_model_id)
        .context("Failed to check secondary model load state")?
    {
        eprintln!("detect_speech: Secondary model {} not loaded, reloading...", audio_model_id);
        model
            .reload_model(audio_model_id)
            .await
            .context("Failed to reload secondary model")?;
        eprintln!("detect_speech: Secondary model reloaded");
    }

    let audio = match mistralrs::AudioInput::from_bytes(audio_chunk) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("detect_speech: Failed to create AudioInput from bytes: {:?}", e);
            return Err(anyhow::anyhow!("Failed to create AudioInput: {}", e));
        }
    };

    let messages = VisionMessages::new().add_multimodal_message(
        TextMessageRole::User,
        "Respond with only 'true' if this audio contains intelligible speech, or 'false' if it does not. Be concise.",
        vec![],
        vec![audio],
    );

    eprintln!("detect_speech: Sending request to secondary model...");
    let response = model
        .send_chat_request_with_model(messages, Some(audio_model_id))
        .await
        .map_err(|e| {
            eprintln!("detect_speech: Full error from mistralrs: {:?}", e);
            anyhow::anyhow!("Speech detection request failed: {}", e)
        })?;

    let content = response.choices[0]
        .message
        .content
        .as_ref()
        .map(|s| s.trim().to_lowercase())
        .unwrap_or_default();

    eprintln!("detect_speech: Model response: {:?}", content);

    let detected = content.contains("true");
    if detected {
        eprintln!("detect_speech: Speech DETECTED");
    } else {
        eprintln!("detect_speech: No speech detected");
    }

    Ok(detected)
}
