//! Audio handling for agentgraph.
//!
//! Real-time audio capture, speech detection, and chunking.

use crate::Args;
use crate::types::{MODEL_SECONDARY, RealtimeListener};
use anyhow::{Context, Result};
use mistralrs::{RequestLike, SamplingParams, TextMessageRole, VisionMessages};
use rand::RngExt;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tokio::fs;
use tokio::sync::{mpsc, oneshot};

/// Create a basic WAV header for a mono 48kHz F32 audio stream
fn create_wav_header(data_len: usize) -> [u8; 44] {
    let mut header = [0u8; 44];
    const SAMPLE_RATE: u32 = 16000;
    const BITS_PER_SAMPLE: u16 = 32;
    const CHANNELS: u16 = 1;
    const BYTES_PER_SECOND: u32 = SAMPLE_RATE * CHANNELS as u32 * (BITS_PER_SAMPLE as u32 / 8);
    const BLOCK_ALIGN: u16 = CHANNELS * (BITS_PER_SAMPLE / 8);
    const AUDIO_FORMAT: u16 = 3; // IEEE Float

    header[0..4].copy_from_slice(b"RIFF");
    header[4..8].copy_from_slice(&(36 + data_len as u32).to_le_bytes());
    header[8..12].copy_from_slice(b"WAVE");
    header[12..16].copy_from_slice(b"fmt ");
    header[16..20].copy_from_slice(&16u32.to_le_bytes());
    header[20..22].copy_from_slice(&AUDIO_FORMAT.to_le_bytes());
    header[22..24].copy_from_slice(&CHANNELS.to_le_bytes());
    header[24..28].copy_from_slice(&SAMPLE_RATE.to_le_bytes());
    header[28..32].copy_from_slice(&BYTES_PER_SECOND.to_le_bytes());
    header[32..34].copy_from_slice(&BLOCK_ALIGN.to_le_bytes());
    header[34..36].copy_from_slice(&BITS_PER_SAMPLE.to_le_bytes());
    header[36..40].copy_from_slice(b"data");
    header[40..44].copy_from_slice(&(data_len as u32).to_le_bytes());

    header
}

/// Spawn a persistent audio listener
pub async fn spawn_realtime_listener(
    source: &str,
    args: &Args,
    model: Arc<mistralrs::Model>,
) -> Result<RealtimeListener> {
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

    if source != "pipewire" {
        anyhow::bail!("Only 'pipewire' source is supported");
    }

    let (chunk_tx, chunk_rx) = mpsc::channel::<Vec<u8>>(32);
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();

    let min_duration = Duration::from_secs_f32(args.audio_chunk_min_secs);
    let max_duration = Duration::from_secs_f32(args.audio_chunk_max_secs);

    let (audio_tx, audio_rx) = flume::unbounded::<Vec<u8>>();
    let (shutdown_tx_blocking, shutdown_rx_blocking) = std::sync::mpsc::channel();

    // is_active is true if we have pending data or are currently processing speech
    let is_active = Arc::new(AtomicBool::new(false));
    let is_active_loop = is_active.clone();

    // 1. Spawn blocking thread for cpal audio capture
    std::thread::spawn(move || {
        let host = cpal::default_host();
        let device = match host.default_input_device() {
            Some(dev) => dev,
            None => {
                eprintln!("Audio: No default input device found");
                return;
            }
        };

        let config = {
            let mut best = None;
            if let Ok(configs) = device.supported_input_configs() {
                for cfg in configs {
                    if cfg.channels() == 1 && cfg.sample_format() == cpal::SampleFormat::F32 {
                        let rate = 16000;
                        if cfg.min_sample_rate() <= rate && rate <= cfg.max_sample_rate() {
                            best = Some(cfg.with_sample_rate(rate));
                            break;
                        }
                    }
                }
            }
            match best {
                Some(cfg) => cfg,
                None => {
                    eprintln!("Audio: No supported 24kHz mono F32 input config");
                    match device.default_input_config() {
                        Ok(cfg) => cfg,
                        Err(e) => {
                            eprintln!("Audio: Failed to get default input config: {}", e);
                            return;
                        }
                    }
                }
            }
        };

        let tx = audio_tx.clone();

        let stream = device.build_input_stream(
            config.into(),
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                let bytes: &[u8] = bytemuck::cast_slice(data);
                let _ = tx.send(bytes.to_vec());
            },
            |err| eprintln!("Audio stream error: {}", err),
            None,
        );

        let stream = match stream {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Audio: Failed to build input stream: {}", e);
                return;
            }
        };

        if let Err(e) = stream.play() {
            eprintln!("Audio: Failed to start audio stream: {}", e);
            return;
        }

        let _ = shutdown_rx_blocking.recv();
    });

    // 2. Spawn the main processing loop
    tokio::spawn(async move {
        let mut chunk_start = std::time::Instant::now();
        let mut current_chunk_duration = rand::rng().random_range(min_duration..=max_duration);
        let mut accumulated_buffer = Vec::new();
        let mut interval = tokio::time::interval(Duration::from_millis(100));

        let mut shutdown_rx = shutdown_rx;

        loop {
            tokio::select! {
                _ = &mut shutdown_rx => {
                    let _ = shutdown_tx_blocking.send(());
                    break;
                }
                _ = interval.tick() => {
                    let mut received = false;
                    while let Ok(bytes) = audio_rx.try_recv() {
                        accumulated_buffer.extend_from_slice(&bytes);
                        received = true;
                    }


                    // If we have data in the buffer, the listener is "active"
                    if !accumulated_buffer.is_empty() {
                        is_active_loop.store(true, Ordering::SeqCst);
                    } else if !received {
                        is_active_loop.store(false, Ordering::SeqCst);
                    }

                    if chunk_start.elapsed() >= current_chunk_duration {
                        if !accumulated_buffer.is_empty() {
                            let model_clone = model.clone();
                            println!("Sending Listener buffer (len = {})", accumulated_buffer.len());
                            let chunk = std::mem::take(&mut accumulated_buffer);
                            let tx = chunk_tx.clone();
                            let is_active_task = is_active_loop.clone();

                            tokio::spawn(async move {

                                let mut wav = create_wav_header(chunk.len()).to_vec();
                                wav.extend_from_slice(&chunk);
                                match detect_speech(&wav, &model_clone, MODEL_SECONDARY).await {
                                    Ok(true) => {
                                        let _ = tx.send(wav).await;
                                        is_active_task.store(false, Ordering::SeqCst);
                                    }
                                    Err(e) => {
                                        println!("detect_speech error: {}",e)
                                    }
                                    _ => {println!("No speech in buffer");}
                                }
                            });
                        }
                        chunk_start = std::time::Instant::now();
                        current_chunk_duration = rand::rng().random_range(min_duration..=max_duration);

                    }
                }
            }
        }
    });

    Ok(RealtimeListener {
        chunk_rx,
        shutdown_tx,
        is_active,
    })
}

/// Detect speech in audio chunk
pub async fn detect_speech(
    audio_chunk: &[u8],
    model: &mistralrs::Model,
    audio_model_id: &str,
) -> Result<bool> {
    if !model
        .is_model_loaded(audio_model_id)
        .context("Failed to check secondary model load state")?
    {
        model
            .reload_model(audio_model_id)
            .await
            .context("Failed to reload secondary model")?;
    }

    pub async fn save_wav_debug(audio_chunk: &[u8], file_path: &Path) -> std::io::Result<()> {
        fs::write(file_path, audio_chunk).await
    }

    let audio = mistralrs::AudioInput::from_bytes(audio_chunk)
        .map_err(|e| anyhow::anyhow!("Failed to create AudioInput: {}", e))?;

    let messages = VisionMessages::new()
        /* .add_message(
            TextMessageRole::System,
            "Transcribe this audio - if there is no legible speech in it, output only \"false\"",
        )*/
        .add_multimodal_message(TextMessageRole::User, "", vec![], vec![audio]);

    let response = model
        .send_chat_request_with_model(messages, Some(audio_model_id))
        .await
        .map_err(|e| anyhow::anyhow!("Speech detection failed: {}", e))?;

    let content = response.choices[0]
        .message
        .content
        .as_ref()
        .map(|s| s.trim().to_lowercase())
        .unwrap_or_default();

    println!("Speech detector: {}", content);
    Ok(content.contains("true"))
}
