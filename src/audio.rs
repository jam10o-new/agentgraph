//! Audio handling for agentgraph.
//!
//! Real-time audio capture, speech detection, and chunking.

use crate::Args;
use crate::types::RealtimeListener;
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
    let model_id = args.secondary_model.clone();

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
            let stream = device.build_input_stream(
                config.into(),
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    let bytes: &[u8] = bytemuck::cast_slice(data);
                    let _ = tx.try_send(bytes.to_vec());
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

            let _ = shutdown_rx_blocking.recv();
        });

        // Main async loop: receive audio bytes and perform chunking
        let mut chunk_start = std::time::Instant::now();
        let mut current_chunk_duration = rand::rng().random_range(min_duration..=max_duration);
        let mut accumulated_buffer = Vec::new();

        loop {
            tokio::select! {
                _ = &mut shutdown_rx => {
                    let _ = shutdown_tx_blocking.send(());
                    break;
                }
                result = audio_rx.recv_async() => {
                    match result {
                        Ok(bytes) => {
                            accumulated_buffer.extend_from_slice(&bytes);

                            if chunk_start.elapsed() >= current_chunk_duration {
                                if !accumulated_buffer.is_empty() {
                                    let model_clone = model_clone.clone();
                                    let speech_detected_tx = speech_detected_tx.clone();
                                    let chunk = accumulated_buffer.clone();
                                    let tx_clone = chunk_tx.clone();
                                    let model_id = model_id.clone();
                                    tokio::spawn(async move {
                                        let mut wav = create_wav_header(chunk.len()).to_vec();
                                        wav.append(&mut chunk.clone());
                                        match detect_speech(&wav, &model_clone, &model_id).await {
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
    if !model
        .is_model_loaded(audio_model_id)
        .context("Failed to check secondary model load state")?
    {
        model
            .reload_model(audio_model_id)
            .await
            .context("Failed to reload secondary model")?;
    }

    let audio = mistralrs::AudioInput::from_bytes(audio_chunk)
        .context("Failed to create AudioInput from bytes")?;

    let messages = VisionMessages::new().add_multimodal_message(
        TextMessageRole::User,
        "Respond with only 'true' if this audio contains intelligible speech, or 'false' if it does not. Be concise.",
        vec![],
        vec![audio],
    );

    let response = model
        .send_chat_request_with_model(messages, Some(audio_model_id))
        .await
        .map_err(|e| {
            eprintln!("Full error from mistralrs: {:?}", e);
            anyhow::anyhow!("Speech detection request failed: {}", e)
        })?;

    let content = response.choices[0]
        .message
        .content
        .as_ref()
        .map(|s| s.trim().to_lowercase())
        .unwrap_or_default();

    Ok(content.contains("true"))
}
