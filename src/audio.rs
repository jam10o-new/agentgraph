use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use anyhow::{anyhow, Result};
use std::path::PathBuf;
use tokio::sync::mpsc;
use std::fs::File;
use std::io::BufWriter;
use hound::{WavSpec, WavWriter, SampleFormat};
use std::collections::VecDeque;

/// Simple energy VAD threshold for f32 samples in [-1.0, 1.0].
/// 0.005 is generous for quiet mics while still filtering pure silence.
const VAD_THRESHOLD: f32 = 0.005;

/// Silence duration required to finalize a recording (samples).
/// At 16 kHz this is 1.5 seconds.
const SILENCE_SAMPLES: usize = 16000 * 3 / 2;

/// Pre-roll ring buffer: capture this many samples *before* speech is detected
/// so the start of the utterance isn't clipped.
const PRE_ROLL_SAMPLES: usize = 16000 / 2; // 0.5 seconds

pub struct AudioListener {
    pub input_dir: PathBuf,
    pub name: String,
}

impl AudioListener {
    pub fn new(name: String, input_dir: PathBuf) -> Self {
        Self { name, input_dir }
    }

    pub async fn start(&self) -> Result<()> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or_else(|| anyhow!("No default input device"))?;

        let default_config = device.default_input_config()?;
        let sample_rate = default_config.sample_rate();
        println!(
            "[audio] Default input config: {:?} (sample_rate={}, channels={})",
            default_config.sample_format(),
            sample_rate,
            default_config.channels()
        );

        // Use the device's preferred sample rate but request 1 channel (mono)
        // to keep things simple. CPAL/PipeWire will resample if needed.
        let mut stream_config = default_config.config();
        stream_config.channels = 1;
        println!("[audio] Using stream config: sample_rate={sample_rate}, channels=1");

        let input_dir = self.input_dir.clone();

        // Bounded channel: large enough to not block the audio callback
        let (tx, mut rx) = mpsc::channel::<f32>(sample_rate as usize * 2);

        // --- Recording task ---
        tokio::spawn(async move {
            let mut is_recording = false;
            let mut writer: Option<WavWriter<BufWriter<File>>> = None;
            let mut silence_count: usize = 0;
            let mut pre_roll: VecDeque<f32> = VecDeque::with_capacity(PRE_ROLL_SAMPLES);
            let mut consecutive_speech = 0usize;
            let mut peak_level: f32 = 0.0;
            let mut frame_count: usize = 0;

            while let Some(sample) = rx.recv().await {
                frame_count += 1;

                // Track peak for diagnostics
                let abs = sample.abs();
                if abs > peak_level {
                    peak_level = abs;
                }

                // Simple energy VAD with hysteresis
                let is_speech = abs > VAD_THRESHOLD;

                if is_speech {
                    consecutive_speech += 1;
                } else {
                    consecutive_speech = consecutive_speech.saturating_sub(1);
                }

                // Logging: every ~5 seconds report peak level so users can see what's happening
                if frame_count % (sample_rate as usize * 5) == 0 {
                    if peak_level < VAD_THRESHOLD {
                        println!(
                            "[audio] Elapsed ~{}s | PEAK LEVEL: {:.4} — BELOW VAD THRESHOLD ({:.4}) — check mic gain!",
                            frame_count / sample_rate as usize,
                            peak_level,
                            VAD_THRESHOLD
                        );
                    } else {
                        println!(
                            "[audio] Elapsed ~{}s | PEAK LEVEL: {:.4} — above threshold",
                            frame_count / sample_rate as usize,
                            peak_level
                        );
                    }
                    peak_level = 0.0;
                }

                // --- State machine ---
                if !is_recording {
                    // Always buffer the latest N samples for pre-roll
                    if pre_roll.len() >= PRE_ROLL_SAMPLES {
                        pre_roll.pop_front();
                    }
                    pre_roll.push_back(sample);

                    // Require 3 consecutive speech samples to start (debounce)
                    if consecutive_speech >= 3 {
                        is_recording = true;
                        silence_count = 0;
                        consecutive_speech = 0;

                        let path = input_dir.join(format!(
                            "audio-{}.wav",
                            chrono::Local::now().format("%Y%m%d%H%M%S")
                        ));
                        println!("[audio] Speech detected! Starting recording -> {}", path.display());

                        let spec = WavSpec {
                            channels: 1,
                            sample_rate,
                            bits_per_sample: 32,
                            sample_format: SampleFormat::Float,
                        };
                        let mut w = WavWriter::create(&path, spec)
                            .expect("Failed to create WAV writer");

                        // Flush pre-roll buffer into the file so we capture the start
                        for &s in &pre_roll {
                            w.write_sample(s).unwrap();
                        }
                        pre_roll.clear();

                        writer = Some(w);
                    }
                } else {
                    // We are recording
                    if !is_speech {
                        silence_count += 1;
                    } else {
                        silence_count = 0;
                    }

                    if let Some(w) = writer.as_mut() {
                        w.write_sample(sample).unwrap();
                    }

                    if silence_count > SILENCE_SAMPLES {
                        is_recording = false;
                        if let Some(w) = writer.take() {
                            w.finalize().unwrap();
                        }
                        println!("[audio] Recording finalized after silence.");
                    }
                }
            }
        });

        // --- CPAL audio stream ---
        let stream = device.build_input_stream(
            stream_config,
            move |data: &[f32], _| {
                for &sample in data {
                    // Don't let backpressure stall the audio thread.
                    // If the channel is full (shouldn't happen with this capacity),
                    // drop the oldest samples silently.
                    if let Err(_) = tx.try_send(sample) {
                        // Channel full — drop sample to keep latency low
                    }
                }
            },
            |err| eprintln!("[audio] Stream error: {}", err),
            None,
        )?;

        stream.play()?;
        println!("[audio] Stream playing. Listening...");

        // Deliberately leak the stream: the audio listener should run for
        // the lifetime of the process (or until the containing task is aborted).
        let _ = Box::leak(Box::new(stream));

        Ok(())
    }
}
