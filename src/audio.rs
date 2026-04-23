use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use anyhow::{anyhow, Result};
use std::path::{PathBuf};
use tokio::sync::mpsc;
use std::fs::File;
use std::io::BufWriter;
use hound::{WavSpec, WavWriter, SampleFormat};

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
        let device = host.default_input_device()
            .ok_or_else(|| anyhow!("No default input device"))?;
        
        let config = device.default_input_config()?;
        let mut supported_config = config.config();
        supported_config.sample_rate = 16000;

        let input_dir = self.input_dir.clone();

        let (tx, mut rx) = mpsc::channel::<f32>(48000 * 2);

        // Recording task
        let input_dir_rec = input_dir.clone();
        tokio::spawn(async move {
            let mut is_recording = false;
            let mut writer: Option<WavWriter<BufWriter<File>>> = None;
            let mut silence_count = 0;

            while let Some(sample) = rx.recv().await {
                // Simple energy VAD
                let is_speech = sample.abs() > 0.05;
                
                if is_speech {
                    if !is_recording {
                        is_recording = true;
                        let path = input_dir_rec.join(format!("audio-{}.wav", chrono::Local::now().format("%Y%m%d%H%M%S")));
                        let spec = WavSpec {
                            channels: 1,
                            sample_rate: 16000,
                            bits_per_sample: 32,
                            sample_format: SampleFormat::Float,
                        };
                        writer = Some(WavWriter::create(path, spec).unwrap());
                    }
                    silence_count = 0;
                } else if is_recording {
                    silence_count += 1;
                    if silence_count > 16000 { // 1 second of silence
                        is_recording = false;
                        if let Some(w) = writer.take() {
                            w.finalize().unwrap();
                        }
                    }
                }

                if let Some(w) = writer.as_mut() {
                    w.write_sample(sample).unwrap();
                }
            }
        });

        let stream = device.build_input_stream(
            supported_config,
            move |data: &[f32], _| {
                for &sample in data {
                    let _ = tx.blocking_send(sample);
                }
            },
            |err| eprintln!("Audio stream error: {}", err),
            None,
        )?;

        stream.play()?;
        
        // Keep stream alive
        std::mem::forget(stream);
        
        Ok(())
    }
}
