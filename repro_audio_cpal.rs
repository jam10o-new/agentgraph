use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::time::{Duration, Instant};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

fn main() {
    println!("=== cpal Audio Capture Diagnostic ===");
    let host = cpal::default_host();
    println!("Host: {}", host.id().name());

    let device = match host.default_input_device() {
        Some(dev) => dev,
        None => {
            println!("ERROR: No default input device found");
            return;
        }
    };
    println!("Default input device: {}", device.name().unwrap_or_else(|_| "Unknown".to_string()));

    let config = {
        let mut best = None;
        if let Ok(configs) = device.supported_input_configs() {
            for cfg in configs {
                println!("  Supported config: {} channels, {} sample format, {}-{} Hz", 
                    cfg.channels(), 
                    match cfg.sample_format() {
                        cpal::SampleFormat::F32 => "F32",
                        cpal::SampleFormat::I16 => "I16",
                        cpal::SampleFormat::U16 => "U16",
                        _ => "Other"
                    },
                    cfg.min_sample_rate().0,
                    cfg.max_sample_rate().0
                );
                
                if cfg.channels() == 1 && cfg.sample_format() == cpal::SampleFormat::F32 {
                    let rate = cpal::SampleRate(48000);
                    if cfg.min_sample_rate() <= rate && rate <= cfg.max_sample_rate() {
                        best = Some(cfg.with_sample_rate(rate));
                        println!("    -> Selected this config");
                    }
                }
            }
        }
        match best {
            Some(cfg) => cfg,
            None => {
                println!("ERROR: No supported 48kHz mono F32 input config");
                return;
            }
        }
    };

    let sample_count = Arc::new(AtomicUsize::new(0));
    let sample_count_clone = sample_count.clone();

    let stream = device.build_input_stream(
        &config.into(),
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            // Count samples to see if we're actually getting data
            let mut non_zero = 0;
            for &s in data {
                if s.abs() > 0.00001 {
                    non_zero += 1;
                }
            }
            sample_count_clone.fetch_add(non_zero, Ordering::SeqCst);
        },
        |err| eprintln!("Audio stream error: {}", err),
        None,
    ).expect("Failed to build input stream");

    stream.play().expect("Failed to start stream");
    println!("Stream started. Listening for 10 seconds...");

    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(10) {
        let count = sample_count.load(Ordering::SeqCst);
        println!("Non-zero samples captured so far: {}", count);
        std::thread::sleep(Duration::from_secs(1));
    }

    println!("Diagnostic complete.");
}
