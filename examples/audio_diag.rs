use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

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

    match device.name() {
        Ok(name) => println!("Default input device: {}", name),
        Err(e) => println!("Failed to get device name: {}", e),
    }

    let config = {
        let mut best = None;
        if let Ok(configs) = device.supported_input_configs() {
            for cfg in configs {
                if cfg.channels() == 1 && cfg.sample_format() == cpal::SampleFormat::F32 {
                    best = Some(cfg.with_sample_rate(48000));
                    break;
                }
            }
        }
        match best {
            Some(cfg) => cfg,
            None => {
                println!("ERROR: No supported mono F32 input config");
                return;
            }
        }
    };

    println!("Selected config: {:?}", config);

    let sample_count = Arc::new(AtomicUsize::new(0));
    let sample_count_clone = sample_count.clone();

    let stream_config: cpal::StreamConfig = config.into();

    let stream = device
        .build_input_stream(
            stream_config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
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
        )
        .expect("Failed to build input stream");

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
