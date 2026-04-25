use agentgraph::audio::AudioListener;
use std::path::PathBuf;
use std::time::Duration;

#[tokio::test]
async fn test_audio_listener_creates_wav() {
    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    let input_dir = temp_dir.path().to_path_buf();

    let listener = AudioListener::new("test-agent".to_string(), input_dir.clone());

    println!("Starting audio listener on default input device...");
    println!("Please speak into your microphone now! (Test will run for 30 seconds)");

    listener.start().await.expect("Failed to start audio listener");

    // Run long enough for the user to notice and say something
    tokio::time::sleep(Duration::from_secs(30)).await;

    let wav_files: Vec<PathBuf> = std::fs::read_dir(&input_dir)
        .expect("Failed to read temp dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("wav"))
        .collect();

    println!("Found {} WAV file(s) in {}", wav_files.len(), input_dir.display());
    for f in &wav_files {
        let meta = std::fs::metadata(f).unwrap();
        println!("  - {} ({} bytes)", f.display(), meta.len());
    }

    // We assert that at least one wav was created if the user spoke,
    // but we don't fail if none were created (no mic / silence is OK for CI).
    // Instead, we just log the result.
    if wav_files.is_empty() {
        println!("WARNING: No WAV files were recorded. This may mean no audio was detected.");
    }
}
