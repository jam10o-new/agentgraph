use image::{ImageBuffer, Rgb};
use std::fs;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;
use tempfile::tempdir;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::time::timeout;

async fn cleanup_sockets() {
    let dir = "/tmp/agentgraph";
    if let Ok(mut entries) = tokio::fs::read_dir(dir).await {
        while let Ok(Some(entry)) = entries.next_entry().await {
            let _ = tokio::fs::remove_file(entry.path()).await;
        }
    }
}

struct LeaderGuard {
    child: tokio::process::Child,
}

impl Drop for LeaderGuard {
    fn drop(&mut self) {
        println!("LeaderGuard: Killing leader process...");
        let _ = self.child.start_kill();
        // Sockets are cleaned up by the next test or manually
    }
}

fn create_test_image(path: &std::path::Path) {
    let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_fn(100, 100, |x, y| {
        if (x + y) % 2 == 0 {
            Rgb([255u8, 0, 0])
        } else {
            Rgb([0, 255u8, 0])
        }
    });
    img.save(path).unwrap();
}

#[tokio::test]
async fn test_multimodal_colony_structured_json() {
    // 0. Cleanup
    cleanup_sockets().await;

    let tmp = tempdir().unwrap();
    let root = tmp.path();

    let researcher_input = root.join("researcher/input");
    let researcher_output = root.join("researcher/output");
    let researcher_system = root.join("researcher/system");
    let coder_input = root.join("coder/input");
    let coder_output = root.join("coder/output");
    let coder_system = root.join("coder/system");

    fs::create_dir_all(&researcher_input).unwrap();
    fs::create_dir_all(&researcher_output).unwrap();
    fs::create_dir_all(&researcher_system).unwrap();
    fs::create_dir_all(&coder_input).unwrap();
    fs::create_dir_all(&coder_output).unwrap();
    fs::create_dir_all(&coder_system).unwrap();

    // 1. Create Config using Qwen-VL for multimodal
    let config_content = format!(
        r#"
models:
  vl:
    id: "qwen3-vl-8b"
    builder: "vision"
    path: "Qwen/Qwen3-VL-8B-Instruct"
    isq: "4"
sampling:
  temperature: 0.1
  top_p: 0.9
  top_k: 40
  max_len: 256
agents:
  researcher:
    inputs: ["{}"]
    output: "{}"
    system: ["{}"]
    model: "vl"
    stream: false
  coder:
    inputs: ["{}", "{}"]
    output: "{}"
    system: ["{}"]
    model: "vl"
    stream: false
"#,
        researcher_input.display(),
        researcher_output.display(),
        researcher_system.display(),
        coder_input.display(),
        researcher_output.display(),
        coder_output.display(),
        coder_system.display()
    );

    let config_path = root.join("colony_config.yaml");
    fs::write(&config_path, config_content).unwrap();

    // 2. Start Leader
    println!("Multimodal Test: Starting leader...");
    let mut child = Command::new("cargo")
        .args([
            "run",
            "--bin",
            "ag",
            "--",
            "leader",
            "--config",
            config_path.to_str().unwrap(),
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn leader");

    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();

    // Wrap in guard for auto-cleanup
    let _guard = LeaderGuard { child };

    let mut stdout_reader = BufReader::new(stdout).lines();

    // Watch stdout for "listening on"
    let wait_ready = async {
        while let Ok(Some(line)) = stdout_reader.next_line().await {
            println!("[LEADER STDOUT] {}", line);
            if line.contains("listening on") {
                return true;
            }
        }
        false
    };

    let ready = timeout(Duration::from_secs(300), wait_ready)
        .await
        .unwrap_or(false);
    if !ready {
        panic!("Leader failed to become ready within timeout");
    }

    tokio::spawn(async move {
        while let Ok(Some(line)) = stdout_reader.next_line().await {
            println!("[LEADER STDOUT] {}", line);
        }
    });
    tokio::spawn(async move {
        let mut reader = BufReader::new(stderr).lines();
        while let Ok(Some(line)) = reader.next_line().await {
            eprintln!("[LEADER STDERR] {}", line);
        }
    });

    // Ensure coder's system prompt is present before any other messages!
    fs::write(coder_system.join("prompt.txt"),
        "When you receive input, output only a brief acknowledgment in strict JSON format, with fields 'status' (string, value 'received' if your input was valid json) and 'domain'."
    ).unwrap();

    // 3. Trigger Researcher with an Image + JSON Prompt
    println!("Multimodal Test: Triggering researcher with image...");
    create_test_image(&researcher_input.join("test.png"));
    fs::write(researcher_input.join("query.txt"),
        "Look at this image. Describe its primary colors and pattern. Output your finding strictly as JSON with fields 'colors' (array) and 'pattern' (string). DO NOT use tools."
    ).unwrap();

    // 4. Wait for Researcher Output and parse it
    let mut researcher_data = None;
    let mut previous_path: PathBuf = PathBuf::new();
    for i in 0..120 {
        tokio::time::sleep(Duration::from_secs(1)).await;
        if let Ok(entries) = fs::read_dir(&researcher_output) {
            let files: Vec<_> = entries.filter_map(|e| e.ok()).collect();
            if !files.is_empty() {
                let mut paths: Vec<_> = files.iter().map(|e| e.path()).collect();
                paths.sort();
                if let Some(latest_path) = paths.last() {
                    let content = fs::read_to_string(latest_path).unwrap();
                    if previous_path != *latest_path {
                        println!("Multimodal Test: Researcher output turn {}: {}", i, content);
                    }
                    previous_path = latest_path.clone();
                    if let Some(start) = content.find('{') {
                        if let Some(end) = content.rfind('}') {
                            let json_str = &content[start..=end];
                            if let Ok(val) = serde_json::from_str::<serde_json::Value>(json_str) {
                                researcher_data = Some(val);
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    let researcher_data = researcher_data.expect("Researcher failed to produce valid JSON output");
    assert!(researcher_data["colors"].is_array());

    // 5. Coder should have been triggered automatically. Wait for its output.
    println!("Multimodal Test: Waiting for coder to acknowledge research...");

    let mut coder_data = None;
    let mut previous_path: PathBuf = PathBuf::new();
    for i in 0..120 {
        tokio::time::sleep(Duration::from_secs(1)).await;
        if let Ok(entries) = fs::read_dir(&coder_output) {
            let files: Vec<_> = entries.filter_map(|e| e.ok()).collect();
            if !files.is_empty() {
                let mut paths: Vec<_> = files.iter().map(|e| e.path()).collect();
                paths.sort();
                if let Some(latest_path) = paths.last() {
                    let content = fs::read_to_string(latest_path).unwrap();
                    if previous_path != *latest_path {
                        println!("Multimodal Test: Coder output turn {}: {}", i, content);
                    }
                    previous_path = latest_path.clone();
                    if let Some(start) = content.find('{') {
                        if let Some(end) = content.rfind('}') {
                            let json_str = &content[start..=end];
                            if let Ok(val) = serde_json::from_str::<serde_json::Value>(json_str) {
                                coder_data = Some(val);
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    // Final check
    assert!(
        coder_data.is_some(),
        "Coder failed to trigger and produce JSON acknowledgment"
    );
    assert_eq!(coder_data.unwrap()["status"], "received");

    // 6. Explicit shutdown via IPC (cleaner than kill)
    let _ = Command::new("cargo")
        .args(["run", "--bin", "ag", "--", "shutdown"])
        .output()
        .await;

    cleanup_sockets().await;
}
