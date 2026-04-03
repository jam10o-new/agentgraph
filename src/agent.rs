use crate::config::{AgentConfig, Config};
use crate::context::{CompressionManager, HistoryTurn, HistoryTurnRole, extract_frontmatter};
use anyhow::{Result, Context};
use std::sync::Arc;
use tokio::sync::{mpsc, watch};
use notify::{Watcher, RecursiveMode, Event};
use std::path::PathBuf;
use mistralrs::{Model, RequestBuilder, SamplingParams, MultimodalMessages, TextMessageRole, Response};
use tokio::fs;
use tokio::io::AsyncWriteExt;

pub struct Agent {
    pub name: String,
    pub config: AgentConfig,
    pub global_config: Config,
    pub model: Arc<Model>,
    pub sampling: SamplingParams,
}

impl Agent {
    pub fn new(name: String, config: AgentConfig, global_config: Config, model: Arc<Model>, sampling: SamplingParams) -> Self {
        Self {
            name,
            config,
            global_config,
            model,
            sampling,
        }
    }

    pub async fn run_loop(&self) -> Result<()> {
        let (tx, mut rx) = mpsc::channel(100);
        let name = self.name.clone();
        
        let mut watcher = notify::recommended_watcher(move |res: Result<Event, notify::Error>| {
            if let Ok(event) = res {
                let _ = tx.blocking_send(event);
            }
        })?;

        let agent_path = PathBuf::from(&self.config.path);
        let input_path = agent_path.join("input");
        let system_path = agent_path.join("system");
        let interrupt_path = agent_path.join("interrupt");

        fs::create_dir_all(&input_path).await?;
        fs::create_dir_all(&system_path).await?;
        
        watcher.watch(&input_path, RecursiveMode::NonRecursive)?;
        watcher.watch(&interrupt_path, RecursiveMode::NonRecursive)?;

        println!("Agent {} watching {}", name, agent_path.display());

        let mut current_inference: Option<tokio::task::JoinHandle<()>> = None;
        let (interrupt_tx, _) = watch::channel(false);

        while let Some(event) = rx.recv().await {
            // Check for interrupt
            if event.paths.iter().any(|p| p.ends_with("interrupt")) {
                println!("Agent {}: Interrupt detected", name);
                let _ = interrupt_tx.send(true);
                if let Some(handle) = current_inference.take() {
                    handle.abort();
                }
                let _ = fs::remove_file(&interrupt_path).await;
                let _ = interrupt_tx.send(false);
                continue;
            }

            // Check for input changes
            if event.kind.is_modify() || event.kind.is_create() {
                if event.paths.iter().any(|p| p.starts_with(&input_path)) {
                    // Filter extensions
                    if let Some(path) = event.paths.first() {
                        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                            if !self.config.allowed_extensions.is_empty() && !self.config.allowed_extensions.contains(&ext.to_string()) {
                                continue;
                            }
                        }
                    }

                    if let Some(handle) = current_inference.take() {
                        handle.abort();
                    }
                    
                    let model = self.model.clone();
                    let config = self.config.clone();
                    let global_config = self.global_config.clone();
                    let sampling = self.sampling.clone();
                    let agent_name = name.clone();
                    let agent_path_clone = agent_path.clone();
                    let interrupt_rx = interrupt_tx.subscribe();
                    let name_for_log = name.clone();
                    
                    current_inference = Some(tokio::spawn(async move {
                        if let Err(e) = run_inference(agent_name, agent_path_clone, model, config, global_config, sampling, interrupt_rx).await {
                            eprintln!("Inference error for agent {}: {:?}", name_for_log, e);
                        }
                    }));
                }
            }
        }
        Ok(())
    }
}

async fn run_inference(
    name: String,
    path: PathBuf,
    model: Arc<Model>,
    config: AgentConfig,
    global_config: Config,
    sampling: SamplingParams,
    interrupt_rx: watch::Receiver<bool>,
) -> Result<()> {
    println!("Agent {}: Starting inference", name);
    
    // 1. Load System Prompts & Skills
    let system_dir = path.join("system");
    let mut combined_history = Vec::new();
    if let Ok(mut entries) = fs::read_dir(system_dir).await {
        let mut files = Vec::new();
        while let Some(entry) = entries.next_entry().await? {
            if entry.path().is_file() {
                files.push(entry.path());
            }
        }
        files.sort();
        for f in files {
            if let Ok(content) = fs::read_to_string(&f).await {
                if let Some((name, desc, body)) = extract_frontmatter(&content) {
                    combined_history.push(HistoryTurn {
                        role: HistoryTurnRole::Skill(name, desc),
                        content: body,
                        turn_index: 0,
                    });
                } else {
                    combined_history.push(HistoryTurn {
                        role: HistoryTurnRole::System,
                        content,
                        turn_index: 0,
                    });
                }
            }
        }
    }

    // 2. Load History from output directory
    let output_dir = path.join("output");
    let mut history = Vec::new();
    if let Ok(mut entries) = fs::read_dir(&output_dir).await {
        let mut files = Vec::new();
        while let Some(entry) = entries.next_entry().await? {
            if entry.path().is_file() && entry.path().extension().map_or(false, |e| e == "txt") {
                files.push(entry.path());
            }
        }
        files.sort();
        
        let limit = config.history_limit.unwrap_or(0);
        let start = if limit > 0 { files.len().saturating_sub(limit) } else { 0 };
        
        for (idx, f) in files.iter().enumerate().skip(start) {
            if let Ok(content) = fs::read_to_string(f).await {
                // Heuristic: alternate User/Assistant based on file index
                let role = if idx % 2 == 0 { HistoryTurnRole::User } else { HistoryTurnRole::Assistant };
                history.push(HistoryTurn {
                    role,
                    content,
                    turn_index: idx + 1,
                });
            }
        }
    }
    combined_history.extend(history);

    // 3. Load Latest User Input
    let input_dir = path.join("input");
    let mut input_files = Vec::new();
    if let Ok(mut entries) = fs::read_dir(input_dir).await {
        while let Some(entry) = entries.next_entry().await? {
            if entry.path().is_file() {
                input_files.push(entry.path());
            }
        }
    }
    input_files.sort();
    let latest_input_path = input_files.last().context("No input files found in input/")?;
    let latest_user_input = fs::read_to_string(latest_input_path).await?;

    // 4. Compression Pass
    let comp_mgr = CompressionManager::new(
        path.clone(),
        global_config.compression.threshold,
        global_config.compression.inverse_probability,
        global_config.compression.resummarize_probability,
    );
    
    let compressed_context = comp_mgr.get_compressed_context(
        model.clone(),
        &combined_history,
        &latest_user_input,
        sampling.clone()
    ).await?;

    // 5. Build Final Request
    let mut messages = MultimodalMessages::new();
    for (role, content) in compressed_context {
        messages = messages.add_message(role, content);
    }
    
    // Add images if present in input directory
    let mut images = Vec::new();
    if let Ok(mut entries) = fs::read_dir(path.join("input")).await {
        while let Some(entry) = entries.next_entry().await? {
            let p = entry.path();
            if let Some(ext) = p.extension().and_then(|e| e.to_str()) {
                match ext.to_lowercase().as_str() {
                    "jpg" | "jpeg" | "png" | "webp" => {
                        if let Ok(img) = image::open(&p) {
                            images.push(img);
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    if !images.is_empty() {
        messages = messages.add_image_message(TextMessageRole::User, &latest_user_input, images);
    } else {
        messages = messages.add_message(TextMessageRole::User, &latest_user_input);
    }

    let request = RequestBuilder::from(messages).set_sampling(sampling);
    let mut stream = model.stream_chat_request(request).await?;

    let output_file_path = path.join("output").join(format!("out-{}.txt", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_millis()));
    fs::create_dir_all(output_file_path.parent().unwrap()).await?;
    let mut file = fs::File::create(&output_file_path).await?;

    while let Some(chunk) = stream.next().await {
        if *interrupt_rx.borrow() {
            println!("Agent {}: Inference interrupted", name);
            break;
        }

        match chunk {
            Response::Chunk(c) => {
                if let Some(choice) = c.choices.first() {
                    if let Some(ref content) = choice.delta.content {
                        file.write_all(content.as_bytes()).await?;
                        file.flush().await?;
                    }
                }
            }
            Response::InternalError(e) => {
                eprintln!("Agent {}: Internal error: {:?}", name, e);
                break;
            }
            Response::ModelError(e, _) => {
                eprintln!("Agent {}: Model error: {:?}", name, e);
                break;
            }
            Response::ValidationError(e) => {
                eprintln!("Agent {}: Validation error: {:?}", name, e);
                break;
            }
            _ => {}
        }
    }

    println!("Agent {}: Inference complete. Output written to {:?}", name, output_file_path);
    Ok(())
}
