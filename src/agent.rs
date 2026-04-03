use crate::config::AgentConfig;
use anyhow::Result;
use std::sync::Arc;
use tokio::sync::{mpsc, watch, Mutex};
use notify::{Watcher, RecursiveMode, Event};
use std::path::PathBuf;
use mistralrs::{Model, RequestBuilder, SamplingParams, MultimodalMessages, TextMessageRole, Response};
use tokio::fs;
use tokio::io::AsyncWriteExt;

pub struct Agent {
    pub name: String,
    pub config: AgentConfig,
    pub model: Arc<Model>,
    pub sampling: SamplingParams,
    pub history: Arc<Mutex<Vec<(TextMessageRole, String)>>>,
}

impl Agent {
    pub fn new(name: String, config: AgentConfig, model: Arc<Model>, sampling: SamplingParams) -> Self {
        Self {
            name,
            config,
            model,
            sampling,
            history: Arc::new(Mutex::new(Vec::new())),
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
                // Reset interrupt file
                let _ = fs::remove_file(&interrupt_path).await;
                let _ = interrupt_tx.send(false);
                continue;
            }

            // Check for input changes
            if event.kind.is_modify() || event.kind.is_create() {
                if event.paths.iter().any(|p| p.starts_with(&input_path)) {
                    // Start inference
                    if let Some(handle) = current_inference.take() {
                        handle.abort();
                    }
                    
                    let model = self.model.clone();
                    let config = self.config.clone();
                    let sampling = self.sampling.clone();
                    let agent_name = name.clone();
                    let agent_path_clone = agent_path.clone();
                    let interrupt_rx = interrupt_tx.subscribe();
                    let history = self.history.clone();
                    
                    current_inference = Some(tokio::spawn(async move {
                        if let Err(e) = run_inference(agent_name, agent_path_clone, model, config, sampling, interrupt_rx, history).await {
                            eprintln!("Inference error: {:?}", e);
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
    sampling: SamplingParams,
    interrupt_rx: watch::Receiver<bool>,
    history: Arc<Mutex<Vec<(TextMessageRole, String)>>>,
) -> Result<()> {
    println!("Agent {}: Starting inference", name);
    
    // Read system prompt
    let system_prompt = fs::read_to_string(path.join("system").join("prompt.txt")).await.unwrap_or_default();
    
    // Build messages using MultimodalMessages builder
    let mut messages = MultimodalMessages::new()
        .add_message(TextMessageRole::System, system_prompt);
    
    // Add history
    {
        let h = history.lock().await;
        for (role, content) in h.iter() {
            messages = messages.add_message(role.clone(), content);
        }
    }

    // Read latest user input
    let user_input = fs::read_to_string(path.join("input").join("user.txt")).await.unwrap_or_default();
    
    // Scan for images in input/
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
        messages = messages.add_image_message(TextMessageRole::User, &user_input, images);
    } else {
        messages = messages.add_message(TextMessageRole::User, &user_input);
    }

    let request = RequestBuilder::from(messages).set_sampling(sampling.clone());
    
    let mut stream = if let Some(ref secondary) = config.secondary_model {
         model.stream_chat_request_with_model(request, Some(secondary)).await?
    } else {
         model.stream_chat_request(request).await?
    };

    let output_file_path = path.join("output").join(format!("out-{}.txt", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_millis()));
    fs::create_dir_all(output_file_path.parent().unwrap()).await?;
    let mut file = fs::File::create(&output_file_path).await?;
    let mut full_response = String::new();

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
                        full_response.push_str(content);
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

    // Update history
    {
        let mut h = history.lock().await;
        h.push((TextMessageRole::User, user_input));
        h.push((TextMessageRole::Assistant, full_response));
        
        // Simple context management: if history too long, summarize
        if h.len() > 10 {
            println!("Agent {}: History long ({} turns), summarizing...", name, h.len());
            let summary: String = crate::context::summarize_history(model.clone(), &h, sampling.clone()).await.unwrap_or_else(|e| format!("Error during summarization: {:?}", e));
            h.clear();
            h.push((TextMessageRole::System, format!("Summary of previous conversation: {}", summary)));
        }
    }

    println!("Agent {}: Inference complete. Output written to {:?}", name, output_file_path);
    Ok(())
}
