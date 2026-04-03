use crate::config::{AgentConfig, Config};
use crate::context::{CompressionManager, HistoryTurn, HistoryTurnRole, extract_frontmatter};
use crate::ipc::Command;
use crate::find_leader_socket;
use anyhow::{Result, Context, anyhow};
use std::sync::Arc;
use tokio::sync::{mpsc, watch, Mutex};
use notify::{Watcher, RecursiveMode, Event};
use std::path::{Path, PathBuf};
use mistralrs::{
    Model, RequestBuilder, SamplingParams, TextMessageRole, Response,
    Tool, ToolType, Function, ToolChoice,
};
use tokio::fs;
use tokio::io::{AsyncWriteExt, AsyncReadExt};
use tokio::net::UnixStream;
use std::time::SystemTime;

macro_rules! json_schema_obj {
    ($($json:tt)+) => {
        {
            let val = serde_json::json!($($json)+);
            if let serde_json::Value::Object(map) = val {
                let mut schema = std::collections::HashMap::new();
                for (k, v) in map { schema.insert(k, v); }
                schema
            } else { std::collections::HashMap::new() }
        }
    };
}

pub struct Agent {
    pub name: String,
    pub config: AgentConfig,
    pub global_config: Config,
    pub model: Arc<Model>,
    pub sampling: SamplingParams,
    pub volatile_context: Arc<Mutex<Vec<(TextMessageRole, String)>>>,
}

impl Agent {
    pub fn new(name: String, config: AgentConfig, global_config: Config, model: Arc<Model>, sampling: SamplingParams) -> Self {
        Self {
            name,
            config,
            global_config,
            model,
            sampling,
            volatile_context: Arc::new(Mutex::new(Vec::new())),
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

        // Watch all input directories
        for input_path in &self.config.inputs {
            let p = PathBuf::from(input_path);
            fs::create_dir_all(&p).await?;
            watcher.watch(&p, RecursiveMode::NonRecursive)?;
        }
        
        // Ensure output and system directories exist
        fs::create_dir_all(&self.config.output).await?;
        for sys_path in &self.config.system {
            fs::create_dir_all(sys_path).await?;
        }

        println!("Agent {} watching inputs: {:?}", name, self.config.inputs);

        let mut current_inference: Option<tokio::task::JoinHandle<()>> = None;
        let (interrupt_tx, _) = watch::channel(false);

        while let Some(event) = rx.recv().await {
            // Any new file or modification in inputs is both an interrupt and a trigger
            if event.kind.is_modify() || event.kind.is_create() {
                let is_input_event = event.paths.iter().any(|p| {
                    self.config.inputs.iter().any(|input_dir| p.starts_with(input_dir))
                });

                if is_input_event {
                    println!("Agent {}: Trigger/Interrupt from input event", name);
                    
                    // Signal interrupt to current inference if any
                    let _ = interrupt_tx.send(true);
                    if let Some(handle) = current_inference.take() {
                        handle.abort();
                    }
                    let _ = interrupt_tx.send(false);

                    let model = self.model.clone();
                    let config = self.config.clone();
                    let global_config = self.global_config.clone();
                    let sampling = self.sampling.clone();
                    let agent_name = name.clone();
                    let interrupt_rx = interrupt_tx.subscribe();
                    let volatile_context = self.volatile_context.clone();
                    let log_name = name.clone();
                    
                    current_inference = Some(tokio::spawn(async move {
                        if let Err(e) = run_inference(agent_name, model, config, global_config, sampling, interrupt_rx, volatile_context).await {
                            eprintln!("Inference error for agent {}: {:?}", log_name, e);
                        }
                    }));
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct FileEntry {
    path: PathBuf,
    created: SystemTime,
    role: HistoryTurnRole,
}

async fn run_inference(
    name: String,
    model: Arc<Model>,
    config: AgentConfig,
    global_config: Config,
    sampling: SamplingParams,
    interrupt_rx: watch::Receiver<bool>,
    volatile_context: Arc<Mutex<Vec<(TextMessageRole, String)>>>,
) -> Result<()> {
    println!("Agent {}: Starting inference", name);
    
    // 1. Collate System Prompts
    let mut system_content = String::new();
    for sys_dir in &config.system {
        if let Ok(mut entries) = fs::read_dir(sys_dir).await {
            let mut files = Vec::new();
            while let Some(entry) = entries.next_entry().await? {
                if entry.path().is_file() { files.push(entry.path()); }
            }
            files.sort();
            for f in files {
                if let Ok(content) = fs::read_to_string(&f).await {
                    if !system_content.is_empty() { system_content.push_str("\n\n"); }
                    system_content.push_str(&content);
                }
            }
        }
    }

    let mut combined_history = Vec::new();
    if !system_content.is_empty() {
        if let Some((n, d, body)) = extract_frontmatter(&system_content) {
            combined_history.push(HistoryTurn { role: HistoryTurnRole::Skill(n, d), content: body, turn_index: 0 });
        } else {
            combined_history.push(HistoryTurn { role: HistoryTurnRole::System, content: system_content, turn_index: 0 });
        }
    }

    // 2. Collate User and Assistant History by Creation Time
    let mut all_files = Vec::new();
    
    // Gather Inputs (User)
    for input_dir in &config.inputs {
        if let Ok(mut entries) = fs::read_dir(input_dir).await {
            while let Some(entry) = entries.next_entry().await? {
                let p = entry.path();
                if p.is_file() {
                    let metadata = fs::metadata(&p).await?;
                    all_files.push(FileEntry {
                        path: p,
                        created: metadata.created()?,
                        role: HistoryTurnRole::User,
                    });
                }
            }
        }
    }

    // Gather Output (Assistant)
    if let Ok(mut entries) = fs::read_dir(&config.output).await {
        while let Some(entry) = entries.next_entry().await? {
            let p = entry.path();
            if p.is_file() {
                let metadata = fs::metadata(&p).await?;
                all_files.push(FileEntry {
                    path: p,
                    created: metadata.created()?,
                    role: HistoryTurnRole::Assistant,
                });
            }
        }
    }

    // Sort by creation time
    all_files.sort_by_key(|f| f.created);

    // Apply history limit
    let limit = config.history_limit.unwrap_or(0);
    let start_idx = if limit > 0 && all_files.len() > limit {
        all_files.len() - limit
    } else {
        0
    };

    let mut turn_idx = 1;
    for entry in all_files.iter().skip(start_idx) {
        if let Ok(content) = fs::read_to_string(&entry.path).await {
            combined_history.push(HistoryTurn {
                role: entry.role.clone(),
                content,
                turn_index: turn_idx,
            });
            turn_idx += 1;
        }
    }

    // The "latest user input" is actually already part of the history if we sorted correctly.
    // However, CompressionManager usually expects history + latest_input separately.
    // Let's pop the last User message if it exists to serve as the trigger.
    let (history, latest_user_input) = if let Some(last_turn) = combined_history.last() {
        if matches!(last_turn.role, HistoryTurnRole::User) {
            let latest = last_turn.content.clone();
            let mut h = combined_history.clone();
            h.pop();
            (h, latest)
        } else {
            (combined_history.clone(), String::new())
        }
    } else {
        (combined_history.clone(), String::new())
    };

    // 3. Compression
    let comp_mgr = CompressionManager::new(
        PathBuf::from(&config.output).parent().unwrap_or_else(|| Path::new(".")).to_path_buf(),
        global_config.compression.threshold,
        global_config.compression.inverse_probability,
        global_config.compression.resummarize_probability,
    );
    let compressed_context = comp_mgr.get_compressed_context(model.clone(), &history, &latest_user_input, sampling.clone()).await?;

    let mut messages = RequestBuilder::new();
    for (role, content) in compressed_context { messages = messages.add_message(role, content); }
    
    {
        let mut v_ctx = volatile_context.lock().await;
        for (role, content) in v_ctx.drain(..) { messages = messages.add_message(role, content); }
    }

    // Latest user input is already in messages? No, get_compressed_context handles history. 
    // We still need to add the latest_user_input to the request.
    messages = messages.add_message(TextMessageRole::User, &latest_user_input);

    let tools = vec![
        Tool {
            tp: ToolType::Function,
            function: Function {
                name: "execute_command".into(),
                description: Some("Execute a shell command on the host".into()),
                parameters: Some(json_schema_obj!({
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "args": {"type": "array", "items": {"type": "string"}}
                    }
                })),
            }
        },
        Tool {
            tp: ToolType::Function,
            function: Function {
                name: "spawn_new_agent".into(),
                description: Some("Spawn a new agent dynamically".into()),
                parameters: Some(json_schema_obj!({
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "inputs": {"type": "array", "items": {"type": "string"}},
                        "output": {"type": "string"},
                        "system": {"type": "array", "items": {"type": "string"}},
                        "model": {"type": "string"},
                        "history_limit": {"type": "integer", "nullable": true}
                    }
                })),
            }
        },
        Tool {
            tp: ToolType::Function,
            function: Function {
                name: "load_into_context".into(),
                description: Some("Load files into context for the next turn".into()),
                parameters: Some(json_schema_obj!({
                    "type": "object",
                    "properties": {
                        "files": {"type": "array", "items": {"type": "string"}}
                    }
                })),
            }
        }
    ];

    let timestamp = SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_millis();
    let output_file_path = PathBuf::from(&config.output).join(format!("out-{}.txt", timestamp));
    let mut file = fs::File::create(&output_file_path).await?;

    let mut request = messages.set_sampling(sampling).set_tools(tools).set_tool_choice(ToolChoice::Auto);
    
    loop {
        let mut model_stream = model.stream_chat_request(request.clone()).await?;
        let mut accumulated_content = String::new();
        let mut tool_calls = Vec::new();

        while let Some(chunk) = model_stream.next().await {
            if *interrupt_rx.borrow() { 
                println!("Agent {}: Inference interrupted", name);
                return Ok(()); 
            }
            match chunk {
                Response::Chunk(c) => {
                    if let Some(choice) = c.choices.first() {
                        if let Some(ref content) = choice.delta.content {
                            accumulated_content.push_str(content);
                            file.write_all(content.as_bytes()).await?;
                            file.flush().await?;
                        }
                        if let Some(ref tcs) = choice.delta.tool_calls {
                            tool_calls.extend(tcs.clone());
                        }
                    }
                }
                _ => {}
            }
        }

        if tool_calls.is_empty() { break; }

        request = request.add_message_with_tool_call(TextMessageRole::Assistant, &accumulated_content, tool_calls.clone());
        for tc in tool_calls {
            let result = match tc.function.name.as_str() {
                "execute_command" => {
                    let args: serde_json::Value = serde_json::from_str(&tc.function.arguments)?;
                    let cmd = args["command"].as_str().unwrap_or_default();
                    let args_vec: Vec<String> = args["args"].as_array().unwrap_or(&vec![]).iter().map(|v| v.as_str().unwrap_or_default().to_string()).collect();
                    match tokio::process::Command::new(cmd).args(args_vec).output().await {
                        Ok(output) => format!("Stdout: {}\nStderr: {}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr)),
                        Err(e) => format!("Error executing command: {}", e),
                    }
                }
                "spawn_new_agent" => {
                    let args: serde_json::Value = serde_json::from_str(&tc.function.arguments)?;
                    let name = args["name"].as_str().unwrap_or_default().to_string();
                    let config = AgentConfig {
                        inputs: args["inputs"].as_array().unwrap_or(&vec![]).iter().map(|v| v.as_str().unwrap_or_default().to_string()).collect(),
                        output: args["output"].as_str().unwrap_or_default().to_string(),
                        system: args["system"].as_array().unwrap_or(&vec![]).iter().map(|v| v.as_str().unwrap_or_default().to_string()).collect(),
                        model: args["model"].as_str().unwrap_or("primary").to_string(),
                        history_limit: args["history_limit"].as_u64().map(|u| u as usize),
                        stream: true,
                        allowed_extensions: vec![],
                    };
                    send_ipc_command(Command::SpawnAgent { name, config }).await.unwrap_or_else(|e| e.to_string())
                }
                "load_into_context" => {
                    let args: serde_json::Value = serde_json::from_str(&tc.function.arguments)?;
                    let empty_vec = vec![];
                    let files = args["files"].as_array().unwrap_or(&empty_vec);
                    let mut loaded = Vec::new();
                    for f in files {
                        let p = f.as_str().unwrap_or_default();
                        if let Ok(c) = fs::read_to_string(p).await {
                            loaded.push((TextMessageRole::System, format!("File {}:\n{}", p, c)));
                        }
                    }
                    volatile_context.lock().await.extend(loaded);
                    "Files loaded into context for next turn".into()
                }
                _ => format!("Unknown tool: {}", tc.function.name),
            };
            request = request.add_tool_message(result, tc.id);
        }
    }

    println!("Agent {}: Inference complete.", name);
    Ok(())
}

async fn send_ipc_command(cmd: Command) -> Result<String> {
    let socket_path = find_leader_socket().await.ok_or_else(|| anyhow!("Leader not found"))?;
    let mut stream = UnixStream::connect(socket_path).await?;
    let payload = serde_json::to_vec(&cmd)?;
    stream.write_all(&payload).await?;
    stream.flush().await?;
    let mut resp = String::new();
    stream.read_to_string(&mut resp).await?;
    Ok(resp)
}
