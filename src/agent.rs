use crate::audio::AudioListener;
use crate::config::{AgentConfig, Config};
use crate::context::{CompressionManager, HistoryTurn, HistoryTurnRole, extract_frontmatter};
use crate::find_leader_socket;
use crate::ipc::Command;
use crate::utils::AgentLogger;
use anyhow::{Context, Result, anyhow};
use mistralrs::{
    Function, Model, MultimodalMessages, RequestBuilder, Response, SamplingParams, TextMessageRole,
    Tool, ToolCallResponse, ToolChoice, ToolType,
};
use notify::{Event, RecursiveMode, Watcher};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::fs;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;
use tokio::sync::{Mutex, mpsc};

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
    pub logger: AgentLogger,
}

impl Agent {
    pub fn new(
        name: String,
        config: AgentConfig,
        global_config: Config,
        model: Arc<Model>,
        sampling: SamplingParams,
    ) -> Self {
        Self {
            name: name.clone(),
            config,
            global_config,
            model,
            sampling,
            volatile_context: Arc::new(Mutex::new(Vec::new())),
            logger: AgentLogger::new(&name),
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

        let mut canonical_inputs = Vec::new();
        for input_path in &self.config.inputs {
            let p = PathBuf::from(input_path);
            fs::create_dir_all(&p)
                .await
                .context(format!("Failed to create input directory: {}", p.display()))?;
            let cp = fs::canonicalize(&p)
                .await
                .context(format!("Failed to canonicalize input path: {}", p.display()))?;
            watcher.watch(&cp, RecursiveMode::NonRecursive)?;
            canonical_inputs.push(cp);
        }

        for output in &self.config.output {
            fs::create_dir_all(output)
                .await
                .context(format!("Failed to create output directory: {}", output))?;
        }
        if let Some(ref stream_dir) = self.config.stream_output {
            let _ = fs::create_dir_all(stream_dir).await;
        }
        if let Some(ref tool_dir) = self.config.tool_output {
            let _ = fs::create_dir_all(tool_dir).await;
        } else if let Some(first_output) = self.config.output.first() {
            let default_tool_dir = format!("{}/tool_output", first_output);
            let _ = fs::create_dir_all(&default_tool_dir).await;
        }
        for sys_path in &self.config.system {
            let _ = fs::create_dir_all(sys_path).await;
        }

        self.logger
            .log(&format!("Watching inputs: {:?}", canonical_inputs))
            .await;

        if self.config.realtime_audio {
            let listener = AudioListener::new(self.name.clone(), canonical_inputs[0].clone());
            tokio::spawn(async move {
                if let Err(e) = listener.start().await {
                    eprintln!("Audio listener error: {:?}", e);
                }
            });
        }

        let (inference_done_tx, mut inference_done_rx) = mpsc::channel::<()>(1);
        let mut inference_in_progress = false;
        let mut retrigger_pending = false;
        let mut debounce_timer = Box::pin(tokio::time::sleep(Duration::MAX));
        let mut timer_active = false;
        let debounce_duration = Duration::from_millis(250);

        loop {
            tokio::select! {
                Some(event) = rx.recv() => {
                    if !event.kind.is_remove() {
                        let is_input_event = event.paths.iter().any(|p| {
                            let p_abs = p.canonicalize().unwrap_or_else(|_| p.clone());
                            canonical_inputs.iter().any(|input_dir| p_abs.starts_with(input_dir))
                        });

                        if is_input_event {
                            if inference_in_progress {
                                retrigger_pending = true;
                            }
                            debounce_timer.as_mut().reset(tokio::time::Instant::now() + debounce_duration);
                            timer_active = true;
                        }
                    }
                }
                _ = &mut debounce_timer, if timer_active => {
                    timer_active = false;
                    while let Ok(_) = rx.try_recv() {}
                    if inference_in_progress {
                        // Already running — just note that new input is waiting.
                        // The completion handler below will retrigger.
                        continue;
                    }
                    self.logger.log("Triggering inference after debounce").await;

                    let model = self.model.clone();
                    let config = self.config.clone();
                    let global_config = self.global_config.clone();
                    let sampling = self.sampling.clone();
                    let agent_name = name.clone();
                    let log_name = name.clone();
                    let volatile_context = self.volatile_context.clone();
                    let logger = AgentLogger::new(&name);
                    let done_tx = inference_done_tx.clone();

                    inference_in_progress = true;
                    retrigger_pending = false;
                    tokio::spawn(async move {
                        let result = run_inference(agent_name, model, config, global_config, sampling, volatile_context, logger).await;
                        if let Err(e) = result {
                            eprintln!("Inference error for agent {}: {:?}", log_name, e);
                        }
                        let _ = done_tx.send(()).await;
                    });
                }
                _ = inference_done_rx.recv() => {
                    inference_in_progress = false;
                    if retrigger_pending {
                        // New input arrived during the previous turn.
                        // Start a fresh inference immediately.
                        retrigger_pending = false;
                        while let Ok(_) = rx.try_recv() {}
                        self.logger.log("Retriggering inference for pending input").await;

                        let model = self.model.clone();
                        let config = self.config.clone();
                        let global_config = self.global_config.clone();
                        let sampling = self.sampling.clone();
                        let agent_name = name.clone();
                        let log_name = name.clone();
                        let volatile_context = self.volatile_context.clone();
                        let logger = AgentLogger::new(&name);
                        let done_tx = inference_done_tx.clone();

                        inference_in_progress = true;
                        tokio::spawn(async move {
                            let result = run_inference(agent_name, model, config, global_config, sampling, volatile_context, logger).await;
                            if let Err(e) = result {
                                eprintln!("Inference error for agent {}: {:?}", log_name, e);
                            }
                            let _ = done_tx.send(()).await;
                        });
                    }
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
struct FileEntry {
    path: PathBuf,
    created: SystemTime,
    role: HistoryTurnRole,
    metadata_str: String,
    excluded: bool,
}

fn format_file_metadata(path: &Path, metadata: &std::fs::Metadata) -> String {
    let abs_path = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    let modified = metadata
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| format!("{}", d.as_secs()))
        .unwrap_or_else(|| "unknown".to_string());
    let size = metadata.len();
    format!(
        "[File: {} | Modified: {} | Size: {} bytes]",
        abs_path.display(),
        modified,
        size
    )
}

/// Heuristic to detect OOM-like errors from their string representation.
/// Matches common patterns from CUDA, Metal, and general allocation failures.
fn looks_like_oom(err_str: &str) -> bool {
    let lower = err_str.to_lowercase();
    lower.contains("out of memory")
        || lower.contains("oom")
        || lower.contains("allocation failed")
        || lower.contains("not enough memory")
        || lower.contains("cuda error")
        || lower.contains("metal error")
        || lower.contains("cannot allocate")
        || lower.contains("memory exhausted")
}

async fn run_inference(
    _name: String,
    model: Arc<Model>,
    config: AgentConfig,
    _global_config: Config,
    sampling: SamplingParams,
    volatile_context: Arc<Mutex<Vec<(TextMessageRole, String)>>>,
    logger: AgentLogger,
) -> Result<()> {
    logger.log("Starting inference turn").await;

    // 1. Collate System Prompts
    let mut system_content = String::new();
    for sys_dir in &config.system {
        if let Ok(mut entries) = fs::read_dir(sys_dir).await {
            let mut files = Vec::new();
            while let Some(entry) = entries.next_entry().await? {
                let file_name = entry.file_name();
                let name_str = file_name.to_string_lossy();
                // Skip hidden files (e.g. tool definitions written by agentgraph)
                if name_str.starts_with('.') {
                    continue;
                }
                if entry.path().is_file() {
                    files.push(entry.path());
                }
            }
            files.sort();
            for f in files {
                if let Ok(content) = fs::read_to_string(&f).await {
                    if !system_content.is_empty() {
                        system_content.push_str("\n\n");
                    }
                    system_content.push_str(&content);
                }
            }
        }
    }

    if let Some(extra_prompt) = &config.prompt {
        if !system_content.is_empty() {
            system_content.push_str("\n\n");
        }
        system_content.push_str(extra_prompt);
    }

    let mut combined_history = Vec::new();
    if !system_content.is_empty() {
        if let Some((n, d, body)) = extract_frontmatter(&system_content) {
            combined_history.push(HistoryTurn {
                role: HistoryTurnRole::Skill(n, d),
                content: body,
                turn_index: 0,
                excluded_from_compression: false,
            });
        } else {
            combined_history.push(HistoryTurn {
                role: HistoryTurnRole::System,
                content: system_content,
                turn_index: 0,
                excluded_from_compression: false,
            });
        }
    }

    // Build canonical excluded directories for comparison
    let mut excluded_canonical = Vec::new();
    for ex in &config.excluded_from_summary {
        if let Ok(c) = fs::canonicalize(ex).await {
            excluded_canonical.push(c);
        }
    }

    // 2. Collate User and Assistant History
    let mut all_files = Vec::new();
    for input_dir in &config.inputs {
        let input_canonical = fs::canonicalize(input_dir)
            .await
            .unwrap_or_else(|_| PathBuf::from(input_dir));
        let is_excluded = excluded_canonical
            .iter()
            .any(|ex| input_canonical.starts_with(ex) || ex.starts_with(&input_canonical));
        if let Ok(mut entries) = fs::read_dir(input_dir).await {
            while let Some(entry) = entries.next_entry().await? {
                let p = entry.path();
                if p.is_file() {
                    let p_display = p.display().to_string();
                    let metadata = fs::metadata(&p)
                        .await
                        .context(format!(
                            "Failed to read metadata for input file: {}",
                            p_display
                        ))?;
                    let meta_str = format_file_metadata(&p, &metadata);
                    all_files.push(FileEntry {
                        path: p,
                        created: metadata
                            .created()
                            .map_err(|e| {
                                anyhow!(
                                    "Failed to get creation time for {}: {e}",
                                    p_display
                                )
                            })?,
                        role: HistoryTurnRole::User,
                        metadata_str: meta_str,
                        excluded: is_excluded,
                    });
                }
            }
        }
    }
    for output in &config.output {
        if let Ok(mut entries) = fs::read_dir(output).await {
            while let Some(entry) = entries.next_entry().await? {
                let p = entry.path();
                if p.is_file() {
                    let p_display = p.display().to_string();
                    let metadata = fs::metadata(&p)
                        .await
                        .context(format!(
                            "Failed to read metadata for output file: {}",
                            p_display
                        ))?;
                    all_files.push(FileEntry {
                        path: p,
                        created: metadata
                            .created()
                            .map_err(|e| {
                                anyhow!(
                                    "Failed to get creation time for {}: {e}",
                                    p_display
                                )
                            })?,
                        role: HistoryTurnRole::Assistant,
                        metadata_str: String::new(),
                        excluded: false,
                    });
                }
            }
        }
    }
    // Tool outputs are loaded as assistant history so the model sees its own prior tool results.
    // When tool_output is not explicitly set, default to the "tool_output" subdirectory
    // within the first output directory.
    let effective_tool_dir = config.tool_output.clone().or_else(|| {
        config
            .output
            .first()
            .map(|o| format!("{}/tool_output", o))
    });
    if let Some(ref tool_dir) = effective_tool_dir {
        if let Ok(mut entries) = fs::read_dir(tool_dir).await {
            while let Some(entry) = entries.next_entry().await? {
                let p = entry.path();
                if p.is_file() {
                    let p_display = p.display().to_string();
                    let metadata = fs::metadata(&p)
                        .await
                        .context(format!(
                            "Failed to read metadata for tool output file: {}",
                            p_display
                        ))?;
                    all_files.push(FileEntry {
                        path: p,
                        created: metadata
                            .created()
                            .map_err(|e| {
                                anyhow!(
                                    "Failed to get creation time for {}: {e}",
                                    p_display
                                )
                            })?,
                        role: HistoryTurnRole::Tool,
                        metadata_str: String::new(),
                        excluded: false,
                    });
                }
            }
        }
    }
    all_files.sort_by_key(|f| f.created);

    let limit = config.history_limit.unwrap_or(0);
    let start_idx = if limit > 0 && all_files.len() > limit {
        all_files.len() - limit
    } else {
        0
    };

    let mut turn_idx = 1;
    for entry in all_files.iter().skip(start_idx) {
        if let Ok(content) = fs::read_to_string(&entry.path).await {
            let final_content =
                if matches!(entry.role, HistoryTurnRole::User) && !entry.metadata_str.is_empty() {
                    format!("{}\n---\n{}", entry.metadata_str, content)
                } else {
                    content
                };
            combined_history.push(HistoryTurn {
                role: entry.role.clone(),
                content: final_content,
                turn_index: turn_idx,
                excluded_from_compression: entry.excluded,
            });
            turn_idx += 1;
        }
    }

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

    // Determine fallback text for multimodal messages that have no text content.
    // This uses the metadata of the actual latest user file, even if it is binary.
    let latest_user_file = all_files
        .iter()
        .filter(|f| matches!(f.role, HistoryTurnRole::User))
        .last();
    let fallback_text = latest_user_file
        .map(|f| f.metadata_str.clone())
        .unwrap_or_default();

    // 3. Compression
    let checkpoint_base = config.output.first()
        .and_then(|o| PathBuf::from(o).parent().map(|p| p.to_path_buf()))
        .or_else(|| config.stream_output.as_ref().and_then(|s| PathBuf::from(s).parent().map(|p| p.to_path_buf())))
        .unwrap_or_else(|| PathBuf::from("."));
    // Save for potential OOM recovery rebuild
    let oom_checkpoint_base = checkpoint_base.clone();
    let oom_compression_config = config.compression.clone();
    let comp_mgr = CompressionManager::new(checkpoint_base, &config.compression);
    // Snapshot of uncompressed history so OOM recovery can re-compress
    // with more aggressive checkpoint limits without losing turns.
    let uncompressed_history: Vec<HistoryTurn> = history.clone();
    let mut history_mut = history;
    let compressed_context = comp_mgr
        .get_compressed_context(
            model.clone(),
            &mut history_mut,
            &latest_user_input,
            sampling.clone(),
            config.context_checkpoint_limit,
        )
        .await?;

    // 4. Build Request
    let mut multimodal = MultimodalMessages::new().enable_thinking(config.enable_thinking);
    for (role, content) in &compressed_context {
        multimodal = multimodal.add_message(role.clone(), content.as_str());
    }
    // Drain volatile context and save for potential OOM rebuild
    let volatile_drained: Vec<(TextMessageRole, String)> = {
        let mut v_ctx = volatile_context.lock().await;
        v_ctx.drain(..).collect()
    };
    for (role, content) in &volatile_drained {
        multimodal = multimodal.add_message(role.clone(), content.as_str());
    }

    let mut images = Vec::new();
    let mut audio_files = Vec::new();
    for input_dir in &config.inputs {
        if let Ok(mut entries) = fs::read_dir(input_dir).await {
            while let Some(entry) = entries.next_entry().await? {
                let p = entry.path();
                if let Some(ext) = p.extension().and_then(|e| e.to_str()) {
                    match ext.to_lowercase().as_str() {
                        "jpg" | "jpeg" | "png" | "webp" => {
                            if let Ok(img) = image::open(&p) {
                                images.push(img);
                            }
                        }
                        "wav" | "mp3" | "ogg" => {
                            audio_files.push(p);
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    let effective_user_text =
        if latest_user_input.is_empty() && (!images.is_empty() || !audio_files.is_empty()) {
            fallback_text
        } else {
            latest_user_input
        };
    // Save for potential OOM recovery recompression
    let oom_latest_input = effective_user_text.clone();

    if !images.is_empty() {
        logger
            .log(&format!("Adding {} images to request", images.len()))
            .await;
        multimodal =
            multimodal.add_image_message(TextMessageRole::User, &effective_user_text, images.clone());
    } else {
        multimodal = multimodal.add_message(TextMessageRole::User, &effective_user_text);
    }

    let tools = if config.tools_enabled {
        vec![
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
                    strict: None,
                },
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
                            "stream_output": {"type": "string", "nullable": true},
                            "tool_output": {"type": "string", "nullable": true},
                            "system": {"type": "array", "items": {"type": "string"}},
                            "model": {"type": "string"},
                            "history_limit": {"type": "integer", "nullable": true},
                            "realtime_audio": {"type": "boolean"},
                            "prompt": {"type": "string", "nullable": true},
                            "tools_enabled": {"type": "boolean"},
                            "excluded_from_summary": {"type": "array", "items": {"type": "string"}},
                            "context_checkpoint_limit": {"type": "integer", "nullable": true}
                        }
                    })),
                    strict: None,
                },
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
                    strict: None,
                },
            },
            Tool {
                tp: ToolType::Function,
                function: Function {
                    name: "read_file".into(),
                    description: Some(
                        "Read the contents of one or more files and return them immediately".into(),
                    ),
                    parameters: Some(json_schema_obj!({
                        "type": "object",
                        "properties": {
                            "files": {"type": "array", "items": {"type": "string"}}
                        }
                    })),
                    strict: None,
                },
            },
            Tool {
                tp: ToolType::Function,
                function: Function {
                    name: "list_directory".into(),
                    description: Some("List files and directories at a given path".into()),
                    parameters: Some(json_schema_obj!({
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"}
                        }
                    })),
                    strict: None,
                },
            },
            Tool {
                tp: ToolType::Function,
                function: Function {
                    name: "list_skills".into(),
                    description: Some("Discover available skills by searching for SKILL.md files in ~/.config and the current working directory. Returns a list of skills with their names, descriptions, and paths.".into()),
                    parameters: Some(json_schema_obj!({
                        "type": "object",
                        "properties": {
                            "search_paths": {"type": "array", "items": {"type": "string"}, "description": "Optional additional paths to search. Defaults to ~/.config and cwd."}
                        }
                    })),
                    strict: None,
                },
            },
            Tool {
                tp: ToolType::Function,
                function: Function {
                    name: "load_skill".into(),
                    description: Some("Load a skill into the agent's system context by copying its directory into a system prompt directory. The skill's SKILL.md and any reference files become part of the agent's system prompt.".into()),
                    parameters: Some(json_schema_obj!({
                        "type": "object",
                        "properties": {
                            "skill_path": {"type": "string", "description": "Path to the skill directory containing SKILL.md"},
                            "system_dir": {"type": "string", "description": "Target system directory to copy the skill into. Defaults to the agent's first system directory."}
                        },
                        "required": ["skill_path"]
                    })),
                    strict: None,
                },
            },
        ]
    } else {
        vec![]
    };

    // When not consuming tool calls (distillation mode), write the tool
    // definitions to a well-known hidden file in the first system directory
    // so downstream harnesses know what tools the model had access to.
    if !config.consume_tool_calls && config.tools_enabled && !tools.is_empty() {
        if let Some(ref sys_dir) = config.system.first() {
            let tools_path = PathBuf::from(sys_dir).join(".agentgraph_tools.json");
            if let Ok(json) = serde_json::to_string_pretty(&tools) {
                let _ = fs::write(&tools_path, &json).await;
            }
        }
    }

    let timestamp = SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_millis();
    let output_file_path: Option<PathBuf> = config.output.first().map(|o| {
        PathBuf::from(o).join(format!("out-{}.txt", timestamp))
    });

    let mut stream_file = if let Some(ref stream_dir) = config.stream_output {
        let stream_path = PathBuf::from(stream_dir).join(format!("out-{}.txt", timestamp));
        fs::create_dir_all(stream_dir).await?;
        logger
            .log(&format!(
                "Streaming enabled, creating stream output file: {:?}",
                stream_path
            ))
            .await;
        Some(
            fs::File::create(&stream_path)
                .await
                .context(format!(
                    "Failed to create stream output file: {}",
                    stream_path.display()
                ))?,
        )
    } else {
        logger
            .log("Streaming disabled, will create output file after completion")
            .await;
        None
    };

    let mut request = RequestBuilder::from(multimodal)
        .set_sampling(sampling.clone())
        .set_tools(tools.clone())
        .set_tool_choice(ToolChoice::Auto);

    let retry_limit = config.inference_retries;
    let retry_delay = Duration::from_millis(config.inference_retry_delay_ms);

    // OOM recovery state: when the normal-strength context OOMs the GPU,
    // we re-compress with a halved context_checkpoint_limit and retry the
    // entire turn. This preserves more history than dropping everything.
    let mut oom_recompression_done = false;
    let mut oom_recovery_pending = false;
    let oom_aggressive_limit = config.context_checkpoint_limit
        .map(|l| (l / 2).max(1));

    loop {
        // If the previous turn failed with OOM, recompress the original
        // uncompressed history with tighter limits before rebuilding the
        // request.
        if oom_recovery_pending {
            logger.log("Recompressing context with aggressive limits after OOM").await;
            let oom_comp_mgr = CompressionManager::new(
                oom_checkpoint_base.clone(),
                &oom_compression_config,
            );
            let mut history_retry = uncompressed_history.clone();
            let recompr_context = match oom_comp_mgr
                .get_compressed_context(
                    model.clone(),
                    &mut history_retry,
                    &oom_latest_input,
                    sampling.clone(),
                    oom_aggressive_limit,
                )
                .await
            {
                Ok(ctx) => ctx,
                Err(e) => {
                    logger.log(&format!("OOM recompression failed: {:?}", e)).await;
                    break; // give up
                }
            };
            // Rebuild the request from the re-compressed context
            let mut new_multimodal =
                MultimodalMessages::new().enable_thinking(config.enable_thinking);
            for (role, content) in &recompr_context {
                new_multimodal = new_multimodal.add_message(role.clone(), content.as_str());
            }
            for (role, content) in &volatile_drained {
                new_multimodal = new_multimodal.add_message(role.clone(), content.as_str());
            }
            if !images.is_empty() {
                new_multimodal = new_multimodal.add_image_message(
                    TextMessageRole::User,
                    &effective_user_text,
                    images.clone(),
                );
            } else {
                new_multimodal = new_multimodal
                    .add_message(TextMessageRole::User, &effective_user_text);
            }
            request = RequestBuilder::from(new_multimodal)
                .set_sampling(sampling.clone())
                .set_tools(tools.clone())
                .set_tool_choice(ToolChoice::Auto);
            oom_recovery_pending = false;
            oom_recompression_done = true;
        }

        let base_request = request.clone();
        let mut accumulated_content = String::new();
        let mut current_tool_calls = Vec::new();
        let mut remaining_retries = retry_limit;
        let mut empty_retry_count: u32 = 0;
        let mut oom_recompression_needed = false;

        // Retry loop: wraps the streaming inference call so that
        // recoverable errors (OOMs, timeouts) trigger a retry with
        // the partial response prefilled.
        loop {
            let mut retry_request = base_request.clone();
            if !accumulated_content.is_empty() {
                retry_request = retry_request.add_message(
                    TextMessageRole::Assistant,
                    accumulated_content.clone(),
                );
                logger
                    .log(&format!(
                        "Retrying inference with {} chars of prefill content",
                        accumulated_content.len()
                    ))
                    .await;
            }

            let mut model_stream = match model.stream_chat_request(retry_request.clone()).await {
                Ok(s) => s,
                Err(e) => {
                    let err_str = format!("{:?}", e);
                    logger
                        .log(&format!(
                            "Inference error: {} (retries left: {})",
                            err_str, remaining_retries
                        ))
                        .await;
                    if remaining_retries > 0 {
                        // On OOM-like errors, flag the outer loop to
                        // re-compress with more aggressive checkpoint limits.
                        if config.enable_oom_recovery
                            && looks_like_oom(&err_str)
                            && accumulated_content.is_empty()
                            && !oom_recompression_done
                        {
                            oom_recompression_needed = true;
                        }
                        let attempt = retry_limit - remaining_retries;
                        let delay = retry_delay * 2u32.pow(attempt as u32);
                        remaining_retries -= 1;
                        tokio::time::sleep(delay).await;
                        continue;
                    }
                    // Final failure — remove any incomplete stream file
                    if let Some(ref stream_dir) = config.stream_output {
                        let stream_path =
                            PathBuf::from(stream_dir).join(format!("out-{}.txt", timestamp));
                        let _ = fs::remove_file(&stream_path).await;
                    }
                    // Drop the stream file handle so we don't leave a
                    // dangling fd; the file is already cleaned above.
                    drop(stream_file.take());
                    return Err(e.into());
                }
            };

            // Consume stream chunks. Track whether we received a
            // finish_reason so we can distinguish a completed response
            // from a prematurely-closed channel.
            let mut got_finish = false;
            let mut stream_error: Option<String> = None;

            while let Some(chunk) = model_stream.next().await {
                match chunk {
                    Response::Chunk(c) => {
                        if let Some(choice) = c.choices.first() {
                            if choice.finish_reason.is_some() {
                                got_finish = true;
                            }
                            if let Some(ref content) = choice.delta.content {
                                accumulated_content.push_str(content);
                                if let Some(ref mut f) = stream_file {
                                    if let Err(e) = f.write_all(content.as_bytes()).await {
                                        logger
                                            .log(&format!(
                                                "Stream write error: {:?}",
                                                e
                                            ))
                                            .await;
                                    }
                                    let _ = f.flush().await;
                                }
                            }
                            if let Some(ref tcs) = choice.delta.tool_calls {
                                current_tool_calls.extend(tcs.clone());
                            }
                        }
                    }
                    Response::ModelError(msg, _) => {
                        stream_error =
                            Some(format!("Model error during streaming: {}", msg));
                        break;
                    }
                    Response::InternalError(e) => {
                        stream_error =
                            Some(format!("Internal error during streaming: {}", e));
                        break;
                    }
                    // Done, CompletionDone, etc. — stream is ending normally
                    _ => {}
                }
            }

            // Check for errors or premature termination and retry if possible.
            if let Some(ref err_msg) = stream_error {
                logger
                    .log(&format!(
                        "{} (retries left: {})",
                        err_msg, remaining_retries
                    ))
                    .await;
                if remaining_retries > 0 {
                    let attempt = retry_limit - remaining_retries;
                    let delay = retry_delay * 2u32.pow(attempt as u32);
                    remaining_retries -= 1;
                    tokio::time::sleep(delay).await;
                    continue;
                }
                // Final failure — remove incomplete stream file and bail
                if let Some(ref stream_dir) = config.stream_output {
                    let stream_path =
                        PathBuf::from(stream_dir).join(format!("out-{}.txt", timestamp));
                    let _ = fs::remove_file(&stream_path).await;
                }
                drop(stream_file.take());
                // If OOM was the likely cause and we haven't tried
                // aggressive compression yet, don't give up — recompress
                // and retry the outer turn loop.
                if oom_recompression_needed {
                    oom_recovery_pending = true;
                    break; // exit inner retry loop, trigger recompression in outer loop
                }
                return Err(anyhow!("{}", err_msg));
            }

            if !got_finish {
                // Stream ended without receiving a finish_reason — the
                // channel was likely closed mid-generation. Retry with
                // the partial content as a prefill so the model can
                // continue from where it left off.
                if !accumulated_content.is_empty() {
                    logger
                        .log(&format!(
                            "Stream ended prematurely with {} chars (retries left: {})",
                            accumulated_content.len(),
                            remaining_retries
                        ))
                        .await;
                    if remaining_retries > 0 {
                        let attempt = retry_limit - remaining_retries;
                        let delay = retry_delay * 2u32.pow(attempt as u32);
                        remaining_retries -= 1;
                        tokio::time::sleep(delay).await;
                        continue;
                    }
                    // All retries exhausted — write whatever partial content
                    // we have as best-effort fallback.
                    logger
                        .log("All retries exhausted on incomplete stream; writing partial output")
                        .await;
                } else {
                    // Empty response — the model streamed nothing at all.
                    // Common for Qwen3.5/Gemma models that emit empty
                    // chunks before real tokens, or concurrent mistralrs
                    // streams that return empty for non-active inferers.
                    // Retry immediately — this is not an error condition.
                    empty_retry_count += 1;
                    logger
                        .log(&format!(
                            "Empty response (attempt {}); retrying immediately",
                            empty_retry_count,
                        ))
                        .await;
                    continue;
                }
            }

            // Streaming completed cleanly for this attempt
            break;
        }

        // Only write the output file when we have a complete, non-empty
        // response. On error we cleaned up and returned above, so by the
        // time we reach this point the response is good.
        //
        // When consume_tool_calls is false, tool call JSON is appended
        // so downstream agents (e.g. distillation harness) can see the
        // raw structured output. Tool call deltas are deduplicated by ID
        // (the last occurrence per ID carries the completed state).
        let mut output_content = accumulated_content.clone();
        if !config.consume_tool_calls && !current_tool_calls.is_empty() {
            let mut seen_ids: std::collections::HashSet<String> =
                std::collections::HashSet::new();
            let mut deduped: Vec<&ToolCallResponse> = Vec::new();
            // Iterate in reverse so the first-seen (last-occurrence) wins
            for tc in current_tool_calls.iter().rev() {
                if seen_ids.insert(tc.id.clone()) {
                    deduped.push(tc);
                }
            }
            deduped.reverse(); // restore original order
            if !output_content.is_empty() {
                output_content.push_str("\n\n");
            }
            for tc in &deduped {
                if let Ok(json) = serde_json::to_string(tc) {
                    output_content.push_str(&json);
                    output_content.push('\n');
                }
            }
        }
        if !output_content.is_empty() {
            if let Some(ref output_path) = output_file_path {
                logger
                    .log(&format!("Turn complete, writing to: {:?}", output_path))
                    .await;
                if let Err(e) = fs::write(output_path, &output_content).await {
                    logger
                        .log(&format!("Failed to write output file: {:?}", e))
                        .await;
                }
            }
        } else if output_file_path.is_some() {
            logger
                .log("Inference produced empty output; no output file written")
                .await;
        }

        if current_tool_calls.is_empty() {
            break;
        }

        logger
            .log(&format!(
                "Executing {} tool calls",
                current_tool_calls.len()
            ))
            .await;
        request = request.add_message_with_tool_call(
            TextMessageRole::Tool,
            &accumulated_content,
            current_tool_calls.clone(),
        );
        for (tool_idx, tc) in current_tool_calls.iter().enumerate() {
            let result = match tc.function.name.as_str() {
                "execute_command" => {
                    let args: serde_json::Value = serde_json::from_str(&tc.function.arguments)?;
                    let cmd = args["command"].as_str().unwrap_or_default();
                    let args_vec: Vec<String> = args["args"]
                        .as_array()
                        .unwrap_or(&vec![])
                        .iter()
                        .map(|v| v.as_str().unwrap_or_default().to_string())
                        .collect();
                    match tokio::process::Command::new(cmd)
                        .args(args_vec)
                        .output()
                        .await
                    {
                        Ok(output) => format!(
                            "Stdout: {}\nStderr: {}",
                            String::from_utf8_lossy(&output.stdout),
                            String::from_utf8_lossy(&output.stderr)
                        ),
                        Err(e) => format!("Error executing command: {}", e),
                    }
                }
                "spawn_new_agent" => {
                    let args: serde_json::Value = serde_json::from_str(&tc.function.arguments)?;
                    let name = args["name"].as_str().unwrap_or_default().to_string();
                    let config = AgentConfig {
                        inputs: args["inputs"]
                            .as_array()
                            .unwrap_or(&vec![])
                            .iter()
                            .map(|v| v.as_str().unwrap_or_default().to_string())
                            .collect(),
                        output: args["output"].as_str().map(|s| vec![s.to_string()]).unwrap_or_default(),
                        stream_output: args["stream_output"].as_str().map(|s| s.to_string()),
                        tool_output: args["tool_output"].as_str().map(|s| s.to_string()),
                        system: args["system"]
                            .as_array()
                            .unwrap_or(&vec![])
                            .iter()
                            .map(|v| v.as_str().unwrap_or_default().to_string())
                            .collect(),
                        model: args["model"].as_str().unwrap_or("primary").to_string(),
                        history_limit: args["history_limit"].as_u64().map(|u| u as usize),
                        realtime_audio: args["realtime_audio"].as_bool().unwrap_or(false),
                        allowed_extensions: vec![],
                        prompt: args["prompt"].as_str().map(|s| s.to_string()),
                        sampling: Default::default(),
                        compression: Default::default(),
                        context_checkpoint_limit: args["context_checkpoint_limit"]
                            .as_u64()
                            .map(|u| u as usize),
                        excluded_from_summary: args["excluded_from_summary"]
                            .as_array()
                            .unwrap_or(&vec![])
                            .iter()
                            .map(|v| v.as_str().unwrap_or_default().to_string())
                            .collect(),
                        tools_enabled: args["tools_enabled"].as_bool().unwrap_or(true),
                        consume_tool_calls: args["consume_tool_calls"].as_bool().unwrap_or(false),
                        enable_thinking: args["enable_thinking"].as_bool().unwrap_or(false),
                        inference_retries: 3,
                        enable_oom_recovery: true,
                        inference_retry_delay_ms: 500,
                    };
                    send_ipc_command(Command::SpawnAgent { name, config })
                        .await
                        .unwrap_or_else(|e| e.to_string())
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
                "read_file" => {
                    let args: serde_json::Value = serde_json::from_str(&tc.function.arguments)?;
                    let empty_vec = vec![];
                    let files = args["files"].as_array().unwrap_or(&empty_vec);
                    let mut out = String::new();
                    for f in files {
                        let p = f.as_str().unwrap_or_default();
                        match fs::read_to_string(p).await {
                            Ok(c) => {
                                out.push_str(&format!("--- {} ---\n{}\n", p, c));
                            }
                            Err(e) => {
                                out.push_str(&format!("--- {} ---\nError: {}\n", p, e));
                            }
                        }
                    }
                    if out.is_empty() {
                        "No files requested.".into()
                    } else {
                        out
                    }
                }
                "list_directory" => {
                    let args: serde_json::Value = serde_json::from_str(&tc.function.arguments)?;
                    let p = args["path"].as_str().unwrap_or(".");
                    let mut out = String::new();
                    match fs::read_dir(p).await {
                        Ok(mut entries) => {
                            while let Some(entry) = entries.next_entry().await? {
                                let meta = entry.metadata().await?;
                                let file_type = if meta.is_dir() { "dir" } else { "file" };
                                out.push_str(&format!(
                                    "[{}] {} ({} bytes)\n",
                                    file_type,
                                    entry.path().display(),
                                    meta.len()
                                ));
                            }
                        }
                        Err(e) => {
                            out.push_str(&format!("Error reading directory: {}", e));
                        }
                    }
                    if out.is_empty() {
                        "Directory is empty or path not found.".into()
                    } else {
                        out
                    }
                }
                "list_skills" => {
                    let args: serde_json::Value = serde_json::from_str(&tc.function.arguments)?;
                    let mut search_roots: Vec<String> = args["search_paths"]
                        .as_array()
                        .map(|arr| arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
                        .unwrap_or_default();
                    if search_roots.is_empty() {
                        if let Ok(home) = std::env::var("HOME") {
                            search_roots.push(format!("{}/.config", home));
                        }
                        if let Ok(cwd) = std::env::current_dir() {
                            search_roots.push(cwd.to_string_lossy().to_string());
                        }
                    }
                    let mut skills = Vec::new();
                    for root in &search_roots {
                        let root_path = PathBuf::from(root);
                        if !root_path.exists() {
                            continue;
                        }
                        let mut stack = vec![root_path];
                        while let Some(dir) = stack.pop() {
                            let skill_md = dir.join("SKILL.md");
                            if skill_md.exists() && skill_md.is_file() {
                                let name = dir.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
                                let mut description = String::new();
                                if let Ok(content) = tokio::fs::read_to_string(&skill_md).await {
                                    if content.starts_with("---") {
                                        if let Some(end) = content[3..].find("---") {
                                            let frontmatter = &content[3..end+3];
                                            for line in frontmatter.lines() {
                                                if line.starts_with("description:") {
                                                    description = line["description:".len()..].trim().to_string();
                                                }
                                            }
                                        }
                                    }
                                    if description.is_empty() && content.len() > 100 {
                                        description = content.lines().skip(1).find(|l| !l.trim().is_empty() && !l.starts_with("---")).unwrap_or("").to_string();
                                    }
                                }
                                skills.push(format!("- {} ({}): {}", name, dir.display(), description));
                                continue; // Don't recurse into skill directories
                            }
                            if let Ok(mut entries) = tokio::fs::read_dir(&dir).await {
                                while let Ok(Some(entry)) = entries.next_entry().await {
                                    let path = entry.path();
                                    if path.is_dir() {
                                        stack.push(path);
                                    }
                                }
                            }
                        }
                    }
                    if skills.is_empty() {
                        "No skills found. Skills are directories containing a SKILL.md file.".into()
                    } else {
                        format!("Found {} skills:\n{}", skills.len(), skills.join("\n"))
                    }
                }
                "load_skill" => {
                    let args: serde_json::Value = serde_json::from_str(&tc.function.arguments)?;
                    let skill_path = args["skill_path"].as_str().unwrap_or_default();
                    let system_dir = args["system_dir"].as_str()
                        .map(|s| s.to_string())
                        .or_else(|| config.system.first().cloned())
                        .unwrap_or_else(|| ".".to_string());
                    if skill_path.is_empty() {
                        "Error: skill_path is required".into()
                    } else {
                        let src = PathBuf::from(skill_path);
                        let skill_md = src.join("SKILL.md");
                        if !skill_md.exists() {
                            format!("Error: No SKILL.md found at {}", skill_md.display())
                        } else {
                            let dest = PathBuf::from(&system_dir).join(src.file_name().unwrap_or_default());
                            match copy_dir(&src, &dest).await {
                                Ok(_) => format!("Skill loaded from {} into {}", src.display(), dest.display()),
                                Err(e) => format!("Error copying skill: {}", e),
                            }
                        }
                    }
                }
                _ => format!("Unknown tool: {}", tc.function.name),
            };

            // Persist tool result to the dedicated tool_output directory if configured.
            // When tool_output is not explicitly set, default to a "tool_output"
            // subdirectory within the first output directory so tool results don't
            // pollute the main output (which would confuse downstream agents).
            let tool_dest_dir = config.tool_output.clone().unwrap_or_else(|| {
                config
                    .output
                    .first()
                    .map(|o| format!("{}/tool_output", o))
                    .unwrap_or_else(|| "/tmp/agentgraph_tool_output".to_string())
            });
            if let Err(e) = fs::create_dir_all(&tool_dest_dir).await {
                logger
                    .log(&format!("Failed to create tool output dir: {:?}", e))
                    .await;
            }
            let tool_output_path = PathBuf::from(tool_dest_dir).join(format!(
                "tool-{}-{}-{}.txt",
                tc.function.name, tool_idx, timestamp
            ));
            if let Err(e) = fs::write(&tool_output_path, &result).await {
                logger
                    .log(&format!("Failed to write tool output: {:?}", e))
                    .await;
            }

            request = request.add_tool_message(result, tc.id.clone());
        }
    }

    logger.log("Inference turn complete").await;
    Ok(())
}

async fn send_ipc_command(cmd: Command) -> Result<String> {
    let socket_path = find_leader_socket()
        .await
        .ok_or_else(|| anyhow!("Leader not found"))?;
    let mut stream = UnixStream::connect(socket_path).await?;
    let payload = serde_json::to_vec(&cmd)?;
    stream.write_all(&payload).await?;
    stream.flush().await?;
    let mut resp = String::new();
    let _ = stream.read_to_string(&mut resp).await;
    Ok(resp)
}

async fn copy_dir(src: &Path, dest: &Path) -> Result<()> {
    fs::create_dir_all(dest)
        .await
        .context(format!("Failed to create directory: {}", dest.display()))?;
    let mut entries = fs::read_dir(src)
        .await
        .context(format!("Failed to read directory: {}", src.display()))?;
    while let Some(entry) = entries.next_entry().await? {
        let src_path = entry.path();
        let dest_path = dest.join(entry.file_name());
        if src_path.is_dir() {
            Box::pin(copy_dir(&src_path, &dest_path)).await?;
        } else {
            fs::copy(&src_path, &dest_path)
                .await
                .context(format!(
                    "Failed to copy {} to {}",
                    src_path.display(),
                    dest_path.display()
                ))?;
        }
    }
    Ok(())
}
