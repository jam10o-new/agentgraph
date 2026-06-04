use crate::audio::AudioListener;
use crate::config::{AgentConfig, Config};
use crate::context::{CompressionManager, HistoryTurn, HistoryTurnRole, extract_frontmatter};
use crate::ipc::Command;
use crate::leader::ModelAccess;
use std::collections::HashMap;
use std::sync::LazyLock;
use std::sync::Mutex as StdMutex;
use crate::utils::AgentLogger;
use crate::utils::find_leader_socket;
use anyhow::{Context, Result, anyhow};
use mistralrs::{
    AudioInput, Constraint, Function, Model, MultimodalMessages, RequestBuilder, Response,
    SamplingParams, TextMessageRole, Tool, ToolCallResponse, ToolChoice, ToolType, VideoInput,
};
use notify::{Event, RecursiveMode, Watcher};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::fs;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;
use tokio::sync::{Mutex, mpsc};
use xxhash_rust::xxh3::xxh3_64;

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
    pub output_forwarder: Arc<Mutex<Option<mpsc::UnboundedSender<String>>>>,
    pub logger: AgentLogger,
    pub model_access: ModelAccess,
}

impl Agent {
    pub fn new(
        name: String,
        config: AgentConfig,
        global_config: Config,
        model: Arc<Model>,
        sampling: SamplingParams,
        model_access: ModelAccess,
    ) -> Self {
        Self {
            name: name.clone(),
            config,
            global_config,
            model,
            sampling,
            volatile_context: Arc::new(Mutex::new(Vec::new())),
            output_forwarder: Arc::new(Mutex::new(None)),
            logger: AgentLogger::new(&name),
            model_access,
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
            let cp = fs::canonicalize(&p).await.context(format!(
                "Failed to canonicalize input path: {}",
                p.display()
            ))?;
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
                    let forwarder = self.output_forwarder.clone();
                    let logger = AgentLogger::new(&name);
                    let model_access = self.model_access.clone();
                    let done_tx = inference_done_tx.clone();

                    inference_in_progress = true;
                    retrigger_pending = false;
                    tokio::spawn(async move {
                        let result = run_inference(agent_name, model, config, global_config, sampling, volatile_context, logger, model_access).await;
                        match result {
                            Ok(content) => {
                                if let Some(tx) = forwarder.lock().await.take() {
                                    let _ = tx.send(content);
                                }
                            }
                            Err(e) => {
                                eprintln!("Inference error for agent {}: {:?}", log_name, e);
                            }
                        }
                        let _ = done_tx.send(()).await;
                    });
                }
                _ = inference_done_rx.recv() => {
                    inference_in_progress = false;
                    // Yield so other agents get a chance to submit
                    // requests to the engine queue before we retrigger.
                    tokio::task::yield_now().await;
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
                        let forwarder = self.output_forwarder.clone();
                        let logger = AgentLogger::new(&name);
                        let done_tx = inference_done_tx.clone();
                        let ma = self.model_access.clone();

                        inference_in_progress = true;
                        tokio::spawn(async move {
                            let result = run_inference(agent_name, model, config, global_config, sampling, volatile_context, logger, ma).await;
                            match result {
                                Ok(content) => {
                                    if let Some(tx) = forwarder.lock().await.take() {
                                        let _ = tx.send(content);
                                    }
                                }
                                Err(e) => {
                                    eprintln!("Inference error for agent {}: {:?}", log_name, e);
                                }
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

/// Parsed output schema from a `.schema-{format}.{ext}[@{constraint}]`
/// file in the system directory. The `{ext}` is the output file extension.
/// The optional `@{constraint}` suffix overrides the constraint type:
///
/// | Constraint | Meaning                |
/// |------------|------------------------|
/// | `json`     | `Constraint::JsonSchema` (default for .json) |
/// | `lark`     | `Constraint::Lark`     |
/// | `regex`    | `Constraint::Regex`    |
/// | `llg`      | `Constraint::Llguidance` |
///
/// Examples:
/// - `.schema-rating.json` → JsonSchema, output `rating_N.json`
/// - `.schema-log.txt@regex` → Regex, output `log_N.txt`
/// - `.schema-story.txt@lark` → Lark grammar, output `story_N.txt`
#[derive(Clone)]
struct OutputSchema {
    format_str: String,
    extension: String,
    constraint: Constraint,
}

/// Intermediate parse result before reading the file content.
struct ParsedSchemaName {
    format_str: String,
    extension: String,
    constraint_hint: Option<String>,
}

impl ParsedSchemaName {
    fn try_parse(filename: &str) -> Option<Self> {
        let name = filename.strip_prefix(".schema-")?;
        // Split on the LAST dot to get the extension (and optional @constraint)
        let dot_pos = name.rfind('.')?;
        let format_and_ext = &name[..dot_pos];
        let ext_and_hint = &name[dot_pos + 1..];

        let (extension, constraint_hint) = if let Some(at_pos) = ext_and_hint.find('@') {
            (
                ext_and_hint[..at_pos].to_string(),
                Some(ext_and_hint[at_pos + 1..].to_string()),
            )
        } else {
            (ext_and_hint.to_string(), None)
        };

        if format_and_ext.is_empty() || extension.is_empty() {
            return None;
        }
        Some(ParsedSchemaName {
            format_str: format_and_ext.to_string(),
            extension,
            constraint_hint,
        })
    }
}

impl OutputSchema {
    /// Build the output filename from the format string, replacing
    /// `{timestamp}` with the epoch millis and `{turn}` with the
    /// auto-incremented turn counter (based on existing files).
    fn build_filename(&self, output_dir: &str, timestamp: u128) -> String {
        let turn = count_matching_files(output_dir, &self.format_str, &self.extension);
        let name = self
            .format_str
            .replace("{timestamp}", &timestamp.to_string())
            .replace("{turn}", &turn.to_string());
        format!("{}.{}", name, self.extension)
    }
}

/// Count files in a directory whose names match the pattern after
/// variable substitution (ignoring `{timestamp}` and `{turn}` — we
/// count by prefix/suffix to determine the next turn number).
fn count_matching_files(dir: &str, format_str: &str, extension: &str) -> usize {
    let dir_path = std::path::Path::new(dir);
    if !dir_path.exists() {
        return 0;
    }
    // Build a prefix to match: everything before the first variable
    let prefix = format_str.split(&['{', '}'][..]).next().unwrap_or("");
    let suffix = format!(".{}", extension);
    let mut count = 0;
    if let Ok(entries) = std::fs::read_dir(dir_path) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if name_str.starts_with(prefix) && name_str.ends_with(&suffix) {
                count += 1;
            }
        }
    }
    count
}

async fn run_inference(
    _name: String,
    model: Arc<Model>,
    config: AgentConfig,
    _global_config: Config,
    sampling: SamplingParams,
    volatile_context: Arc<Mutex<Vec<(TextMessageRole, String)>>>,
    logger: AgentLogger,
    model_access: ModelAccess,
) -> Result<String> {
    logger.log("Starting inference turn").await;

    // Mark model as recently accessed so the idle eviction timer knows it's in use.
    model_access.touch(&config.model).await;

    // 1. Collate System Prompts
    let mut system_content = String::new();
    // Schema-driven output: if a .schema-{format}.{ext} file exists in the
    // system directory, use llguidance constrained decoding and a custom
    // output filename pattern instead of the default out-{timestamp}.txt.
    let mut output_schema: Option<OutputSchema> = None;
    for sys_dir in &config.system {
        if let Ok(mut entries) = fs::read_dir(sys_dir).await {
            let mut files = Vec::new();
            while let Some(entry) = entries.next_entry().await? {
                let file_name = entry.file_name();
                let name_str = file_name.to_string_lossy();
                // Check for schema files before the hidden-file skip
                if let Some(parsed) = ParsedSchemaName::try_parse(&name_str) {
                    if let Ok(content) = fs::read_to_string(entry.path()).await {
                        let constraint_type =
                            parsed.constraint_hint.as_deref().unwrap_or_else(|| {
                                // Default: JsonSchema for .json, nothing otherwise
                                if parsed.extension == "json" {
                                    "json"
                                } else {
                                    ""
                                }
                            });
                        let constraint = match constraint_type {
                            "json" => serde_json::from_str::<serde_json::Value>(&content)
                                .map(Constraint::JsonSchema),
                            "lark" => Ok(Constraint::Lark(content)),
                            "regex" => Ok(Constraint::Regex(content)),
                            "llg" => serde_json::from_str(&content).map(Constraint::Llguidance),
                            _ => continue, // unknown or no default, skip
                        };
                        if let Ok(c) = constraint {
                            logger
                                .log(&format!(
                                    "Loaded output schema: .schema-{}.{}",
                                    parsed.format_str, parsed.extension
                                ))
                                .await;
                            output_schema = Some(OutputSchema {
                                format_str: parsed.format_str,
                                extension: parsed.extension,
                                constraint: c,
                            });
                        }
                    }
                    continue; // don't load as system prompt
                }
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

    // ── Tool discovery (dynamic, per-binary --describe / --help) ──────────
    // Run early so guidance text is injected into the system prompt before
    // it gets consumed by the history builder.
    let mut tool_registry: Option<ToolRegistry> = None;
    let mut tools = vec![];
    if !config.tools.is_empty() && output_schema.is_none() {
        match discover_tools(&config.tools, &logger).await {
            Ok(registry) => {
                if !registry.guidance.is_empty() {
                    system_content.push_str("\n\n== Tool Usage Guidance ==\n");
                    system_content.push_str(&registry.guidance);
                }
                tools = registry.tools.clone();
                tool_registry = Some(registry);
            }
            Err(e) => {
                logger.log(&format!("Tool discovery error: {e}")).await;
            }
        }
    } else if output_schema.is_some() && !config.tools.is_empty() {
        logger.log("Tools disabled — schema constraint is active (tools conflict with llguidance)").await;
    }

    let mut combined_history = Vec::new();
    if !system_content.is_empty() {
        if let Some((n, d, body)) = extract_frontmatter(&system_content) {
            combined_history.push(HistoryTurn {
                role: HistoryTurnRole::Skill(n, d),
                content: body,
                turn_index: 0,
                excluded_from_compression: true,
                file_key: "_system_".into(),
                file_path: String::new(),
                content_hash: String::new(),
            });
        } else {
            combined_history.push(HistoryTurn {
                role: HistoryTurnRole::System,
                content: system_content,
                turn_index: 0,
                excluded_from_compression: true,
                file_key: "_system_".into(),
                file_path: String::new(),
                content_hash: String::new(),
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
                    let metadata = fs::metadata(&p).await.context(format!(
                        "Failed to read metadata for input file: {}",
                        p_display
                    ))?;
                    let meta_str = format_file_metadata(&p, &metadata);
                    all_files.push(FileEntry {
                        path: p,
                        created: metadata.created().map_err(|e| {
                            anyhow!("Failed to get creation time for {}: {e}", p_display)
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
                    let metadata = fs::metadata(&p).await.context(format!(
                        "Failed to read metadata for output file: {}",
                        p_display
                    ))?;
                    all_files.push(FileEntry {
                        path: p,
                        created: metadata.created().map_err(|e| {
                            anyhow!("Failed to get creation time for {}: {e}", p_display)
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
    let effective_tool_dir = config
        .tool_output
        .clone()
        .or_else(|| config.output.first().map(|o| format!("{}/tool_output", o)));
    if let Some(ref tool_dir) = effective_tool_dir {
        if let Ok(mut entries) = fs::read_dir(tool_dir).await {
            while let Some(entry) = entries.next_entry().await? {
                let p = entry.path();
                if p.is_file() {
                    let p_display = p.display().to_string();
                    let metadata = fs::metadata(&p).await.context(format!(
                        "Failed to read metadata for tool output file: {}",
                        p_display
                    ))?;
                    all_files.push(FileEntry {
                        path: p,
                        created: metadata.created().map_err(|e| {
                            anyhow!("Failed to get creation time for {}: {e}", p_display)
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
            let file_key = {
                let path_str = entry.path.to_string_lossy();
                format!("{:016x}", xxh3_64(path_str.as_bytes()))
            };
            let content_hash = { format!("{:016x}", xxh3_64(final_content.as_bytes())) };
            combined_history.push(HistoryTurn {
                role: entry.role.clone(),
                content: final_content,
                turn_index: turn_idx,
                excluded_from_compression: entry.excluded,
                file_key,
                file_path: entry.path.to_string_lossy().to_string(),
                content_hash,
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
    let checkpoint_base = config
        .output
        .first()
        .and_then(|o| PathBuf::from(o).parent().map(|p| p.to_path_buf()))
        .or_else(|| {
            config
                .stream_output
                .as_ref()
                .and_then(|s| PathBuf::from(s).parent().map(|p| p.to_path_buf()))
        })
        .unwrap_or_else(|| PathBuf::from("."));
    // Save for potential OOM recovery rebuild
    let oom_db_path = config
        .compression_db_path
        .as_ref()
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            checkpoint_base
                .join(".agent_context")
                .join("compression.db")
        });
    let oom_compression_config = config.compression.clone();
    let comp_mgr = CompressionManager::new(&oom_db_path, &config.compression)?;
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
    let mut audios: Vec<AudioInput> = Vec::new();
    let mut videos: Vec<VideoInput> = Vec::new();
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
                        "mp4" | "avi" | "mov" | "mkv" | "webm" | "m4v" | "gif" => {
                            logger
                                .log(&format!(
                                    "Decoding video {} via ffmpeg",
                                    p.display()
                                ))
                                .await;
                            match decode_video(&p).await {
                                Ok(video) => videos.push(video),
                                Err(e) => logger
                                    .log(&format!(
                                        "Failed to decode video {}: {:?}",
                                        p.display(),
                                        e
                                    ))
                                    .await,
                            }
                        }
                        "wav" => {
                            match AudioInput::read_wav(&p.to_string_lossy()) {
                                Ok(audio) => audios.push(audio),
                                Err(e) => logger
                                    .log(&format!(
                                        "Failed to decode WAV {}: {:?}",
                                        p.display(),
                                        e
                                    ))
                                    .await,
                            }
                        }
                        "mp3" | "ogg" | "flac" | "m4a" | "aac" | "opus" => {
                            let decoded = match std::fs::read(&p) {
                                Ok(bytes) => AudioInput::from_bytes(&bytes),
                                Err(e) => Err(anyhow!("read: {}", e)),
                            };
                            // For .ogg and .opus, the bundled symphonia may lack the
                            // opus feature.  Fall back to ffmpeg → WAV conversion.
                            if decoded.is_ok()
                                || ext == "mp3"
                                || ext == "flac"
                                || ext == "m4a"
                                || ext == "aac"
                            {
                                match decoded {
                                    Ok(audio) => audios.push(audio),
                                    Err(e) => logger
                                        .log(&format!(
                                            "Failed to decode audio {}: {:?}",
                                            p.display(), e
                                        ))
                                        .await,
                                }
                            } else {
                                logger
                                    .log(&format!(
                                        "symphonia decode failed for {}; falling back to ffmpeg",
                                        p.display()
                                    ))
                                    .await;
                                match convert_audio_to_wav(&p).await {
                                    Ok(wav_path) => {
                                        match AudioInput::read_wav(&wav_path.to_string_lossy()) {
                                            Ok(audio) => audios.push(audio),
                                            Err(e) => logger
                                                .log(&format!(
                                                    "Failed to decode ffmpeg-converted WAV {}: {:?}",
                                                    p.display(), e
                                                ))
                                                .await,
                                        }
                                    }
                                    Err(e) => logger
                                        .log(&format!(
                                            "ffmpeg conversion failed for {}: {:?}",
                                            p.display(), e
                                        ))
                                        .await,
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    let effective_user_text =
        if latest_user_input.is_empty()
            && (!images.is_empty() || !audios.is_empty() || !videos.is_empty())
        {
            fallback_text
        } else {
            latest_user_input
        };
    // Save for potential OOM recovery recompression
    let oom_latest_input = effective_user_text.clone();

    if !images.is_empty() || !audios.is_empty() || !videos.is_empty() {
        if !images.is_empty() {
            logger
                .log(&format!("Adding {} images to request", images.len()))
                .await;
        }
        if !audios.is_empty() {
            logger
                .log(&format!("Adding {} audio inputs to request", audios.len()))
                .await;
        }
        if !videos.is_empty() {
            logger
                .log(&format!("Adding {} video inputs to request", videos.len()))
                .await;
        }
        multimodal = multimodal.add_multimodal_message(
            TextMessageRole::User,
            &effective_user_text,
            images.clone(),
            audios.clone(),
            videos.clone(),
        );
    } else {
        multimodal = multimodal.add_message(TextMessageRole::User, &effective_user_text);
    }

    // When not consuming tool calls (distillation mode), write the tool
    // definitions to a well-known hidden file in the first system directory
    // so downstream harnesses know what tools the model had access to.
    if !config.consume_tool_calls && !tools.is_empty() {
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
        if let Some(ref schema) = output_schema {
            PathBuf::from(o).join(schema.build_filename(o, timestamp))
        } else {
            PathBuf::from(o).join(format!("out-{}.txt", timestamp))
        }
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
        Some(fs::File::create(&stream_path).await.context(format!(
            "Failed to create stream output file: {}",
            stream_path.display()
        ))?)
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
    if let Some(ref schema) = output_schema {
        request = request.set_constraint(schema.constraint.clone());
    }

    let retry_limit = config.inference_retries;
    let retry_delay = Duration::from_millis(config.inference_retry_delay_ms);

    // OOM recovery state: when the normal-strength context OOMs the GPU,
    // we re-compress with a halved context_checkpoint_limit and retry the
    // entire turn. This preserves more history than dropping everything.
    let mut oom_recompression_done = false;
    let mut oom_recovery_pending = false;
    let oom_aggressive_limit = config.context_checkpoint_limit.map(|l| (l / 2).max(1));

    let mut final_output = String::new();
    let mut tools_executed = false;
    let mut nudged = false; // prevent more than one empty-output nudge
    loop {
        // If the previous turn failed with OOM, recompress the original
        // uncompressed history with tighter limits before rebuilding the
        // request.
        if oom_recovery_pending {
            logger
                .log("Recompressing context with aggressive limits after OOM")
                .await;
            let oom_comp_mgr = CompressionManager::new(&oom_db_path, &oom_compression_config)?;
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
                    logger
                        .log(&format!("OOM recompression failed: {:?}", e))
                        .await;
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
            if !images.is_empty() || !audios.is_empty() || !videos.is_empty() {
                new_multimodal = new_multimodal.add_multimodal_message(
                    TextMessageRole::User,
                    &effective_user_text,
                    images.clone(),
                    audios.clone(),
                    videos.clone(),
                );
            } else {
                new_multimodal =
                    new_multimodal.add_message(TextMessageRole::User, &effective_user_text);
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
                retry_request = retry_request
                    .add_message(TextMessageRole::Assistant, accumulated_content.clone());
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
                                        logger.log(&format!("Stream write error: {:?}", e)).await;
                                    }
                                    let _ = f.flush().await;
                                }
                            }
                            if let Some(ref tcs) = choice.delta.tool_calls {
                                current_tool_calls.extend(tcs.clone());
                                // When tool calls are unconsumed, write them
                                // to the stream file so the caller can see
                                // what the model is doing in real time.
                                if !config.consume_tool_calls {
                                    if let Some(ref mut f) = stream_file {
                                        for tc in tcs {
                                            if let Ok(json) = serde_json::to_string(tc) {
                                                let _ = f.write_all(json.as_bytes()).await;
                                                let _ = f.write_all(b"\n").await;
                                            }
                                        }
                                        let _ = f.flush().await;
                                    }
                                }
                            }
                        }
                    }
                    Response::ModelError(msg, _) => {
                        stream_error = Some(format!("Model error during streaming: {}", msg));
                        break;
                    }
                    Response::InternalError(e) => {
                        stream_error = Some(format!("Internal error during streaming: {}", e));
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

        // Prepare output content.  When consume_tool_calls is false, tool
        // call JSON is appended so downstream agents (e.g. distillation
        // harness) can see the raw structured output.  Tool call deltas
        // are deduplicated by ID (the last occurrence per ID carries the
        // completed state).
        let mut output_content = accumulated_content.clone();
        if !config.consume_tool_calls && !current_tool_calls.is_empty() && output_schema.is_none() {
            let mut seen_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
            let mut deduped: Vec<&ToolCallResponse> = Vec::new();
            for tc in current_tool_calls.iter().rev() {
                if seen_ids.insert(tc.id.clone()) {
                    deduped.push(tc);
                }
            }
            deduped.reverse();
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

        final_output = output_content.clone();

        if current_tool_calls.is_empty() {
            // If the model went silent after executing tools, inject a
            // nudge and retry once instead of writing empty output.
            if output_content.is_empty() && tools_executed && !nudged {
                nudged = true;
                logger
                    .log("Empty output after tool execution; nudging model to respond")
                    .await;
                request = request.add_message(
                    TextMessageRole::System,
                    "Tool execution complete. Provide your response based on the tool results.",
                );
                continue;
            }
            // Turn is complete — write the output file so that downstream
            // watchers (wait_for_output, the streaming background task)
            // can detect completion.
            if let Some(ref output_path) = output_file_path {
                logger
                    .log(&format!("Turn complete, writing {} chars to: {:?}", output_content.len(), output_path))
                    .await;
                if let Err(e) = fs::write(output_path, &output_content).await {
                    logger
                        .log(&format!("Failed to write output file: {:?}", e))
                        .await;
                }
            }
            break;
        }

        logger
            .log(&format!(
                "Executing {} tool calls",
                current_tool_calls.len()
            ))
            .await;
        request = request.add_message_with_tool_call(
            TextMessageRole::Assistant,
            &accumulated_content,
            current_tool_calls.clone(),
        );
        for (tool_idx, tc) in current_tool_calls.iter().enumerate() {
            let result = if let Some(ref registry) = tool_registry {
                if let Some(binary_name) = registry.name_to_binary.get(&tc.function.name) {
                    match execute_tool_binary(binary_name, &tc.function.arguments).await {
                        Ok(output) => {
                            // load_into_context has special semantics: store
                            // output in volatile context for the next turn.
                            if tc.function.name == "load_into_context" {
                                volatile_context.lock().await.push((
                                    TextMessageRole::System,
                                    output,
                                ));
                                "Files loaded into context for next turn.".into()
                            } else {
                                output
                            }
                        }
                        Err(e) => {
                            format!("Error executing {}: {}", tc.function.name, e)
                        }
                    }
                } else {
                    format!("Unknown tool: {}", tc.function.name)
                }
            } else {
                "Tools are not enabled for this agent.".into()
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
        tools_executed = true;
    }

    logger.log("Inference turn complete").await;
    Ok(final_output)
}

// ── Tool plugin system ──────────────────────────────────────────────────

/// Cached tool schema + guidance, keyed by binary name.
struct CachedTool {
    tool: Tool,
    guidance: String,
}

/// Global tool cache — avoids spawning `--describe`/`--help` on every turn.
static TOOL_CACHE: LazyLock<StdMutex<HashMap<String, CachedTool>>> =
    LazyLock::new(|| StdMutex::new(HashMap::new()));

/// Built tool registry for a single inference run.
struct ToolRegistry {
    tools: Vec<Tool>,
    guidance: String,
    /// Maps function name (e.g. `read_file`) → binary name (e.g. `ag-tool-read`).
    name_to_binary: HashMap<String, String>,
}

/// Discover tools by running `--describe` and `--help` on each configured binary.
/// Results are cached globally.
async fn discover_tools(
    tool_names: &[String],
    logger: &AgentLogger,
) -> anyhow::Result<ToolRegistry> {
    let mut tools = Vec::new();
    let mut guidance_parts = Vec::new();
    let mut name_to_binary = HashMap::new();

    // Phase 1: load from cache (short lock)
    let uncached: Vec<String> = {
        let cache = TOOL_CACHE.lock().unwrap();
        for name in tool_names {
            if let Some(cached) = cache.get(name) {
                tools.push(cached.tool.clone());
                guidance_parts.push(cached.guidance.clone());
                let func_name = cached.tool.function.name.clone();
                name_to_binary.insert(func_name, name.clone());
            }
        }
        tool_names
            .iter()
            .filter(|n| !cache.contains_key(*n))
            .cloned()
            .collect()
    };

    // Phase 2: discover uncached tools (no lock held across await)
    for name in &uncached {
        let binary_path = match which::which(name) {
            Ok(p) => p,
            Err(_) => {
                logger
                    .log(&format!(
                        "Tool binary '{name}' not found on PATH — skipping"
                    ))
                    .await;
                continue;
            }
        };

        // --describe → JSON Function schema
        let describe = tokio::process::Command::new(&binary_path)
            .arg("--describe")
            .output()
            .await;
        let (func_name, tool) = match describe {
            Ok(out) if out.status.success() => {
                let raw = String::from_utf8_lossy(&out.stdout);
                match serde_json::from_str::<serde_json::Value>(&raw) {
                    Ok(schema) => {
                        let fn_name =
                            schema["name"].as_str().unwrap_or(name).to_string();
                        let tool = Tool {
                            tp: ToolType::Function,
                            function: Function {
                                name: fn_name.clone(),
                                description: schema["description"]
                                    .as_str()
                                    .map(String::from),
                                parameters: schema
                                    .get("parameters")
                                    .and_then(|p| p.as_object())
                                    .map(|obj| {
                                        obj.iter()
                                            .map(|(k, v)| (k.clone(), v.clone()))
                                            .collect::<HashMap<String, serde_json::Value>>()
                                    }),
                                strict: None,
                            },
                        };
                        (fn_name, tool)
                    }
                    Err(e) => {
                        logger
                            .log(&format!(
                                "Failed to parse --describe from '{name}': {e}"
                            ))
                            .await;
                        continue;
                    }
                }
            }
            _ => {
                logger
                    .log(&format!(
                        "Tool '{name}' --describe failed"
                    ))
                    .await;
                continue;
            }
        };

        // --help → LLM guidance
        let guidance = match tokio::process::Command::new(&binary_path)
            .arg("--help")
            .output()
            .await
        {
            Ok(out) if out.status.success() => {
                String::from_utf8_lossy(&out.stdout).into_owned()
            }
            _ => String::new(),
        };

        name_to_binary.insert(func_name.clone(), name.clone());
        tools.push(tool.clone());
        guidance_parts.push(guidance.clone());

        // Insert into global cache (short lock)
        TOOL_CACHE.lock().unwrap().insert(
            name.clone(),
            CachedTool {
                tool,
                guidance,
            },
        );
    }

    Ok(ToolRegistry {
        tools,
        guidance: guidance_parts.join("\n\n"),
        name_to_binary,
    })
}

/// Execute a tool by spawning its binary, writing arguments to stdin,
/// and capturing stdout.
async fn execute_tool_binary(binary_name: &str, arguments: &str) -> Result<String, anyhow::Error> {
    let path = which::which(binary_name)?;
    let mut child = tokio::process::Command::new(&path)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()?;

    if let Some(mut stdin) = child.stdin.take() {
        use tokio::io::AsyncWriteExt;
        stdin.write_all(arguments.as_bytes()).await?;
    }
    drop(child.stdin.take());

    let output = child.wait_with_output().await?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Ok(format!(
            "Error: tool '{binary_name}' exited with status {}.\nStderr: {stderr}",
            output.status
        ));
    }

    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
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
            fs::copy(&src_path, &dest_path).await.context(format!(
                "Failed to copy {} to {}",
                src_path.display(),
                dest_path.display()
            ))?;
        }
    }
    Ok(())
}

/// Decode a video file into a `VideoInput` by shelling out to FFmpeg.
/// Extracts 32 uniformly sampled frames.
async fn decode_video(path: &Path) -> Result<VideoInput> {
    // Probe FPS and total frames with ffprobe
    let probe = tokio::process::Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate,nb_frames",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
        ])
        .arg(path)
        .output()
        .await
        .ok();

    let fps: f64 = probe
        .as_ref()
        .and_then(|o| String::from_utf8(o.stdout.clone()).ok())
        .and_then(|s| {
            s.lines()
                .next()
                .and_then(|l| {
                    let mut parts = l.split('/');
                    let num: f64 = parts.next()?.parse().ok()?;
                    let den: f64 = parts.next()?.parse().ok()?;
                    if den == 0.0 { None } else { Some(num / den) }
                })
        })
        .unwrap_or(24.0);

    let total_frames: usize = probe
        .as_ref()
        .and_then(|o| String::from_utf8(o.stdout.clone()).ok())
        .and_then(|s| s.lines().nth(1)?.parse().ok())
        .unwrap_or(0);

    let num_frames = 32usize;
    let sampled_indices: Vec<usize> = sample_frame_indices(total_frames, num_frames);

    // Extract frames as PNG with ffmpeg
    let tmpdir = tempfile::tempdir()?;
    let tmp_path = tmpdir.path().join("frame_%04d.png");
    let select_expr = sampled_indices
        .iter()
        .map(|i| format!("eq(n,{})", i))
        .collect::<Vec<_>>()
        .join("+");
    let output = tokio::process::Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(path)
        .args(["-vf", &format!("select='{}'", select_expr), "-vsync", "vfr"])
        .arg(tmp_path.to_str().unwrap())
        .output()
        .await
        .context("ffmpeg frame extraction failed")?;

    if !output.status.success() {
        return Err(anyhow!(
            "ffmpeg failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    // Read extracted frames
    let mut frames: Vec<image::DynamicImage> = Vec::new();
    for i in 1..=(num_frames as u32) {
        let frame_path = tmpdir.path().join(format!("frame_{:04}.png", i));
        match image::open(&frame_path) {
            Ok(img) => frames.push(img),
            Err(_) => break,
        }
    }
    if frames.is_empty() {
        return Err(anyhow!("No frames extracted from video"));
    }

    Ok(VideoInput::from_frames(frames, fps, Some(sampled_indices)))
}

/// Uniformly sample `n` frame indices from `total` frames.
/// Mimics torch.arange(0, total, total/n).int()
fn sample_frame_indices(total: usize, n: usize) -> Vec<usize> {
    if total == 0 || n == 0 {
        return Vec::new();
    }
    let step = total as f64 / n as f64;
    (0..n)
        .map(|i| ((i as f64) * step) as usize)
        .take(total)
        .collect()
}

/// Convert any audio file to WAV via ffmpeg, returning the path to the
/// output file.  Used as a fallback for formats that the bundled symphonia
/// build doesn't support (e.g. Opus in OGG).
async fn convert_audio_to_wav(path: &Path) -> Result<PathBuf> {
    let tmpdir = tempfile::tempdir()?;
    let wav_path = tmpdir.path().join("output.wav");
    let output = tokio::process::Command::new("ffmpeg")
        .args([
            "-v", "error",
            "-i",
        ])
        .arg(path)
        .args([
            "-acodec", "pcm_f32le",
            "-ar", "16000",
            "-ac", "1",
            "-y",
        ])
        .arg(&wav_path)
        .output()
        .await
        .context("ffmpeg audio conversion failed")?;

    if !output.status.success() {
        return Err(anyhow!(
            "ffmpeg audio: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    // Keep the tempdir alive by leaking it — the file will be cleaned up
    // when the agent completes its turn.
    let wav = wav_path.clone();
    std::mem::forget(tmpdir);
    Ok(wav)
}
