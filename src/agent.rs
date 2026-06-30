#[cfg(feature = "audio")]
use crate::audio::AudioListener;
use crate::config::{AgentConfig, Config, SystemPromptMode};
use crate::context::{CompressionManager, HistoryTurn, HistoryTurnRole, extract_frontmatter};
use crate::inference_provider::{
    ChatMessage, ChatRequest, GuidanceConstraint, InferenceProvider, ProviderEvent, Role,
    SamplingConfig as ProviderSamplingConfig, ToolCall, ToolCallDelta, ToolChoice as ProviderToolChoice,
    ToolDef,
};
use crate::ipc::{Command, HeavyToolState};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::Mutex as StdMutex;
use crate::utils::AgentLogger;
use crate::utils::find_leader_socket;
use anyhow::{Context, Result, anyhow};
use notify::{Event, RecursiveMode, Watcher};
use std::path::{Path, PathBuf};
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
    pub volatile_context: Arc<Mutex<Vec<(Role, String)>>>,
    pub output_forwarder: Arc<Mutex<Option<mpsc::UnboundedSender<String>>>>,
    pub logger: AgentLogger,
    pub provider: Arc<dyn InferenceProvider>,
}

impl Agent {
    pub fn new(
        name: String,
        config: AgentConfig,
        global_config: Config,
        provider: Arc<dyn InferenceProvider>,
    ) -> Self {
        Self {
            name: name.clone(),
            config,
            global_config,
            volatile_context: Arc::new(Mutex::new(Vec::new())),
            output_forwarder: Arc::new(Mutex::new(None)),
            logger: AgentLogger::new(&name),
            provider,
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

        #[cfg(feature = "audio")]
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

                    let config = self.config.clone();
                    let global_config = self.global_config.clone();
                    let agent_name = name.clone();
                    let log_name = name.clone();
                    let volatile_context = self.volatile_context.clone();
                    let forwarder = self.output_forwarder.clone();
                    let logger = AgentLogger::new(&name);
                    let done_tx = inference_done_tx.clone();

                    inference_in_progress = true;
                    retrigger_pending = false;

                    let provider = self.provider.clone();
                    tokio::spawn(async move {
                        let result = run_inference_with_provider(
                            agent_name, provider, config, global_config, volatile_context, logger,
                        ).await;
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

                        let config = self.config.clone();
                        let global_config = self.global_config.clone();
                        let agent_name = name.clone();
                        let log_name = name.clone();
                        let volatile_context = self.volatile_context.clone();
                        let forwarder = self.output_forwarder.clone();
                        let logger = AgentLogger::new(&name);
                        let done_tx = inference_done_tx.clone();

                        inference_in_progress = true;

                        let provider = self.provider.clone();
                        tokio::spawn(async move {
                            let result = run_inference_with_provider(
                                agent_name, provider, config, global_config, volatile_context, logger,
                            ).await;
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
    /// For Tool-role entries, the tool_call_id parsed from the filename.
    tool_call_id: Option<String>,
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

/// Parsed output schema from a `.schema-{format}.{ext}[@{constraint}]`
/// file in the system directory. The `{ext}` is the output file extension.
/// The optional `@{constraint}` suffix overrides the constraint type:
///
/// | GuidanceConstraint | Meaning                |
/// |------------|------------------------|
/// | `json`     | `GuidanceConstraint::JsonSchema` (default for .json) |
/// | `lark`     | `GuidanceConstraint::Lark`     |
/// | `regex`    | `GuidanceConstraint::Regex`    |
/// | `llg`      | `GuidanceConstraint::Llguidance` |
///
/// Examples:
/// - `.schema-rating.json` → JsonSchema, output `rating_N.json`
/// - `.schema-log.txt@regex` → Regex, output `log_N.txt`
/// - `.schema-story.txt@lark` → Lark grammar, output `story_N.txt`
#[derive(Clone)]
struct OutputSchema {
    format_str: String,
    extension: String,
    constraint: GuidanceConstraint,
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


// ── Tool plugin system ──────────────────────────────────────────────────

/// Cached tool schema + guidance, keyed by binary name.
struct CachedTool {
    tool: ToolDef,
    guidance: String,
}

/// Global tool cache — avoids spawning `--describe`/`--help` on every turn.
static TOOL_CACHE: LazyLock<StdMutex<HashMap<String, CachedTool>>> =
    LazyLock::new(|| StdMutex::new(HashMap::new()));

/// Built tool registry for a single inference run.
struct ToolRegistry {
    tools: Vec<ToolDef>,
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
                let func_name = cached.tool.name.clone();
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
        let (func_name, tool): (String, ToolDef) = match describe {
            Ok(out) if out.status.success() => {
                let raw = String::from_utf8_lossy(&out.stdout);
                match serde_json::from_str::<serde_json::Value>(&raw) {
                    Ok(schema) => {
                        let fn_name =
                            schema["name"].as_str().unwrap_or(name).to_string();
                        let params = schema
                            .get("parameters")
                            .and_then(|p| p.as_object())
                            .map(|obj| {
                                obj.iter()
                                    .map(|(k, v)| (k.clone(), v.clone()))
                                    .collect::<serde_json::Map<String, serde_json::Value>>()
                            })
                            .unwrap_or_default();
                        let description = schema["description"]
                            .as_str()
                            .unwrap_or("")
                            .to_string();
                        let tool = ToolDef::new(
                            &fn_name,
                            description,
                            serde_json::Value::Object(params),
                        );
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

/// Save heavy tool state to disk for the heavy-tool-runner to pick up.
/// Returns the path to the saved state file.
fn save_heavy_tool_state(
    agent_name: &str,
    config: &AgentConfig,
    global_config: &Config,
    tool_binary: &str,
    tool_name: &str,
    tool_args: &str,
    tool_call_id: &str,
    accumulated_content: &str,
    timestamp: u128,
) -> anyhow::Result<String> {
    let resume_dir = home_dir().join(".agentgraph").join("resume").join(agent_name);
    std::fs::create_dir_all(&resume_dir)?;
    let path = resume_dir.join(format!("heavy_tool_{}.json", timestamp));
    let state = HeavyToolState {
        agent_name: agent_name.to_string(),
        agent_config: config.clone(),
        global_config: global_config.clone(),
        tool_binary: tool_binary.to_string(),
        tool_name: tool_name.to_string(),
        tool_args: tool_args.to_string(),
        tool_call_id: tool_call_id.to_string(),
        accumulated_content: accumulated_content.to_string(),
        timestamp,
    };
    let json = serde_json::to_string_pretty(&state)?;
    std::fs::write(&path, &json)?;
    Ok(path.to_string_lossy().to_string())
}

/// Write a signal file that tells the leader's watchdog to persist
/// sessions and hand off to the heavy-tool-runner.
fn write_heavy_tool_signal(agent_name: &str, state_path: &str) {
    let signal_dir = PathBuf::from("/tmp/agentgraph");
    let _ = std::fs::create_dir_all(&signal_dir);
    let signal_path = signal_dir.join(format!("heavy_tool_signal_{}.json", agent_name));
    let signal = serde_json::json!({
        "agent": agent_name,
        "state_path": state_path,
    });
    if let Ok(json) = serde_json::to_string(&signal) {
        let _ = std::fs::write(&signal_path, &json);
    }
}

fn home_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp"))
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

/// Uniformly sample `n` frame indices from `total` frames.
/// Mimics torch.arange(0, total, total/n).int()

/// Convert any audio file to WAV via ffmpeg, returning the path to the
/// output file.  Used as a fallback for formats that the bundled symphonia
/// build doesn't support (e.g. Opus in OGG).

// ── Provider-based inference ───────────────────────────────────────────

/// Run a single inference turn using an [`InferenceProvider`].
/// Builds provider-agnostic [`ChatRequest`] values and processes
/// [`ProviderEvent`] streams.
async fn run_inference_with_provider(
    name: String,
    provider: Arc<dyn InferenceProvider>,
    config: AgentConfig,
    global_config: Config,
    volatile_context: Arc<Mutex<Vec<(Role, String)>>>,
    logger: AgentLogger,
) -> Result<String> {
    logger.log("Starting inference turn (provider)").await;

    // Resolve system prompt mode: agent override > model config > Merged
    let system_mode = config
        .system_prompt_mode
        .clone()
        .or_else(|| {
            global_config
                .models
                .get(&config.model)
                .map(|m| m.system_prompt_mode.clone())
        })
        .unwrap_or_default();

    // ── 1. Schema detection ──────────────────────────────────────────
    let mut output_schema: Option<(String, String, GuidanceConstraint)> = None;
    let mut schema_file_names: std::collections::HashSet<String> = std::collections::HashSet::new();

    for sys_dir in &config.system {
        if let Ok(mut entries) = fs::read_dir(sys_dir).await {
            while let Some(entry) = entries.next_entry().await? {
                let name_str = entry.file_name().to_string_lossy().to_string();
                if let Some(parsed) = ParsedSchemaName::try_parse(&name_str) {
                    schema_file_names.insert(name_str.clone());
                    if let Ok(content) = fs::read_to_string(entry.path()).await {
                        let constraint_type =
                            parsed.constraint_hint.as_deref().unwrap_or_else(|| {
                                if parsed.extension == "json" { "json" } else { "" }
                            });
                        let constraint = match constraint_type {
                            "json" => serde_json::from_str::<serde_json::Value>(&content)
                                .map(GuidanceConstraint::JsonSchema),
                            "lark" => Ok(GuidanceConstraint::Lark(content)),
                            "regex" => Ok(GuidanceConstraint::Regex(content)),
                            "llg" => Ok(GuidanceConstraint::Llguidance(content)),
                            _ => continue,
                        };
                        if let Ok(c) = constraint {
                            logger.log(&format!(
                                "Loaded output schema: .schema-{}.{}",
                                parsed.format_str, parsed.extension
                            )).await;
                            output_schema = Some((
                                parsed.format_str,
                                parsed.extension,
                                c,
                            ));
                        }
                    }
                }
            }
        }
    }

    // ── 2. Tool discovery ────────────────────────────────────────────
    let mut tool_registry: Option<ToolRegistry> = None;
    let mut tools: Vec<ToolDef> = vec![];
    let mut tool_guidance = String::new();

    let mut all_tool_names: Vec<String> = config.tools.clone();
    for ht in &config.heavy_tools {
        if !all_tool_names.contains(ht) {
            all_tool_names.push(ht.clone());
        }
    }

    if !all_tool_names.is_empty() && output_schema.is_none() {
        match discover_tools(&all_tool_names, &logger).await {
            Ok(registry) => {
                if !registry.guidance.is_empty() {
                    tool_guidance.push_str("== Tool Usage Guidance ==\n");
                    tool_guidance.push_str(&registry.guidance);
                }
                for t in &registry.tools {
                    tools.push(t.clone());
                }
                tool_registry = Some(registry);
            }
            Err(e) => {
                logger.log(&format!("Tool discovery error: {e}")).await;
            }
        }
    } else if output_schema.is_some() && !config.tools.is_empty() {
        logger.log("Tools disabled — schema constraint is active").await;
    }

    // ── 3. File collation ────────────────────────────────────────────
    let mut all_files = Vec::new();

    for sys_dir in &config.system {
        if let Ok(mut entries) = fs::read_dir(sys_dir).await {
            let mut sys_files: Vec<PathBuf> = Vec::new();
            while let Some(entry) = entries.next_entry().await? {
                let name_str = entry.file_name().to_string_lossy().to_string();
                if name_str.starts_with('.') || schema_file_names.contains(&name_str) {
                    continue;
                }
                if entry.path().is_file() {
                    sys_files.push(entry.path());
                }
            }
            sys_files.sort();
            for f in sys_files {
                if let Ok(content) = fs::read_to_string(&f).await {
                    let role = match extract_frontmatter(&content) {
                        Some((n, d, body)) => {
                            let _ = fs::write(&f, &body).await;
                            HistoryTurnRole::Skill(n, d)
                        }
                        None => HistoryTurnRole::System,
                    };
                    let md = fs::metadata(&f).await.ok();
                    all_files.push(FileEntry {
                        path: f.clone(),
                        created: md.as_ref().and_then(|m| m.created().ok()).unwrap_or(SystemTime::UNIX_EPOCH),
                        role,
                        metadata_str: String::new(),
                        excluded: false,
                        tool_call_id: None,
                    });
                }
            }
        }
    }

    if let Some(ref extra_prompt) = config.prompt {
        let tmp = tempfile::tempdir()?;
        let prompt_path = tmp.path().join("config-prompt.txt");
        fs::write(&prompt_path, extra_prompt).await?;
        all_files.push(FileEntry {
            path: prompt_path,
            created: SystemTime::UNIX_EPOCH,
            role: HistoryTurnRole::System,
            metadata_str: String::new(),
            excluded: false,
            tool_call_id: None,
        });
        std::mem::forget(tmp);
    }

    if !tool_guidance.is_empty() {
        let tmp = tempfile::tempdir()?;
        let tg_path = tmp.path().join("tool-guidance.txt");
        fs::write(&tg_path, &tool_guidance).await?;
        all_files.push(FileEntry {
            path: tg_path,
            created: SystemTime::UNIX_EPOCH + Duration::from_secs(1),
            role: HistoryTurnRole::System,
            metadata_str: String::new(),
            excluded: false,
            tool_call_id: None,
        });
        std::mem::forget(tmp);
    }

    let mut excluded_canonical = Vec::new();
    for ex in &config.excluded_from_summary {
        if let Ok(c) = fs::canonicalize(ex).await {
            excluded_canonical.push(c);
        }
    }

    // Collect User / Assistant / Tool files
    for input_dir in &config.inputs {
        let input_canonical = fs::canonicalize(input_dir).await.unwrap_or_else(|_| PathBuf::from(input_dir));
        let is_excluded = excluded_canonical.iter().any(|ex| {
            input_canonical.starts_with(ex) || ex.starts_with(&input_canonical)
        });
        if let Ok(mut entries) = fs::read_dir(input_dir).await {
            while let Some(entry) = entries.next_entry().await? {
                let p = entry.path();
                if p.is_file() {
                    let metadata = fs::metadata(&p).await?;
                    let meta_str = format_file_metadata(&p, &metadata);
                    all_files.push(FileEntry {
                        path: p,
                        created: metadata.created()?,
                        role: HistoryTurnRole::User,
                        metadata_str: meta_str,
                        excluded: is_excluded,
                        tool_call_id: None,
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
                    let metadata = fs::metadata(&p).await?;
                    all_files.push(FileEntry {
                        path: p,
                        created: metadata.created()?,
                        role: HistoryTurnRole::Assistant,
                        metadata_str: String::new(),
                        excluded: false,
                        tool_call_id: None,
                    });
                }
            }
        }
    }
    let effective_tool_dir = config.tool_output.clone()
        .or_else(|| config.output.first().map(|o| format!("{}/tool_output", o)));
    if let Some(ref tool_dir) = effective_tool_dir {
        if let Ok(mut entries) = fs::read_dir(tool_dir).await {
            while let Some(entry) = entries.next_entry().await? {
                let p = entry.path();
                if p.is_file() {
                    let metadata = fs::metadata(&p).await?;
                    let tool_call_id = p.file_stem()
                        .and_then(|s| s.to_str())
                        .and_then(|stem| {
                            let parts: Vec<&str> = stem.rsplitn(4, '-').collect();
                            if parts.len() >= 4 { Some(parts[0].to_string()) } else { None }
                        });
                    all_files.push(FileEntry {
                        path: p,
                        created: metadata.created()?,
                        role: HistoryTurnRole::Tool,
                        metadata_str: String::new(),
                        excluded: false,
                        tool_call_id,
                    });
                }
            }
        }
    }
    all_files.sort_by_key(|f| f.created);

    // ── 4. Apply system_prompt_mode ──────────────────────────────────
    let is_system_role = |r: &HistoryTurnRole| -> bool {
        matches!(r, HistoryTurnRole::System | HistoryTurnRole::Skill(..))
    };
    match system_mode {
        SystemPromptMode::Merged => {
            let mut merged = String::new();
            all_files.retain(|f| {
                if is_system_role(&f.role) {
                    if let Ok(content) = std::fs::read_to_string(&f.path) {
                        if !merged.is_empty() { merged.push_str("\n\n"); }
                        merged.push_str(&content);
                    }
                    false
                } else {
                    true
                }
            });
            if !merged.is_empty() {
                let tmp = tempfile::tempdir().expect("temp dir");
                let mp = tmp.path().join("_system_merged_.txt");
                let _ = std::fs::write(&mp, &merged);
                all_files.insert(0, FileEntry {
                    path: mp,
                    created: SystemTime::UNIX_EPOCH,
                    role: HistoryTurnRole::System,
                    metadata_str: String::new(),
                    excluded: true,
                    tool_call_id: None,
                });
                std::mem::forget(tmp);
            }
        }
        SystemPromptMode::Frontloaded | SystemPromptMode::Summarized => {
            let excluded = matches!(system_mode, SystemPromptMode::Frontloaded);
            let mut sys: Vec<FileEntry> = Vec::new();
            let mut rest: Vec<FileEntry> = Vec::new();
            for f in all_files {
                if is_system_role(&f.role) {
                    sys.push(FileEntry { excluded, ..f });
                } else {
                    rest.push(f);
                }
            }
            all_files = sys;
            all_files.extend(rest);
        }
        SystemPromptMode::Interleaved => {}
    }

    let limit = config.history_limit.unwrap_or(0);
    let start_idx = if limit > 0 && all_files.len() > limit {
        all_files.len() - limit
    } else {
        0
    };

    // ── 5. Build history turns ───────────────────────────────────────
    let mut combined_history = Vec::new();
    let mut turn_idx = 1;
    for entry in all_files.iter().skip(start_idx) {
        if let Ok(content) = fs::read_to_string(&entry.path).await {
            let final_content = if matches!(entry.role, HistoryTurnRole::User)
                && !entry.metadata_str.is_empty()
                && config.prepend_file_metadata
            {
                format!("{}\n---\n{}", entry.metadata_str, content)
            } else {
                content
            };
            let file_key = if is_system_role(&entry.role) {
                "_system_".to_string()
            } else {
                format!("{:016x}", xxh3_64(entry.path.to_string_lossy().as_bytes()))
            };
            combined_history.push(HistoryTurn {
                role: entry.role.clone(),
                content: final_content,
                turn_index: turn_idx,
                excluded_from_compression: entry.excluded,
                file_key,
                file_path: if is_system_role(&entry.role) { String::new() } else { entry.path.to_string_lossy().to_string() },
                content_hash: String::new(),
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

    // ── 6. Compression ───────────────────────────────────────────────
    let checkpoint_base = config.output.first()
        .and_then(|o| PathBuf::from(o).parent().map(|p| p.to_path_buf()))
        .or_else(|| config.stream_output.as_ref()
            .and_then(|s| PathBuf::from(s).parent().map(|p| p.to_path_buf())))
        .unwrap_or_else(|| PathBuf::from("."));
    let oom_db_path = config.compression_db_path.as_ref()
        .map(PathBuf::from)
        .unwrap_or_else(|| checkpoint_base.join(".agent_context").join("compression.db"));
    let oom_compression_config = config.compression.clone();
    let comp_mgr = CompressionManager::new(&oom_db_path, &config.compression)?;
    let uncompressed_history: Vec<HistoryTurn> = history.clone();
    let mut history_mut = history;
    let provider_max_seq_len = config.provider.as_ref().and_then(|p| match p {
        crate::config::ProviderConfig::OpenAi { max_seq_len, .. } => *max_seq_len,
        crate::config::ProviderConfig::Plugin { max_seq_len, .. } => *max_seq_len,
    });
    let compressed_context = comp_mgr
        .get_compressed_context(
            provider.as_ref(),
            &mut history_mut,
            &latest_user_input,
            config.context_checkpoint_limit,
            provider_max_seq_len,
        )
        .await?;

    // ── 7. Drain volatile context ────────────────────────────────────
    let volatile_drained: Vec<(Role, String)> = {
        let mut v_ctx = volatile_context.lock().await;
        v_ctx.drain(..).collect()
    };

    // ── 8. Build ChatRequest ─────────────────────────────────────────
    let mut messages: Vec<ChatMessage> = Vec::new();
    for (role, content) in &compressed_context {
        messages.push(ChatMessage {
            role: role.clone(),
            content: content.clone(),
            tool_calls: None,
            tool_call_id: None,
        });
    }
    for (role, content) in &volatile_drained {
        messages.push(ChatMessage {
            role: role.clone(),
            content: content.clone(),
            tool_calls: None,
            tool_call_id: None,
        });
    }
    messages.push(ChatMessage {
        role: Role::User,
        content: latest_user_input,
        tool_calls: None,
        tool_call_id: None,
    });

    let sampling = ProviderSamplingConfig::new(None);

    // ── 9. Prepare output paths ──────────────────────────────────────
    let timestamp = SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_millis();
    let output_file_path: Option<PathBuf> = config.output.first().map(|o| {
        PathBuf::from(o).join(format!("out-{}.txt", timestamp))
    });

    let mut stream_file = if let Some(ref stream_dir) = config.stream_output {
        let stream_path = PathBuf::from(stream_dir).join(format!("out-{}.txt", timestamp));
        fs::create_dir_all(stream_dir).await?;
        Some(fs::File::create(&stream_path).await?)
    } else {
        None
    };

    // ── 10. Constraint ───────────────────────────────────────────────
    let constraint = output_schema.map(|(_, _, c)| c);

    let tool_choice = if tools.is_empty() {
        ProviderToolChoice::None
    } else {
        ProviderToolChoice::Auto
    };

    // ── 11. Inference loop ───────────────────────────────────────────
    let mut final_output = String::new();
    let mut tools_executed = false;

    loop {
        let request = ChatRequest {
            messages: messages.clone(),
            sampling: sampling.clone(),
            tools: if tools.is_empty() { None } else { Some(tools.clone()) },
            tool_choice: tool_choice.clone(),
            constraint: constraint.clone(),
            enable_thinking: config.enable_thinking,
            images: Vec::new(),
            audios: Vec::new(),
            videos: Vec::new(),
            model: None,
        };

        let mut accumulated_content = String::new();
        let mut current_tool_calls: Vec<ToolCallDelta> = Vec::new();
        let mut stream = provider.stream_chat(request).await?;

        use futures::StreamExt;
        while let Some(event) = stream.next().await {
            match event? {
                ProviderEvent::Chunk(text) => {
                    accumulated_content.push_str(&text);
                    if let Some(ref mut f) = stream_file {
                        let _ = f.write_all(text.as_bytes()).await;
                        let _ = f.flush().await;
                    }
                }
                ProviderEvent::Reasoning(reasoning) => {
                    if let Some(ref mut f) = stream_file {
                        let _ = f.write_all(reasoning.as_bytes()).await;
                        let _ = f.flush().await;
                    }
                }
                ProviderEvent::ToolCall(tc) => {
                    current_tool_calls.push(tc);
                }
                ProviderEvent::Error(msg) => {
                    logger.log(&format!("Provider error: {}", msg)).await;
                }
                ProviderEvent::Done => break,
            }
        }

        final_output = accumulated_content.clone();

        // ── 12. Tool calls ───────────────────────────────────────────
        if current_tool_calls.is_empty() {
            if !final_output.is_empty() {
                if let Some(ref output_path) = output_file_path {
                    logger.log(&format!("Turn complete, writing {} chars to: {:?}", final_output.len(), output_path)).await;
                    let _ = fs::write(output_path, &final_output).await;
                }
            }
            break;
        }

        logger.log(&format!("Executing {} tool calls", current_tool_calls.len())).await;

        // Add assistant message with tool calls
        let tc_response: Vec<ToolCall> = current_tool_calls.iter().map(|tc| ToolCall {
            id: tc.id.clone(),
            name: tc.name.clone(),
            arguments: tc.arguments.clone(),
        }).collect();
        messages.push(ChatMessage {
            role: Role::Assistant,
            content: accumulated_content,
            tool_calls: Some(tc_response),
            tool_call_id: None,
        });

        // Execute each tool
        for (tool_idx, tc) in current_tool_calls.iter().enumerate() {
            let result = if let Some(ref registry) = tool_registry {
                if let Some(binary_name) = registry.name_to_binary.get(&tc.name) {
                    match execute_tool_binary(binary_name, &tc.arguments).await {
                        Ok(output) => {
                            if tc.name == "load_into_context" {
                                volatile_context.lock().await.push((
                                    Role::System,
                                    output,
                                ));
                                "Files loaded into context for next turn.".into()
                            } else {
                                output
                            }
                        }
                        Err(e) => format!("Error executing {}: {}", tc.name, e),
                    }
                } else {
                    format!("Unknown tool: {}", tc.name)
                }
            } else {
                "Tools are not enabled for this agent.".into()
            };

            let tool_dest_dir = config.tool_output.clone().unwrap_or_else(|| {
                config.output.first()
                    .map(|o| format!("{}/tool_output", o))
                    .unwrap_or_else(|| "/tmp/agentgraph_tool_output".to_string())
            });
            let _ = fs::create_dir_all(&tool_dest_dir).await;
            let tool_output_path = PathBuf::from(tool_dest_dir).join(format!(
                "tool-{}-{}-{}-{}.txt",
                tc.name, tool_idx, timestamp, tc.id
            ));
            let _ = fs::write(&tool_output_path, &result).await;

            messages.push(ChatMessage {
                role: Role::Tool,
                content: result,
                tool_calls: None,
                tool_call_id: Some(tc.id.clone()),
            });
        }
        tools_executed = true;
    }

    logger.log("Inference turn complete (provider)").await;
    Ok(final_output)
}
