use crate::agent::Agent;
use crate::config::{AgentConfig, Config};
use crate::ipc::{Command, IpcResponse, SessionStep};
use crate::model_loader::load_models;
use crate::remote_session::{ConversationStep, RemoteSessionState};
use crate::utils::{LEADER_PID_FILE, LeaderStatus, is_leader_alive};
use anyhow::{Context, Result, anyhow};
use mistralrs::{SamplingParams, TextMessageRole};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixListener;
use tokio::sync::Mutex;

pub struct AgentEntry {
    pub handle: tokio::task::JoinHandle<()>,
    pub volatile_context: Arc<Mutex<Vec<(TextMessageRole, String)>>>,
    pub output_forwarder: Arc<Mutex<Option<tokio::sync::mpsc::UnboundedSender<String>>>>,
    pub trigger_path: PathBuf,
}

/// Tracks the last inference time per model alias for idle eviction.
#[derive(Clone, Default)]
pub struct ModelAccess {
    pub last_access: Arc<tokio::sync::RwLock<HashMap<String, tokio::time::Instant>>>,
}

impl ModelAccess {
    pub async fn touch(&self, alias: &str) {
        let mut map = self.last_access.write().await;
        map.insert(alias.to_string(), tokio::time::Instant::now());
    }
}

pub struct Leader {
    pub config: Arc<Mutex<Config>>,
    pub config_path: String,
    /// The binary's mtime at leader startup. Used to detect when
    /// the on-disk binary has been replaced (e.g. by a new build)
    /// so the leader can self-restart with the updated version.
    pub binary_mtime: SystemTime,
    pub model: Option<Arc<mistralrs::Model>>,
    pub agents: Arc<Mutex<HashMap<String, AgentEntry>>>,
    pub model_access: ModelAccess,
    /// Shared session-tree state accessible via IPC by API frontends.
    pub sessions: Arc<RemoteSessionState>,
}

impl Leader {
    pub async fn new(config: Config, config_path: String) -> Result<Self> {
        let model = if config.models.is_empty() {
            None
        } else {
            Some(Arc::new(load_models(&config.models).await?))
        };

        let binary_mtime = std::env::current_exe()
            .ok()
            .and_then(|p| std::fs::metadata(&p).ok())
            .and_then(|m| m.modified().ok())
            .unwrap_or(SystemTime::UNIX_EPOCH);

        Ok(Self {
            config: Arc::new(Mutex::new(config)),
            config_path,
            binary_mtime,
            model,
            agents: Arc::new(Mutex::new(HashMap::new())),
            model_access: ModelAccess::default(),
            sessions: Arc::new(RemoteSessionState::new()),
        })
    }

    pub async fn spawn_agent(&self, name: String, agent_config: AgentConfig) -> Result<()> {
        let mut agents = self.agents.lock().await;
        if agents.contains_key(&name) {
            println!("Agent {} already exists, skipping.", name);
            return Ok(());
        }

        let model = self
            .model
            .as_ref()
            .ok_or_else(|| anyhow!("No models loaded in leader. Cannot spawn agent {}.", name))?;

        let sampling = SamplingParams {
            temperature: agent_config.sampling.temperature,
            top_p: agent_config.sampling.top_p,
            top_k: agent_config.sampling.top_k,
            min_p: agent_config.sampling.min_p,
            repetition_penalty: agent_config.sampling.repetition_penalty,
            frequency_penalty: agent_config.sampling.frequency_penalty,
            presence_penalty: agent_config.sampling.presence_penalty,
            max_len: agent_config.sampling.max_len,
            top_n_logprobs: 0,
            stop_toks: None,
            logits_bias: None,
            n_choices: 1,
            dry_params: None,
        };

        let config = self.config.lock().await.clone();
        let mut agent = Agent::new(
            name.clone(),
            agent_config.clone(),
            config,
            model.clone(),
            sampling,
            self.model_access.clone(),
        );

        // Shared output forwarder: the leader installs a sender when it
        // wants to capture output (e.g. `ag run --quiet`). The agent
        // sends its inference result through it after each turn.
        let forwarder: Arc<Mutex<Option<tokio::sync::mpsc::UnboundedSender<String>>>> =
            Arc::new(Mutex::new(None));
        agent.output_forwarder = forwarder.clone();

        // We use the first input directory as the "trigger" path for manual runs
        let trigger_path = PathBuf::from(
            agent_config
                .inputs
                .first()
                .cloned()
                .unwrap_or_else(|| ".".into()),
        );
        let volatile_context = agent.volatile_context.clone();

        let name_for_log = name.clone();
        let handle = tokio::spawn(async move {
            if let Err(e) = agent.run_loop().await {
                eprintln!("Agent {} loop error: {:?}", name_for_log, e);
            }
        });

        agents.insert(
            name,
            AgentEntry {
                handle,
                volatile_context,
                output_forwarder: forwarder,
                trigger_path,
            },
        );
        Ok(())
    }

    /// Check whether the on-disk binary has been replaced since startup.
    /// Returns `true` if the binary changed and the leader should restart.
    fn binary_changed(&self) -> bool {
        std::env::current_exe()
            .ok()
            .and_then(|p| std::fs::metadata(&p).ok())
            .and_then(|m| m.modified().ok())
            .is_some_and(|mtime| mtime != self.binary_mtime)
    }

    /// Prepare a leader restart with the new binary. Writes `cmd` to a
    /// temporary file and spawns the new leader.  The old socket is left
    /// in place — `find_leader_socket()` handles stale cleanup via the
    /// `/proc/{pid}` liveness check.  The PID file is removed so the new
    /// leader can claim it.
    /// Does NOT exit — the caller should flush the IPC response and then
    /// call `std::process::exit(0)`.
    fn prepare_restart(&self, cmd: &Command) {
        let pending_dir = PathBuf::from("/tmp/agentgraph");
        let _ = std::fs::create_dir_all(&pending_dir);

        let pid = std::process::id();
        // Remove PID file so the new leader can write a fresh one
        let _ = std::fs::remove_file(LEADER_PID_FILE);

        let pending_path = pending_dir.join(format!("pending_command_{}.json", pid));
        if let Ok(json) = serde_json::to_vec(cmd) {
            let _ = std::fs::write(&pending_path, &json);
        }

        // Spawn the new leader process with the updated binary.
        if let Ok(exe) = std::env::current_exe() {
            let log_dir = std::path::Path::new("/tmp/agentgraph");
            let _ = std::fs::create_dir_all(log_dir);
            if let Ok(log_file) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(log_dir.join("leader.log"))
            {
                let _ = std::process::Command::new(&exe)
                    .arg("leader")
                    .arg("--config")
                    .arg(&self.config_path)
                    .arg("--pending-command")
                    .arg(pending_path.to_string_lossy().as_ref())
                    .env("AGENTGRAPH_BACKGROUND", "1")
                    .stdout(std::process::Stdio::from(log_file.try_clone().unwrap()))
                    .stderr(std::process::Stdio::from(log_file))
                    .spawn();
            }
        }
    }

    /// Process a single pending command (called on startup after a restart).
    async fn process_pending_command(&self, cmd: Command) {
        match cmd {
            Command::SpawnAgent {
                name,
                config: agent_config,
            } => {
                if let Err(e) = self.spawn_agent(name, agent_config).await {
                    eprintln!("Pending command spawn error: {}", e);
                }
            }
            Command::RunAgent(name, message, _quiet) => {
                let agents_map = self.agents.lock().await;
                if let Some(entry) = agents_map.get(&name) {
                    if let Some(msg) = message {
                        entry
                            .volatile_context
                            .lock()
                            .await
                            .push((TextMessageRole::User, msg));
                    }
                    let trigger_file = entry.trigger_path.join(format!(
                        ".trigger-{}",
                        SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_millis()
                    ));
                    let _ = tokio::fs::write(&trigger_file, b"").await;
                    let _ = tokio::fs::remove_file(&trigger_file).await;
                }
            }
            Command::StopAgent(name) => {
                let mut agents_map = self.agents.lock().await;
                if let Some(e) = agents_map.remove(&name) {
                    e.handle.abort();
                }
            }
            // Session and other stateless commands — no need to replay.
            _ => {}
        }
    }

    /// Resolve an agent name for a chat context. Returns the agent name
    /// to use, falling back to the default "api" agent.
    pub async fn resolve_chat_agent(&self, _chat_id: &str) -> String {
        // Per-user overrides are handled by the frontend binary (telegram/http).
        // The leader just needs to ensure a fallback "api" agent exists.
        {
            let mut config = self.config.lock().await;
            if !config.agents.contains_key("api") {
                let first_model = config
                    .models
                    .keys()
                    .next()
                    .cloned()
                    .unwrap_or_else(|| "primary".to_string());
                config.agents.insert(
                    "api".to_string(),
                    AgentConfig {
                        inputs: vec![],
                        output: vec![],
                        stream_output: None,
                        tool_output: None,
                        consume_tool_calls: false,
                        system: vec![],
                        model: first_model,
                        history_limit: None,
                        realtime_audio: false,
                        allowed_extensions: vec![],
                        prompt: None,
                        sampling: Default::default(),
                        compression: Default::default(),
                        context_checkpoint_limit: Some(10000),
                        excluded_from_summary: vec![],
                        tools: vec![],
                        enable_thinking: false,
                        inference_retries: 3,
                        enable_oom_recovery: true,
                        inference_retry_delay_ms: 500,
                        compression_db_path: None,
                    },
                );
            }
        }
        "api".to_string()
    }

    /// Handle a session-tree IPC command. Returns the JSON-serialized IpcResponse.
    async fn handle_session_command(&self, cmd: Command) -> String {
        let resp = match cmd {
            Command::SessionCreate { session_id } => {
                self.sessions.get_or_create_tree(&session_id).await;
                IpcResponse::ok_str("created")
            }
            Command::SessionDelete { session_id } => {
                self.sessions.remove_tree(&session_id).await;
                IpcResponse::ok_str("deleted")
            }
            Command::SessionList => {
                let ids = self.sessions.list_ids().await;
                IpcResponse::ok_json(&ids)
            }
            Command::SessionBuild { session_id, steps } => {
                let tree = self.sessions.get_or_create_tree(&session_id).await;
                let conv_steps: Vec<ConversationStep> = steps
                    .into_iter()
                    .map(|s| ConversationStep {
                        role: s.role,
                        content: s.content,
                    })
                    .collect();
                match crate::remote_session::build_conversation(&tree, &conv_steps).await {
                    Ok(state) => IpcResponse::ok_json(&state),
                    Err(e) => IpcResponse::err(e),
                }
            }
            Command::SessionSetupDirs {
                session_id,
                system_msgs,
            } => {
                let tree = self.sessions.get_or_create_tree(&session_id).await;
                match crate::remote_session::setup_request_dirs(&tree, &system_msgs).await {
                    Ok((stream_dir, tools_dir, system_dir)) => {
                        let info = serde_json::json!({
                            "stream_dir": stream_dir.to_string_lossy(),
                            "tools_dir": tools_dir.to_string_lossy(),
                            "system_dir": system_dir.to_string_lossy(),
                        });
                        IpcResponse::ok_json(&info)
                    }
                    Err(e) => IpcResponse::err(e),
                }
            }
            Command::SessionCreateResponseDir {
                session_id,
                current_hash,
            } => {
                let tree = self.sessions.get_or_create_tree(&session_id).await;
                match crate::remote_session::create_response_dir(&tree, &current_hash).await {
                    Ok(dir) => IpcResponse::ok_str(dir.to_string_lossy().to_string()),
                    Err(e) => IpcResponse::err(e),
                }
            }
            Command::SessionCacheResponse {
                session_id,
                parent_hash,
                content,
                response_dir,
            } => {
                let tree = self.sessions.get_or_create_tree(&session_id).await;
                crate::remote_session::cache_response(
                    &tree,
                    &parent_hash,
                    &content,
                    PathBuf::from(&response_dir),
                )
                .await;
                IpcResponse::ok_str("cached")
            }
            Command::SessionChat {
                session_id,
                steps,
                model,
                ..
            } => match self.run_session_chat(&session_id, &steps, &model).await {
                Ok(content) => IpcResponse {
                    ok: true,
                    data: Some(content),
                    error: None,
                },
                Err(e) => IpcResponse::err(e.to_string()),
            },
            Command::SessionListChildren { session_id, hash } => {
                let children = self.sessions.list_children(&session_id, &hash).await;
                IpcResponse::ok_json(&children)
            }
            Command::SessionPath { session_id, hash } => {
                let path = self.sessions.get_path(&session_id, &hash).await;
                IpcResponse::ok_json(&path)
            }
            _ => IpcResponse::err("unexpected command in session handler"),
        };
        serde_json::to_string(&resp)
            .unwrap_or_else(|_| r#"{"ok":false,"error":"serialize"}"#.to_string())
    }

    /// Run a full chat turn: build conversation state, spawn an ephemeral agent,
    /// collect output. Returns the assistant's response text.
    async fn run_session_chat(
        &self,
        session_id: &str,
        steps: &[SessionStep],
        model: &str,
    ) -> Result<String> {
        let config = self.config.lock().await;
        let base_agent = config
            .agents
            .get(model)
            .ok_or_else(|| anyhow!("agent '{}' not found in config", model))?
            .clone();
        let global_config = config.clone();
        drop(config);

        let tree = self.sessions.get_or_create_tree(session_id).await;

        let conv_steps: Vec<ConversationStep> = steps
            .iter()
            .map(|s| ConversationStep {
                role: s.role.clone(),
                content: s.content.clone(),
            })
            .collect();

        let state = crate::remote_session::build_conversation(&tree, &conv_steps)
            .await
            .map_err(|e| anyhow!("build_conversation: {}", e))?;

        let (_stream_dir, _tools_dir, system_dir) =
            crate::remote_session::setup_request_dirs(&tree, &state.system_msgs)
                .await
                .map_err(|e| anyhow!("setup_request_dirs: {}", e))?;

        let response_dir = crate::remote_session::create_response_dir(&tree, &state.current_hash)
            .await
            .map_err(|e| anyhow!("create_response_dir: {}", e))?;

        // Build isolated agent config
        let mut isolated_config = base_agent.clone();
        isolated_config.inputs = state.user_dirs.clone();
        let mut output_dirs = vec![response_dir.to_string_lossy().to_string()];
        output_dirs.extend(state.assistant_dirs);
        isolated_config.output = output_dirs;
        isolated_config.system = if state.system_msgs.is_empty() {
            base_agent.system.clone()
        } else {
            let mut s = vec![system_dir.to_string_lossy().to_string()];
            s.extend(base_agent.system.iter().cloned());
            s
        };

        let model_arc = self
            .model
            .as_ref()
            .ok_or_else(|| anyhow!("no model loaded"))?
            .clone();

        let sampling = SamplingParams {
            temperature: isolated_config.sampling.temperature,
            top_p: isolated_config.sampling.top_p,
            top_k: isolated_config.sampling.top_k,
            min_p: isolated_config.sampling.min_p,
            repetition_penalty: isolated_config.sampling.repetition_penalty,
            frequency_penalty: isolated_config.sampling.frequency_penalty,
            presence_penalty: isolated_config.sampling.presence_penalty,
            max_len: isolated_config.sampling.max_len,
            top_n_logprobs: 0,
            stop_toks: None,
            logits_bias: None,
            n_choices: 1,
            dry_params: None,
        };

        let agent_name = format!("session-{}-{}", session_id, uuid::Uuid::new_v4());
        let agent = crate::Agent::new(
            agent_name,
            isolated_config,
            global_config,
            model_arc,
            sampling,
            self.model_access.clone(),
        );

        let handle = tokio::spawn(async move {
            if let Err(e) = agent.run_loop().await {
                eprintln!("Session agent error: {:?}", e);
            }
        });

        tokio::time::sleep(Duration::from_millis(150)).await;

        // Write trigger
        let trigger_dir = if state.user_dirs.is_empty() {
            &response_dir
        } else {
            &PathBuf::from(state.user_dirs.last().unwrap())
        };
        let trigger_path = trigger_dir.join(format!(
            "session-latest-{}.txt",
            SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis()
        ));
        tokio::fs::write(&trigger_path, &state.latest_user_msg)
            .await
            .map_err(|e| anyhow!("write trigger: {}", e))?;

        let output_path = response_dir.to_string_lossy().to_string();
        let content = wait_for_output(Some(output_path.clone()), SystemTime::now())
            .await
            .map_err(|e| anyhow!("wait_for_output: {}", e))?;

        handle.abort();

        // Cache response
        crate::remote_session::cache_response(&tree, &state.current_hash, &content, response_dir)
            .await;

        Ok(content)
    }
}

/// Wait for a new output file to appear in `output_dir` after `after` time.
async fn wait_for_output(
    output_dir_maybe: Option<String>,
    after: SystemTime,
) -> anyhow::Result<String> {
    use tokio::time::timeout;
    if let Some(output_dir) = output_dir_maybe {
        let output_path = PathBuf::from(output_dir);
        timeout(Duration::from_secs(120), async {
            loop {
                tokio::time::sleep(Duration::from_millis(100)).await;
                let mut candidates = Vec::new();
                if let Ok(mut entries) = tokio::fs::read_dir(&output_path).await {
                    while let Ok(Some(entry)) = entries.next_entry().await {
                        let path = entry.path();
                        if path.is_file() {
                            if let Ok(metadata) = entry.metadata().await {
                                let modified = metadata
                                    .modified()
                                    .or_else(|_| metadata.created())
                                    .map_err(|e| anyhow!("mtime: {}: {}", path.display(), e))?;
                                if modified >= after {
                                    candidates.push((modified, path));
                                }
                            }
                        }
                    }
                }
                if let Some((_, path)) = candidates.into_iter().max_by_key(|(t, _)| *t) {
                    tokio::time::sleep(Duration::from_millis(200)).await;
                    return anyhow::Result::Ok(
                        tokio::fs::read_to_string(&path)
                            .await
                            .map_err(|e| anyhow!("read output: {}", e))?,
                    );
                }
            }
        })
        .await
        .map_err(|_| anyhow!("Timeout waiting for agent output"))?
    } else {
        Err(anyhow!("No output directory configured"))
    }
}

impl Leader {
    /// Spawn API frontend binaries discovered via the `api-*` config convention.
    ///
    /// Scans `config.plugins` for keys matching `api-*`.  For each found key,
    /// looks up `ag-{key}` in PATH and spawns it with `--config` and `--section`,
    /// passing the section's raw YAML value as the section argument.  Missing
    /// binaries log a warning but never prevent the leader from starting.
    ///
    /// Third-party API plugins need only a matching key in the config file and
    /// a `ag-api-<name>` binary on PATH — no leader changes required.
    async fn spawn_api_binaries(&self) {
        let pipe_path =
            PathBuf::from("/tmp/agentgraph").join(format!("ag-{}.sock", std::process::id()));

        let config = self.config.lock().await;

        for (key, value) in &config.plugins {
            // Only process keys matching `api-*`
            let section = match key.strip_prefix("api-") {
                Some(s) => s,
                None => continue,
            };
            let binary_name = format!("ag-api-{}", section);

            // Serialize the section's YAML value to a JSON string so the child
            // binary can parse it as its own config struct.
            let section_json = serde_json::to_string(value).unwrap_or_else(|_| "{}".to_string());

            if let Ok(path) = which::which(&binary_name) {
                if let Err(e) = std::process::Command::new(path)
                    .arg("--config")
                    .arg(&self.config_path)
                    .arg("--socket")
                    .arg(pipe_path.to_string_lossy().as_ref())
                    .arg("--section")
                    .arg(&section_json)
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::inherit())
                    .spawn()
                {
                    eprintln!("Leader: failed to spawn {}: {}", binary_name, e);
                } else {
                    println!("Leader: spawned {} (section api-{})", binary_name, section);
                }
            } else {
                eprintln!(
                    "Leader: {} not found in PATH — api-{} frontend unavailable",
                    binary_name, section
                );
            }
        }
    }

    pub async fn run(self, pending_command_path: Option<String>) -> Result<()> {
        // 1. Ensure Leader Uniqueness
        let status = is_leader_alive().await;
        match status {
            LeaderStatus::Ready { socket, pid } => {
                return Err(anyhow!(
                    "Another leader is already running (PID {}, socket: {:?})",
                    pid,
                    socket
                ));
            }
            LeaderStatus::Degraded { pid } => {
                return Err(anyhow!(
                    "Another leader process is running (PID {}) but has no IPC socket. \
                     The existing leader may be in a degraded state. \
                     Check `/tmp/agentgraph/leader.log`. Kill PID {} manually if needed.",
                    pid,
                    pid
                ));
            }
            LeaderStatus::NotRunning => {}
        }

        // Write PID file for multi-tier process detection
        let pid = std::process::id();
        let pid_file = std::path::Path::new(LEADER_PID_FILE);
        if let Some(parent) = pid_file.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Err(e) = std::fs::write(pid_file, pid.to_string()) {
            eprintln!("Warning: failed to write leader PID file: {}", e);
        }

        // 2. Initial agents from config
        let initial_agents = {
            let config = self.config.lock().await;
            config.agents.clone()
        };
        for (name, agent_config) in initial_agents {
            self.spawn_agent(name.clone(), agent_config.clone()).await?;
        }

        // 3. Process pending command from a previous version-mismatch restart.
        if let Some(ref pending_path) = pending_command_path {
            if let Ok(buf) = tokio::fs::read(pending_path).await {
                if let Ok(cmd) = serde_json::from_slice::<Command>(&buf) {
                    self.process_pending_command(cmd).await;
                }
            }
            let _ = tokio::fs::remove_file(pending_path).await;
        }

        // 4. Spawn API frontend binaries if their config sections are enabled.
        let _ = self.spawn_api_binaries().await;

        // 5. IPC listener (self-healing: watchdog recreates socket if deleted)
        let pid = std::process::id();
        let pipe_path = PathBuf::from("/tmp/agentgraph").join(format!("ag-{}.sock", pid));
        let parent = pipe_path.parent().unwrap().to_path_buf();
        tokio::fs::create_dir_all(&parent).await.context(format!(
            "Failed to create IPC socket dir: {}",
            parent.display()
        ))?;
        let _ = tokio::fs::remove_file(&pipe_path).await;
        let initial_listener = UnixListener::bind(&pipe_path)?;
        println!("Leader PID {} listening on {:?}", pid, pipe_path);

        // Shared listener that the watchdog can replace when the
        // socket file is deleted externally (e.g. by cleanup scripts).
        // Because connections are one-shot (no persistent IPC), dropping
        // the old listener is safe — already-accepted handlers run in
        // their own spawned tasks.
        let listener_ref: Arc<Mutex<Option<UnixListener>>> =
            Arc::new(Mutex::new(Some(initial_listener)));
        let pipe_path_clone = pipe_path.clone();

        let agents = self.agents.clone();
        let model_opt = self.model.clone();
        let config_mutex = self.config.clone();
        let sessions = self.sessions.clone();
        let leader_for_ipc = Arc::new(self);

        // Accept loop — polls the shared listener; handles None (during
        // socket recovery) by sleeping briefly and retrying.
        tokio::spawn({
            let listener_ref = listener_ref.clone();
            async move {
                loop {
                    let accept_result = {
                        let mut guard = listener_ref.lock().await;
                        match guard.as_mut() {
                            Some(listener) => listener.accept().await,
                            None => {
                                drop(guard);
                                tokio::time::sleep(Duration::from_millis(250)).await;
                                continue;
                            }
                        }
                    };

                    let (mut stream, _) = match accept_result {
                        Ok(v) => v,
                        Err(_) => {
                            // Listener likely closed during swap — retry
                            tokio::time::sleep(Duration::from_millis(100)).await;
                            continue;
                        }
                    };

                    let agents = agents.clone();
                    let model_opt = model_opt.clone();
                    let config_mutex = config_mutex.clone();
                    let leader = leader_for_ipc.clone();
                    let sessions = sessions.clone();
                    let _ = &sessions; // used by session command handler closure capture

                    tokio::spawn(async move {
                        let mut buf = Vec::new();
                        if let Ok(_) = stream.read_to_end(&mut buf).await {
                            if let Ok(cmd) = serde_json::from_slice::<Command>(&buf) {
                                // Check if the binary has been updated since startup.
                                // If so, restart with the new binary and replay the
                                // current command on the new leader.
                                if leader.binary_changed() {
                                    leader.prepare_restart(&cmd);
                                    let _ = stream.write_all(b"RESTARTING").await;
                                    let _ = stream.flush().await;
                                    std::process::exit(0);
                                }

                                match cmd {
                                    Command::UpdateConfig(new_config) => {
                                        let config = config_mutex.lock().await;
                                        // Ignore existing models
                                        for (name, _) in &new_config.models {
                                            if !config.models.contains_key(name) {
                                                eprintln!(
                                                    "Warning: Dynamic model loading not yet supported for model '{}'.",
                                                    name
                                                );
                                            }
                                        }
                                        // Add new agents
                                        for (name, agent_config) in new_config.agents {
                                            if !agents.lock().await.contains_key(&name) {
                                                if let Err(e) =
                                                    leader.spawn_agent(name, agent_config).await
                                                {
                                                    let _ = stream
                                                        .write_all(
                                                            format!("Error spawning agent: {}", e)
                                                                .as_bytes(),
                                                        )
                                                        .await;
                                                }
                                            }
                                        }
                                        let _ = stream.write_all(b"Config updated").await;
                                    }
                                    Command::SpawnAgent {
                                        name,
                                        config: agent_config,
                                    } => {
                                        if let Err(e) = leader.spawn_agent(name, agent_config).await
                                        {
                                            let _ = stream
                                                .write_all(
                                                    format!("Error spawning agent: {}", e)
                                                        .as_bytes(),
                                                )
                                                .await;
                                        } else {
                                            let _ = stream.write_all(b"Agent Spawned").await;
                                        }
                                    }
                                    Command::RunAgent(name, message, quiet) => {
                                        let forwarder_opt = {
                                            let agents_map = agents.lock().await;
                                            if let Some(entry) = agents_map.get(&name) {
                                                if let Some(msg) = message {
                                                    entry
                                                        .volatile_context
                                                        .lock()
                                                        .await
                                                        .push((TextMessageRole::User, msg));
                                                }
                                                Some((
                                                    entry.output_forwarder.clone(),
                                                    entry.trigger_path.clone(),
                                                ))
                                            } else {
                                                None
                                            }
                                        };

                                        match forwarder_opt {
                                            Some((forwarder, trigger_path)) => {
                                                let (tx, mut rx) =
                                                    tokio::sync::mpsc::unbounded_channel::<String>(
                                                    );
                                                forwarder.lock().await.replace(tx);

                                                let trigger_file = trigger_path.join(format!(
                                                    ".trigger-{}",
                                                    std::time::SystemTime::now()
                                                        .duration_since(std::time::UNIX_EPOCH)
                                                        .unwrap()
                                                        .as_millis()
                                                ));
                                                let _ = tokio::fs::write(&trigger_file, b"").await;
                                                let _ = tokio::fs::remove_file(&trigger_file).await;

                                                if !quiet {
                                                    let _ = stream
                                                        .write_all(
                                                            format!("Triggered agent {}\n", name)
                                                                .as_bytes(),
                                                        )
                                                        .await;
                                                }

                                                let output = tokio::time::timeout(
                                                    Duration::from_secs(120),
                                                    rx.recv(),
                                                )
                                                .await
                                                .ok()
                                                .flatten()
                                                .unwrap_or_default();

                                                forwarder.lock().await.take();

                                                let _ = stream.write_all(output.as_bytes()).await;
                                            }
                                            None => {
                                                let _ = stream
                                                    .write_all(
                                                        format!("Agent {} not found", name)
                                                            .as_bytes(),
                                                    )
                                                    .await;
                                            }
                                        }
                                    }
                                    Command::StopAgent(name) => {
                                        let mut agents_map = agents.lock().await;
                                        if let Some(e) = agents_map.remove(&name) {
                                            e.handle.abort();
                                            let _ = stream.write_all(b"Agent Stopped").await;
                                        } else {
                                            let _ = stream.write_all(b"Agent Not Found").await;
                                        }
                                    }
                                    Command::Status => {
                                        let agents_map = agents.lock().await;
                                        let status = format!(
                                            "Active Agents: {:?}\nModels Loaded: {}",
                                            agents_map.keys().collect::<Vec<_>>(),
                                            model_opt.is_some()
                                        );
                                        let _ = stream.write_all(status.as_bytes()).await;
                                    }
                                    Command::Shutdown => {
                                        println!("Shutdown requested via IPC");
                                        let _ = stream.write_all(b"Shutting down...").await;
                                        std::process::exit(0);
                                    }
                                    // ── Session-tree commands ────────
                                    Command::SessionCreate { .. }
                                    | Command::SessionDelete { .. }
                                    | Command::SessionList
                                    | Command::SessionBuild { .. }
                                    | Command::SessionSetupDirs { .. }
                                    | Command::SessionCreateResponseDir { .. }
                                    | Command::SessionCacheResponse { .. } => {
                                        let resp = leader.handle_session_command(cmd).await;
                                        let _ = stream.write_all(resp.as_bytes()).await;
                                    }
                                    _ => {
                                        let _ = stream
                                            .write_all(
                                                format!(
                                                    "Command Received (Unimplemented: {:?})",
                                                    cmd
                                                )
                                                .as_bytes(),
                                            )
                                            .await;
                                    }
                                }
                            }
                        }
                    });
                }
            }
        });

        // 6. Socket watchdog — periodically checks that the socket file
        //    still exists on disk.  If it was deleted externally (cleanup
        //    scripts, filesystem issues), rebind to recreate it.
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(10)).await;
                if !pipe_path_clone.exists() {
                    // Take the old listener (closes fd), remove any
                    // leftover socket inode, then rebind.
                    let old = listener_ref.lock().await.take();
                    drop(old);
                    let _ = tokio::fs::remove_file(&pipe_path_clone).await;
                    match UnixListener::bind(&pipe_path_clone) {
                        Ok(new_listener) => {
                            *listener_ref.lock().await = Some(new_listener);
                            println!(
                                "Socket recreated at {:?} (was deleted externally)",
                                pipe_path_clone
                            );
                        }
                        Err(e) => {
                            eprintln!("Failed to recreate socket {:?}: {}", pipe_path_clone, e);
                        }
                    }
                }
                // Also ensure the PID file exists
                let pid_file = std::path::Path::new(LEADER_PID_FILE);
                if !pid_file.exists() {
                    let _ = std::fs::write(pid_file, pid.to_string());
                }
            }
        });

        tokio::signal::ctrl_c().await?;
        println!("Shutting down...");
        let _ = std::fs::remove_file(LEADER_PID_FILE);
        Ok(())
    }
}
