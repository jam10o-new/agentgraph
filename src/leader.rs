use crate::config::{Config, AgentConfig};
use crate::model_loader::load_models;
use crate::agent::Agent;
use crate::ipc::Command;
use crate::utils::find_leader_socket;
use anyhow::{Context, Result, anyhow};
use std::sync::Arc;
use std::collections::HashMap;
use mistralrs::{SamplingParams, TextMessageRole};
use tokio::io::{AsyncWriteExt, AsyncReadExt};
use tokio::net::UnixListener;
use tokio::sync::Mutex;
use std::path::PathBuf;
use std::time::SystemTime;

pub struct AgentEntry {
    pub handle: tokio::task::JoinHandle<()>,
    pub volatile_context: Arc<Mutex<Vec<(TextMessageRole, String)>>>,
    pub trigger_path: PathBuf,
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
        })
    }

    pub async fn spawn_agent(&self, name: String, agent_config: AgentConfig) -> Result<()> {
        let mut agents = self.agents.lock().await;
        if agents.contains_key(&name) {
            println!("Agent {} already exists, skipping.", name);
            return Ok(());
        }

        let model = self.model.as_ref().ok_or_else(|| anyhow!("No models loaded in leader. Cannot spawn agent {}.", name))?;
        
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
        let agent = Agent::new(
            name.clone(),
            agent_config.clone(),
            config,
            model.clone(),
            sampling,
        );
        
        // We use the first input directory as the "trigger" path for manual runs
        let trigger_path = PathBuf::from(agent_config.inputs.first().cloned().unwrap_or_else(|| ".".into()));
        let volatile_context = agent.volatile_context.clone();
        
        let name_for_log = name.clone();
        let handle = tokio::spawn(async move {
            if let Err(e) = agent.run_loop().await {
                eprintln!("Agent {} loop error: {:?}", name_for_log, e);
            }
        });

        agents.insert(name, AgentEntry {
            handle,
            volatile_context,
            trigger_path,
        });
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
    /// temporary file, removes our socket, and spawns the new leader.
    /// Does NOT exit — the caller should flush the IPC response and then
    /// call `std::process::exit(0)`.
    fn prepare_restart(&self, cmd: &Command) {
        let pending_dir = PathBuf::from("/tmp/agentgraph");
        let _ = std::fs::create_dir_all(&pending_dir);

        // Remove our socket so find_leader_socket() won't see us
        // as a living leader when the new process starts.
        let pid = std::process::id();
        let socket_path = pending_dir.join(format!("ag-{}.sock", pid));
        let _ = std::fs::remove_file(&socket_path);

        let pending_path = pending_dir.join(format!(
            "pending_command_{}.json",
            pid
        ));
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
            Command::SpawnAgent { name, config: agent_config } => {
                if let Err(e) = self.spawn_agent(name, agent_config).await {
                    eprintln!("Pending command spawn error: {}", e);
                }
            }
            Command::RunAgent(name, message) => {
                let agents_map = self.agents.lock().await;
                if let Some(entry) = agents_map.get(&name) {
                    if let Some(msg) = message {
                        entry.volatile_context.lock().await.push((TextMessageRole::User, msg));
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
            // Other commands are stateless (Status, Shutdown, Reload) —
            // no need to replay them from a pending file.
            _ => {}
        }
    }

    pub async fn run(self, pending_command_path: Option<String>) -> Result<()> {
        // 1. Ensure Leader Uniqueness
        if let Some(socket) = find_leader_socket().await {
            return Err(anyhow!("Another leader is already running (socket: {:?})", socket));
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

        // 4. Start API server if enabled
        let api_config = {
            let config = self.config.lock().await;
            config.api.clone()
        };
        if let Some(ref api) = api_config {
            if api.enabled {
                // Inject a default "api" agent template so consumers can
                // unconditionally POST to /v1/chat/completions with model "api".
                {
                    let mut config = self.config.lock().await;
                    if !config.agents.contains_key("api") {
                        let first_model = config.models.keys().next().cloned().unwrap_or_else(|| "primary".to_string());
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
                                tools_enabled: true,
                                enable_thinking: false,
                                inference_retries: 3,
                                inference_retry_delay_ms: 500,
                            },
                        );
                    }
                }

                let api_state = Arc::new(crate::api::ApiState {
                    config: self.config.clone(),
                    model: self.model.clone(),
                    trees: tokio::sync::Mutex::new(std::collections::HashMap::new()),
                });
                let bind_addr = format!("{}:{}", api.bind_address, api.port);
                tokio::spawn(async move {
                    let app = crate::api::router(api_state);
                    match tokio::net::TcpListener::bind(&bind_addr).await {
                        Ok(listener) => {
                            println!("API server listening on {}", bind_addr);
                            if let Err(e) = axum::serve(listener, app).await {
                                eprintln!("API server error: {}", e);
                            }
                        }
                        Err(e) => {
                            eprintln!("Failed to bind API server to {}: {}", bind_addr, e);
                        }
                    }
                });
            }
        }

        // 5. IPC listener
        let pid = std::process::id();
        let pipe_path = PathBuf::from("/tmp/agentgraph").join(format!("ag-{}.sock", pid));
        let parent = pipe_path.parent().unwrap().to_path_buf();
        tokio::fs::create_dir_all(&parent)
            .await
            .context(format!("Failed to create IPC socket dir: {}", parent.display()))?;
        let _ = tokio::fs::remove_file(&pipe_path).await;
        let listener = UnixListener::bind(&pipe_path)?;
        println!("Leader PID {} listening on {:?}", pid, pipe_path);

        let agents = self.agents.clone();
        let model_opt = self.model.clone();
        let config_mutex = self.config.clone();
        let leader_for_ipc = Arc::new(self);

        tokio::spawn(async move {
            loop {
                if let Ok((mut stream, _)) = listener.accept().await {
                    let agents = agents.clone();
                    let model_opt = model_opt.clone();
                    let config_mutex = config_mutex.clone();
                    let leader = leader_for_ipc.clone();
                    
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
                                                eprintln!("Warning: Dynamic model loading not yet supported for model '{}'.", name);
                                            }
                                        }
                                        // Add new agents
                                        for (name, agent_config) in new_config.agents {
                                            if !agents.lock().await.contains_key(&name) {
                                                if let Err(e) = leader.spawn_agent(name, agent_config).await {
                                                    let _ = stream.write_all(format!("Error spawning agent: {}", e).as_bytes()).await;
                                                }
                                            }
                                        }
                                        let _ = stream.write_all(b"Config updated").await;
                                    }
                                    Command::SpawnAgent { name, config: agent_config } => {
                                        if let Err(e) = leader.spawn_agent(name, agent_config).await {
                                            let _ = stream.write_all(format!("Error spawning agent: {}", e).as_bytes()).await;
                                        } else {
                                            let _ = stream.write_all(b"Agent Spawned").await;
                                        }
                                    }
                                    Command::RunAgent(name, message) => {
                                        let agents_map = agents.lock().await;
                                        if let Some(entry) = agents_map.get(&name) {
                                            if let Some(msg) = message {
                                                entry.volatile_context.lock().await.push((TextMessageRole::User, msg));
                                            }
                                            // Trigger by touching a dummy file in the first input dir
                                            let trigger_file = entry.trigger_path.join(format!(".trigger-{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis()));
                                            let _ = tokio::fs::write(&trigger_file, b"").await;
                                            let _ = tokio::fs::remove_file(&trigger_file).await;
                                            let _ = stream.write_all(format!("Triggered agent {}", name).as_bytes()).await;
                                        } else {
                                            let _ = stream.write_all(format!("Agent {} not found", name).as_bytes()).await;
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
                                        let status = format!("Active Agents: {:?}\nModels Loaded: {}", agents_map.keys().collect::<Vec<_>>(), model_opt.is_some());
                                        let _ = stream.write_all(status.as_bytes()).await;
                                    }
                                    Command::Shutdown => {
                                        println!("Shutdown requested via IPC");
                                        let _ = stream.write_all(b"Shutting down...").await;
                                        std::process::exit(0);
                                    }
                                    _ => {
                                        let _ = stream.write_all(b"Command Received (Unimplemented)").await;
                                    }
                                }
                            }
                        }
                    });
                }
            }
        });

        tokio::signal::ctrl_c().await?;
        println!("Shutting down...");
        Ok(())
    }
}
