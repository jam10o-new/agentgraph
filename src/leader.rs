use crate::config::{Config, AgentConfig};
use crate::model_loader::load_models;
use crate::agent::Agent;
use crate::ipc::Command;
use crate::utils::find_leader_socket;
use anyhow::{Result, anyhow};
use std::sync::Arc;
use std::collections::HashMap;
use mistralrs::{SamplingParams, TextMessageRole};
use tokio::io::{AsyncWriteExt, AsyncReadExt};
use tokio::net::UnixListener;
use tokio::sync::Mutex;
use std::path::PathBuf;

pub struct AgentEntry {
    pub handle: tokio::task::JoinHandle<()>,
    pub volatile_context: Arc<Mutex<Vec<(TextMessageRole, String)>>>,
    pub trigger_path: PathBuf,
}

pub struct Leader {
    pub config: Arc<Mutex<Config>>,
    pub model: Option<Arc<mistralrs::Model>>,
    pub agents: Arc<Mutex<HashMap<String, AgentEntry>>>,
}

impl Leader {
    pub async fn new(config: Config) -> Result<Self> {
        let model = if config.models.is_empty() {
            None
        } else {
            Some(Arc::new(load_models(&config.models).await?))
        };
        
        Ok(Self { 
            config: Arc::new(Mutex::new(config)), 
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

    pub async fn run(self) -> Result<()> {
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

        // 3. Start API server if enabled
        let api_config = {
            let config = self.config.lock().await;
            config.api.clone()
        };
        if let Some(ref api) = api_config {
            if api.enabled {
                let session_temp = Arc::new(tempfile::tempdir().expect("Failed to create API session temp dir"));
                let session_temp = Arc::new(tempfile::tempdir().expect("Failed to create API session temp dir"));
                let api_state = Arc::new(crate::api::ApiState {
                    config: self.config.clone(),
                    model: self.model.clone(),
                    session_temp: Some(session_temp),
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

        // 4. IPC listener
        let pid = std::process::id();
        let pipe_path = PathBuf::from("/tmp/agentgraph").join(format!("ag-{}.sock", pid));
        tokio::fs::create_dir_all(pipe_path.parent().unwrap()).await?;
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
