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
    pub config: Config,
    pub model: Option<Arc<mistralrs::Model>>,
    pub agents: Arc<Mutex<HashMap<String, AgentEntry>>>,
    pub sampling: SamplingParams,
}

impl Leader {
    pub async fn new(config: Config) -> Result<Self> {
        let model = if config.models.is_empty() {
            None
        } else {
            Some(Arc::new(load_models(&config.models).await?))
        };
        
        let sampling = SamplingParams {
            temperature: config.sampling.temperature,
            top_p: config.sampling.top_p,
            top_k: config.sampling.top_k,
            min_p: config.sampling.min_p,
            repetition_penalty: config.sampling.repetition_penalty,
            frequency_penalty: config.sampling.frequency_penalty,
            presence_penalty: config.sampling.presence_penalty,
            max_len: config.sampling.max_len,
            top_n_logprobs: 0,
            stop_toks: None,
            logits_bias: None,
            n_choices: 1,
            dry_params: None,
        };

        Ok(Self { 
            config, 
            model, 
            agents: Arc::new(Mutex::new(HashMap::new())),
            sampling,
        })
    }

    pub async fn spawn_agent(&self, name: String, agent_config: AgentConfig) -> Result<()> {
        let model = self.model.as_ref().ok_or_else(|| anyhow!("No models loaded in leader. Cannot spawn agent {}.", name))?;
        
        let agent = Agent::new(
            name.clone(),
            agent_config.clone(),
            self.config.clone(),
            model.clone(),
            self.sampling.clone(),
        );
        
        let mut agents = self.agents.lock().await;
        if let Some(old_entry) = agents.remove(&name) {
            old_entry.handle.abort();
        }

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

    pub async fn run(&self) -> Result<()> {
        // 1. Ensure Leader Uniqueness
        if let Some(socket) = find_leader_socket().await {
            return Err(anyhow!("Another leader is already running (socket: {:?})", socket));
        }

        // 2. Initial agents from config
        for (name, agent_config) in &self.config.agents {
            self.spawn_agent(name.clone(), agent_config.clone()).await?;
        }

        // 3. IPC listener
        let pid = std::process::id();
        let pipe_path = PathBuf::from("/tmp/agentgraph").join(format!("ag-{}.sock", pid));
        tokio::fs::create_dir_all(pipe_path.parent().unwrap()).await?;
        let _ = tokio::fs::remove_file(&pipe_path).await;
        let listener = UnixListener::bind(&pipe_path)?;
        println!("Leader PID {} listening on {:?}", pid, pipe_path);

        let agents = self.agents.clone();
        let model_opt = self.model.clone();
        let global_config = self.config.clone();
        let sampling = self.sampling.clone();

        tokio::spawn(async move {
            loop {
                if let Ok((mut stream, _)) = listener.accept().await {
                    let agents = agents.clone();
                    let model_opt = model_opt.clone();
                    let global_config = global_config.clone();
                    let sampling = sampling.clone();
                    
                    tokio::spawn(async move {
                        let mut buf = Vec::new();
                        if let Ok(_) = stream.read_to_end(&mut buf).await {
                            if let Ok(cmd) = serde_json::from_slice::<Command>(&buf) {
                                match cmd {
                                    Command::SpawnAgent { name, config } => {
                                        if let Some(model) = &model_opt {
                                            println!("IPC: Spawning agent {}", name);
                                            let agent = Agent::new(name.clone(), config.clone(), global_config, model.clone(), sampling);
                                            let trigger_path = PathBuf::from(config.inputs.first().cloned().unwrap_or_else(|| ".".into()));
                                            let volatile_context = agent.volatile_context.clone();
                                            
                                            let mut agents_map = agents.lock().await;
                                            if let Some(e) = agents_map.remove(&name) { e.handle.abort(); }
                                            
                                            let name_for_log = name.clone();
                                            let handle = tokio::spawn(async move {
                                                if let Err(err) = agent.run_loop().await {
                                                    eprintln!("Agent {} loop error: {:?}", name_for_log, err);
                                                }
                                            });
                                            agents_map.insert(name, AgentEntry { handle, volatile_context, trigger_path });
                                            let _ = stream.write_all(b"Agent Spawned").await;
                                        } else {
                                            let _ = stream.write_all(b"Error: No models loaded in leader").await;
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
