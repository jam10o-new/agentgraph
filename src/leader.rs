use crate::config::Config;
use crate::model_loader::load_models;
use crate::agent::Agent;
use anyhow::Result;
use std::sync::Arc;
use mistralrs::SamplingParams;
use tokio::io::{AsyncWriteExt, AsyncReadExt};
use tokio::net::UnixListener;
use std::path::PathBuf;

pub struct Leader {
    pub config: Config,
    pub model: Arc<mistralrs::Model>,
}

impl Leader {
    pub async fn new(config: Config) -> Result<Self> {
        let model = Arc::new(load_models(&config.models).await?);
        Ok(Self { config, model })
    }

    pub async fn run(&self) -> Result<()> {
        let mut agent_handles = Vec::new();

        let sampling = SamplingParams {
            temperature: self.config.sampling.temperature,
            top_p: self.config.sampling.top_p,
            top_k: self.config.sampling.top_k,
            min_p: self.config.sampling.min_p,
            repetition_penalty: self.config.sampling.repetition_penalty,
            frequency_penalty: self.config.sampling.frequency_penalty,
            presence_penalty: self.config.sampling.presence_penalty,
            max_len: self.config.sampling.max_len,
            top_n_logprobs: 0,
            stop_toks: None,
            logits_bias: None,
            n_choices: 1,
            dry_params: None,
        };

        for (name, agent_config) in &self.config.agents {
            let agent = Agent::new(
                name.clone(),
                agent_config.clone(),
                self.model.clone(),
                sampling.clone(),
            );
            
            let handle = tokio::spawn(async move {
                if let Err(e) = agent.run_loop().await {
                    eprintln!("Agent loop error: {:?}", e);
                }
            });
            agent_handles.push(handle);
        }

        // IPC listener for commands (status, reload, etc.)
        let pid = std::process::id();
        let pipe_path = PathBuf::from("/tmp/agentgraph").join(format!("ag-{}.sock", pid));
        tokio::fs::create_dir_all(pipe_path.parent().unwrap()).await?;
        let _ = tokio::fs::remove_file(&pipe_path).await;
        let listener = UnixListener::bind(&pipe_path)?;
        println!("Leader PID {} listening on {:?}", pid, pipe_path);

        let model = self.model.clone();
        tokio::spawn(async move {
            loop {
                if let Ok((mut stream, _)) = listener.accept().await {
                    let _model = model.clone();
                    tokio::spawn(async move {
                        let mut buf = [0u8; 1024];
                        if let Ok(n) = stream.read(&mut buf).await {
                            if let Ok(cmd) = serde_json::from_slice::<crate::ipc::Command>(&buf[..n]) {
                                println!("Received IPC command: {:?}", cmd);
                                // TODO: Actually execute command
                                // For now just acknowledge
                                let _ = stream.write_all(b"OK").await;
                            }
                        }
                    });
                }
            }
        });

        // Keep running until Ctrl-C
        tokio::signal::ctrl_c().await?;
        println!("Shutting down...");
        
        Ok(())
    }
}
