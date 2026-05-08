use agentgraph::config::{Config, AgentConfig, SamplingConfig, CompressionConfig};
use agentgraph::leader::Leader;
use agentgraph::ipc::Command;
use agentgraph::{is_leader_alive, LeaderStatus};
use clap::{Parser, Subcommand};
use anyhow::{Result, anyhow};
use tokio::io::{AsyncWriteExt, AsyncReadExt};
use tokio::net::UnixStream;
use tokio::time::timeout;
use std::time::Duration;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the agentgraph leader
    Leader {
        /// Path to the config file
        #[arg(short, long, default_value = "config.yaml")]
        config: String,
        /// Path to a pending command file to process on startup
        #[arg(long)]
        pending_command: Option<String>,
    },
    /// Run a specific agent turn, optionally injecting a message
    Run {
        /// Agent name
        agent: String,
        /// Optional message to inject into volatile context
        message: Option<String>,
        /// Suppress all non-output logging; emit ONLY the model's
        /// response on stdout (suitable for piping to execution).
        #[arg(long)]
        quiet: bool,
    },
    /// Stop a specific agent
    Stop {
        /// Agent name
        agent: String,
    },
    /// Get leader status
    Status,
    /// Reload leader config
    Reload,
    /// Shutdown the leader
    Shutdown,
    /// Spawn a new agent dynamically
    Spawn {
        /// Agent name
        name: String,
        /// Input directories (comma separated)
        #[arg(short, long, value_delimiter = ',')]
        inputs: Vec<String>,
        /// Output directory (accepts single value; use --outputs for multiple)
        #[arg(short, long)]
        output: Option<String>,
        /// Additional output directories (comma separated)
        #[arg(long, value_delimiter = ',')]
        outputs: Vec<String>,
        /// System prompt directories (comma separated)
        #[arg(short, long, value_delimiter = ',')]
        system: Vec<String>,
        /// Model alias to use
        #[arg(short, long, default_value = "primary")]
        model: String,
        /// History limit (latest N turns)
        #[arg(short, long)]
        limit: Option<usize>,
        /// Optional streaming output directory
        #[arg(long)]
        stream_output: Option<String>,
        /// Optional tool output directory
        #[arg(long)]
        tool_output: Option<String>,
        /// When set, tool call content is hidden from output (consumed)
        #[arg(long, default_value_t = false)]
        consume_tool_calls: bool,
        /// Extra system prompt text
        #[arg(short, long)]
        prompt: Option<String>,

        // ── Sampling ──────────────────────────────────────────────
        /// Temperature (0.0–2.0)
        #[arg(long)]
        temperature: Option<f64>,
        /// Top-p nucleus sampling
        #[arg(long)]
        top_p: Option<f64>,
        /// Top-k sampling
        #[arg(long)]
        top_k: Option<usize>,
        /// Min-p sampling
        #[arg(long)]
        min_p: Option<f64>,
        /// Repetition penalty
        #[arg(long)]
        repetition_penalty: Option<f32>,
        /// Frequency penalty
        #[arg(long)]
        frequency_penalty: Option<f32>,
        /// Presence penalty
        #[arg(long)]
        presence_penalty: Option<f32>,
        /// Maximum output tokens (None = model default)
        #[arg(long)]
        max_len: Option<usize>,

        // ── Compression ───────────────────────────────────────────
        /// Compression threshold (0.0–1.0, default 0.5)
        #[arg(long)]
        compression_threshold: Option<f64>,
        /// Compression inverse probability (default 0.9)
        #[arg(long)]
        compression_inverse_prob: Option<f64>,
        /// Compression resummarize probability (default 0.1)
        #[arg(long)]
        compression_resummarize_prob: Option<f64>,

        // ── Other AgentConfig fields ──────────────────────────────
        /// Context checkpoint limit (char count before metasummary)
        #[arg(long)]
        context_checkpoint_limit: Option<usize>,
        /// Directories excluded from summarization (comma separated)
        #[arg(long, value_delimiter = ',')]
        excluded_from_summary: Vec<String>,
        /// Allowed file extensions for input (comma separated)
        #[arg(long, value_delimiter = ',')]
        allowed_extensions: Vec<String>,
        /// Enable realtime audio
        #[arg(long)]
        realtime_audio: bool,
        /// Enable tool usage (default true)
        #[arg(long, default_value_t = true)]
        tools_enabled: bool,
        /// Enable extended thinking / chain-of-thought
        #[arg(long)]
        enable_thinking: bool,
        /// Number of inference retries on error (default 3)
        #[arg(long)]
        inference_retries: Option<u32>,
        /// Delay between inference retries in ms (default 500)
        #[arg(long)]
        inference_retry_delay_ms: Option<u64>,
        /// Enable OOM recovery via aggressive recompression (default true)
        #[arg(long, default_value_t = true)]
        enable_oom_recovery: bool,
    },
    /// Print the full version string (including git commit hash)
    Version,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Check if we are already in the background
    if std::env::var("AGENTGRAPH_BACKGROUND").is_ok() {
        if let Commands::Leader { config, pending_command } = cli.command {
            let config_obj = Config::load(&config)?;
            let config_path = std::path::absolute(&config)
                .unwrap_or_else(|_| std::path::PathBuf::from(&config));
            let leader = Leader::new(config_obj, config_path.to_string_lossy().to_string()).await?;
            leader.run(pending_command).await?;
            return Ok(());
        }
    }

    match cli.command {
        Commands::Leader { config, .. } => {
            match is_leader_alive().await {
                LeaderStatus::Ready { .. } => {
                    let config_obj = Config::load(&config)?;
                    let cmd = Command::UpdateConfig(config_obj);
                    send_command(cmd).await?;
                }
                LeaderStatus::Degraded { pid } => {
                    eprintln!(
                        "Leader process is running (PID {}) but has no IPC socket. \
                         It may be loading models or in a degraded state. \
                         Check /tmp/agentgraph/leader.log.",
                        pid
                    );
                    std::process::exit(1);
                }
                LeaderStatus::NotRunning => {
                    spawn_background_leader(&config, None)?;
                    println!("Leader started in background. Logs: /tmp/agentgraph/leader.log");
                }
            }
        }
        Commands::Version => {
            println!("{}", agentgraph::version());
        }
        Commands::Status => {
            match is_leader_alive().await {
                LeaderStatus::Ready { .. } => {
                    send_command(Command::Status).await?;
                }
                LeaderStatus::Degraded { pid } => {
                    println!(
                        "Leader process detected (PID {}) but no IPC socket available. \
                         It may still be loading models or is in a degraded state. \
                         Check `/tmp/agentgraph/leader.log`.",
                        pid
                    );
                }
                LeaderStatus::NotRunning => {
                    println!("No leader is present.");
                }
            }
        }
        Commands::Shutdown => {
            match is_leader_alive().await {
                LeaderStatus::Ready { .. } => {
                    send_command(Command::Shutdown).await?;
                }
                LeaderStatus::Degraded { pid } => {
                    eprintln!(
                        "Leader process (PID {}) has no socket — cannot send shutdown. \
                         Kill it manually with `kill {}` if needed.",
                        pid, pid
                    );
                    std::process::exit(1);
                }
                LeaderStatus::NotRunning => {
                    println!("No leader is present.");
                }
            }
        }
        _ => {
            ensure_leader().await?;
            let cmd = match cli.command {
                Commands::Run { agent, message, quiet } => Command::RunAgent(agent, message, quiet),
                Commands::Stop { agent } => Command::StopAgent(agent),
                Commands::Reload => Command::ReloadConfig,
                Commands::Spawn {
                    name, inputs, output, outputs,
                    stream_output, tool_output, consume_tool_calls,
                    system, model, limit, prompt,
                    temperature, top_p, top_k, min_p,
                    repetition_penalty, frequency_penalty, presence_penalty, max_len,
                    compression_threshold, compression_inverse_prob, compression_resummarize_prob,
                    context_checkpoint_limit, excluded_from_summary,
                    allowed_extensions, realtime_audio,
                    tools_enabled, enable_thinking,
                    inference_retries, inference_retry_delay_ms,
                    enable_oom_recovery,
                    ..  // future-proof
                } => {
                    let mut output_dirs = output.map(|s| vec![s]).unwrap_or_default();
                    output_dirs.extend(outputs);
                    let config = AgentConfig {
                        inputs,
                        output: output_dirs,
                        stream_output,
                        tool_output,
                        consume_tool_calls,
                        system,
                        model,
                        history_limit: limit,
                        realtime_audio,
                        allowed_extensions,
                        prompt,
                        sampling: SamplingConfig {
                            temperature,
                            top_p,
                            top_k,
                            min_p,
                            repetition_penalty,
                            frequency_penalty,
                            presence_penalty,
                            max_len,
                        },
                        compression: CompressionConfig {
                            threshold: compression_threshold.unwrap_or(0.5),
                            inverse_probability: compression_inverse_prob.unwrap_or(0.9),
                            resummarize_probability: compression_resummarize_prob.unwrap_or(0.1),
                        },
                        context_checkpoint_limit,
                        excluded_from_summary,
                        tools_enabled,
                        enable_thinking,
                        inference_retries: inference_retries.unwrap_or(3),
                        inference_retry_delay_ms: inference_retry_delay_ms.unwrap_or(500),
                        enable_oom_recovery,
                        compression_db_path: None,
                    };
                    Command::SpawnAgent { name, config }
                }
                _ => unreachable!(),
            };
            send_command(cmd).await?;
        }
    }

    Ok(())
}

fn spawn_background_leader(config_path: &str, pending_command: Option<&str>) -> Result<()> {
    let exe = std::env::current_exe()?;
    let log_dir = std::path::Path::new("/tmp/agentgraph");
    if !log_dir.exists() {
        std::fs::create_dir_all(log_dir)?;
    }
    let log_path = log_dir.join("leader.log");
    let log_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)?;

    let mut cmd = std::process::Command::new(exe);
    cmd.arg("leader")
        .arg("--config")
        .arg(config_path)
        .env("AGENTGRAPH_BACKGROUND", "1")
        .stdout(std::process::Stdio::from(log_file.try_clone()?))
        .stderr(std::process::Stdio::from(log_file));
    if let Some(pc) = pending_command {
        cmd.arg("--pending-command").arg(pc);
    }
    cmd.spawn()?;
    Ok(())
}

async fn ensure_leader() -> Result<()> {
    match is_leader_alive().await {
        LeaderStatus::Ready { .. } => return Ok(()),
        LeaderStatus::Degraded { pid } => {
            return Err(anyhow!(
                "A leader process (PID {}) is running but has no IPC socket. \
                 It may be loading models. Check `/tmp/agentgraph/leader.log`.",
                pid
            ));
        }
        LeaderStatus::NotRunning => {}
    }

    spawn_background_leader("config.yaml", None)?;
    // Wait for leader to become ready (up to 90 s for model loading)
    for _ in 0..450 {
        tokio::time::sleep(Duration::from_millis(200)).await;
        match is_leader_alive().await {
            LeaderStatus::Ready { .. } => return Ok(()),
            LeaderStatus::Degraded { .. } => {
                // Process exists but no socket yet — keep waiting
            }
            LeaderStatus::NotRunning => {}
        }
    }
    Err(anyhow!(
        "Failed to start leader within 90s. \
         Check `/tmp/agentgraph/leader.log` for errors."
    ))
}

async fn send_command(cmd: Command) -> Result<()> {
    let max_retries = 3;
    for attempt in 0..=max_retries {
        let status = is_leader_alive().await;
        let socket_path = match status {
            LeaderStatus::Ready { socket, .. } => socket,
            LeaderStatus::Degraded { pid } => {
                if attempt < max_retries {
                    // Leader process exists but no socket yet — may still
                    // be loading models.  Retry.
                    tokio::time::sleep(Duration::from_secs(2)).await;
                    continue;
                }
                return Err(anyhow!(
                    "Leader process (PID {}) is running but has no IPC socket. \
                     It may still be loading models or is in a degraded state. \
                     Check `/tmp/agentgraph/leader.log`.",
                    pid
                ));
            }
            LeaderStatus::NotRunning => {
                return Err(anyhow!("No leader process found. Start one with `ag leader`."));
            }
        };

        let mut stream = match UnixStream::connect(&socket_path).await {
            Ok(s) => s,
            Err(e) => {
                if attempt < max_retries {
                    tokio::time::sleep(Duration::from_millis(1500)).await;
                    continue;
                }
                return Err(anyhow!("Failed to connect to leader socket: {}", e));
            }
        };

        let payload = serde_json::to_vec(&cmd)?;
        stream.write_all(&payload).await?;
        stream.flush().await?;

        // Shutdown writing so leader knows we're done sending
        stream.shutdown().await?;

        let mut resp = String::new();
        // Timeout the read: if the leader is busy with inference it may
        // take a while to respond.  30 s is generous for any IPC command.
        match timeout(Duration::from_secs(30), stream.read_to_string(&mut resp)).await {
            Ok(Ok(_)) => {}
            Ok(Err(_)) | Err(_) => {
                if attempt < max_retries {
                    tokio::time::sleep(Duration::from_millis(1500)).await;
                    continue;
                }
                return Err(anyhow!(
                    "Leader connection timed out (30s). The leader may be \
                     under heavy inference load. Check `/tmp/agentgraph/leader.log`."
                ));
            }
        }

        let resp = resp.trim().to_string();
        if resp == "RESTARTING" {
            // Leader is restarting with a new binary version.
            // Wait for new leader and retry.
            if attempt < max_retries {
                for _ in 0..100 {
                    tokio::time::sleep(Duration::from_millis(200)).await;
                    match is_leader_alive().await {
                        LeaderStatus::Ready { .. } => break,
                        LeaderStatus::Degraded { .. } => {
                            // Process present, keep waiting for socket
                        }
                        LeaderStatus::NotRunning => {}
                    }
                }
                continue;
            }
            return Err(anyhow!("Leader restart timed out after {} attempts", max_retries));
        }

        println!("{}", resp);
        return Ok(());
    }

    Err(anyhow!("Failed to send command after {} retries", max_retries))
}
