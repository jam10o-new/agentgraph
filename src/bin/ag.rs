use agentgraph::config::{Config, AgentConfig};
use agentgraph::leader::Leader;
use agentgraph::ipc::Command;
use agentgraph::find_leader_socket;
use clap::{Parser, Subcommand};
use anyhow::{Result, anyhow};
use tokio::io::{AsyncWriteExt, AsyncReadExt};
use tokio::net::UnixStream;

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
    },
    /// Run a specific agent turn, optionally injecting a message
    Run {
        /// Agent name
        agent: String,
        /// Optional message to inject into volatile context
        message: Option<String>,
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
        /// Output directory
        #[arg(short, long)]
        output: String,
        /// System prompt directories (comma separated)
        #[arg(short, long, value_delimiter = ',')]
        system: Vec<String>,
        /// Model alias to use
        #[arg(short, long, default_value = "primary")]
        model: String,
        /// History limit (latest N)
        #[arg(short, long)]
        limit: Option<usize>,
    /// Optional streaming output directory
    #[arg(short, long)]
    stream_output: Option<String>,
        /// Extra system prompt
        #[arg(short, long)]
        prompt: Option<String>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Check if we are already in the background
    if std::env::var("AGENTGRAPH_BACKGROUND").is_ok() {
        if let Commands::Leader { config } = cli.command {
            let config = Config::load(config)?;
            let leader = Leader::new(config).await?;
            leader.run().await?;
            return Ok(());
        }
    }

    match cli.command {
        Commands::Leader { config } => {
            if find_leader_socket().await.is_some() {
                let config_obj = Config::load(&config)?;
                let cmd = Command::UpdateConfig(config_obj);
                send_command(cmd).await?;
            } else {
                spawn_background_leader(&config)?;
                println!("Leader started in background. Logs: /tmp/agentgraph/leader.log");
            }
        }
        Commands::Status => {
            if find_leader_socket().await.is_none() {
                println!("No leader is present.");
            } else {
                send_command(Command::Status).await?;
            }
        }
        Commands::Shutdown => {
            if find_leader_socket().await.is_some() {
                send_command(Command::Shutdown).await?;
            } else {
                println!("No leader is present.");
            }
        }
        _ => {
            ensure_leader().await?;
            let cmd = match cli.command {
                Commands::Run { agent, message } => Command::RunAgent(agent, message),
                Commands::Stop { agent } => Command::StopAgent(agent),
                Commands::Reload => Command::ReloadConfig,
                Commands::Spawn { name, inputs, output, stream_output, system, model, limit, prompt, .. } => {
                    let config = AgentConfig {
                        inputs,
                        output,
                        stream_output,
                        system,
                        model,
                        history_limit: limit,
                        allowed_extensions: vec![],
                        realtime_audio: false,
                        prompt,
                        sampling: Default::default(),
                        compression: Default::default(),
                        context_checkpoint_limit: None,
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

fn spawn_background_leader(config_path: &str) -> Result<()> {
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

    std::process::Command::new(exe)
        .arg("leader")
        .arg("--config")
        .arg(config_path)
        .env("AGENTGRAPH_BACKGROUND", "1")
        .stdout(std::process::Stdio::from(log_file.try_clone()?))
        .stderr(std::process::Stdio::from(log_file))
        .spawn()?;
    Ok(())
}

async fn ensure_leader() -> Result<()> {
    if find_leader_socket().await.is_none() {
        spawn_background_leader("config.yaml")?;
        // Wait a bit for it to start
        for _ in 0..300 {
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
            if find_leader_socket().await.is_some() {
                return Ok(());
            }
        }
        return Err(anyhow!("Failed to start leader in background within 60s"));
    }
    Ok(())
}

async fn send_command(cmd: Command) -> Result<()> {
    let socket_path = find_leader_socket().await.ok_or_else(|| anyhow!("Leader not found. Is it running?"))?;
    let mut stream = UnixStream::connect(socket_path).await?;
    let payload = serde_json::to_vec(&cmd)?;
    stream.write_all(&payload).await?;
    stream.flush().await?;
    
    // Shutdown writing so leader knows we're done sending
    stream.shutdown().await?;

    let mut resp = String::new();
    stream.read_to_string(&mut resp).await?;
    println!("{}", resp);
    Ok(())
}
