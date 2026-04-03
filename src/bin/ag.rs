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
        /// Enable streaming
        #[arg(short, long, default_value_t = true)]
        stream: bool,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Leader { config } => {
            let config = Config::load(config)?;
            let leader = Leader::new(config).await?;
            leader.run().await?;
        }
        Commands::Spawn { name, inputs, output, system, model, limit, stream } => {
            let config = AgentConfig {
                inputs,
                output,
                system,
                model,
                history_limit: limit,
                stream,
                allowed_extensions: vec![],
            };
            let cmd = Command::SpawnAgent { name, config };
            send_command(cmd).await?;
        }
        _ => {
            let cmd = match cli.command {
                Commands::Run { agent, message } => Command::RunAgent(agent, message),
                Commands::Stop { agent } => Command::StopAgent(agent),
                Commands::Status => Command::Status,
                Commands::Reload => Command::ReloadConfig,
                Commands::Shutdown => Command::Shutdown,
                _ => unreachable!(),
            };
            send_command(cmd).await?;
        }
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
