use agentgraph::config::Config;
use agentgraph::leader::Leader;
use agentgraph::ipc::Command;
use agentgraph::find_leader_socket;
use clap::{Parser, Subcommand};
use anyhow::{Result, anyhow};
use tokio::io::AsyncWriteExt;
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
    /// Run a specific agent once
    Run {
        /// Agent name
        agent: String,
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
        _ => {
            let cmd = match cli.command {
                Commands::Run { agent } => Command::RunAgent(agent),
                Commands::Stop { agent } => Command::StopAgent(agent),
                Commands::Status => Command::Status,
                Commands::Reload => Command::ReloadConfig,
                Commands::Shutdown => Command::Shutdown,
                _ => unreachable!(),
            };

            let socket_path = find_leader_socket().await.ok_or_else(|| anyhow!("Leader not found"))?;
            let mut stream = UnixStream::connect(socket_path).await?;
            let payload = serde_json::to_vec(&cmd)?;
            stream.write_all(&payload).await?;
            stream.flush().await?;
            println!("Command sent to leader.");
        }
    }

    Ok(())
}
