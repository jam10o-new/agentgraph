use serde::{Deserialize, Serialize};
use crate::config::AgentConfig;

#[derive(Debug, Serialize, Deserialize)]
pub enum Command {
    /// Run an agent, optionally injecting a message into volatile context.
    RunAgent(String, Option<String>),
    StopAgent(String),
    Status,
    ReloadConfig,
    Shutdown,
    SpawnAgent { name: String, config: AgentConfig },
}
