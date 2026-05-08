use serde::{Deserialize, Serialize};
use crate::config::AgentConfig;

#[derive(Debug, Serialize, Deserialize)]
pub enum Command {
    /// Run an agent, optionally injecting a message into volatile context.
    /// When `quiet` is true, the output stream contains only the model's
    /// response text so it can be piped directly to other commands.
    RunAgent(String, Option<String>, bool),
    StopAgent(String),
    Status,
    ReloadConfig,
    Shutdown,
    SpawnAgent { name: String, config: AgentConfig },
    UpdateConfig(crate::config::Config),
}
