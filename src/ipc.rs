use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub enum Command {
    RunAgent(String),
    StopAgent(String),
    Status,
    ReloadConfig,
    Shutdown,
}
