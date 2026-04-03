use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Config {
    pub models: HashMap<String, ModelConfig>,
    pub sampling: SamplingConfig,
    pub agents: HashMap<String, AgentConfig>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModelConfig {
    pub id: String,
    pub path: Option<String>,
    pub gguf: Option<String>,
    pub isq: Option<String>, // "4", "8", etc.
    pub dtype: Option<String>, // "f32", "f16", "bf16"
    pub builder: String, // "vision", "gguf", "audio"
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SamplingConfig {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    pub min_p: Option<f64>,
    pub repetition_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub max_len: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AgentConfig {
    pub path: String,
    pub primary_model: String,
    pub secondary_model: Option<String>,
    pub max_runtime_secs: Option<u64>,
    pub stream: bool,
}

impl Config {
    pub fn load<P: AsRef<std::path::Path>>(path: P) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Config = serde_yaml::from_str(&content)?;
        Ok(config)
    }
}
