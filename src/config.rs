use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Config {
    pub models: HashMap<String, ModelConfig>,
    pub sampling: SamplingConfig,
    pub agents: HashMap<String, AgentConfig>,
    #[serde(default = "default_compression")]
    pub compression: CompressionConfig,
    #[serde(default)]
    pub shutdown_on_idle: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModelConfig {
    pub id: String,
    pub path: Option<String>,
    pub gguf: Option<String>,
    pub isq: Option<String>,
    pub dtype: Option<String>,
    pub builder: String,
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
    pub model: String,
    /// Latest N turns to load. 0 or None means unbound (all).
    pub history_limit: Option<usize>,
    pub stream: bool,
    #[serde(default)]
    pub allowed_extensions: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CompressionConfig {
    pub threshold: f64,
    pub inverse_probability: f64,
    pub resummarize_probability: f64,
}

fn default_compression() -> CompressionConfig {
    CompressionConfig {
        threshold: 0.5,
        inverse_probability: 0.9,
        resummarize_probability: 0.1,
    }
}

impl Config {
    pub fn load<P: AsRef<std::path::Path>>(path: P) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Config = serde_yaml::from_str(&content)?;
        Ok(config)
    }
}
