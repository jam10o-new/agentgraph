use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModelConfig {
    pub id: String,
    pub path: Option<String>,
    pub gguf: Option<String>,
    pub isq: Option<String>,
    pub dtype: Option<String>,
    pub builder: String,
    pub chat_template: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ApiConfig {
    pub enabled: bool,
    #[serde(default = "default_bind_address")]
    pub bind_address: String,
    #[serde(default = "default_port")]
    pub port: u16,
}

fn default_bind_address() -> String {
    "127.0.0.1".to_string()
}

fn default_port() -> u16 {
    3000
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Config {
    pub models: HashMap<String, ModelConfig>,
    pub agents: HashMap<String, AgentConfig>,
    #[serde(default)]
    pub shutdown_on_idle: bool,
    #[serde(default)]
    pub api: Option<ApiConfig>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AgentConfig {
    pub inputs: Vec<String>,
    /// Primary output directory. Written only at turn completion (or interrupt).
    /// Other agents should watch this directory for triggered downstream work.
    pub output: String,
    /// Optional streaming output directory. When set, assistant tokens are
    /// written live to this directory for human display, in addition to the
    /// primary `output` directory receiving the final result.
    #[serde(default)]
    pub stream_output: Option<String>,
    /// Optional directory to write tool results into. When set, tool outputs
    /// are persisted here instead of the primary output directory, keeping
    /// downstream agent inputs clean from tool noise.
    #[serde(default)]
    pub tool_output: Option<String>,
    pub system: Vec<String>,
    pub model: String,
    /// Latest N turns to load. 0 or None means unbound (all).
    pub history_limit: Option<usize>,
    #[serde(default)]
    pub realtime_audio: bool,
    #[serde(default)]
    pub allowed_extensions: Vec<String>,
    pub prompt: Option<String>,
    #[serde(default)]
    pub sampling: SamplingConfig,
    /// Per-agent compression settings.
    #[serde(default = "default_compression")]
    pub compression: CompressionConfig,
    /// When the total character count of compressed history exceeds this value, a metasummary checkpoint is triggered.
    /// 0 or None disables checkpointing.
    #[serde(default)]
    pub context_checkpoint_limit: Option<usize>,
    /// Input directories whose files should never be compressed or folded into metasummaries.
    /// Useful for mutable/ephemeral inputs that must always reach the model verbatim.
    #[serde(default)]
    pub excluded_from_summary: Vec<String>,
    /// Whether this agent is allowed to use tools. Defaults to true.
    #[serde(default = "default_true")]
    pub tools_enabled: bool,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
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
pub struct CompressionConfig {
    pub threshold: f64,
    pub inverse_probability: f64,
    pub resummarize_probability: f64,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        CompressionConfig {
            threshold: 0.5,
            inverse_probability: 0.9,
            resummarize_probability: 0.1,
        }
    }
}

fn default_compression() -> CompressionConfig {
    CompressionConfig::default()
}

impl Config {
    pub fn load<P: AsRef<std::path::Path>>(path: P) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Config = serde_yaml::from_str(&content)?;
        Ok(config)
    }
}
