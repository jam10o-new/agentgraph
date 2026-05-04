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
    /// Primary output directories. The agent writes its response to the first
    /// directory, and reads all directories for assistant context. Other agents
    /// should watch the first directory for triggered downstream work.
    /// Accepts either a single string (backward-compatible) or a list of strings.
    #[serde(default, deserialize_with = "deserialize_output")]
    pub output: Vec<String>,
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
    /// Enable extended thinking / chain-of-thought for models that support it
    /// (e.g. Qwen3.5). Defaults to false because thinking mode can interfere
    /// with streaming output reliability.
    #[serde(default)]
    pub enable_thinking: bool,
    /// Number of times to retry inference on recoverable errors (OOMs, timeouts).
    /// Defaults to 3. Set to 0 to disable retries.
    /// Each retry includes the previously accumulated content as an assistant prefill,
    /// so the model picks up where it left off rather than restarting from scratch.
    #[serde(default = "default_inference_retries")]
    pub inference_retries: u32,
    /// Delay in milliseconds between inference retry attempts.
    /// Defaults to 500ms. Larger values help OOM recovery as VRAM may take time to free.
    #[serde(default = "default_inference_retry_delay_ms")]
    pub inference_retry_delay_ms: u64,
}

fn default_inference_retries() -> u32 {
    3
}

fn default_inference_retry_delay_ms() -> u64 {
    500
}

fn default_true() -> bool {
    true
}

fn deserialize_output<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct OutputVisitor;
    impl<'de> serde::de::Visitor<'de> for OutputVisitor {
        type Value = Vec<String>;
        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("a string or a list of strings")
        }
        fn visit_str<E>(self, value: &str) -> Result<Vec<String>, E> {
            Ok(vec![value.to_string()])
        }
        fn visit_string<E>(self, value: String) -> Result<Vec<String>, E> {
            Ok(vec![value])
        }
        fn visit_seq<A>(self, mut seq: A) -> Result<Vec<String>, A::Error>
        where
            A: serde::de::SeqAccess<'de>,
        {
            let mut vec = Vec::new();
            while let Some(elem) = seq.next_element::<String>()? {
                vec.push(elem);
            }
            Ok(vec)
        }
    }
    deserializer.deserialize_any(OutputVisitor)
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
