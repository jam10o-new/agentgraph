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
    #[serde(default = "default_max_num_seqs")]
    pub max_num_seqs: usize,
}

fn default_max_num_seqs() -> usize {
    32
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ApiConfig {
    #[serde(default)]
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

/// Telegram bot configuration — consumed by the `ag-api-telegram` binary.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TelegramConfig {
    /// Whether the Telegram bot should be spawned.
    #[serde(default)]
    pub enabled: bool,
    /// Telegram bot token from @BotFather.
    pub bot_token: String,
    /// Default agent to use for chats that don't have a per-chat override.
    #[serde(default = "default_telegram_agent")]
    pub default_agent: String,
    /// Per-user agent overrides.  Key is the Telegram user ID (as a string).
    /// Falls back to `default_agent` when a chat has no explicit override.
    #[serde(default)]
    pub user_agents: HashMap<String, String>,
    /// Per-group agent overrides.  Key is the Telegram group ID.
    #[serde(default)]
    pub group_agents: HashMap<String, String>,
    /// Per-channel agent overrides.  Key is the Telegram channel ID.
    #[serde(default)]
    pub channel_agents: HashMap<String, String>,
    /// Comma-separated list of allowed user IDs. When non-empty, only these
    /// users may interact with the bot.  Empty = allow all.
    #[serde(default)]
    pub allowed_users: Vec<String>,
}

fn default_telegram_agent() -> String {
    "api".to_string()
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Config {
    pub models: HashMap<String, ModelConfig>,
    pub agents: HashMap<String, AgentConfig>,
    #[serde(default)]
    pub shutdown_on_idle: bool,
    #[serde(default)]
    pub model_idle_secs: Option<u64>,

    /// Dynamic API plugin configurations.  Keys following the pattern
    /// `api-*` (e.g. `api-http`, `api-telegram`) are treated as API
    /// frontend sections.  Each value is an opaque YAML node that is
    /// serialized verbatim and passed to the corresponding binary via
    /// `--section`.  This lets third-party API plugins to be discovered
    /// without any changes to the leader or shared config types.
    #[serde(flatten)]
    pub plugins: HashMap<String, serde_yaml::Value>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AgentConfig {
    pub inputs: Vec<String>,
    #[serde(default, deserialize_with = "deserialize_output")]
    pub output: Vec<String>,
    #[serde(default)]
    pub stream_output: Option<String>,
    #[serde(default)]
    pub tool_output: Option<String>,
    pub system: Vec<String>,
    pub model: String,
    pub history_limit: Option<usize>,
    #[serde(default)]
    pub realtime_audio: bool,
    #[serde(default)]
    pub allowed_extensions: Vec<String>,
    pub prompt: Option<String>,
    #[serde(default)]
    pub sampling: SamplingConfig,
    #[serde(default = "default_compression")]
    pub compression: CompressionConfig,
    #[serde(default)]
    pub context_checkpoint_limit: Option<usize>,
    #[serde(default)]
    pub compression_db_path: Option<String>,
    #[serde(default)]
    pub excluded_from_summary: Vec<String>,
    #[serde(default = "default_true")]
    pub tools_enabled: bool,
    #[serde(default)]
    pub consume_tool_calls: bool,
    #[serde(default = "default_true")]
    pub enable_oom_recovery: bool,
    #[serde(default)]
    pub enable_thinking: bool,
    #[serde(default = "default_inference_retries")]
    pub inference_retries: u32,
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
