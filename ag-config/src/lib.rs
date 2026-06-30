use serde::{Deserialize, Deserializer, Serialize, de};
use std::collections::HashMap;

/// Controls how system prompts are placed in the final prompt array.
/// Values: "merged" (default), "frontloaded", "interleaved", "summarized".
#[derive(Debug, Clone, PartialEq)]
pub enum SystemPromptMode {
    /// All system content merged into one message placed first.
    /// (default, safest for templates requiring messages[0] == system)
    Merged,
    /// Multiple system messages, all first (turn_index 0, excluded from compression).
    Frontloaded,
    /// System messages appear wherever they were inserted; no reordering, no exclusion.
    Interleaved,
    /// System messages are ordinary history entries; they may be compressed.
    Summarized,
}

impl Default for SystemPromptMode {
    fn default() -> Self { SystemPromptMode::Merged }
}

impl Serialize for SystemPromptMode {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(match self {
            SystemPromptMode::Merged => "merged",
            SystemPromptMode::Frontloaded => "frontloaded",
            SystemPromptMode::Interleaved => "interleaved",
            SystemPromptMode::Summarized => "summarized",
        })
    }
}

impl<'de> Deserialize<'de> for SystemPromptMode {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?.to_lowercase();
        match s.as_str() {
            "merged" => Ok(SystemPromptMode::Merged),
            "frontloaded" => Ok(SystemPromptMode::Frontloaded),
            "interleaved" => Ok(SystemPromptMode::Interleaved),
            "summarized" => Ok(SystemPromptMode::Summarized),
            other => Err(de::Error::unknown_variant(other, &["merged", "frontloaded", "interleaved", "summarized"])),
        }
    }
}

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
    /// How system prompts are placed in the message array.
    /// "merged" (default), "frontloaded", "interleaved", "summarized".
    #[serde(default)]
    pub system_prompt_mode: SystemPromptMode,
    /// Override the auto-detected model loader type.
    /// For multimodal models, this maps to mistralrs `MultimodalLoaderType`
    /// (e.g., "gemma4", "phi3v", "qwen2vl").  When set, bypasses the
    /// HuggingFace config.json `architectures` field lookup, which is
    /// useful when HF renamed a class but the underlying architecture
    /// hasn't changed.
    #[serde(default)]
    pub loader_type: Option<String>,
    /// Enable online calibration for K-quant ISQ types.
    /// When true (default), the model will:
    ///   1. Load a previously-saved importance matrix if one exists
    ///      at ~/.config/agentgraph/imatrix/{alias}.cimatrix
    ///   2. Begin collecting activation statistics from live inference
    ///   3. Apply and save the collected imatrix when the model is
    ///      unloaded due to idle timeout
    /// Set to false to disable calibration entirely.
    #[serde(default = "default_calibration_enabled")]
    pub calibration_enabled: bool,
    /// Maximum sequence length for the auto device map.
    /// If unset, mistralrs defaults to 4096 for multimodal models.
    /// Set to a larger value (e.g. 32768) to allow longer context.
    #[serde(default)]
    pub max_seq_len: Option<usize>,
    /// Optional inference provider.  When set, this model uses the
    /// provider plugin instead of loading via mistralrs.  Agents that
    /// reference this model will automatically inherit the provider
    /// (unless the agent explicitly overrides with its own provider).
    #[serde(default)]
    pub provider: Option<ProviderConfig>,
}

fn default_max_num_seqs() -> usize {
    32
}

fn default_calibration_enabled() -> bool {
    true
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
    /// When set to a number of seconds, the leader will spawn a cold
    /// replacement process and exit after all models have been idle
    /// for at least this long.  The new leader starts with no model
    /// loaded, loading it on the first request (reducing idle VRAM
    /// consumption to zero between uses).
    #[serde(default)]
    pub shutdown_on_idle: Option<u64>,
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

/// Provider configuration for an agent.
///
/// When set, the agent uses an external inference provider (e.g. an
/// OpenAI-compatible API) instead of loading a model directly via
/// mistralrs.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "type")]
pub enum ProviderConfig {
    /// OpenAI-compatible HTTP API (ag-provider-openai crate).
    #[serde(rename = "openai")]
    OpenAi {
        /// Base URL of the OpenAI-compatible API.
        #[serde(default = "default_provider_api_base")]
        api_base: String,
        /// API key (empty for local servers that don't require auth).
        #[serde(default)]
        api_key: String,
        /// Model name to use in API requests.
        #[serde(default = "default_provider_model")]
        model: String,
        /// Shell command to start the server automatically.
        /// The provider will spawn this as a subprocess and wait for
        /// the API to become healthy before servicing requests.
        #[serde(default)]
        server_command: Option<String>,
        /// Override max sequence length (for context window management).
        #[serde(default)]
        max_seq_len: Option<usize>,
        /// Timeout in seconds waiting for the server to become healthy
        /// after spawning `server_command`.
        #[serde(default = "default_startup_timeout")]
        startup_timeout_secs: u64,
    },
    /// External plugin binary discovered via `config.plugins.provider-{name}`.
    /// The binary name is `ag-provider-{name}` (found on PATH).
    #[serde(rename = "plugin")]
    Plugin {
        /// Plugin name — maps to `ag-provider-{name}` on PATH.
        name: String,
        /// Override max sequence length.
        #[serde(default)]
        max_seq_len: Option<usize>,
    },
}

fn default_provider_api_base() -> String {
    "http://127.0.0.1:8338".into()
}

fn default_provider_model() -> String {
    "default".into()
}

fn default_startup_timeout() -> u64 {
    300
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
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
    /// Inference provider to use instead of loading a model via mistralrs.
    /// When set, `model` is still required (it controls agent config lookup),
    /// but the model is not loaded into memory — the provider serves inference.
    #[serde(default)]
    pub provider: Option<ProviderConfig>,
    /// When true, prepend file metadata (`[File: /path | Modified: ts | Size: N bytes]`)
    /// before each user message's content. When false (default), skip metadata.
    #[serde(default)]
    pub prepend_file_metadata: bool,
    /// Tool binaries to make available to this agent. Each entry is the
    /// full binary name (e.g. `ag-tool-bash`, `ag-tool-read`).
    /// An empty list means no tools are available at all.
    #[serde(default)]
    pub tools: Vec<String>,
    /// Heavy tool binaries.  When the model calls a tool whose binary
    /// name appears in this list, the leader offloads the call to a
    /// background process (unloading the model first), then respawns
    /// and resumes inference once the tool completes.  This avoids GPU
    /// memory contention between the LLM and tools that also need
    /// accelerator resources (e.g. image generation, video processing).
    #[serde(default)]
    pub heavy_tools: Vec<String>,
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
    /// Override the model-level system_prompt_mode.  When set, takes
    /// precedence over `ModelConfig.system_prompt_mode`.
    #[serde(default)]
    pub system_prompt_mode: Option<SystemPromptMode>,
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

#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq)]
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

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
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
