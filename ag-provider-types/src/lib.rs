use async_trait::async_trait;
use futures::Stream;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Provider-generic role for chat messages.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Role {
    User,
    Assistant,
    System,
    Tool,
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Role::User => write!(f, "user"),
            Role::Assistant => write!(f, "assistant"),
            Role::System => write!(f, "system"),
            Role::Tool => write!(f, "tool"),
        }
    }
}

/// A tool call emitted by the model during streaming.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallDelta {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

/// A complete tool call (for persisting in messages).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

/// Tool definition sent to the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

impl ToolDef {
    pub fn new(name: impl Into<String>, description: impl Into<String>, parameters: serde_json::Value) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
        }
    }
}

/// How the model should decide to use tools.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ToolChoice {
    Auto,
    Required,
    None,
    Named(String),
}

/// Structured output / grammar constraint.
#[derive(Debug, Clone)]
pub enum GuidanceConstraint {
    JsonSchema(serde_json::Value),
    Regex(String),
    Lark(String),
    Llguidance(String),
}

/// Sampling parameters for inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingConfig {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<u32>,
    pub top_logprobs: Option<u32>,
    pub max_tokens: Option<u32>,
    pub stop: Option<Vec<String>>,
    pub frequency_penalty: Option<f64>,
    pub presence_penalty: Option<f64>,
    pub seed: Option<u64>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: None,
            top_p: None,
            top_k: None,
            top_logprobs: None,
            max_tokens: None,
            stop: None,
            frequency_penalty: None,
            presence_penalty: None,
            seed: None,
        }
    }
}

impl SamplingConfig {
    pub fn new(max_tokens: Option<u32>) -> Self {
        Self {
            max_tokens,
            ..Default::default()
        }
    }
}

/// A single message in a chat conversation, provider-agnostic.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
    pub tool_calls: Option<Vec<ToolCall>>,
    pub tool_call_id: Option<String>,
}

/// Image data for multimodal input.
#[derive(Debug, Clone)]
pub enum ImageSource {
    Base64(String, String),
    Url(String),
    Path(std::path::PathBuf),
}

/// Audio data for multimodal input.
#[derive(Debug, Clone)]
pub enum AudioSource {
    Base64(String, String),
    Url(String),
    Path(std::path::PathBuf),
}

/// Video data for multimodal input.
#[derive(Debug, Clone)]
pub enum VideoSource {
    Frames(Vec<ImageSource>),
    Path(std::path::PathBuf),
}

/// Full request to the inference provider.
#[derive(Debug, Clone)]
pub struct ChatRequest {
    pub messages: Vec<ChatMessage>,
    pub sampling: SamplingConfig,
    pub tools: Option<Vec<ToolDef>>,
    pub tool_choice: ToolChoice,
    pub constraint: Option<GuidanceConstraint>,
    pub enable_thinking: bool,
    pub images: Vec<ImageSource>,
    pub audios: Vec<AudioSource>,
    pub videos: Vec<VideoSource>,
    pub model: Option<String>,
}

impl Default for ChatRequest {
    fn default() -> Self {
        Self {
            messages: Vec::new(),
            sampling: SamplingConfig::default(),
            tools: None,
            tool_choice: ToolChoice::Auto,
            constraint: None,
            enable_thinking: false,
            images: Vec::new(),
            audios: Vec::new(),
            videos: Vec::new(),
            model: None,
        }
    }
}

/// Events emitted during a streaming chat response.
#[derive(Debug, Clone)]
pub enum ProviderEvent {
    /// A text delta chunk.
    Chunk(String),
    /// A reasoning/thinking delta chunk.
    Reasoning(String),
    /// A tool call delta (complete tool call, not streaming partial).
    ToolCall(ToolCallDelta),
    /// A non-fatal error that may trigger a retry.
    Error(String),
    /// The stream finished normally.
    Done,
}

/// Supported input modalities.
#[derive(Debug, Clone, PartialEq)]
pub enum Modality {
    Text,
    Image,
    Audio,
    Video,
}

/// Result of a provider diagnostics query.
#[derive(Debug, Clone)]
pub struct ProviderInfo {
    pub name: String,
    pub model: Option<String>,
    pub max_seq_len: Option<usize>,
    pub supports_modalities: Vec<Modality>,
    pub supports_tools: bool,
    pub supports_constraints: bool,
    pub supports_embeddings: bool,
}

/// The core abstraction for inference backends.
#[async_trait]
pub trait InferenceProvider: Send + Sync {
    /// Human-readable name for diagnostics.
    fn name(&self) -> &str;

    /// Metadata about this provider.
    async fn info(&self) -> ProviderInfo;

    /// Streaming chat completion.
    async fn stream_chat(
        &self,
        request: ChatRequest,
    ) -> anyhow::Result<Box<dyn Stream<Item = anyhow::Result<ProviderEvent>> + Send + Unpin>>;

    /// Non-streaming chat completion (used by compression system).
    async fn chat(&self, request: ChatRequest) -> anyhow::Result<String>;

    /// Generate an embedding vector for the given text.
    async fn embed(&self, text: &str) -> anyhow::Result<Vec<f32>>;

    /// Count tokens in the given text.
    async fn count_tokens(&self, text: &str) -> anyhow::Result<usize>;

    /// Health check — returns true if the provider is operational.
    async fn health_check(&self) -> bool;
}
