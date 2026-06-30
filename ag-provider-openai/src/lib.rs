use ag_provider_types::{
    ChatRequest, GuidanceConstraint, InferenceProvider, Modality, ProviderEvent,
    ProviderInfo, Role, ToolCallDelta, ToolChoice, ToolDef,
};
use anyhow::{Context, Result, anyhow};
use futures::Stream;
use serde::{Deserialize, Serialize};
use std::process::Stdio;
use std::time::Duration;
use tokio::process::{Child, Command};
use tokio::sync::Mutex;
use tokio::time::sleep;

// ---------------------------------------------------------------------------
// OpenAI wire-format types
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<Message>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenaiTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
    // Extended / vLLM-style guided decoding
    #[serde(skip_serializing_if = "Option::is_none", rename = "guided_json")]
    guided_json: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "guided_regex")]
    guided_regex: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "guided_grammar")]
    guided_grammar: Option<String>,
    // Reasoning effort
    #[serde(skip_serializing_if = "Option::is_none", rename = "reasoning_effort")]
    reasoning_effort: Option<String>,
}

#[derive(Debug, Serialize)]
struct Message {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenaiToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Debug, Serialize)]
struct OpenaiTool {
    #[serde(rename = "type")]
    type_: String,
    function: OpenaiFunction,
}

#[derive(Debug, Serialize)]
struct OpenaiFunction {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct OpenaiToolCall {
    id: String,
    #[serde(rename = "type")]
    type_: String,
    function: OpenaiFunctionCall,
}

#[derive(Debug, Serialize)]
struct OpenaiFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionChunk {
    #[allow(dead_code)]
    id: String,
    choices: Vec<ChunkChoice>,
}

#[derive(Debug, Deserialize)]
struct ChunkChoice {
    #[serde(default)]
    delta: ChunkDelta,
    #[allow(dead_code)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct ChunkDelta {
    #[serde(default)]
    role: Option<String>,
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ChunkToolCall>>,
}

#[derive(Debug, Deserialize)]
struct ChunkToolCall {
    index: i64,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    r#type: Option<String>,
    #[serde(default)]
    function: Option<ChunkFunction>,
}

#[derive(Debug, Deserialize, Default)]
struct ChunkFunction {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<NonStreamChoice>,
}

#[derive(Debug, Deserialize)]
struct NonStreamChoice {
    message: ResponseMessage,
}

#[derive(Debug, Deserialize)]
struct ResponseMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ResponseToolCall>>,
}

#[derive(Debug, Deserialize)]
struct ResponseToolCall {
    id: String,
    #[serde(rename = "type")]
    type_: String,
    function: ResponseFunction,
}

#[derive(Debug, Deserialize)]
struct ResponseFunction {
    name: String,
    arguments: String,
}

#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f64>,
}

#[derive(Debug, Serialize)]
struct EmbeddingRequest {
    model: String,
    input: String,
}

#[derive(Debug, Deserialize)]
struct ModelListResponse {
    data: Vec<ModelInfo>,
}

#[derive(Debug, Deserialize)]
struct ModelInfo {
    id: String,
}

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

pub struct OpenAiProviderConfig {
    pub api_base: String,
    pub api_key: String,
    pub model: String,
    pub max_seq_len: Option<usize>,
    /// Optional shell command to spawn the server (e.g. `mistralrs server`).
    pub server_command: Option<String>,
    /// How long to wait for the server to be ready (seconds).
    pub startup_timeout_secs: u64,
}

impl Default for OpenAiProviderConfig {
    fn default() -> Self {
        Self {
            api_base: "http://127.0.0.1:8338".into(),
            api_key: String::new(),
            model: "default".into(),
            max_seq_len: None,
            server_command: None,
            startup_timeout_secs: 300,
        }
    }
}

pub struct OpenAiProvider {
    config: OpenAiProviderConfig,
    client: reqwest::Client,
    server_process: Mutex<Option<Child>>,
}

impl OpenAiProvider {
    pub fn new(config: OpenAiProviderConfig) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(600))
            .build()
            .expect("reqwest client");
        Self {
            config,
            client,
            server_process: Mutex::new(None),
        }
    }

    /// Start the managed server subprocess (if `server_command` is set) and
    /// wait for it to become healthy.
    pub async fn start(&self) -> Result<()> {
        if let Some(cmd) = &self.config.server_command {
            let mut proc = Command::new("sh")
                .arg("-c")
                .arg(cmd)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .kill_on_drop(true)
                .spawn()
                .context("failed to spawn server process")?;

            let deadline = tokio::time::Instant::now()
                + Duration::from_secs(self.config.startup_timeout_secs);

            let mut logged = false;
            loop {
                if tokio::time::Instant::now() > deadline {
                    return Err(anyhow!("server did not become ready within timeout"));
                }
                if let Ok(Some(status)) = proc.try_wait() {
                    let stderr = match proc.stderr.take() {
                        Some(mut s) => {
                            let mut buf = String::new();
                            tokio::io::AsyncReadExt::read_to_string(&mut s, &mut buf).await.ok();
                            buf
                        }
                        None => String::new(),
                    };
                    return Err(anyhow!(
                        "server process exited during startup (status: {}): {}",
                        status,
                        stderr.trim()
                    ));
                }
                if self.health_check().await {
                    tracing::info!("OpenAI provider server is ready");
                    break;
                }
                if !logged {
                    tracing::info!("waiting for OpenAI provider server to become ready …");
                    logged = true;
                }
                sleep(Duration::from_secs(2)).await;
            }

            *self.server_process.lock().await = Some(proc);
        }
        Ok(())
    }

    /// Gracefully shut down the managed server subprocess.
    pub async fn shutdown(&self) {
        if let Some(mut proc) = self.server_process.lock().await.take() {
            _ = proc.start_kill();
        }
    }

    // ---- helpers ----

    fn auth_req(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        if self.config.api_key.is_empty() {
            req
        } else {
            req.header("Authorization", format!("Bearer {}", self.config.api_key))
        }
    }

    fn api_url(&self, path: &str) -> String {
        format!("{}{}", self.config.api_base.trim_end_matches('/'), path)
    }

    fn build_messages(&self, request: &ChatRequest) -> Vec<Message> {
        let mut out: Vec<Message> = Vec::new();
        for msg in &request.messages {
            let openai_role = match msg.role {
                Role::User => "user",
                Role::Assistant => "assistant",
                Role::System => "system",
                Role::Tool => "tool",
            };
            let mut m = Message {
                role: openai_role.to_string(),
                content: Some(serde_json::Value::String(msg.content.clone())),
                tool_calls: None,
                tool_call_id: msg.tool_call_id.clone(),
            };
            if let Some(tcs) = &msg.tool_calls {
                let calls: Vec<OpenaiToolCall> = tcs
                    .iter()
                    .map(|tc| OpenaiToolCall {
                        id: tc.id.clone(),
                        type_: "function".into(),
                        function: OpenaiFunctionCall {
                            name: tc.name.clone(),
                            arguments: tc.arguments.clone(),
                        },
                    })
                    .collect();
                m.tool_calls = Some(calls);
            }
            out.push(m);
        }
        out
    }

    fn map_tool_choice(&self, choice: &ToolChoice) -> Option<serde_json::Value> {
        match choice {
            ToolChoice::Auto => None,
            ToolChoice::Required => Some(serde_json::Value::String("required".into())),
            ToolChoice::None => Some(serde_json::Value::String("none".into())),
            ToolChoice::Named(name) => Some(serde_json::json!({"type": "function", "function": {"name": name}})),
        }
    }

    fn map_tools(&self, tools: &[ToolDef]) -> Vec<OpenaiTool> {
        tools
            .iter()
            .map(|t| OpenaiTool {
                type_: "function".into(),
                function: OpenaiFunction {
                    name: t.name.clone(),
                    description: t.description.clone(),
                    parameters: t.parameters.clone(),
                },
            })
            .collect()
    }

    fn map_constraint(&self, constraint: &GuidanceConstraint) -> ChatCompletionRequestMapper {
        let mut m = ChatCompletionRequestMapper::default();
        match constraint {
            GuidanceConstraint::JsonSchema(schema) => {
                m.guided_json = Some(schema.clone());
            }
            GuidanceConstraint::Regex(re) => {
                m.guided_regex = Some(re.clone());
            }
            GuidanceConstraint::Lark(grammar) => {
                m.guided_grammar = Some(grammar.clone());
            }
            GuidanceConstraint::Llguidance(_s) => {
                // llguidance binary constraints are not portable over HTTP;
                // silently ignore.
                tracing::warn!("llguidance constraints are not supported by the OpenAI provider");
            }
        }
        m
    }

    async fn stream_chat_inner(
        &self,
        request: ChatRequest,
    ) -> Result<Box<dyn Stream<Item = Result<ProviderEvent>> + Send + Unpin>> {
        let body = self.build_chat_request(&request, true)?;

        let resp = self
            .auth_req(self.client.post(self.api_url("/v1/chat/completions")))
            .json(&body)
            .send()
            .await
            .context("OpenAI chat request failed")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(anyhow!("OpenAI returned HTTP {}: {}", status, text));
        }

        let (tx, rx) = tokio::sync::mpsc::channel::<Result<ProviderEvent>>(64);
        let byte_stream = resp.bytes_stream();

        tokio::spawn(Self::drain_sse(byte_stream, tx));

        Ok(Box::new(tokio_stream::wrappers::ReceiverStream::new(rx)))
    }

    async fn drain_sse(
        mut byte_stream: impl futures::Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send + Unpin + 'static,
        tx: tokio::sync::mpsc::Sender<Result<ProviderEvent>>,
    ) {
        use futures::StreamExt;
        let mut buf: Vec<u8> = Vec::new();

        loop {
            // Try to extract a complete line from the buffer
            if let Some(pos) = buf.iter().position(|&b| b == b'\n') {
                let line: Vec<u8> = buf.drain(..=pos).collect();
                let line_str = String::from_utf8_lossy(&line);
                let trimmed = line_str.trim();
                if let Some(payload) = trimmed.strip_prefix("data: ") {
                    let payload = payload.trim();
                    if payload == "[DONE]" {
                        break;
                    }
                    let event = parse_sse_line(payload);
                    if tx.send(event).await.is_err() {
                        break;
                    }
                }
                continue;
            }

            // Read more bytes
            match byte_stream.next().await {
                Some(Ok(chunk)) => buf.extend_from_slice(&chunk),
                Some(Err(e)) => {
                    _ = tx.send(Err(anyhow!("stream error: {}", e))).await;
                    break;
                }
                None => break,
            }
        }

        // Flush remaining buffer
        if !buf.is_empty() {
            let line_str = String::from_utf8_lossy(&buf);
            let trimmed = line_str.trim();
            if let Some(payload) = trimmed.strip_prefix("data: ") {
                let payload = payload.trim();
                if payload != "[DONE]" {
                    let event = parse_sse_line(payload);
                    _ = tx.send(event).await;
                }
            }
        }
    }

    async fn chat_inner(&self, request: ChatRequest) -> Result<String> {
        let body = self.build_chat_request(&request, false)?;

        let resp = self
            .auth_req(self.client.post(self.api_url("/v1/chat/completions")))
            .json(&body)
            .send()
            .await
            .context("OpenAI non-streaming chat request failed")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(anyhow!("OpenAI returned HTTP {}: {}", status, text));
        }

        let data: ChatCompletionResponse = resp
            .json()
            .await
            .context("failed to parse non-streaming response")?;

        data.choices
            .into_iter()
            .next()
            .and_then(|c| c.message.content)
            .ok_or_else(|| anyhow!("no content in non-streaming response"))
    }

    fn build_chat_request(
        &self,
        request: &ChatRequest,
        stream: bool,
    ) -> Result<ChatCompletionRequest> {
        let mut guided_json = None;
        let mut guided_regex = None;
        let mut guided_grammar = None;
        if let Some(constraint) = &request.constraint {
            let m = self.map_constraint(constraint);
            guided_json = m.guided_json;
            guided_regex = m.guided_regex;
            guided_grammar = m.guided_grammar;
        }

        let reasoning_effort = if request.enable_thinking {
            Some("high".to_string())
        } else {
            None
        };

        let tools = request.tools.as_ref().map(|t| self.map_tools(t));
        let tool_choice = self.map_tool_choice(&request.tool_choice);

        Ok(ChatCompletionRequest {
            model: request.model.clone().unwrap_or_else(|| self.config.model.clone()),
            messages: self.build_messages(request),
            stream,
            temperature: request.sampling.temperature,
            top_p: request.sampling.top_p,
            max_tokens: request.sampling.max_tokens,
            stop: request.sampling.stop.clone(),
            frequency_penalty: request.sampling.frequency_penalty,
            presence_penalty: request.sampling.presence_penalty,
            seed: request.sampling.seed,
            tools,
            tool_choice,
            guided_json,
            guided_regex,
            guided_grammar,
            reasoning_effort,
        })
    }
}

#[derive(Default)]
struct ChatCompletionRequestMapper {
    guided_json: Option<serde_json::Value>,
    guided_regex: Option<String>,
    guided_grammar: Option<String>,
}

// ---------------------------------------------------------------------------
// SSE parsing
// ---------------------------------------------------------------------------

fn parse_sse_line(line: &str) -> Result<ProviderEvent> {
    let chunk: ChatCompletionChunk =
        serde_json::from_str(line).context("failed to parse SSE JSON")?;

    for choice in chunk.choices {
        let delta = choice.delta;

        // Reasoning content (DeepSeek-style, some mistralrs models)
        if let Some(reasoning) = delta.reasoning_content {
            if !reasoning.is_empty() {
                return Ok(ProviderEvent::Reasoning(reasoning));
            }
        }

        // Text content
        if let Some(content) = delta.content {
            if !content.is_empty() {
                return Ok(ProviderEvent::Chunk(content));
            }
        }

        // Tool calls
        if let Some(tcs) = delta.tool_calls {
            for tc in tcs {
                if let Some(id) = tc.id {
                    if let Some(func) = tc.function {
                        return Ok(ProviderEvent::ToolCall(ToolCallDelta {
                            id,
                            name: func.name.unwrap_or_default(),
                            arguments: func.arguments.unwrap_or_default(),
                        }));
                    }
                }
            }
        }
    }

    // If we didn't match anything, emit minimal chunk or empty.
    // This can happen for the final chunk that only has a finish_reason.
    Ok(ProviderEvent::Chunk(String::new()))
}

// ---------------------------------------------------------------------------
// InferenceProvider impl
// ---------------------------------------------------------------------------

#[async_trait::async_trait]
impl InferenceProvider for OpenAiProvider {
    fn name(&self) -> &str {
        "openai"
    }

    async fn info(&self) -> ProviderInfo {
        // Try to fetch model list
        let models = self
            .auth_req(self.client.get(self.api_url("/v1/models")))
            .send()
            .await;

        let model_name = if let Ok(resp) = models {
            if let Ok(list) = resp.json::<ModelListResponse>().await {
                list.data.into_iter().map(|m| m.id).next()
            } else {
                None
            }
        } else {
            None
        };

        ProviderInfo {
            name: self.name().to_string(),
            model: model_name.or(Some(self.config.model.clone())),
            max_seq_len: self.config.max_seq_len,
            supports_modalities: vec![Modality::Text],
            supports_tools: true,
            supports_constraints: true,
            supports_embeddings: true,
        }
    }

    async fn stream_chat(
        &self,
        request: ChatRequest,
    ) -> Result<Box<dyn Stream<Item = Result<ProviderEvent>> + Send + Unpin>> {
        self.stream_chat_inner(request).await
    }

    async fn chat(&self, request: ChatRequest) -> Result<String> {
        self.chat_inner(request).await
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let body = EmbeddingRequest {
            model: self.config.model.clone(),
            input: text.to_string(),
        };

        let resp = self
            .auth_req(self.client.post(self.api_url("/v1/embeddings")))
            .json(&body)
            .send()
            .await
            .context("embedding request failed")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(anyhow!("embedding HTTP {}: {}", status, text));
        }

        let data: EmbeddingResponse = resp
            .json()
            .await
            .context("failed to parse embedding response")?;

        data.data
            .into_iter()
            .next()
            .map(|d| d.embedding.into_iter().map(|v| v as f32).collect())
            .ok_or_else(|| anyhow!("no embedding data returned"))
    }

    async fn count_tokens(&self, text: &str) -> Result<usize> {
        // Fallback: rough character-based estimate (~4 chars per token for English).
        // Providers that expose a tokenize endpoint could do better.
        Ok((text.len() + 3) / 4)
    }

    async fn health_check(&self) -> bool {
        self.auth_req(self.client.get(self.api_url("/v1/models")))
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false)
    }
}

impl Drop for OpenAiProvider {
    fn drop(&mut self) {
        // Best-effort kill; we can't await in drop.
        if let Ok(mut guard) = self.server_process.try_lock() {
            if let Some(ref mut proc) = *guard {
                _ = proc.start_kill();
            }
        }
    }
}
