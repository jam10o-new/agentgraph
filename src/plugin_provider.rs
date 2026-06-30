use crate::inference_provider::{
    ChatMessage as IpChatMessage, ChatRequest as IpChatRequest, InferenceProvider,
    ProviderEvent, ProviderInfo, ToolCallDelta, ToolChoice, ToolDef,
};
use ag_provider_common::{
    self as proto, ChatRequest, Constraint, ContentPart, CountTokensRequest, EmbedRequest,
    Request, RequestKind, ToolCall,
};
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use futures::Stream;
use std::path::PathBuf;
use std::process::Stdio;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

/// Inference provider that delegates to a plugin binary (`ag-provider-*`).
///
/// Protocol: binary is spawned per-request.  A single JSON request is
/// written to stdin, then streaming type-prefixed lines are read from
/// stdout (see [`ag_provider_common`] for the wire format).
pub struct PluginProvider {
    binary_path: PathBuf,
    max_seq_len: Option<usize>,
}

impl PluginProvider {
    pub fn new(binary_path: PathBuf, max_seq_len: Option<usize>) -> Self {
        Self {
            binary_path,
            max_seq_len,
        }
    }

    /// Spawn the binary, send a request, and return a stream of response events.
    async fn spawn_and_stream(
        &self,
        kind: RequestKind,
    ) -> Result<tokio::sync::mpsc::Receiver<Result<proto::ProviderEvent>>> {
        let request = Request { kind, request_id: 0 };
        let input = serde_json::to_string(&request)?;

        let mut child = Command::new(&self.binary_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| anyhow!("failed to spawn {}: {}", self.binary_path.display(), e))?;

        // Write request to stdin
        if let Some(mut child_stdin) = child.stdin.take() {
            child_stdin.write_all(input.as_bytes()).await?;
            drop(child_stdin);
        }

        let (tx, rx) = tokio::sync::mpsc::channel(64);
        let stdout = child.stdout.take().expect("piped stdout");

        // Spawn a reader task: read lines from stdout, parse events, send to channel
        tokio::spawn(async move {
            use tokio::io::AsyncBufReadExt;
            let reader = tokio::io::BufReader::new(stdout);
            let mut lines = reader.lines();

            loop {
                match lines.next_line().await {
                    Ok(Some(line)) => {
                        match proto::parse_event_line(&line) {
                            Ok(Some(ev)) => {
                                let is_terminal = matches!(
                                    &ev,
                                    proto::ProviderEvent::Done | proto::ProviderEvent::Error(_)
                                );
                                if tx.send(Ok(ev)).await.is_err() {
                                    break;
                                }
                                if is_terminal {
                                    break;
                                }
                            }
                            Ok(None) => {} // blank line, skip
                            Err(e) => {
                                let _ = tx.send(Err(anyhow!("parse: {}", e))).await;
                                break;
                            }
                        }
                    }
                    Ok(None) => break, // EOF
                    Err(e) => {
                        let _ = tx.send(Err(anyhow!("read: {}", e))).await;
                        break;
                    }
                }
            }

            // Wait for the process to exit (collect zombie)
            let _ = child.wait().await;
        });

        Ok(rx)
    }

    /// Send a request and collect all text content from the response stream.
    async fn send_and_collect_text(&self, kind: RequestKind) -> Result<String> {
        let mut rx = self.spawn_and_stream(kind).await?;
        let mut output = String::new();

        while let Some(ev) = rx.recv().await {
            match ev {
                Ok(proto::ProviderEvent::Chunk(t)) => output.push_str(&t),
                Ok(proto::ProviderEvent::Reasoning(t)) => output.push_str(&t),
                Ok(proto::ProviderEvent::Done) => return Ok(output),
                Ok(proto::ProviderEvent::Error(e)) => return Err(anyhow!("plugin error: {}", e)),
                _ => {} // tool calls, etc.
            }
        }
        Ok(output)
    }
}

// ── Message mapping (crate types → wire types) ────────────────────────

fn map_chat_messages(messages: &[IpChatMessage]) -> Vec<proto::ChatMessage> {
    messages
        .iter()
        .map(|m| {
            let tool_calls = m.tool_calls.as_ref().map(|tcs| {
                tcs.iter()
                    .map(|tc| ToolCall {
                        id: tc.id.clone(),
                        name: tc.name.clone(),
                        arguments: tc.arguments.clone(),
                    })
                    .collect()
            });
            proto::ChatMessage {
                role: m.role.to_string(),
                content: vec![ContentPart::Text {
                    text: m.content.clone(),
                }],
                tool_calls,
                tool_call_id: m.tool_call_id.clone(),
            }
        })
        .collect()
}

fn map_tools(tools: &[ToolDef]) -> Vec<proto::ToolDef> {
    tools
        .iter()
        .map(|t| proto::ToolDef {
            name: t.name.clone(),
            description: t.description.clone(),
            parameters: t.parameters.clone(),
        })
        .collect()
}

fn map_constraint(request: &IpChatRequest) -> Option<Constraint> {
    request.constraint.as_ref().map(|c| match c {
        crate::inference_provider::GuidanceConstraint::JsonSchema(schema) => Constraint {
            schema: Some(schema.clone()),
            regex: None,
            grammar: None,
        },
        crate::inference_provider::GuidanceConstraint::Regex(re) => Constraint {
            schema: None,
            regex: Some(re.clone()),
            grammar: None,
        },
        crate::inference_provider::GuidanceConstraint::Lark(g) => Constraint {
            schema: None,
            regex: None,
            grammar: Some(g.clone()),
        },
        crate::inference_provider::GuidanceConstraint::Llguidance(_) => Constraint {
            schema: None,
            regex: None,
            grammar: None,
        },
    })
}

fn map_tool_choice(choice: &ToolChoice) -> Option<serde_json::Value> {
    match choice {
        ToolChoice::Auto => None,
        ToolChoice::Required => Some(serde_json::Value::String("required".into())),
        ToolChoice::None => Some(serde_json::Value::String("none".into())),
        ToolChoice::Named(name) => {
            Some(serde_json::json!({"type": "function", "function": {"name": name}}))
        }
    }
}

fn map_modality(s: &str) -> crate::inference_provider::Modality {
    match s {
        "text" => crate::inference_provider::Modality::Text,
        "image" => crate::inference_provider::Modality::Image,
        "audio" => crate::inference_provider::Modality::Audio,
        "video" => crate::inference_provider::Modality::Video,
        _ => crate::inference_provider::Modality::Text,
    }
}

fn info_from_proto(info: &proto::InfoResponse) -> ProviderInfo {
    ProviderInfo {
        name: info.name.clone(),
        model: info.model.clone(),
        max_seq_len: info.max_seq_len,
        supports_tools: info.supports_tools,
        supports_embeddings: info.supports_embeddings,
        supports_constraints: info.supports_constraints,
        supports_modalities: info
            .supports_modalities
            .iter()
            .map(|s| map_modality(s))
            .collect(),
    }
}

fn fallback_info() -> ProviderInfo {
    ProviderInfo {
        name: "plugin".into(),
        model: None,
        max_seq_len: None,
        supports_tools: true,
        supports_embeddings: true,
        supports_constraints: true,
        supports_modalities: vec![crate::inference_provider::Modality::Text],
    }
}

// ── InferenceProvider impl ────────────────────────────────────────────

#[async_trait]
impl InferenceProvider for PluginProvider {
    fn name(&self) -> &str {
        "plugin"
    }

    async fn info(&self) -> ProviderInfo {
        match self.spawn_and_stream(RequestKind::Info).await {
            Ok(mut rx) => {
        while let Some(ev) = rx.recv().await {
                    match ev {
                        Ok(proto::ProviderEvent::Info(info)) => return info_from_proto(&info),
                        Ok(proto::ProviderEvent::Error(e)) => {
                            eprintln!("plugin info error: {}", e);
                            return fallback_info();
                        }
                        _ => {}
                    }
                }
                fallback_info()
            }
            Err(_) => fallback_info(),
        }
    }

    async fn stream_chat(
        &self,
        request: IpChatRequest,
    ) -> Result<Box<dyn Stream<Item = Result<ProviderEvent>> + Send + Unpin>> {
        let chat_req = ChatRequest {
            messages: map_chat_messages(&request.messages),
            tools: request.tools.as_ref().map(|t| map_tools(t)),
            tool_choice: map_tool_choice(&request.tool_choice),
            constraint: map_constraint(&request),
            enable_thinking: request.enable_thinking,
            model: request.model.clone(),
            max_tokens: request.sampling.max_tokens,
            temperature: request.sampling.temperature,
            top_p: request.sampling.top_p,
        };

        let mut rx = self
            .spawn_and_stream(RequestKind::Chat(chat_req))
            .await?;

        let (tx, out_rx) = tokio::sync::mpsc::channel(64);
        tokio::spawn(async move {
            while let Some(ev) = rx.recv().await {
                let is_done = matches!(&ev, Ok(proto::ProviderEvent::Done));
                let mapped = match ev {
                    Ok(proto::ProviderEvent::Chunk(t)) => Ok(ProviderEvent::Chunk(t)),
                    Ok(proto::ProviderEvent::Reasoning(t)) => Ok(ProviderEvent::Reasoning(t)),
                    Ok(proto::ProviderEvent::ToolCall(tc)) => {
                        Ok(ProviderEvent::ToolCall(ToolCallDelta {
                            id: tc.id,
                            name: tc.name,
                            arguments: tc.arguments,
                        }))
                    }
                    Ok(proto::ProviderEvent::Done) => Ok(ProviderEvent::Done),
                    Ok(proto::ProviderEvent::Error(e)) => {
                        Err(anyhow!("plugin: {}", e))
                    }
                    Ok(other) => {
                        // Non-chat event in a chat stream — ignore
                        continue;
                    }
                    Err(e) => Err(e),
                };
                if tx.send(mapped).await.is_err() {
                    break;
                }
                if is_done {
                    break;
                }
            }
        });

        Ok(Box::new(tokio_stream::wrappers::ReceiverStream::new(out_rx)))
    }

    async fn chat(&self, request: IpChatRequest) -> Result<String> {
        let chat_req = ChatRequest {
            messages: map_chat_messages(&request.messages),
            tools: request.tools.as_ref().map(|t| map_tools(t)),
            tool_choice: map_tool_choice(&request.tool_choice),
            constraint: map_constraint(&request),
            enable_thinking: request.enable_thinking,
            model: request.model.clone(),
            max_tokens: request.sampling.max_tokens,
            temperature: request.sampling.temperature,
            top_p: request.sampling.top_p,
        };

        self.send_and_collect_text(RequestKind::Chat(chat_req))
            .await
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let req = EmbedRequest {
            text: text.to_string(),
        };
        let mut rx = self.spawn_and_stream(RequestKind::Embed(req)).await?;

        while let Some(ev) = rx.recv().await {
            match ev {
                Ok(proto::ProviderEvent::Embedding(emb)) => return Ok(emb),
                Ok(proto::ProviderEvent::Error(e)) => {
                    return Err(anyhow!("plugin embed error: {}", e))
                }
                _ => {}
            }
        }
        Err(anyhow!("no embedding in plugin response"))
    }

    async fn count_tokens(&self, text: &str) -> Result<usize> {
        let req = CountTokensRequest {
            text: text.to_string(),
        };
        let mut rx = self.spawn_and_stream(RequestKind::CountTokens(req)).await?;

        while let Some(ev) = rx.recv().await {
            match ev {
                Ok(proto::ProviderEvent::TokenCount(n)) => return Ok(n),
                Ok(proto::ProviderEvent::Error(e)) => {
                    return Err(anyhow!("plugin count_tokens error: {}", e))
                }
                _ => {}
            }
        }
        Ok(text.len() / 4)
    }

    async fn health_check(&self) -> bool {
        let mut rx = match self.spawn_and_stream(RequestKind::Health).await {
            Ok(r) => r,
            Err(_) => return false,
        };

        while let Some(ev) = rx.recv().await {
            match ev {
                Ok(proto::ProviderEvent::Healthy(h)) => return h,
                _ => {}
            }
        }
        false
    }
}
