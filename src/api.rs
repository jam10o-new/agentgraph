use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Sse, sse::Event},
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::convert::Infallible;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::fs;
use tokio::sync::Mutex;
use tokio::time::{sleep, timeout};
use tokio_stream::wrappers::ReceiverStream;
use tower_http::cors::CorsLayer;

use crate::config::Config;

/// Metadata for a node in the conversation tree.
pub struct NodeMeta {
    pub dir: PathBuf,
    pub role: String,
    pub parent_hash: Option<String>,
}

/// A tree of conversation states for a specific agent. Each state represents
/// a prefix of the message history and has its own directory. States with
/// shared prefixes share directories.
pub struct SessionTree {
    pub temp_dir: Arc<tempfile::TempDir>,
    pub nodes: Mutex<HashMap<String, NodeMeta>>,
}

pub struct ApiState {
    pub config: Arc<Mutex<Config>>,
    /// The loaded Mistral.rs model. When `None` the API falls back to writing into
    /// the agent's configured directories directly (used in tests and when no model
    /// is configured).
    pub model: Option<Arc<mistralrs::Model>>,
    /// Per-agent conversation trees for API requests. Each agent/model gets its
    /// own temp workspace and directory-sharing graph.
    pub trees: Mutex<HashMap<String, Arc<SessionTree>>>,
}

pub fn router(state: Arc<ApiState>) -> Router {
    Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

/// Cumulative hash for a conversation state. Two requests with the same
/// message prefix will produce the same hash chain, allowing them to share
/// on-disk directories.
fn hash_state(parent_hash: &str, role: &str, content: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    parent_hash.hash(&mut hasher);
    role.hash(&mut hasher);
    content.hash(&mut hasher);
    format!("{:x}", hasher.finish())
}

// ---------- OpenAI-compatible request/response types ----------

#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    stream: bool,
}

#[derive(Debug, Deserialize, Clone)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct ChatCompletionResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<Choice>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Choice {
    index: usize,
    message: MessageResponse,
    finish_reason: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct MessageResponse {
    role: String,
    content: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct ChatCompletionChunk {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<ChunkChoice>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ChunkChoice {
    index: usize,
    delta: Delta,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Default)]
struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ModelsListResponse {
    object: String,
    data: Vec<ModelObject>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ModelObject {
    id: String,
    object: String,
    created: u64,
    owned_by: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    context_window: Option<usize>,
}

// ---------- Handlers ----------

async fn list_models(
    State(state): State<Arc<ApiState>>,
) -> Result<Json<ModelsListResponse>, StatusCode> {
    let config = state.config.lock().await;
    let created = SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let data: Vec<ModelObject> = config
        .agents
        .iter()
        .map(|(name, agent_config)| ModelObject {
            id: name.clone(),
            object: "model".to_string(),
            created,
            owned_by: "agentgraph".to_string(),
            context_window: None, // Virtual agents have no fixed token cap; automatic summarization adapts context
        })
        .collect();

    Ok(Json(ModelsListResponse {
        object: "list".to_string(),
        data,
    }))
}

async fn chat_completions(
    State(state): State<Arc<ApiState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<impl IntoResponse, StatusCode> {
    let config_guard = state.config.lock().await;
    let base_agent_config = config_guard
        .agents
        .get(&req.model)
        .ok_or(StatusCode::NOT_FOUND)?
        .clone();
    let global_config = config_guard.clone();
    drop(config_guard);

    let start_time = SystemTime::now();

    let id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let created = start_time
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // ------------------------------------------------------------------
    //  Session-tree path: shared conversation directories across requests.
    // ------------------------------------------------------------------
    if let Some(ref model) = state.model {
        // Get or create the session tree for this model.
        let tree = {
            let mut trees = state.trees.lock().await;
            trees
                .entry(req.model.clone())
                .or_insert_with(|| {
                    Arc::new(SessionTree {
                        temp_dir: Arc::new(
                            tempfile::tempdir()
                                .expect("Failed to create API session temp dir"),
                        ),
                        nodes: Mutex::new(HashMap::new()),
                    })
                })
                .clone()
        };

        // Walk the message history, creating shared nodes where needed.
        let mut current_hash = String::new();
        let mut system_msgs: Vec<String> = Vec::new();
        let mut latest_user_msg = String::new();

        for msg in &req.messages {
            if msg.role == "system" {
                system_msgs.push(msg.content.clone());
                continue;
            }
            let parent_hash = current_hash.clone();
            current_hash = hash_state(&parent_hash, &msg.role, &msg.content);
            let mut nodes = tree.nodes.lock().await;
            if !nodes.contains_key(&current_hash) {
                let dir = tree
                    .temp_dir
                    .path()
                    .join(format!("{}-{}", msg.role, &current_hash[..16]));
                fs::create_dir_all(&dir)
                    .await
                    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
                let path = dir.join("msg.txt");
                fs::write(&path, &msg.content)
                    .await
                    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
                nodes.insert(
                    current_hash.clone(),
                    NodeMeta {
                        dir,
                        role: msg.role.clone(),
                        parent_hash: if parent_hash.is_empty() {
                            None
                        } else {
                            Some(parent_hash)
                        },
                    },
                );
            }
            if msg.role == "user" {
                latest_user_msg = msg.content.clone();
            }
        }

        // Collect all user dirs and assistant dirs from the path.
        let mut user_dirs: Vec<String> = Vec::new();
        let mut assistant_dirs: Vec<String> = Vec::new();
        {
            let nodes = tree.nodes.lock().await;
            let mut hash = current_hash.clone();
            while !hash.is_empty() {
                if let Some(node) = nodes.get(&hash) {
                    match node.role.as_str() {
                        "user" => user_dirs.push(node.dir.to_string_lossy().to_string()),
                        "assistant" => {
                            assistant_dirs.push(node.dir.to_string_lossy().to_string())
                        }
                        _ => {}
                    }
                    hash = node.parent_hash.clone().unwrap_or_default();
                } else {
                    break;
                }
            }
        }
        user_dirs.reverse();
        assistant_dirs.reverse();

        // Create a fresh response directory for this request.
        let response_hash = hash_state(&current_hash, "assistant", "");
        let response_dir = tree
            .temp_dir
            .path()
            .join(format!("assistant-{}", &response_hash[..16]));
        fs::create_dir_all(&response_dir)
            .await
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

        // Per-request stream/tools/system dirs.
        let api_stream = tree
            .temp_dir
            .path()
            .join(format!("stream-{}", uuid::Uuid::new_v4()));
        let api_tools = tree
            .temp_dir
            .path()
            .join(format!("tools-{}", uuid::Uuid::new_v4()));
        let api_system = tree
            .temp_dir
            .path()
            .join(format!("system-{}", uuid::Uuid::new_v4()));

        fs::create_dir_all(&api_stream)
            .await
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        fs::create_dir_all(&api_tools)
            .await
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        fs::create_dir_all(&api_system)
            .await
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

        for (idx, sys_msg) in system_msgs.iter().enumerate() {
            let path = api_system.join(format!("sys-{:02}.txt", idx));
            fs::write(&path, sys_msg)
                .await
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        }

        // Build isolated agent config.
        let mut isolated_config = base_agent_config.clone();
        isolated_config.inputs = user_dirs.clone();
        let mut output_dirs = vec![response_dir.to_string_lossy().to_string()];
        output_dirs.extend(assistant_dirs);
        isolated_config.output = output_dirs;
        isolated_config.stream_output = Some(api_stream.to_string_lossy().to_string());
        isolated_config.tool_output = Some(api_tools.to_string_lossy().to_string());
        isolated_config.system = if system_msgs.is_empty() {
            base_agent_config.system.clone()
        } else {
            let mut s = vec![api_system.to_string_lossy().to_string()];
            s.extend(base_agent_config.system.iter().cloned());
            s
        };

        let model = model.clone();
        let sampling = mistralrs::SamplingParams {
            temperature: isolated_config.sampling.temperature,
            top_p: isolated_config.sampling.top_p,
            top_k: isolated_config.sampling.top_k,
            min_p: isolated_config.sampling.min_p,
            repetition_penalty: isolated_config.sampling.repetition_penalty,
            frequency_penalty: isolated_config.sampling.frequency_penalty,
            presence_penalty: isolated_config.sampling.presence_penalty,
            max_len: isolated_config.sampling.max_len,
            top_n_logprobs: 0,
            stop_toks: None,
            logits_bias: None,
            n_choices: 1,
            dry_params: None,
        };

        let agent_name = format!("api-{}-{}", req.model, uuid::Uuid::new_v4());
        let agent = crate::Agent::new(agent_name, isolated_config, global_config, model, sampling);

        let handle = tokio::spawn(async move {
            if let Err(e) = agent.run_loop().await {
                eprintln!("API agent loop error: {:?}", e);
            }
        });

        sleep(Duration::from_millis(150)).await;

        if latest_user_msg.is_empty() {
            handle.abort();
            return Err(StatusCode::BAD_REQUEST);
        }

        // Write trigger to the last user dir.
        let trigger_dir = if user_dirs.is_empty() {
            &response_dir
        } else {
            &PathBuf::from(user_dirs.last().unwrap())
        };
        let trigger_path = trigger_dir.join(format!(
            "api-latest-{}.txt",
            SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis()
        ));
        fs::write(&trigger_path, &latest_user_msg)
            .await
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

        let output_path = response_dir.to_string_lossy().to_string();
        let stream_path = api_stream.to_string_lossy().to_string();
        let model_name = req.model;

        if req.stream {
            let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(100);

            tokio::spawn(async move {
                let result = stream_from_dir(
                    &id,
                    &model_name,
                    &stream_path,
                    &output_path,
                    start_time,
                    tx.clone(),
                )
                .await;

                if let Err(e) = result {
                    let err_json = serde_json::json!({"error": e.to_string()}).to_string();
                    let _ = tx.send(Ok(Event::default().data(err_json))).await;
                }

                handle.abort();
            });

            Ok(Sse::new(ReceiverStream::new(rx)).into_response())
        } else {
            let content = wait_for_output(Some(output_path), start_time)
                .await
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

            handle.abort();

            // Cache the response in the session tree so future requests
            // with this assistant turn can reuse the directory.
            let response_hash = hash_state(&current_hash, "assistant", &content);
            let mut nodes = tree.nodes.lock().await;
            nodes.insert(
                response_hash,
                NodeMeta {
                    dir: response_dir,
                    role: "assistant".to_string(),
                    parent_hash: if current_hash.is_empty() {
                        None
                    } else {
                        Some(current_hash)
                    },
                },
            );

            Ok(Json(ChatCompletionResponse {
                id,
                object: "chat.completion".to_string(),
                created,
                model: model_name,
                choices: vec![Choice {
                    index: 0,
                    message: MessageResponse {
                        role: "assistant".to_string(),
                        content,
                    },
                    finish_reason: "stop".to_string(),
                }],
            })
            .into_response())
        }
    } else {
        // ------------------------------------------------------------------
        //  Fallback: no model available (tests / no-model config).
        //  Writes directly into the configured agent directories.
        // ------------------------------------------------------------------
        let input_dir = base_agent_config
            .inputs
            .first()
            .ok_or(StatusCode::BAD_REQUEST)?;

        let latest_user_msg = req
            .messages
            .iter()
            .rev()
            .find(|m| m.role == "user")
            .map(|m| m.content.clone())
            .ok_or(StatusCode::BAD_REQUEST)?;

        let input_file = PathBuf::from(input_dir).join(format!(
            "api-{}.txt",
            start_time
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis()
        ));
        let _ = fs::create_dir_all(input_dir).await;
        fs::write(&input_file, &latest_user_msg)
            .await
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

        if req.stream {
            let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(100);
            let model_name = req.model;

            tokio::spawn(async move {
                match wait_for_output(base_agent_config.output.first().cloned(), start_time).await {
                    Ok(content) => {
                        let chunk = ChatCompletionChunk {
                            id: id.clone(),
                            object: "chat.completion.chunk".to_string(),
                            created,
                            model: model_name.clone(),
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: Delta {
                                    role: Some("assistant".to_string()),
                                    content: Some(content),
                                    ..Default::default()
                                },
                                finish_reason: None,
                            }],
                        };
                        let data = serde_json::to_string(&chunk).unwrap_or_default();
                        let _ = tx.send(Ok(Event::default().data(data))).await;

                        let finish_chunk = ChatCompletionChunk {
                            id,
                            object: "chat.completion.chunk".to_string(),
                            created,
                            model: model_name,
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: Delta::default(),
                                finish_reason: Some("stop".to_string()),
                            }],
                        };
                        let data = serde_json::to_string(&finish_chunk).unwrap_or_default();
                        let _ = tx.send(Ok(Event::default().data(data))).await;
                        let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
                    }
                    Err(e) => {
                        let err_json = serde_json::json!({"error": e.to_string()}).to_string();
                        let _ = tx.send(Ok(Event::default().data(err_json))).await;
                    }
                }
            });

            Ok(Sse::new(ReceiverStream::new(rx)).into_response())
        } else {
            let content = wait_for_output(base_agent_config.output.first().cloned(), start_time)
                .await
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

            Ok(Json(ChatCompletionResponse {
                id,
                object: "chat.completion".to_string(),
                created,
                model: req.model,
                choices: vec![Choice {
                    index: 0,
                    message: MessageResponse {
                        role: "assistant".to_string(),
                        content,
                    },
                    finish_reason: "stop".to_string(),
                }],
            })
            .into_response())
        }
    }
}

// ---------- Helpers ----------

async fn wait_for_output(
    output_dir_maybe: Option<String>,
    after: SystemTime,
) -> anyhow::Result<String> {
    if let Some(output_dir) = output_dir_maybe {
        let output_path = PathBuf::from(output_dir);
        let result = timeout(Duration::from_secs(120), async {
            loop {
                sleep(Duration::from_millis(100)).await;

                let mut candidates = Vec::new();
                if let Ok(mut entries) = fs::read_dir(&output_path).await {
                    while let Ok(Some(entry)) = entries.next_entry().await {
                        let path = entry.path();
                        if path.is_file() {
                            if let Ok(metadata) = entry.metadata().await {
                                let modified =
                                    metadata.modified().or_else(|_| metadata.created())?;
                                if modified >= after {
                                    candidates.push((modified, path));
                                }
                            }
                        }
                    }
                }

                if let Some((_, path)) = candidates.into_iter().max_by_key(|(t, _)| *t) {
                    // Give the agent a moment to finish flushing.
                    sleep(Duration::from_millis(200)).await;
                    return anyhow::Result::Ok(fs::read_to_string(&path).await?);
                }
            }
        })
        .await;

        result.map_err(|_| anyhow::anyhow!("Timeout waiting for agent output"))?
    } else {
        Err(anyhow::anyhow!("No non-streaming output set"))?
    }
}

async fn stream_from_dir(
    id: &str,
    model: &str,
    stream_dir: &str,
    output_dir: &str,
    after: SystemTime,
    tx: tokio::sync::mpsc::Sender<Result<Event, Infallible>>,
) -> anyhow::Result<()> {
    let stream_path = wait_for_stream_file(stream_dir, after).await?;

    // The agent names both stream and output files with the same `out-{timestamp}.txt`
    // prefix, so we can derive the matching output file from the stream file stem.
    let filename = stream_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    let timestamp_str = filename.strip_prefix("out-").unwrap_or("");
    let output_file = PathBuf::from(output_dir).join(format!("out-{}.txt", timestamp_str));

    let created = SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs();

    // Send the role chunk.
    let role_chunk = ChatCompletionChunk {
        id: id.to_string(),
        object: "chat.completion.chunk".to_string(),
        created,
        model: model.to_string(),
        choices: vec![ChunkChoice {
            index: 0,
            delta: Delta {
                role: Some("assistant".to_string()),
                ..Default::default()
            },
            finish_reason: None,
        }],
    };
    let data = serde_json::to_string(&role_chunk)?;
    tx.send(Ok(Event::default().data(data))).await.ok();

    let mut last_pos = 0;
    loop {
        if output_file.exists() {
            // Inference is complete — flush any remaining stream content.
            if let Ok(content) = fs::read_to_string(&stream_path).await {
                if content.len() > last_pos {
                    let new_content = &content[last_pos..];
                    let chunk = ChatCompletionChunk {
                        id: id.to_string(),
                        object: "chat.completion.chunk".to_string(),
                        created,
                        model: model.to_string(),
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: Delta {
                                content: Some(new_content.to_string()),
                                ..Default::default()
                            },
                            finish_reason: None,
                        }],
                    };
                    let data = serde_json::to_string(&chunk)?;
                    tx.send(Ok(Event::default().data(data))).await.ok();
                }
            }

            // Send finish chunk.
            let finish_chunk = ChatCompletionChunk {
                id: id.to_string(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model.to_string(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: Delta::default(),
                    finish_reason: Some("stop".to_string()),
                }],
            };
            let data = serde_json::to_string(&finish_chunk)?;
            tx.send(Ok(Event::default().data(data))).await.ok();
            tx.send(Ok(Event::default().data("[DONE]"))).await.ok();
            break;
        }

        if let Ok(content) = fs::read_to_string(&stream_path).await {
            if content.len() > last_pos {
                let new_content = &content[last_pos..];
                let chunk = ChatCompletionChunk {
                    id: id.to_string(),
                    object: "chat.completion.chunk".to_string(),
                    created,
                    model: model.to_string(),
                    choices: vec![ChunkChoice {
                        index: 0,
                        delta: Delta {
                            content: Some(new_content.to_string()),
                            ..Default::default()
                        },
                        finish_reason: None,
                    }],
                };
                let data = serde_json::to_string(&chunk)?;
                tx.send(Ok(Event::default().data(data))).await.ok();
                last_pos = content.len();
            }
        }

        sleep(Duration::from_millis(50)).await;
    }

    Ok(())
}

async fn wait_for_stream_file(stream_dir: &str, after: SystemTime) -> anyhow::Result<PathBuf> {
    let stream_path = PathBuf::from(stream_dir);
    let result = timeout(Duration::from_secs(120), async {
        loop {
            sleep(Duration::from_millis(100)).await;

            let mut candidates = Vec::new();
            if let Ok(mut entries) = fs::read_dir(&stream_path).await {
                while let Ok(Some(entry)) = entries.next_entry().await {
                    let path = entry.path();
                    if path.is_file() {
                        if let Ok(metadata) = entry.metadata().await {
                            let modified = metadata.modified().or_else(|_| metadata.created())?;
                            if modified >= after {
                                candidates.push((modified, path));
                            }
                        }
                    }
                }
            }

            if let Some((_, path)) = candidates.into_iter().max_by_key(|(t, _)| *t) {
                return anyhow::Result::Ok(path);
            }
        }
    })
    .await;

    result.map_err(|_| anyhow::anyhow!("Timeout waiting for stream file"))?
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use http_body_util::BodyExt;
    use std::collections::HashMap;
    use std::time::Duration;
    use tempfile::TempDir;
    use tokio::time::sleep;
    use tower::ServiceExt;

    async fn setup_test_dirs(
        input_dir: &std::path::Path,
        output_dir: &std::path::Path,
        stream_dir: Option<&std::path::Path>,
    ) {
        let _ = fs::create_dir_all(input_dir).await;
        let _ = fs::create_dir_all(output_dir).await;
        if let Some(d) = stream_dir {
            let _ = fs::create_dir_all(d).await;
        }
    }

    fn make_test_config(
        input_dir: &std::path::Path,
        output_dir: &std::path::Path,
        stream_dir: Option<&std::path::Path>,
    ) -> Config {
        let mut agents = HashMap::new();
        agents.insert(
            "test_agent".to_string(),
            crate::config::AgentConfig {
                inputs: vec![input_dir.to_string_lossy().to_string()],
                output: vec![output_dir.to_string_lossy().to_string()],
                stream_output: stream_dir.map(|p| p.to_string_lossy().to_string()),
                tool_output: None,
                system: vec![],
                model: "primary".to_string(),
                history_limit: None,
                realtime_audio: false,
                allowed_extensions: vec![],
                prompt: None,
                sampling: Default::default(),
                compression: Default::default(),
                context_checkpoint_limit: Some(10000),
                excluded_from_summary: vec![],
                tools_enabled: true,
                enable_thinking: false,
                inference_retries: 3,
                inference_retry_delay_ms: 500,
            },
        );

        crate::config::Config {
            models: HashMap::new(),
            agents,
            shutdown_on_idle: false,
            api: None,
        }
    }

    #[tokio::test]
    async fn test_list_models() {
        let temp_dir = TempDir::new().unwrap();
        let input_dir = temp_dir.path().join("input");
        let output_dir = temp_dir.path().join("output");

        setup_test_dirs(&input_dir, &output_dir, None).await;

        let config = Arc::new(Mutex::new(make_test_config(&input_dir, &output_dir, None)));
        let state = Arc::new(ApiState {
            config,
            model: None,
            trees: Mutex::new(HashMap::new()),
        });
        let app = router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = response.into_body().collect().await.unwrap().to_bytes();
        let models: ModelsListResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(models.object, "list");
        assert_eq!(models.data.len(), 1);
        assert_eq!(models.data[0].id, "test_agent");
        assert_eq!(models.data[0].context_window, None);
    }

    #[tokio::test]
    async fn test_chat_completions_non_streaming() {
        let temp_dir = TempDir::new().unwrap();
        let input_dir = temp_dir.path().join("input");
        let output_dir = temp_dir.path().join("output");

        setup_test_dirs(&input_dir, &output_dir, None).await;

        let config = Arc::new(Mutex::new(make_test_config(&input_dir, &output_dir, None)));
        let state = Arc::new(ApiState {
            config,
            model: None,
            trees: Mutex::new(HashMap::new()),
        });
        let app = router(state);

        // Simulate a "model" writing the response file after a short delay.
        let output_dir_clone = output_dir.clone();
        tokio::spawn(async move {
            sleep(Duration::from_millis(300)).await;
            let output_file = output_dir_clone.join("out-1234567890123.txt");
            fs::write(&output_file, "Hello from agent").await.unwrap();
        });

        let request_body = serde_json::json!({
            "model": "test_agent",
            "messages": [{"role": "user", "content": "Hello"}]
        });

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(request_body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = response.into_body().collect().await.unwrap().to_bytes();
        let completion: ChatCompletionResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(completion.choices[0].message.content, "Hello from agent");
        assert_eq!(completion.object, "chat.completion");
    }

    #[tokio::test]
    async fn test_chat_completions_streaming() {
        let temp_dir = TempDir::new().unwrap();
        let input_dir = temp_dir.path().join("input");
        let output_dir = temp_dir.path().join("output");
        let stream_dir = temp_dir.path().join("stream");

        setup_test_dirs(&input_dir, &output_dir, Some(&stream_dir)).await;

        let config = Arc::new(Mutex::new(make_test_config(
            &input_dir,
            &output_dir,
            Some(&stream_dir),
        )));
        let state = Arc::new(ApiState {
            config,
            model: None,
            trees: Mutex::new(HashMap::new()),
        });
        let app = router(state);

        // Simulate a streaming model writing pieces into the stream dir,
        // then finishing with the output dir.
        let stream_dir_clone = stream_dir.clone();
        let output_dir_clone = output_dir.clone();
        tokio::spawn(async move {
            sleep(Duration::from_millis(200)).await;
            let _ = fs::create_dir_all(&stream_dir_clone).await;
            let stream_file = stream_dir_clone.join("out-1234567890123.txt");
            fs::write(&stream_file, "Hello ").await.unwrap();

            sleep(Duration::from_millis(100)).await;
            fs::write(&stream_file, "world").await.unwrap();

            sleep(Duration::from_millis(100)).await;
            let output_file = output_dir_clone.join("out-1234567890123.txt");
            fs::write(&output_file, "Hello world").await.unwrap();
        });

        let request_body = serde_json::json!({
            "model": "test_agent",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": true
        });

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(request_body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = response.into_body().collect().await.unwrap().to_bytes();
        let text = String::from_utf8_lossy(&body);

        // Should contain SSE data lines.
        assert!(text.contains("chat.completion.chunk"));
        assert!(text.contains("[DONE]"));
    }

    #[tokio::test]
    async fn test_chat_completions_model_not_found() {
        let temp_dir = TempDir::new().unwrap();
        let input_dir = temp_dir.path().join("input");
        let output_dir = temp_dir.path().join("output");

        let config = Arc::new(Mutex::new(make_test_config(&input_dir, &output_dir, None)));
        let state = Arc::new(ApiState {
            config,
            model: None,
            trees: Mutex::new(HashMap::new()),
        });
        let app = router(state);

        let request_body = serde_json::json!({
            "model": "nonexistent_agent",
            "messages": [{"role": "user", "content": "Hello"}]
        });

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(request_body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }
}
