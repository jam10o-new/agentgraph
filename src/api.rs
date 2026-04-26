use axum::{
    extract::State,
    http::StatusCode,
    response::{sse::Event, IntoResponse, Sse},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::fs;
use tokio::sync::Mutex;
use tokio::time::sleep;
use tokio_stream::wrappers::ReceiverStream;
use tower_http::cors::CorsLayer;

use crate::config::Config;

pub struct ApiState {
    pub config: Arc<Mutex<Config>>,
}

pub fn router(state: Arc<ApiState>) -> Router {
    Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .layer(CorsLayer::permissive())
        .with_state(state)
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
            context_window: agent_config.context_checkpoint_limit,
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
    let config = state.config.lock().await;

    let agent_config = config
        .agents
        .get(&req.model)
        .ok_or(StatusCode::NOT_FOUND)?
        .clone();

    drop(config); // Release lock early

    let latest_user_msg = req
        .messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| m.content.clone())
        .ok_or(StatusCode::BAD_REQUEST)?;

    let input_dir = agent_config
        .inputs
        .first()
        .ok_or(StatusCode::BAD_REQUEST)?;

    let start_time = SystemTime::now();

    let timestamp = start_time
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();

    let input_file = PathBuf::from(input_dir).join(format!("api-{}.txt", timestamp));
    let _ = fs::create_dir_all(input_dir).await;
    fs::write(&input_file, &latest_user_msg)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let model = req.model;
    let created = timestamp as u64 / 1000;

    if req.stream {
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(100);

        tokio::spawn(async move {
            if let Some(ref stream_dir) = agent_config.stream_output {
                if let Err(e) = stream_from_dir(&id, &model, stream_dir, &agent_config.output, start_time, tx.clone()).await {
                    let err_json = serde_json::json!({"error": e.to_string()}).to_string();
                    let _ = tx.send(Ok(Event::default().data(err_json))).await;
                }
            } else {
                // Fallback: wait for output and send as single chunk
                match wait_for_output(&agent_config.output, start_time).await {
                    Ok(content) => {
                        let chunk = ChatCompletionChunk {
                            id: id.clone(),
                            object: "chat.completion.chunk".to_string(),
                            created,
                            model: model.clone(),
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
                            id: id.clone(),
                            object: "chat.completion.chunk".to_string(),
                            created,
                            model: model.clone(),
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
            }
        });

        Ok(Sse::new(ReceiverStream::new(rx)).into_response())
    } else {
        let content = wait_for_output(&agent_config.output, start_time)
            .await
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

        let response = ChatCompletionResponse {
            id,
            object: "chat.completion".to_string(),
            created,
            model,
            choices: vec![Choice {
                index: 0,
                message: MessageResponse {
                    role: "assistant".to_string(),
                    content,
                },
                finish_reason: "stop".to_string(),
            }],
        };

        Ok(Json(response).into_response())
    }
}

// ---------- Helpers ----------

async fn wait_for_output(output_dir: &str, after: SystemTime) -> anyhow::Result<String> {
    let output_path = PathBuf::from(output_dir);
    loop {
        sleep(Duration::from_millis(100)).await;

        let mut candidates = Vec::new();
        if let Ok(mut entries) = fs::read_dir(&output_path).await {
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
            sleep(Duration::from_millis(200)).await;
            return Ok(fs::read_to_string(&path).await?);
        }
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

    // Extract timestamp from stream filename like "out-1234567890123.txt"
    let filename = stream_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    let timestamp_str = filename.strip_prefix("out-").unwrap_or("");
    let output_file = PathBuf::from(output_dir).join(format!("out-{}.txt", timestamp_str));

    let created = SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs();

    // Send the role chunk
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
            // Inference is complete, send remaining content
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

            // Send finish chunk
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
            return Ok(path);
        }
    }
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
                output: output_dir.to_string_lossy().to_string(),
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
        let state = Arc::new(ApiState { config });
        let app = router(state);

        let response = app
            .oneshot(Request::builder().uri("/v1/models").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = response.into_body().collect().await.unwrap().to_bytes();
        let models: ModelsListResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(models.object, "list");
        assert_eq!(models.data.len(), 1);
        assert_eq!(models.data[0].id, "test_agent");
        assert_eq!(models.data[0].context_window, Some(10000));
    }

    #[tokio::test]
    async fn test_chat_completions_non_streaming() {
        let temp_dir = TempDir::new().unwrap();
        let input_dir = temp_dir.path().join("input");
        let output_dir = temp_dir.path().join("output");

        setup_test_dirs(&input_dir, &output_dir, None).await;

        let config = Arc::new(Mutex::new(make_test_config(&input_dir, &output_dir, None)));
        let state = Arc::new(ApiState { config });
        let app = router(state);

        // Spawn a task to create the output file after a delay
        let output_dir_clone = output_dir.clone();
        tokio::spawn(async move {
            sleep(Duration::from_millis(300)).await;
            let output_file = output_dir_clone.join("out-1234567890123.txt");
            fs::write(&output_file, "Hello from agent")
                .await
                .unwrap();
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

        // Verify input file was created
        let mut entries = Vec::new();
        let mut dir = fs::read_dir(&input_dir).await.unwrap();
        while let Ok(Some(entry)) = dir.next_entry().await {
            entries.push(entry);
        }
        assert_eq!(entries.len(), 1);

        let input_content = fs::read_to_string(&entries[0].path()).await.unwrap();
        assert_eq!(input_content, "Hello");

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
        let state = Arc::new(ApiState { config });
        let app = router(state);

        // Spawn a task to simulate streaming output
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

        // Should contain SSE data lines
        assert!(text.contains("chat.completion.chunk"));
        assert!(text.contains("[DONE]"));
    }

    #[tokio::test]
    async fn test_chat_completions_model_not_found() {
        let temp_dir = TempDir::new().unwrap();
        let input_dir = temp_dir.path().join("input");
        let output_dir = temp_dir.path().join("output");

        let config = Arc::new(Mutex::new(make_test_config(&input_dir, &output_dir, None)));
        let state = Arc::new(ApiState { config });
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
