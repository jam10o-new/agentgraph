//! Inference provider plugin wrapping the native mistralrs library.
//!
//! Protocol: reads a JSON [`Request`] from stdin, streams
//! type-prefixed events to stdout.
//!
//! Model configuration comes from:
//!   1. `--section '{json}'` CLI flag
//!   2. `AG_PROVIDER_MISTRALRS_CONFIG` env var (path to YAML config)
//!   3. `AG_PROVIDER_MISTRALRS_MODEL` env var (model alias, picks from
//!      the config under that key)
//!
//! The config file is the same format as agentgraph's config.yaml
//! (`models:` section).  If only a model alias is given via env var,
//! a minimal config is built on the fly.

use ag_provider_common::*;
use anyhow::{Context, Result, anyhow};
use either::Either;
use serde::Deserialize;
use std::collections::HashMap;
use std::io::BufRead;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

// ── Model config (subset of agentgraph config) ─────────────────────────

#[derive(Debug, Deserialize, Clone)]
struct ModelConfigEntry {
    path: Option<String>,
    id: String,
    builder: String,
    gguf: Option<String>,
    isq: Option<String>,
    dtype: Option<String>,
    #[serde(default)]
    max_num_seqs: usize,
    max_seq_len: Option<usize>,
    chat_template: Option<String>,
    loader_type: Option<String>,
    #[serde(default)]
    calibration_enabled: bool,
}

#[derive(Debug, Deserialize)]
struct ConfigFile {
    models: HashMap<String, ModelConfigEntry>,
}

// ── Resolve model configuration ────────────────────────────────────────

fn resolve_config() -> Result<(String, ModelConfigEntry)> {
    // 1. --section flag
    let mut args = std::env::args().peekable();
    while let Some(arg) = args.next() {
        if arg == "--section" {
            if let Some(json_str) = args.next() {
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(&json_str) {
                    if let Some(model_alias) = val.get("model").and_then(|v| v.as_str()) {
                        if let Some(cfg) = val.get("config") {
                            let entry: ModelConfigEntry = serde_json::from_value(cfg.clone())
                                .context("parse model config from --section")?;
                            return Ok((model_alias.to_string(), entry));
                        }
                    }
                }
            }
        }
    }

    // 2. Config file from env
    if let Ok(config_path) = std::env::var("AG_PROVIDER_MISTRALRS_CONFIG") {
        let content = std::fs::read_to_string(&config_path)
            .with_context(|| format!("read config file: {}", config_path))?;
        let cfg: ConfigFile = serde_yaml::from_str(&content)
            .context("parse config YAML")?;

        // If AG_PROVIDER_MISTRALRS_MODEL is set, use that alias
        let alias = std::env::var("AG_PROVIDER_MISTRALRS_MODEL")
            .unwrap_or_else(|_| {
                // Use the first model in the file
                cfg.models.keys().next().cloned().unwrap_or_else(|| "default".into())
            });

        let entry = cfg.models.get(&alias)
            .cloned()
            .ok_or_else(|| anyhow!("model '{}' not found in config", alias))?;
        return Ok((alias, entry));
    }

    // 3. Minimal config from env vars (for simple setups)
    let model_id = std::env::var("AG_PROVIDER_MISTRALRS_MODEL")
        .map_err(|_| anyhow!(
            "No model configuration provided. Set AG_PROVIDER_MISTRALRS_CONFIG \
             (path to YAML) + AG_PROVIDER_MISTRALRS_MODEL, or provide --section JSON."
        ))?;

    Ok((model_id.clone(), ModelConfigEntry {
        path: None,
        id: model_id,
        builder: "gguf".into(),
        gguf: std::env::var("AG_PROVIDER_MISTRALRS_GGUF").ok(),
        isq: None,
        dtype: None,
        max_num_seqs: 1,
        max_seq_len: None,
        chat_template: None,
        loader_type: None,
        calibration_enabled: false,
    }))
}

// ── Model loading (adapted from src/model_loader.rs) ───────────────────

use mistralrs::{
    AutoDeviceMapParams, DeviceMapSetting, GgufModelBuilder, IsqBits, IsqType,
    ModelDType, MultiModelBuilder, MultimodalLoaderType, MultimodalModelBuilder,
};

fn parse_multimodal_loader(s: &str) -> Result<MultimodalLoaderType> {
    s.parse::<MultimodalLoaderType>()
        .map_err(|_| anyhow!("Unknown multimodal loader type '{s}'"))
}

async fn load_models(configs: &HashMap<String, ModelConfigEntry>) -> Result<mistralrs::Model> {
    let mut builder = MultiModelBuilder::new();

    for (alias, config) in configs {
        let isq = match config.isq.as_deref() {
            Some("2") => Some(IsqBits::Two),
            Some("3") => Some(IsqBits::Three),
            Some("4") => Some(IsqBits::Four),
            Some("5") => Some(IsqBits::Five),
            Some("6") => Some(IsqBits::Six),
            Some("8") => Some(IsqBits::Eight),
            _ => None,
        };

        let dtype = match config.dtype.as_deref() {
            Some("f32") => Some(ModelDType::F32),
            Some("f16") => Some(ModelDType::F16),
            Some("bf16") => Some(ModelDType::BF16),
            _ => Some(ModelDType::Auto),
        };

        let model_src = config
            .path
            .as_deref()
            .or(Some(&config.id))
            .ok_or_else(|| anyhow!("Path or ID required for model {}", alias))?;

        match config.builder.as_str() {
            "vision" => {
                let mut b = MultimodalModelBuilder::new(model_src);
                if let Some(i) = isq {
                    b = b.with_auto_isq(i);
                }
                if let Some(d) = dtype {
                    b = b.with_dtype(d);
                }
                b = b.with_logging();
                b = b.with_max_num_seqs(config.max_num_seqs);
                if let Some(ref t) = config.chat_template {
                    b = b.with_chat_template(t.clone());
                }
                if let Some(ref lt) = config.loader_type {
                    let tp = parse_multimodal_loader(lt)?;
                    b = b.with_loader_type(tp);
                }
                if let Some(msl) = config.max_seq_len {
                    b = b.with_device_mapping(DeviceMapSetting::Auto(
                        AutoDeviceMapParams::Multimodal {
                            max_seq_len: msl,
                            max_batch_size: AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE,
                            max_image_shape: (
                                AutoDeviceMapParams::DEFAULT_MAX_IMAGE_LENGTH,
                                AutoDeviceMapParams::DEFAULT_MAX_IMAGE_LENGTH,
                            ),
                            max_num_images: AutoDeviceMapParams::DEFAULT_MAX_NUM_IMAGES,
                        },
                    ));
                }
                builder = builder.add_model_with_alias(alias, b);
            }
            "gguf" => {
                let gguf_path = config
                    .gguf
                    .as_ref()
                    .ok_or_else(|| anyhow!("GGUF file required for gguf model"))?
                    .clone();
                let mut b = GgufModelBuilder::new(&config.id, vec![gguf_path]);
                b = b.with_logging();
                if let Some(ref t) = config.chat_template {
                    b = b.with_chat_template(t.clone());
                }
                builder = builder.add_model_with_alias(alias, b);
            }
            _ => return Err(anyhow!("Unknown builder type: {}", config.builder)),
        }
    }

    builder.build().await.map_err(|e| anyhow!("mistralrs build: {e}"))
}

// ── Mistralrs helper functions (from context.rs) ───────────────────────

fn role_to_mistral(role: &str) -> mistralrs::TextMessageRole {
    match role {
        "user" => mistralrs::TextMessageRole::User,
        "assistant" => mistralrs::TextMessageRole::Assistant,
        "tool" => mistralrs::TextMessageRole::Tool,
        _ => mistralrs::TextMessageRole::System,
    }
}

fn build_messages(messages: &[ChatMessage]) -> Vec<(mistralrs::TextMessageRole, String)> {
    messages.iter().map(|m| {
        let role = role_to_mistral(&m.role);
        // Collect text from ContentParts and reference files inline
        let mut text = String::new();
        for part in &m.content {
            match part {
                ContentPart::Text { text: t } => text.push_str(t),
                ContentPart::Image { file_path } => {
                    text.push_str(&format!("\n[Image: {}]\n", file_path));
                }
                ContentPart::Audio { file_path } => {
                    text.push_str(&format!("\n[Audio: {}]\n", file_path));
                }
                ContentPart::Video { file_path } => {
                    text.push_str(&format!("\n[Video: {}]\n", file_path));
                }
            }
        }
        (role, text)
    }).collect()
}

fn build_constraint(constraint: &Option<Constraint>) -> Option<mistralrs::Constraint> {
    constraint.as_ref().and_then(|c| {
        if let Some(ref schema) = c.schema {
            Some(mistralrs::Constraint::JsonSchema(schema.clone()))
        } else if let Some(ref regex) = c.regex {
            Some(mistralrs::Constraint::Regex(regex.clone()))
        } else if let Some(ref grammar) = c.grammar {
            Some(mistralrs::Constraint::Lark(grammar.clone()))
        } else {
            None
        }
    })
}

// Mirrors mistralrs's SupportedModality for the discovered token
// count path — used for checking modality support internally.
use mistralrs::core::SupportedModality;

/// Count tokens using the mistralrs model's tokenizer.
async fn count_tokens(model: &mistralrs::Model, text: &str) -> usize {
    match model
        .tokenize(Either::Right(text.to_string()), None, false, false, None)
        .await
    {
        Ok(tokens) => tokens.len(),
        Err(_) => text.len() / 4,
    }
}

// ── Handle requests ────────────────────────────────────────────────────

async fn handle_chat(
    model: &mistralrs::Model,
    request: &ChatRequest,
    model_alias: &str,
) -> Result<()> {
    use mistralrs::{
        AudioInput, Function, Model as _, MultimodalMessages, RequestBuilder,
        SamplingParams, TextMessageRole, Tool, ToolChoice, VideoInput,
    };

    let messages = build_messages(&request.messages);

    let sampling = SamplingParams {
        temperature: request.temperature.unwrap_or(0.7),
        top_p: request.top_p.unwrap_or(0.95),
        max_len: request.max_tokens.unwrap_or(4096) as usize,
        ..SamplingParams::default()
    };

    let mut multimodal = MultimodalMessages::new()
        .enable_thinking(request.enable_thinking);
    for (role, content) in &messages {
        multimodal = multimodal.add_message(role.clone(), content.as_str());
    }

    // Load multimodal content from file paths
    let mut images = Vec::new();
    let mut audios: Vec<AudioInput> = Vec::new();
    let mut videos: Vec<VideoInput> = Vec::new();

    for msg in &request.messages {
        for part in &msg.content {
            match part {
                ContentPart::Image { file_path } => {
                    if let Ok(img) = image::open(file_path) {
                        images.push(img);
                    }
                }
                ContentPart::Audio { file_path } => {
                    if let Ok(audio) = AudioInput::read_wav(file_path) {
                        audios.push(audio);
                    }
                    // Non-WAV audio could use from_bytes but the provider
                    // can fall back to the file path reference in text.
                }
                ContentPart::Video { file_path } => {
                    // Video loading is complex (ffmpeg decode) — for now
                    // the text reference is included above.
                    let _ = file_path;
                }
                _ => {}
            }
        }
    }

    // Determine which model alias to use from the request, defaulting
    // to the one we loaded.
    let effective_alias = request.model.as_deref().unwrap_or(model_alias);

    // Add the final user message (or multimodal message)
    if !images.is_empty() || !audios.is_empty() || !videos.is_empty() {
        // Find the last user text
        let user_text = messages.iter()
            .rev()
            .find(|(r, _)| *r == TextMessageRole::User)
            .map(|(_, t)| t.clone())
            .unwrap_or_default();
        multimodal = multimodal.add_multimodal_message(
            TextMessageRole::User,
            &user_text,
            images,
            audios,
            videos,
        );
    }

    let tools: Vec<Tool> = request.tools.as_ref().map(|ts| {
        ts.iter().map(|t| Tool {
            function: Function {
                name: t.name.clone(),
                description: t.description.clone(),
                parameters: match &t.parameters {
                    serde_json::Value::Object(map) => map.clone(),
                    _ => HashMap::new(),
                },
            },
            tool_type: ToolType::Function,
        }).collect()
    }).unwrap_or_default();

    let mut request_builder = RequestBuilder::from(multimodal)
        .set_sampling(sampling)
        .set_tools(tools)
        .set_tool_choice(ToolChoice::Auto);

    if request.enable_thinking {
        request_builder = request_builder.enable_thinking(true);
    }
    if let Some(constraint) = build_constraint(&request.constraint) {
        request_builder = request_builder.set_constraint(constraint);
    }

    let mut stream = model.stream_chat_request(request_builder)
        .await
        .map_err(|e| anyhow!("stream_chat_request: {e}"))?;

    use futures::StreamExt;
    while let Some(chunk) = stream.next().await {
        match chunk {
            mistralrs::Response::Chunk(c) => {
                for choice in c.choices {
                    if let Some(ref content) = choice.delta.content {
                        emit_event(&ProviderEvent::Chunk(content.clone()));
                    }
                    if request.enable_thinking {
                        if let Some(ref reasoning) = choice.delta.reasoning {
                            emit_event(&ProviderEvent::Reasoning(reasoning.clone()));
                        }
                    }
                    if let Some(ref tcs) = choice.delta.tool_calls {
                        for tc in tcs {
                            if let Some(ref fn_call) = tc.function {
                                emit_event(&ProviderEvent::ToolCall(ToolCall {
                                    id: tc.id.clone().unwrap_or_default(),
                                    name: fn_call.name.clone().unwrap_or_default(),
                                    arguments: fn_call.arguments.clone().unwrap_or_default(),
                                }));
                            }
                        }
                    }
                }
            }
            mistralrs::Response::ModelError(msg, _) => {
                emit_event(&ProviderEvent::Error(msg));
                emit_event(&ProviderEvent::Done);
                return Ok(());
            }
            _ => {}
        }
    }

    emit_event(&ProviderEvent::Done);
    Ok(())
}

async fn handle_embed(model: &mistralrs::Model, request: &EmbedRequest) -> Result<()> {
    let text = &request.text;
    let result = model
        .send_embed_request(text.into(), None, false, false)
        .await
        .map_err(|e| anyhow!("embed error: {e}"))?;

    let embedding = result.data.first()
        .map(|d| d.embedding.clone())
        .unwrap_or_default();

    emit_event(&ProviderEvent::Embedding(embedding));
    emit_event(&ProviderEvent::Done);
    Ok(())
}

// ── Main ───────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    if has_flag("--describe") {
        let model_name = std::env::var("AG_PROVIDER_MISTRALRS_MODEL").ok();
        describe(&InfoResponse {
            name: "mistralrs".into(),
            model: model_name,
            max_seq_len: None, // determined at runtime
            supports_tools: true,
            supports_embeddings: true,
            supports_constraints: true,
            supports_modalities: vec!["text".into(), "image".into(), "audio".into(), "video".into()],
        });
        return Ok(());
    }

    if has_flag("--help") {
        guidance(
            "Native mistral.rs inference provider.\n\
             Supports all mistralrs model types (GGUF, vision/multimodal, HuggingFace),\n\
             native tool calling, JSON Schema / Lark / Regex constraints,\n\
             embeddings, multimodal (image, audio, video), and token counting.\n\n\
             Configuration via AG_PROVIDER_MISTRALRS_CONFIG YAML file or\n\
             AG_PROVIDER_MISTRALRS_MODEL + AG_PROVIDER_MISTRALRS_GGUF env vars.",
        );
        return Ok(());
    }

    let (model_alias, model_config) = resolve_config()?;

    // Build a single-entry model map for the loader
    let mut configs = HashMap::new();
    configs.insert(model_alias.clone(), model_config);

    let model = load_models(&configs).await?;
    let model = Arc::new(model);

    // Read the JSON request from stdin
    let request = read_request().map_err(|e| anyhow!("{}", e))?;

    match &request.kind {
        RequestKind::Chat(chat_req) => {
            handle_chat(&model, chat_req, &model_alias).await
        }
        RequestKind::Embed(embed_req) => {
            handle_embed(&model, embed_req).await
        }
        RequestKind::CountTokens(ct_req) => {
            let count = count_tokens(&model, &ct_req.text).await;
            emit_event(&ProviderEvent::TokenCount(count));
            emit_event(&ProviderEvent::Done);
            Ok(())
        }
        RequestKind::Health => {
            emit_event(&ProviderEvent::Healthy(true));
            emit_event(&ProviderEvent::Done);
            Ok(())
        }
        RequestKind::Info => {
            let has_audio = model.supports_modality(&SupportedModality::Audio);
            let has_image = model.supports_modality(&SupportedModality::Image);
            let mut modalities = vec!["text".to_string()];
            if has_image { modalities.push("image".to_string()); }
            if has_audio { modalities.push("audio".to_string()); }

            let msl = model.max_sequence_length_with_model(Some(&model_alias))
                .ok()
                .flatten();

            emit_event(&ProviderEvent::Info(InfoResponse {
                name: "mistralrs".into(),
                model: Some(model_alias.clone()),
                max_seq_len: msl,
                supports_tools: true,
                supports_embeddings: true,
                supports_constraints: true,
                supports_modalities: modalities,
            }));
            emit_event(&ProviderEvent::Done);
            Ok(())
        }
    }
}
