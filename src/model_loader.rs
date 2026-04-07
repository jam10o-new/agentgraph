use crate::config::ModelConfig;
use anyhow::{Result, anyhow};
use mistralrs::{GgufModelBuilder, IsqBits, ModelDType, MultiModelBuilder, MultimodalModelBuilder};
use std::path::{Path, PathBuf};

fn has_model_template(model_src: &str) -> bool {
    let check_file = |p: &Path| -> bool {
        let tp = p.join("tokenizer_config.json");
        if let Ok(c) = std::fs::read_to_string(tp) {
            c.contains("\"chat_template\"")
        } else {
            false
        }
    };

    let p = Path::new(model_src);
    if p.exists() && p.is_dir() {
        return check_file(p);
    }

    // Check HF cache
    if let Ok(home) = std::env::var("HOME") {
        let cache_base = PathBuf::from(home).join(".cache/huggingface/hub");
        let model_dir_name = format!("models--{}", model_src.replace('/', "--"));
        let model_path = cache_base.join(model_dir_name).join("snapshots");
        if let Ok(mut entries) = std::fs::read_dir(model_path) {
            if let Some(Ok(entry)) = entries.next() {
                return check_file(&entry.path());
            }
        }
    }
    false
}

pub async fn load_models(
    configs: &std::collections::HashMap<String, ModelConfig>,
) -> Result<mistralrs::Model> {
    let mut builder = MultiModelBuilder::new();

    let default_chat_template = "mistralrs-fork/chat_templates/chatml.json";

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
            _ => None,
        };

        let model_src = config
            .path
            .as_deref()
            .or(Some(&config.id))
            .ok_or_else(|| anyhow!("Path or ID required for model {}", alias))?;

        // 1. User provided in config -> Always use it
        // 2. Model has one naturally -> Don't call with_chat_template (mistralrs handles it)
        // 3. Neither -> Use fallback
        let chat_template_override = if let Some(t) = &config.chat_template {
            Some(t.clone())
        } else if !has_model_template(model_src) {
            Some(default_chat_template.to_string())
        } else {
            None
        };

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
                if let Some(t) = chat_template_override {
                    b = b.with_chat_template(t);
                }
                builder = builder.add_model_with_alias(alias, b);
            }
            "gguf" => {
                let mut b = GgufModelBuilder::new(
                    &config.id,
                    vec![
                        config
                            .gguf
                            .as_ref()
                            .ok_or_else(|| anyhow!("GGUF file required for gguf model"))?
                            .clone(),
                    ],
                );
                b = b.with_logging();
                if let Some(t) = chat_template_override {
                    b = b.with_chat_template(t);
                }
                builder = builder.add_model_with_alias(alias, b);
            }
            _ => return Err(anyhow!("Unknown builder type: {}", config.builder)),
        }
    }

    builder.build().await
}
