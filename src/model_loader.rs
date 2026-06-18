use crate::config::ModelConfig;
use anyhow::{Result, anyhow};
use mistralrs::{
    GgufModelBuilder, IsqBits, IsqType, ModelDType, MultiModelBuilder, MultimodalLoaderType,
    MultimodalModelBuilder,
};
use std::path::PathBuf;

/// Map a user-supplied loader type string to the mistralrs enum.
fn parse_multimodal_loader(s: &str) -> Result<MultimodalLoaderType> {
    s.parse::<MultimodalLoaderType>()
        .map_err(|_| anyhow!("Unknown multimodal loader type '{s}'"))
}

/// Resolve an `isq` config string to the platform-preferred `IsqType`.
pub fn resolve_isq_type(isq: Option<&str>) -> Option<IsqType> {
    let bits = match isq {
        Some("2") => IsqBits::Two,
        Some("3") => IsqBits::Three,
        Some("4") => IsqBits::Four,
        Some("5") => IsqBits::Five,
        Some("6") => IsqBits::Six,
        Some("8") => IsqBits::Eight,
        _ => return None,
    };
    Some(bits.expand()[0])
}

/// Whether the resolved ISQ type can benefit from importance matrix calibration.
/// Only K-quant variants (Q2K–Q6K) consume imatrix data.
pub fn isq_supports_imatrix(isq: Option<&str>) -> bool {
    const K_QUANTS: &[IsqType] = &[
        IsqType::Q2K,
        IsqType::Q3K,
        IsqType::Q4K,
        IsqType::Q5K,
        IsqType::Q6K,
    ];
    resolve_isq_type(isq).is_some_and(|t| K_QUANTS.contains(&t))
}

/// Path to the persisted importance matrix file for a model alias.
/// Returns `~/.config/agentgraph/imatrix/{alias}.cimatrix`.
pub fn imatrix_path(alias: &str) -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    let dir = PathBuf::from(home).join(".config/agentgraph/imatrix");
    dir.join(format!("{}.cimatrix", alias))
}

pub async fn load_models(
    configs: &std::collections::HashMap<String, ModelConfig>,
) -> Result<mistralrs::Model> {
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

        let chat_template_override = if let Some(t) = &config.chat_template {
            Some(t.clone())
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
                b = b.with_max_num_seqs(config.max_num_seqs);
                if let Some(t) = chat_template_override {
                    b = b.with_chat_template(t);
                }
                if let Some(ref lt) = config.loader_type {
                    let tp = parse_multimodal_loader(lt)?;
                    b = b.with_loader_type(tp);
                }

                // If calibration is enabled and a saved importance matrix
                // exists, pass it at build time so ISQ applies with
                // importance weights from the get-go.
                if config.calibration_enabled {
                    let im_path = imatrix_path(alias);
                    if im_path.exists() {
                        println!(
                            "Loading saved importance matrix for '{}' from {:?}",
                            alias, im_path
                        );
                        b = b.with_imatrix(im_path);
                    }
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