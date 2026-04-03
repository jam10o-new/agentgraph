use crate::config::ModelConfig;
use anyhow::{Result, anyhow};
use mistralrs::{
    GgufModelBuilder, IsqBits, ModelDType, MultiModelBuilder, MultimodalModelBuilder,
};

pub async fn load_models(configs: &std::collections::HashMap<String, ModelConfig>) -> Result<mistralrs::Model> {
    let mut builder = MultiModelBuilder::new();

    for (alias, config) in configs {
        let isq = match config.isq.as_deref() {
            Some("4") => Some(IsqBits::Four),
            Some("8") => Some(IsqBits::Eight),
            _ => None,
        };

        let dtype = match config.dtype.as_deref() {
            Some("f32") => Some(ModelDType::F32),
            Some("f16") => Some(ModelDType::F16),
            Some("bf16") => Some(ModelDType::BF16),
            _ => None,
        };

        match config.builder.as_str() {
            "vision" => {
                let model_src = config.path.as_deref().or(Some(&config.id)).ok_or_else(|| anyhow!("Path or ID required for vision model"))?;
                let mut b = MultimodalModelBuilder::new(model_src);
                if let Some(i) = isq {
                    b = b.with_auto_isq(i);
                }
                if let Some(d) = dtype {
                    b = b.with_dtype(d);
                }
                builder = builder.add_model_with_alias(alias, b);
            }
            "gguf" => {
                let b = GgufModelBuilder::new(
                    &config.id,
                    vec![config.gguf.as_ref().ok_or_else(|| anyhow!("GGUF file required for gguf model"))?.clone()],
                );
                builder = builder.add_model_with_alias(alias, b);
            }
            _ => return Err(anyhow!("Unknown builder type: {}", config.builder)),
        }
    }

    builder.build().await
}
