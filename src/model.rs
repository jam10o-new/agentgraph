//! Model loading for agentgraph.

use crate::Args;
use crate::types::{MODEL_PRIMARY, MODEL_SECONDARY};
use anyhow::Result;
use mistralrs::{GgufModelBuilder, IsqBits, VisionModelBuilder};

/// Load primary and optionally secondary models
pub async fn load_model(args: &Args) -> Result<mistralrs::Model> {
    println!(
        "primary_model={}{}",
        args.model,
        args.model_path
            .as_ref()
            .map(|p| format!("(local: {})", p))
            .unwrap_or_default()
    );
    println!("secondary_model={}", args.secondary_model);

    let mut builder = mistralrs::MultiModelBuilder::new().with_default_model(MODEL_PRIMARY);

    // Add primary model
    if args.verbose {
        eprintln!("Loading Primary Model");
    }
    if let Some(path) = &args.model_path {
        if args.verbose {
            eprintln!("Using local model file for primary: {}", path);
        }
        let primary_builder = VisionModelBuilder::new(path)
            .with_auto_isq(IsqBits::Four)
            .with_logging();
        builder = builder.add_model_with_alias(MODEL_PRIMARY, primary_builder);
    } else if !args.gguf.is_empty() {
        println!("Loading primary as GGUF!");
        let primary_builder = GgufModelBuilder::new(&args.model, args.gguf.clone());
        builder = builder.add_model_with_alias(MODEL_PRIMARY, primary_builder);
    } else {
        let primary_builder = VisionModelBuilder::new(&args.model)
            .with_auto_isq(IsqBits::Four)
            .with_logging();
        builder = builder.add_model_with_alias(MODEL_PRIMARY, primary_builder);
    }

    // Only load secondary (audio) model when needed
    if args.secondary_model != "none" && args.realtime_listener.is_some() {
        if args.verbose {
            eprintln!("Loading Secondary Model");
        }
        let secondary_builder = mistralrs::VisionModelBuilder::new(&args.secondary_model)
            .with_auto_isq(IsqBits::Four)
            .with_dtype(mistralrs::ModelDType::F32)
            .with_logging();
        builder = builder.add_model_with_alias(MODEL_SECONDARY, secondary_builder);
    } else if args.verbose {
        eprintln!("Skipping secondary model (not needed).");
    }

    if args.verbose {
        eprintln!("Building Models");
    }
    let model = builder.build().await?;
    if args.verbose {
        eprintln!("Models Built");
    }
    Ok(model)
}
