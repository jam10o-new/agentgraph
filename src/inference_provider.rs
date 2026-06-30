// Re-export all types and the trait from ag-provider-types.
pub use ag_provider_types::*;

use crate::config::ProviderConfig;
use crate::plugin_provider::PluginProvider;
use std::sync::Arc;

/// Create an inference provider from a config, connect to the API, and
/// optionally start the server subprocess if the initial connection fails.
pub async fn create_provider_from_config(
    config: &ProviderConfig,
) -> anyhow::Result<Option<Arc<dyn InferenceProvider>>> {
    match config {
        ProviderConfig::Plugin { name, max_seq_len } => {
            let binary_name = format!("ag-provider-{}", name);
            let binary_path = which::which(&binary_name)
                .map_err(|_| anyhow::anyhow!("Provider plugin binary '{}' not found on PATH", binary_name))?;
            Ok(Some(Arc::new(PluginProvider::new(binary_path, *max_seq_len)) as Arc<dyn InferenceProvider>))
        }
        ProviderConfig::OpenAi {
            api_base,
            api_key,
            model,
            server_command,
            max_seq_len,
            startup_timeout_secs,
        } => {
            let provider_config = ag_provider_openai::OpenAiProviderConfig {
                api_base: api_base.clone(),
                api_key: api_key.clone(),
                model: model.clone(),
                max_seq_len: *max_seq_len,
                server_command: server_command.clone(),
                startup_timeout_secs: *startup_timeout_secs,
            };
            let provider = ag_provider_openai::OpenAiProvider::new(provider_config);

            if provider.health_check().await {
                println!("OpenAI provider reachable at {}", api_base);
                return Ok(Some(Arc::new(provider)));
            }

            if server_command.is_some() {
                println!(
                    "OpenAI provider not reachable at {}; starting server …",
                    api_base
                );
                provider.start().await?;
                Ok(Some(Arc::new(provider)))
            } else {
                anyhow::bail!(
                    "OpenAI provider not reachable at {} and no server_command configured",
                    api_base
                );
            }
        }
    }
}
