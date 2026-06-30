use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

/// Info about a discovered provider plugin binary.
#[derive(Clone, Debug)]
pub struct ProviderPluginInfo {
    pub binary_path: PathBuf,
    pub description: serde_json::Value,
}

/// Registry of discovered `ag-provider-*` plugin binaries.
/// Populated at startup by scanning `config.plugins` for `provider-*` keys.
#[derive(Clone, Default)]
pub struct ProviderRegistry {
    providers: Arc<std::sync::Mutex<HashMap<String, ProviderPluginInfo>>>,
}

impl ProviderRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Look up a cached provider plugin by name (without the `ag-provider-` prefix).
    pub fn get(&self, name: &str) -> Option<ProviderPluginInfo> {
        self.providers.lock().ok()?.get(name).cloned()
    }

    /// Register a discovered provider.
    pub fn register(&self, name: String, info: ProviderPluginInfo) {
        if let Ok(mut guard) = self.providers.lock() {
            guard.insert(name, info);
        }
    }
}
