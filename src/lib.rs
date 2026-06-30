pub mod config;
pub mod leader;
pub mod agent;
pub mod context;
pub mod utils;
pub mod ipc;
#[cfg(feature = "audio")]
pub mod audio;
pub mod api;
pub mod remote_session;
pub mod contacts;
pub mod inference_provider;
pub mod plugin_provider;
pub mod provider_registry;

// Re-export key types
pub use config::Config;
pub use leader::Leader;
pub use agent::Agent;
pub use utils::{find_leader_socket, is_leader_alive, LeaderStatus, AgentLogger, LEADER_PID_FILE};
pub use ipc::{Command, SessionStep, IpcResponse};
pub use api::ApiState;
pub use remote_session::RemoteSessionState;
pub use contacts::ContactBook;
pub use inference_provider::InferenceProvider;

/// Full version string including git commit hash.
/// Format: `<cargo-version>-<short-hash>` or `<cargo-version>-<short-hash>-dirty`.
pub fn version() -> String {
    let pkg_version = env!("CARGO_PKG_VERSION");
    let hash = option_env!("GIT_COMMIT_HASH").unwrap_or("unknown");
    let dirty = option_env!("GIT_DIRTY").map(|_| "-dirty").unwrap_or("");
    format!("{}-{}{}", pkg_version, hash, dirty)
}
