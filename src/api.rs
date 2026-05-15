//! API bridge — shared state used by all remote frontends.
//!
//! The HTTP and Telegram frontends are separate binaries (`ag-api-http`,
//! `ag-api-telegram`) spawned by the leader.  They own their own config
//! fields, connect to the leader's Unix socket for session-tree operations,
//! and spawn inference agents via the `Session{Create,Build,...}` IPC commands.

use crate::config::Config;
use crate::leader::ModelAccess;
use crate::remote_session::RemoteSessionState;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Shared state for all remote frontends (HTTP, Telegram, ...).
/// Each frontend binary receives this and uses the Unix socket for IPC.
pub struct ApiState {
    pub config: Arc<Mutex<Config>>,
    /// The loaded Mistral.rs model. When `None`, the frontend falls back to
    /// writing into the agent's configured directories directly.
    pub model: Option<Arc<mistralrs::Model>>,
    /// Shared session-tree state.
    pub sessions: Arc<RemoteSessionState>,
    /// Tracks per-model access times for idle eviction.
    pub model_access: ModelAccess,
}
