use ag_config::AgentConfig;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub enum Command {
    /// Run an agent, optionally injecting a message into volatile context.
    /// When `quiet` is true, the output stream contains only the model's
    /// response text so it can be piped directly to other commands.
    RunAgent(String, Option<String>, bool),
    StopAgent(String),
    Status,
    ReloadConfig,
    Shutdown,
    SpawnAgent {
        name: String,
        config: AgentConfig,
    },
    UpdateConfig(ag_config::Config),

    // ── Session-tree commands (used by API frontends via IPC) ──────────
    /// Create a new session tree for a given identifier.
    SessionCreate {
        session_id: String,
    },

    /// Delete a session tree.
    SessionDelete {
        session_id: String,
    },

    /// List all active session IDs.
    SessionList,

    /// Walk conversation steps through a tree, creating nodes.
    /// If `agent_name` is provided, the leader also reads the agent's
    /// config-level system directories and returns their content in
    /// `config_system_msgs`.
    SessionBuild {
        session_id: String,
        steps: Vec<SessionStep>,
        agent_name: Option<String>,
    },

    /// Create per-request stream/tools/system dirs and populate system messages.
    SessionSetupDirs {
        session_id: String,
        system_msgs: Vec<String>,
    },

    /// Create a fresh response directory for the given hash position.
    SessionCreateResponseDir {
        session_id: String,
        current_hash: String,
    },

    /// Cache an assistant response in the tree so future requests can reuse it.
    SessionCacheResponse {
        session_id: String,
        parent_hash: String,
        content: String,
        response_dir: String,
    },

    /// High-level command: run a chat turn within a session. The leader handles
    /// all session management, agent spawning, inference, and caching internally.
    SessionChat {
        session_id: String,
        steps: Vec<SessionStep>,
        model: String,
        stream: bool,
    },

    /// List all children of a given node hash. Returns JSON array of
    /// `[{ hash, role, content_preview }]` — shows diverging branches.
    SessionListChildren {
        session_id: String,
        hash: String,
    },

    /// Walk parent chain from a hash to root, returning path in chronological
    /// order. Returns JSON array of `[{ hash, role, content_preview }]`.
    SessionPath {
        session_id: String,
        hash: String,
    },
}

/// Response from a SessionChat command.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionChatResponse {
    pub ok: bool,
    pub content: Option<String>,
    pub stream_path: Option<String>,
    pub error: Option<String>,
}

/// One step in a conversation sent over IPC.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionStep {
    pub role: String,
    pub content: String,
}

/// Structured response returned by session commands over IPC.
#[derive(Debug, Serialize, Deserialize)]
pub struct IpcResponse {
    pub ok: bool,
    pub data: Option<String>,
    pub error: Option<String>,
}

impl IpcResponse {
    pub fn ok_json<T: Serialize>(v: &T) -> Self {
        Self {
            ok: true,
            data: Some(serde_json::to_string(v).unwrap_or_default()),
            error: None,
        }
    }

    pub fn ok_str(s: impl Into<String>) -> Self {
        Self {
            ok: true,
            data: Some(s.into()),
            error: None,
        }
    }

    pub fn err(s: impl Into<String>) -> Self {
        Self {
            ok: false,
            data: None,
            error: Some(s.into()),
        }
    }
}
