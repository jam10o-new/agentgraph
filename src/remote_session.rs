//! RemoteSession layer — shared session-tree management used by both the
//! HTTP API and the Telegram bot.  Each frontend registers sessions keyed by
//! its own identifier (model name for HTTP, chat_id for Telegram) and
//! benefits from content-addressed directory sharing across requests.

use serde::{Serialize};

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::fs;
use tokio::sync::Mutex;

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// Metadata for a node in the conversation tree.
#[derive(Debug, Clone)]
pub struct NodeMeta {
    pub dir: PathBuf,
    pub role: String,
    pub parent_hash: Option<String>,
}

/// A tree of conversation states for a specific session.  Each state
/// represents a prefix of the message history and has its own directory.
/// States with shared prefixes share directories (content-addressed).
pub struct SessionTree {
    /// Temp directory where all nodes for this tree live.
    pub temp_dir: Arc<tempfile::TempDir>,
    /// hash → node lookup.
    pub nodes: Mutex<HashMap<String, NodeMeta>>,
}

/// Central session state shared across all remote interfaces (HTTP, Telegram).
pub struct RemoteSessionState {
    /// Per-identifier session trees.  The identifier is frontend-specific:
    /// model name for the HTTP API, chat_id for Telegram.
    pub trees: Mutex<HashMap<String, Arc<SessionTree>>>,
}

/// One step in a conversation — a (role, content) pair used to walk the tree.
#[derive(Debug, Clone)]
pub struct ConversationStep {
    pub role: String,
    pub content: String,
}

/// Accumulated state after walking a message history through the tree.
#[derive(Debug, Clone, Serialize)]
pub struct ConversationState {
    /// Directories for user messages in chronological order.
    pub user_dirs: Vec<String>,
    /// Directories for assistant messages in chronological order.
    pub assistant_dirs: Vec<String>,
    /// Content of the most recent (last) user message.
    pub latest_user_msg: String,
    /// System messages collected from conversation steps (role=="system").
    pub system_msgs: Vec<String>,
    /// System messages read from the agent's config-level system directories.
    /// Populated when the leader has an agent_name to look up.
    pub config_system_msgs: Vec<String>,
    /// Hash of the last message (current position in the tree).
    pub current_hash: String,
}

// ---------------------------------------------------------------------------
// RemoteSessionState
// ---------------------------------------------------------------------------

impl RemoteSessionState {
    /// Create a new (empty) session state.
    pub fn new() -> Self {
        Self {
            trees: Mutex::new(HashMap::new()),
        }
    }

    /// Get or create the session tree for the given identifier.
    pub async fn get_or_create_tree(&self, id: &str) -> Arc<SessionTree> {
        let mut trees = self.trees.lock().await;
        trees
            .entry(id.to_string())
            .or_insert_with(|| {
                Arc::new(SessionTree {
                    temp_dir: Arc::new(
                        tempfile::tempdir().expect("Failed to create API session temp dir"),
                    ),
                    nodes: Mutex::new(HashMap::new()),
                })
            })
            .clone()
    }

    /// Check whether a tree exists for the given id.
    pub async fn has_tree(&self, id: &str) -> bool {
        self.trees.lock().await.contains_key(id)
    }

    /// Remove a session tree (e.g. when a chat is cleared).
    pub async fn remove_tree(&self, id: &str) {
        self.trees.lock().await.remove(id);
    }

    /// List all active session identifiers.
    pub async fn list_ids(&self) -> Vec<String> {
        self.trees.lock().await.keys().cloned().collect()
    }

    /// List all child nodes of `parent_hash` within the tree identified by `id`.
    /// Returns a Vec of (hash, role, content_preview).
    pub async fn list_children(&self, id: &str, parent_hash: &str) -> Vec<(String, String, String)> {
        let trees_guard = self.trees.lock().await;
        let Some(tree) = trees_guard.get(id) else {
            return Vec::new();
        };
        let nodes = tree.nodes.lock().await;
        let parent_pattern = if parent_hash.is_empty() {
            None
        } else {
            Some(parent_hash.to_string())
        };
        let mut children: Vec<_> = nodes
            .iter()
            .filter(|(_, meta)| {
                match (&meta.parent_hash, &parent_pattern) {
                    (Some(ph), Some(pp)) => ph == pp,
                    (None, None) => true,  // both root
                    _ => false,
                }
            })
            .map(|(hash, meta)| {
                let preview = meta
                    .dir
                    .join("msg.txt")
                    .to_string_lossy()
                    .to_string();
                (hash.clone(), meta.role.clone(), preview)
            })
            .collect();
        children.sort_by(|a, b| a.0.cmp(&b.0));
        children
    }

    /// Walk the parent chain from `hash` to the root, returning the path
    /// as a chronological Vec of (hash, role, content_preview).
    pub async fn get_path(&self, id: &str, hash: &str) -> Vec<(String, String, String)> {
        let trees_guard = self.trees.lock().await;
        let Some(tree) = trees_guard.get(id) else {
            return Vec::new();
        };
        let nodes = tree.nodes.lock().await;
        let mut path = Vec::new();
        let mut current = hash.to_string();
        while !current.is_empty() {
            if let Some(meta) = nodes.get(&current) {
                let preview = format!(
                    "{}",
                    meta.dir.join("msg.txt").to_string_lossy()
                );
                path.push((current.clone(), meta.role.clone(), preview));
                current = meta.parent_hash.clone().unwrap_or_default();
            } else {
                break;
            }
        }
        path.reverse();
        path
    }
}

// ---------------------------------------------------------------------------
// Hashing
// ---------------------------------------------------------------------------

/// Cumulative hash for a conversation state.  Two requests with the same
/// message prefix will produce the same hash chain, allowing them to share
/// on-disk directories.
pub fn hash_state(parent_hash: &str, role: &str, content: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    parent_hash.hash(&mut hasher);
    role.hash(&mut hasher);
    content.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

// ---------------------------------------------------------------------------
// Building conversation state
// ---------------------------------------------------------------------------

/// Walk a sequence of conversation steps through the session tree, creating
/// nodes on disk where they don't yet exist.  Returns the accumulated
/// directory paths and the final hash position.
///
/// System-role steps are collected into `system_msgs` without creating tree
/// nodes (they're handled separately as system prompt directories).
pub async fn build_conversation(
    tree: &SessionTree,
    steps: &[ConversationStep],
) -> Result<ConversationState, String> {
    let mut current_hash = String::new();
    let mut system_msgs: Vec<String> = Vec::new();
    let mut latest_user_msg = String::new();

    for step in steps {
        if step.role == "system" {
            system_msgs.push(step.content.clone());
            continue;
        }

        let parent_hash = current_hash.clone();
        current_hash = hash_state(&parent_hash, &step.role, &step.content);

        // Lock, create node if needed, unlock.
        {
            let mut nodes = tree.nodes.lock().await;
            if !nodes.contains_key(&current_hash) {
                let dir = tree
                    .temp_dir
                    .path()
                    .join(format!("{}-{}", step.role, &current_hash[..16]));
                fs::create_dir_all(&dir).await.map_err(|e| {
                    format!("create_dir_all({}): {}", dir.display(), e)
                })?;
                let path = dir.join("msg.txt");
                fs::write(&path, &step.content).await.map_err(|e| {
                    format!("write({}): {}", path.display(), e)
                })?;
                nodes.insert(
                    current_hash.clone(),
                    NodeMeta {
                        dir,
                        role: step.role.clone(),
                        parent_hash: if parent_hash.is_empty() {
                            None
                        } else {
                            Some(parent_hash)
                        },
                    },
                );
            }
        }

        if step.role == "user" {
            latest_user_msg = step.content.clone();
        }
    }

    // Collect user/assistant dirs from the hash chain.
    let mut user_dirs: Vec<String> = Vec::new();
    let mut assistant_dirs: Vec<String> = Vec::new();
    {
        let nodes = tree.nodes.lock().await;
        let mut hash = current_hash.clone();
        while !hash.is_empty() {
            if let Some(node) = nodes.get(&hash) {
                match node.role.as_str() {
                    "user" => user_dirs.push(node.dir.to_string_lossy().to_string()),
                    "assistant" => {
                        assistant_dirs.push(node.dir.to_string_lossy().to_string())
                    }
                    _ => {}
                }
                hash = node.parent_hash.clone().unwrap_or_default();
            } else {
                break;
            }
        }
    }
    user_dirs.reverse();
    assistant_dirs.reverse();

    Ok(ConversationState {
        user_dirs,
        assistant_dirs,
        latest_user_msg,
        system_msgs,
        config_system_msgs: Vec::new(),
        current_hash,
    })
}

/// Create per-request directories (stream, tools, system) inside the tree
/// and populate the system dir with system messages.
pub async fn setup_request_dirs(
    tree: &SessionTree,
    system_msgs: &[String],
) -> Result<(PathBuf, PathBuf, PathBuf), String> {
    let api_stream = tree
        .temp_dir
        .path()
        .join(format!("stream-{}", uuid::Uuid::new_v4()));
    let api_tools = tree
        .temp_dir
        .path()
        .join(format!("tools-{}", uuid::Uuid::new_v4()));
    let api_system = tree
        .temp_dir
        .path()
        .join(format!("system-{}", uuid::Uuid::new_v4()));

    fs::create_dir_all(&api_stream).await.map_err(|e| {
        format!("create_dir_all({}): {}", api_stream.display(), e)
    })?;
    fs::create_dir_all(&api_tools).await.map_err(|e| {
        format!("create_dir_all({}): {}", api_tools.display(), e)
    })?;
    fs::create_dir_all(&api_system).await.map_err(|e| {
        format!("create_dir_all({}): {}", api_system.display(), e)
    })?;

    for (idx, sys_msg) in system_msgs.iter().enumerate() {
        let path = api_system.join(format!("sys-{:02}.txt", idx));
        fs::write(&path, sys_msg).await.map_err(|e| {
            format!("write({}): {}", path.display(), e)
        })?;
    }

    Ok((api_stream, api_tools, api_system))
}

/// Read all visible (non-hidden) files from a list of system directories and
/// return their concatenated content.  This mirrors the logic in `agent.rs`
/// that loads system prompts during inference.
pub async fn read_system_dirs(dirs: &[String]) -> Vec<String> {
    let mut messages = Vec::new();
    for sys_dir in dirs {
        if let Ok(mut entries) = fs::read_dir(sys_dir).await {
            let mut files = Vec::new();
            while let Ok(Some(entry)) = entries.next_entry().await {
                let name_str = entry.file_name().to_string_lossy().to_string();
                // Skip hidden files (same convention as agent.rs).
                if name_str.starts_with('.') {
                    continue;
                }
                if entry.path().is_file() {
                    files.push(entry.path());
                }
            }
            files.sort();
            for f in files {
                if let Ok(content) = fs::read_to_string(&f).await {
                    messages.push(content);
                }
            }
        }
    }
    messages
}

/// Create a fresh response directory in the tree for this request.
pub async fn create_response_dir(
    tree: &SessionTree,
    current_hash: &str,
) -> Result<PathBuf, String> {
    let response_hash = hash_state(current_hash, "assistant", "");
    let response_dir = tree
        .temp_dir
        .path()
        .join(format!("assistant-{}", &response_hash[..16]));
    fs::create_dir_all(&response_dir).await.map_err(|e| {
        format!("create_dir_all({}): {}", response_dir.display(), e)
    })?;
    Ok(response_dir)
}

/// Cache an assistant response in the session tree so future requests can
/// reuse the directory.
pub async fn cache_response(
    tree: &SessionTree,
    parent_hash: &str,
    content: &str,
    response_dir: PathBuf,
) {
    let response_hash = hash_state(parent_hash, "assistant", content);
    let mut nodes = tree.nodes.lock().await;
    nodes.insert(
        response_hash,
        NodeMeta {
            dir: response_dir,
            role: "assistant".to_string(),
            parent_hash: if parent_hash.is_empty() {
                None
            } else {
                Some(parent_hash.to_string())
            },
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_determinism() {
        let a = hash_state("", "user", "hello");
        let b = hash_state("", "user", "hello");
        assert_eq!(a, b);

        let c = hash_state("", "user", "world");
        assert_ne!(a, c);

        let d = hash_state("", "assistant", "hello");
        assert_ne!(a, d);
    }

    #[tokio::test]
    async fn test_build_conversation() {
        let temp_dir = tempfile::tempdir().unwrap();
        let tree = SessionTree {
            temp_dir: Arc::new(temp_dir),
            nodes: Mutex::new(HashMap::new()),
        };

        let steps = vec![
            ConversationStep {
                role: "user".into(),
                content: "hello".into(),
            },
            ConversationStep {
                role: "assistant".into(),
                content: "hi there".into(),
            },
            ConversationStep {
                role: "user".into(),
                content: "how are you".into(),
            },
        ];

        let state = build_conversation(&tree, &steps).await.unwrap();
        assert_eq!(state.user_dirs.len(), 2);
        assert_eq!(state.assistant_dirs.len(), 1);
        assert_eq!(state.latest_user_msg, "how are you");
        assert_eq!(state.system_msgs.len(), 0);

        // Second walk with a shared prefix should reuse directories.
        let steps2 = vec![
            ConversationStep {
                role: "user".into(),
                content: "hello".into(),
            },
            ConversationStep {
                role: "assistant".into(),
                content: "hi there".into(),
            },
            ConversationStep {
                role: "user".into(),
                content: "what's up".into(),
            },
        ];
        let state2 = build_conversation(&tree, &steps2).await.unwrap();
        // First two dirs should be shared, third is new.
        assert_eq!(state2.user_dirs[0], state.user_dirs[0]);
        assert_eq!(state2.assistant_dirs[0], state.assistant_dirs[0]);
        assert_ne!(state2.user_dirs[1], state.user_dirs[1]);
    }

    #[tokio::test]
    async fn test_system_messages() {
        let temp_dir = tempfile::tempdir().unwrap();
        let tree = SessionTree {
            temp_dir: Arc::new(temp_dir),
            nodes: Mutex::new(HashMap::new()),
        };

        let steps = vec![
            ConversationStep {
                role: "system".into(),
                content: "You are helpful.".into(),
            },
            ConversationStep {
                role: "user".into(),
                content: "hi".into(),
            },
        ];

        let state = build_conversation(&tree, &steps).await.unwrap();
        assert_eq!(state.system_msgs.len(), 1);
        assert_eq!(state.system_msgs[0], "You are helpful.");
        assert_eq!(state.user_dirs.len(), 1);
        assert_eq!(state.latest_user_msg, "hi");
    }

    #[tokio::test]
    async fn test_remote_session_state_management() {
        let state = RemoteSessionState::new();
        assert!(!state.has_tree("chat-1").await);

        let tree1 = state.get_or_create_tree("chat-1").await;
        assert!(state.has_tree("chat-1").await);

        let tree2 = state.get_or_create_tree("chat-1").await;
        // Same Arc pointer (same tree).
        assert!(Arc::ptr_eq(&tree1, &tree2));

        state.remove_tree("chat-1").await;
        assert!(!state.has_tree("chat-1").await);
    }
}
