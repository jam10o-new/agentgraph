//! RemoteSession layer — shared session-tree management used by both the
//! HTTP API and the Telegram bot.  Each frontend registers sessions keyed by
//! its own identifier (model name for HTTP, chat_id for Telegram) and
//! benefits from content-addressed directory sharing across requests.
//!
//! # Persistence model
//!
//! A session has an **active tree** (the current live conversation) and
//! optionally an **archive** (a persisted snapshot on disk under
//! `~/.agentgraph/sessions/`).
//!
//! * **`/persist`** — copies only the active branch (the path from current_hash
//!   to root, no siblings) into `~/.agentgraph/sessions/`.  The active tree
//!   stays as a tempdir and the conversation continues uninterrupted.
//!
//! * **`/reset`** — drops the active tree, creates a fresh tempdir.  The archive
//!   (if any) is kept on disk.
//!
//! * **`/delete`** — drops the active tree AND removes the archive from disk
//!   + index.
//!
//! On leader restart, `get_or_create_tree` checks the persistence index and
//! loads the archived tree as the new active root.

use serde::{Deserialize, Serialize};

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::fs;
use tokio::sync::Mutex;

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// Metadata for a node in the conversation tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeMeta {
    pub dir: PathBuf,
    pub role: String,
    pub parent_hash: Option<String>,
}

/// A live conversation tree rooted at some directory on disk.
/// The root may be a tempdir (ephemeral) or a persistent path.
pub struct SessionTree {
    /// Root directory — all node dirs live under here.
    pub root: PathBuf,
    /// Keeps the tempdir alive; `None` when this tree uses a persistent path
    /// (loaded from archive or created fresh at a persistent root).
    pub temp_cleanup: Option<tempfile::TempDir>,
    /// hash → node lookup.
    pub nodes: Mutex<HashMap<String, NodeMeta>>,
}

/// A frozen snapshot of a previously persisted conversation branch.
#[derive(Debug, Clone)]
pub struct SessionArchive {
    /// Root directory under `~/.agentgraph/sessions/`.
    pub root: PathBuf,
    /// The agent name this archive belongs to.
    pub agent: String,
    /// All nodes on the branch (loaded from meta.json).
    pub nodes: HashMap<String, NodeMeta>,
}

/// A session entry, combining an active tree with an optional archive.
pub struct SessionEntry {
    /// Current live tree where new writes go.
    pub active: Arc<SessionTree>,
    /// Frozen persisted snapshot.  Populated after `/persist` and kept
    /// through `/reset`; `None` when active IS the persistent tree
    /// (i.e. after loading from archive on restart).
    pub archive: Option<SessionArchive>,
}

/// Central session state shared across all remote interfaces (HTTP, Telegram).
pub struct RemoteSessionState {
    /// Per-identifier session entries.  The identifier is frontend-specific:
    /// model name for the HTTP API, chat_id for Telegram.
    pub sessions: Mutex<HashMap<String, SessionEntry>>,
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
// Index helpers  (~/.agentgraph/sessions/index.json)
// ---------------------------------------------------------------------------

/// Index entry for a single persisted session.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct IndexEntry {
    agent: String,
    path: String,
}

/// On-disk index mapping session_id → {agent, path}.
#[derive(Debug, Serialize, Deserialize)]
struct PersistIndex {
    sessions: HashMap<String, IndexEntry>,
}

fn home_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp"))
}

fn index_path() -> PathBuf {
    home_dir()
        .join(".agentgraph")
        .join("sessions")
        .join("index.json")
}

async fn read_index() -> PersistIndex {
    let path = index_path();
    match fs::read_to_string(&path).await {
        Ok(json) => serde_json::from_str(&json).unwrap_or(PersistIndex {
            sessions: HashMap::new(),
        }),
        Err(_) => PersistIndex {
            sessions: HashMap::new(),
        },
    }
}

async fn write_index(index: &PersistIndex) {
    let path = index_path();
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent).await;
    }
    if let Ok(json) = serde_json::to_string_pretty(index) {
        let _ = fs::write(&path, &json).await;
    }
}

/// Return the agent-nested path for a persisted session, e.g.
/// `~/.agentgraph/sessions/{agent}/{session_id}/`.
fn archive_path(agent: &str, session_id: &str) -> PathBuf {
    home_dir()
        .join(".agentgraph")
        .join("sessions")
        .join(agent)
        .join(session_id)
}

// ---------------------------------------------------------------------------
// RemoteSessionState
// ---------------------------------------------------------------------------

impl RemoteSessionState {
    /// Create a new (empty) session state.
    pub fn new() -> Self {
        Self {
            sessions: Mutex::new(HashMap::new()),
        }
    }

    /// Get or create the session entry for the given identifier.
    ///
    /// On leader start, checks the persistence index first.  If the session
    /// was previously persisted, loads the archive and creates an active tree
    /// rooted at the persisted path (no separate archive needed since the
    /// active tree IS the archive).  Otherwise creates a fresh tempdir.
    pub async fn get_or_create_tree(&self, id: &str) -> Arc<SessionTree> {
        let index = read_index().await;
        let mut sessions = self.sessions.lock().await;
        Arc::clone(
            &sessions
                .entry(id.to_string())
                .or_insert_with(|| {
                    // Check persistence index for a previously archived session.
                    if let Some(entry) = index.sessions.get(id) {
                        let persist_root = PathBuf::from(&entry.path);
                        // Load meta.json if it exists.
                        let nodes = Self::load_archive_nodes(&persist_root);
                        Self::log_info(&format!(
                            "Loaded persisted tree for {id} from {} ({} nodes)",
                            persist_root.display(),
                            nodes.len()
                        ));
                        let active_root = persist_root.clone();
                        let archive_nodes = nodes.clone();
                        SessionEntry {
                            active: Arc::new(SessionTree {
                                root: active_root,
                                temp_cleanup: None,
                                nodes: Mutex::new(nodes),
                            }),
                            // Keep an archive reference so `/reset` doesn't
                            // orphan the persisted data — it remains
                            // queryable via list_children / get_path.
                            archive: Some(SessionArchive {
                                root: persist_root,
                                agent: entry.agent.clone(),
                                nodes: archive_nodes,
                            }),
                        }
                    } else {
                        let td = tempfile::tempdir()
                            .expect("Failed to create API session temp dir");
                        let root = td.path().to_path_buf();
                        SessionEntry {
                            active: Arc::new(SessionTree {
                                root,
                                temp_cleanup: Some(td),
                                nodes: Mutex::new(HashMap::new()),
                            }),
                            archive: None,
                        }
                    }
                })
                .active,
        )
    }

    /// Return the active tree AND a reference to the archive (for merged
    /// lookups).  Callers that only need the tree should use
    /// `get_or_create_tree`.
    pub async fn get_entry(&self, id: &str) -> Option<(Arc<SessionTree>, Option<SessionArchive>)> {
        let sessions = self.sessions.lock().await;
        sessions.get(id).map(|e| {
            (
                Arc::clone(&e.active),
                e.archive.clone(),
            )
        })
    }

    /// Check whether a tree exists for the given id.
    pub async fn has_tree(&self, id: &str) -> bool {
        self.sessions.lock().await.contains_key(id)
    }

    /// Remove a session entry (e.g. when a chat is cleared).  Does NOT remove
    /// the archive from disk — see `delete_persisted`.
    pub async fn remove_tree(&self, id: &str) {
        self.sessions.lock().await.remove(id);
    }

    /// List all active session identifiers.
    pub async fn list_ids(&self) -> Vec<String> {
        self.sessions.lock().await.keys().cloned().collect()
    }

    /// Persist the active branch (the chain from `current_hash` to root) to
    /// `~/.agentgraph/sessions/{agent}/{session_id}/`.
    ///
    /// After persisting, the active tree remains a tempdir (unchanged).  An
    /// archive entry is created so branches can be scanned and so the session
    /// can be resumed after restart.
    pub async fn persist_branch(
        &self,
        session_id: &str,
        agent: &str,
        current_hash: &str,
    ) -> Result<(), String> {
        let mut sessions = self.sessions.lock().await;
        let entry = sessions
            .get(session_id)
            .ok_or_else(|| format!("session {session_id} not found"))?;

        let dest_root = archive_path(agent, session_id);
        let tree = &entry.active;
        let nodes = tree.nodes.lock().await;

        // Walk from current_hash to root to find the active branch.
        let branch_hashes = walk_to_root(&nodes, current_hash);
        if branch_hashes.is_empty() {
            return Err("no nodes on active branch".into());
        }

        // Create dest root.
        fs::create_dir_all(&dest_root)
            .await
            .map_err(|e| format!("create_dir_all({}): {}", dest_root.display(), e))?;

        // Copy each node directory.
        let mut archive_nodes = HashMap::new();
        for hash in &branch_hashes {
            let Some(meta) = nodes.get(hash) else {
                continue;
            };
            let src = &meta.dir;
            let dir_name = src
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            let dest = dest_root.join(&dir_name);
            // Only copy if the dest doesn't already exist (content-addressed
            // means re-persisting the same branch is a no-op for each node).
            if !dest.exists() {
                if let Err(e) = copy_dir(src, &dest).await {
                    return Err(format!(
                        "copy_dir({}, {}): {}",
                        src.display(),
                        dest.display(),
                        e
                    ));
                }
            }
            archive_nodes.insert(
                hash.clone(),
                NodeMeta {
                    dir: dest,
                    role: meta.role.clone(),
                    parent_hash: meta.parent_hash.clone(),
                },
            );
        }

        // Write meta.json.
        let meta_json = serde_json::to_string(&archive_nodes)
            .map_err(|e| format!("serialize meta.json: {e}"))?;
        fs::write(dest_root.join("meta.json"), &meta_json)
            .await
            .map_err(|e| format!("write meta.json: {e}"))?;

        // Drop nodes lock before awaiting index write.
        drop(nodes);

        // Upsert index.
        let mut index = read_index().await;
        index.sessions.insert(
            session_id.to_string(),
            IndexEntry {
                agent: agent.to_string(),
                path: dest_root.to_string_lossy().to_string(),
            },
        );
        write_index(&index).await;

        // Set archive on the entry (active tree stays as tempdir).
        // We must re-lock and re-get entry because we dropped sessions lock
        // briefly... actually we still hold sessions lock. That's fine.
        if let Some(entry) = sessions.get_mut(session_id) {
            entry.archive = Some(SessionArchive {
                root: dest_root,
                agent: agent.to_string(),
                nodes: archive_nodes,
            });
        }

        Ok(())
    }

    /// Reset: drop the active tree and create a fresh tempdir.  The archive
    /// (if any) is kept on disk and remains queryable via list_children.
    pub async fn reset_tree(&self, id: &str) {
        let mut sessions = self.sessions.lock().await;
        if let Some(entry) = sessions.get_mut(id) {
            let td = tempfile::tempdir().expect("Failed to create temp dir");
            let root = td.path().to_path_buf();
            entry.active = Arc::new(SessionTree {
                root,
                temp_cleanup: Some(td),
                nodes: Mutex::new(HashMap::new()),
            });
        }
    }

    /// Delete: remove the active tree AND the persistent archive from disk
    /// + index.  The entire session entry is removed.
    pub async fn delete_tree(&self, id: &str) {
        let mut sessions = self.sessions.lock().await;
        if let Some(entry) = sessions.remove(id) {
            // Determine which on-disk roots need cleanup.
            let mut roots_to_remove = Vec::new();

            // Active tree root — if it's persistent (no temp_cleanup),
            // remove it from disk.
            if entry.active.temp_cleanup.is_none() {
                roots_to_remove.push(entry.active.root.clone());
            }

            // Archive root — if it's a different path from the active root,
            // remove it too.
            if let Some(ref archive) = entry.archive {
                if !roots_to_remove.contains(&archive.root) {
                    roots_to_remove.push(archive.root.clone());
                }
            }

            drop(entry);
            drop(sessions);

            for root in &roots_to_remove {
                let _ = fs::remove_dir_all(root).await;
            }

            // Remove from persistence index.
            let mut index = read_index().await;
            index.sessions.remove(id);
            write_index(&index).await;
        }
    }

    /// List all child nodes of `parent_hash` within the session identified by
    /// `id`.  Merges results from both the active tree and the archive.
    pub async fn list_children(
        &self,
        id: &str,
        parent_hash: &str,
    ) -> Vec<(String, String, String)> {
        let sessions = self.sessions.lock().await;
        let Some(entry) = sessions.get(id) else {
            return Vec::new();
        };

        let mut seen = HashMap::new();

        // Collect from active tree.
        {
            let nodes = entry.active.nodes.lock().await;
            for (hash, meta) in nodes.iter() {
                if parent_hash_matches(meta, parent_hash) {
                    seen.insert(
                        hash.clone(),
                        (
                            hash.clone(),
                            meta.role.clone(),
                            meta.dir.join("msg.txt").to_string_lossy().to_string(),
                        ),
                    );
                }
            }
        }

        // Collect from archive (if any and different from active).
        if let Some(ref archive) = entry.archive {
            for (hash, meta) in &archive.nodes {
                if parent_hash_matches(meta, parent_hash) && !seen.contains_key(hash) {
                    seen.insert(
                        hash.clone(),
                        (
                            hash.clone(),
                            meta.role.clone(),
                            meta.dir.join("msg.txt").to_string_lossy().to_string(),
                        ),
                    );
                }
            }
        }

        let mut children: Vec<_> = seen.into_values().collect();
        children.sort_by(|a, b| a.0.cmp(&b.0));
        children
    }

    /// Walk the parent chain from `hash` to the root, returning the path
    /// as a chronological Vec of (hash, role, content_preview).  Checks both
    /// active and archive nodes.
    pub async fn get_path(&self, id: &str, hash: &str) -> Vec<(String, String, String)> {
        let sessions = self.sessions.lock().await;
        let Some(entry) = sessions.get(id) else {
            return Vec::new();
        };

        let mut path = Vec::new();
        let mut current = hash.to_string();

        loop {
            if current.is_empty() {
                break;
            }
            // Check active first.
            let found = {
                let nodes = entry.active.nodes.lock().await;
                nodes.get(&current).cloned()
            };
            let meta = match found {
                Some(m) => m,
                None => {
                    // Check archive.
                    match entry.archive.as_ref().and_then(|a| a.nodes.get(&current)) {
                        Some(m) => m.clone(),
                        None => break,
                    }
                }
            };
            path.push((
                current.clone(),
                meta.role.clone(),
                meta.dir.join("msg.txt").to_string_lossy().to_string(),
            ));
            current = meta.parent_hash.unwrap_or_default();
        }

        path.reverse();
        path
    }

    // ── Private helpers ──────────────────────────────────────────────

    fn load_archive_nodes(root: &Path) -> HashMap<String, NodeMeta> {
        let meta_path = root.join("meta.json");
        match std::fs::read_to_string(&meta_path) {
            Ok(json) => serde_json::from_str(&json).unwrap_or_default(),
            Err(_) => HashMap::new(),
        }
    }

    fn log_info(msg: &str) {
        eprintln!("[remote_session] {msg}");
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
                    .root
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
        .root
        .join(format!("stream-{}", uuid::Uuid::new_v4()));
    let api_tools = tree
        .root
        .join(format!("tools-{}", uuid::Uuid::new_v4()));
    let api_system = tree
        .root
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
        .root
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

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Walk the parent chain from `hash` to root, returning hashes in
/// root-to-leaf order (chronological).
fn walk_to_root(nodes: &HashMap<String, NodeMeta>, hash: &str) -> Vec<String> {
    let mut chain = Vec::new();
    let mut current = hash.to_string();
    while !current.is_empty() {
        if let Some(meta) = nodes.get(&current) {
            chain.push(current.clone());
            current = meta.parent_hash.clone().unwrap_or_default();
        } else {
            break;
        }
    }
    chain.reverse();
    chain
}

fn parent_hash_matches(meta: &NodeMeta, parent_hash: &str) -> bool {
    match (&meta.parent_hash, parent_hash) {
        (Some(ph), pp) if pp.is_empty() => false, // non-root can't match root query
        (Some(ph), pp) => ph == pp,
        (None, "") => true,  // both root
        (None, _) => false,  // root can't match non-root query
    }
}

/// Recursive directory copy.
async fn copy_dir(src: &Path, dest: &Path) -> Result<(), String> {
    fs::create_dir_all(dest)
        .await
        .map_err(|e| format!("create_dir_all({}): {}", dest.display(), e))?;
    let mut entries = fs::read_dir(src)
        .await
        .map_err(|e| format!("read_dir({}): {}", src.display(), e))?;
    while let Some(entry) = entries
        .next_entry()
        .await
        .map_err(|e| format!("next_entry({}): {}", src.display(), e))?
    {
        let src_path = entry.path();
        let dest_path = dest.join(entry.file_name());
        if src_path.is_dir() {
            Box::pin(copy_dir(&src_path, &dest_path)).await?;
        } else {
            fs::copy(&src_path, &dest_path)
                .await
                .map_err(|e| format!("copy({}, {}): {}", src_path.display(), dest_path.display(), e))?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

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
        let td = tempfile::tempdir().unwrap();
        let root = td.path().to_path_buf();
        let tree = SessionTree {
            root,
            temp_cleanup: Some(td),
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
        let td = tempfile::tempdir().unwrap();
        let root = td.path().to_path_buf();
        let tree = SessionTree {
            root,
            temp_cleanup: Some(td),
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

    #[tokio::test]
    async fn test_walk_to_root() {
        let td = tempfile::tempdir().unwrap();
        let root = td.path().to_path_buf();
        let tree = SessionTree {
            root,
            temp_cleanup: Some(td),
            nodes: Mutex::new(HashMap::new()),
        };

        let steps = vec![
            ConversationStep { role: "user".into(), content: "a".into() },
            ConversationStep { role: "assistant".into(), content: "b".into() },
            ConversationStep { role: "user".into(), content: "c".into() },
        ];
        let state = build_conversation(&tree, &steps).await.unwrap();

        let nodes = tree.nodes.lock().await;
        let chain = walk_to_root(&nodes, &state.current_hash);
        assert_eq!(chain.len(), 3, "should include all 3 nodes");
        // First hash in chain should be the root (user "a").
        let first = nodes.get(&chain[0]).unwrap();
        assert_eq!(first.role, "user");
        assert!(first.parent_hash.is_none());
    }
}
