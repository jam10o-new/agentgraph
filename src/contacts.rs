use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// A single contact entry in the agent's contact book.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContactEntry {
    /// Human-readable name for this contact.
    pub name: String,
    /// Remote identifier (e.g., HTTP session ID, Telegram chat ID).
    pub remote_id: String,
    /// Optional free-form metadata (URL, description, etc.).
    pub metadata: Option<String>,
}

/// On-disk contact book stored at `~/.agentgraph/contacts/contacts.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContactBook {
    /// keyed by contact name (unique, case-sensitive).
    pub contacts: HashMap<String, ContactEntry>,
}

impl ContactBook {
    fn path() -> PathBuf {
        let home = std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("/tmp"));
        home.join(".agentgraph").join("contacts").join("contacts.json")
    }

    /// Load the contact book from disk.  Returns an empty book if the
    /// file does not exist or is corrupt.
    pub fn load() -> Self {
        let path = Self::path();
        match std::fs::read_to_string(&path) {
            Ok(json) => serde_json::from_str(&json).unwrap_or_else(|_| Self {
                contacts: HashMap::new(),
            }),
            Err(_) => Self {
                contacts: HashMap::new(),
            },
        }
    }

    /// Save the contact book to disk.
    pub fn save(&self) {
        let path = Self::path();
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(json) = serde_json::to_string_pretty(self) {
            let _ = std::fs::write(&path, &json);
        }
    }

    /// Add or update a contact.
    pub fn add(&mut self, entry: ContactEntry) {
        self.contacts.insert(entry.name.clone(), entry);
    }

    /// Remove a contact by name.
    pub fn remove(&mut self, name: &str) {
        self.contacts.remove(name);
    }

    /// Look up a contact by name.
    pub fn get(&self, name: &str) -> Option<&ContactEntry> {
        self.contacts.get(name)
    }

    /// List all contacts.
    pub fn list(&self) -> Vec<&ContactEntry> {
        let mut entries: Vec<_> = self.contacts.values().collect();
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        entries
    }
}
