//! Summary types and storage for context management.

use serde::{Deserialize, Serialize};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// A compressed summary of a conversational turn
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextSummary {
    /// Short title describing the turn's content
    pub title: String,
    /// Brief description of what this turn contains
    pub micro_summary: String,
    /// Turn number when this summary was created
    pub turn_created: usize,
    /// Turn number when this summary was last accessed
    pub turn_last_accessed: usize,
    /// Hash of the original content for cache invalidation
    pub content_hash: u64,
}

/// The content of a relevance check - extracted snippets or a micro-summary
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RelevanceResult {
    /// Relevant snippets from the turn (verbatim excerpts)
    Snippets {
        /// List of relevant text snippets
        snippets: Vec<String>,
        /// Brief explanation of why these are relevant
        relevance_reason: String,
    },
    /// A micro-summary (single sentence description)
    MicroSummary {
        /// Brief one-sentence summary
        text: String,
    },
}

impl ContextSummary {
    /// Create a new micro-summary
    pub fn new_microsummary(title: String, micro_summary: String, turn: usize, content_hash: u64) -> Self {
        Self {
            title,
            micro_summary,
            turn_created: turn,
            turn_last_accessed: turn,
            content_hash,
        }
    }

    /// Get the summary text as a string for injection into context
    pub fn as_context_string(&self) -> String {
        format!("[Turn {} - {}]: {}", self.turn_created, self.title, self.micro_summary)
    }

    /// Get a short description for display
    pub fn brief(&self) -> String {
        format!("{}: {}", self.title, self.micro_summary)
    }
}

/// Cache for storing and retrieving context summaries
pub struct SummaryCache {
    /// Base directory for summary cache
    cache_dir: PathBuf,
}

impl SummaryCache {
    /// Create a new summary cache
    pub fn new(cache_dir: PathBuf) -> Self {
        Self { cache_dir }
    }

    /// Get the cache directory for a specific file
    pub fn get_cache_dir_for_file(&self, file_path: &Path) -> PathBuf {
        // Create a unique subdirectory based on the file's path hash
        let file_hash = xxhash_rust::xxh3::xxh3_64(file_path.to_string_lossy().as_bytes());
        self.cache_dir.join(format!("{:016x}", file_hash))
    }

    /// Get the summary file path for a given source file
    fn get_summary_path(&self, file_path: &Path) -> PathBuf {
        let cache_dir = self.get_cache_dir_for_file(file_path);
        let file_hash = xxhash_rust::xxh3::xxh3_64(file_path.to_string_lossy().as_bytes());
        cache_dir.join(format!("{:016x}.summary.json", file_hash))
    }

    /// Check if a summary exists for a file and is valid (hash matches)
    pub fn has_summary(&self, file_path: &Path) -> bool {
        let summary_path = self.get_summary_path(file_path);
        if !summary_path.exists() {
            return false;
        }
        
        // Also verify hash matches (cache invalidation)
        if let Ok(content) = fs::read_to_string(&summary_path) {
            if let Ok(summary) = serde_json::from_str::<ContextSummary>(&content) {
                let current_hash = self.compute_content_hash(file_path);
                return summary.content_hash == current_hash;
            }
        }
        false
    }

    /// Load a summary for a file if it exists and is valid
    pub fn load_summary(&self, file_path: &Path) -> Option<ContextSummary> {
        let summary_path = self.get_summary_path(file_path);
        
        if !summary_path.exists() {
            return None;
        }

        // Read and parse the summary
        let content = fs::read_to_string(&summary_path).ok()?;
        let summary: ContextSummary = serde_json::from_str(&content).ok()?;

        // Verify content hash matches (cache invalidation)
        let current_hash = self.compute_content_hash(file_path);
        if summary.content_hash != current_hash {
            // Cache miss - content has changed
            return None;
        }

        Some(summary)
    }

    /// Save a summary for a file
    pub fn save_summary(&self, file_path: &Path, summary: &ContextSummary) -> io::Result<()> {
        let summary_path = self.get_summary_path(file_path);
        let cache_dir = summary_path.parent().unwrap();

        // Create cache directory if it doesn't exist
        fs::create_dir_all(cache_dir)?;

        // Serialize and write the summary
        let content = serde_json::to_string_pretty(summary)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

        fs::write(&summary_path, content)?;
        Ok(())
    }

    /// Compute hash of file content for cache invalidation
    fn compute_content_hash(&self, file_path: &Path) -> u64 {
        if let Ok(content) = fs::read_to_string(file_path) {
            xxhash_rust::xxh3::xxh3_64(content.as_bytes())
        } else {
            0
        }
    }

    /// Get the global cache directory
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// Clear old summaries (older than specified turns)
    pub fn clear_old_summaries(&self, max_age: usize, current_turn: usize) -> io::Result<usize> {
        let mut cleared = 0;

        if !self.cache_dir.exists() {
            return Ok(0);
        }

        for entry in fs::read_dir(&self.cache_dir)? {
            let entry = entry?;
            let path = entry.path();
            
            if path.is_dir() {
                for summary_file in fs::read_dir(path)? {
                    let summary_file = summary_file?;
                    if summary_file.path().extension().map(|e| e == "json").unwrap_or(false) {
                        if let Ok(content) = fs::read_to_string(summary_file.path()) {
                            if let Ok(summary) = serde_json::from_str::<ContextSummary>(&content) {
                                if current_turn.saturating_sub(summary.turn_last_accessed) > max_age {
                                    fs::remove_file(summary_file.path())?;
                                    cleared += 1;
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(cleared)
    }
}

/// Strategy for loading message content
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SummaryLoadStrategy {
    /// Load full content
    Full,
    /// Load from existing microsummary
    MicroSummary,
    /// Generate new summary via compression
    Compress,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_summary_creation() {
        let summary = ContextSummary::new_microsummary(
            "User Question".to_string(),
            "User asked about context management".to_string(),
            5,
            12345,
        );

        assert_eq!(summary.title, "User Question");
        assert_eq!(summary.micro_summary, "User asked about context management");
        assert_eq!(summary.turn_created, 5);
    }

    #[test]
    fn test_summary_cache_roundtrip() {
        let temp_dir = TempDir::new().unwrap();
        let cache = SummaryCache::new(temp_dir.path().to_path_buf());

        let test_file = temp_dir.path().join("test.txt");
        fs::write(&test_file, "Test content").unwrap();

        let summary = ContextSummary::new_microsummary(
            "Test".to_string(),
            "Test summary".to_string(),
            1,
            cache.compute_content_hash(&test_file),
        );

        cache.save_summary(&test_file, &summary).unwrap();
        assert!(cache.has_summary(&test_file));

        let loaded = cache.load_summary(&test_file).unwrap();
        assert_eq!(loaded.title, summary.title);
    }

    #[test]
    fn test_cache_invalidation() {
        let temp_dir = TempDir::new().unwrap();
        let cache = SummaryCache::new(temp_dir.path().to_path_buf());

        let test_file = temp_dir.path().join("test.txt");
        fs::write(&test_file, "Original content").unwrap();

        let summary = ContextSummary::new_microsummary(
            "Test".to_string(),
            "Test summary".to_string(),
            1,
            cache.compute_content_hash(&test_file),
        );

        cache.save_summary(&test_file, &summary).unwrap();
        assert!(cache.has_summary(&test_file));

        // Modify the file
        fs::write(&test_file, "Modified content").unwrap();

        // Cache should be invalidated
        assert!(!cache.has_summary(&test_file));
        assert!(cache.load_summary(&test_file).is_none());
    }

    #[test]
    fn test_context_string_formatting() {
        let summary = ContextSummary::new_microsummary(
            "Discussion".to_string(),
            "Agents discussed the architecture".to_string(),
            10,
            0,
        );

        let context = summary.as_context_string();
        assert!(context.contains("Turn 10"));
        assert!(context.contains("Discussion"));
        assert!(context.contains("Agents discussed the architecture"));
    }
}
