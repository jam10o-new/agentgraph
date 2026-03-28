//! Context management for long conversations.
//!
//! This module provides context compression and summarization capabilities
//! to handle long conversations with small local models. It implements:
//! - Turn-age-based probability scaling for loading strategies
//! - Summary caching in hidden folders alongside source files
//! - Compression agent for rapid context summarization
//! - Multi-modal support (text, images, audio)

mod compression;
mod summary;

pub use compression::{CompressionAgent, CompressionRequest, CompressionResponse, simple_compress};
pub use summary::{ContextSummary, RelevanceResult, SummaryCache, SummaryLoadStrategy};

use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// A message with its turn index and relevance information
#[derive(Debug, Clone)]
pub struct ContextMessage {
    /// The turn number when this message was created
    pub turn: usize,
    /// The role (user, assistant, system)
    pub role: String,
    /// The full content (text, image, or audio)
    pub content: String,
    /// Whether this is text, image, or audio content
    pub content_type: ContextMessageType,
}

/// Type of message content
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ContextMessageType {
    Text,
    Image,
    Audio,
}

/// Result of relevance extraction for a historical message
#[derive(Debug, Clone)]
pub enum RelevanceExtraction {
    /// Full content should be used (recent message)
    Full { content: String },
    /// Micro-summary should be used (old, not relevant)
    MicroSummary { summary: String },
    /// Relevant snippets extracted (old, relevant to baseline)
    RelevantSnippets {
        snippets: Vec<String>,
        relevance_reason: String,
    },
}

impl ContextMessage {
    /// Convert a ContextMessage to a relevance extraction result (synchronous, heuristic-based)
    pub fn extract_relevance(
        &self,
        baseline_turn: &str,
        compression_agent: Option<&CompressionAgent>,
    ) -> RelevanceExtraction {
        // For now, use simple heuristic - in production, this would use the compression agent
        if let Some(_agent) = compression_agent {
            // Use the compression agent to extract relevant snippets
            // This is async, so we'd need to handle it differently
            // For now, fall through to simple heuristic
        }

        // Simple heuristic: if content mentions keywords from baseline, it's relevant
        let baseline_lower = baseline_turn.to_lowercase();
        let content_lower = self.content.to_lowercase();
        
        // Check for keyword overlap (very simple heuristic)
        let baseline_words: Vec<&str> = baseline_lower
            .split_whitespace()
            .filter(|w| w.len() > 4)
            .collect();
        
        let mut overlap = 0;
        for word in &baseline_words {
            if content_lower.contains(word) {
                overlap += 1;
            }
        }

        if overlap >= 2 {
            // Potentially relevant - extract snippets (simple version)
            let snippets: Vec<String> = self.content
                .split('.')
                .filter(|s| {
                    baseline_words.iter().any(|w| s.to_lowercase().contains(w))
                })
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .take(3)
                .collect();
            
            if !snippets.is_empty() {
                RelevanceExtraction::RelevantSnippets {
                    snippets,
                    relevance_reason: format!("Shares {} key terms with baseline", overlap),
                }
            } else {
                RelevanceExtraction::MicroSummary {
                    summary: format!("[Turn {}] {}", self.turn, self.content.chars().take(100).collect::<String>()),
                }
            }
        } else {
            // Not relevant - use micro-summary
            RelevanceExtraction::MicroSummary {
                summary: format!("[Turn {}] {}", self.turn, self.content.chars().take(100).collect::<String>()),
            }
        }
    }

    /// Convert a ContextMessage to a relevance extraction result (async, uses compression agent)
    pub async fn extract_relevance_async(
        &self,
        baseline_turn: &str,
        compression_agent: &CompressionAgent,
        model_slot: crate::types::ModelSlot,
    ) -> anyhow::Result<RelevanceExtraction> {
        use compression::{CompressionRequest, RelevanceResult};
        
        let request = CompressionRequest {
            baseline_turn: baseline_turn.to_string(),
            historical_turn: (self.turn, self.content.clone()),
            image: None, // ContextMessage content is currently String-only
            model_slot,
        };

        let response = compression_agent.compress(request).await?;
        
        Ok(match response.relevance {
            RelevanceResult::Snippets { snippets, relevance_reason } => {
                RelevanceExtraction::RelevantSnippets {
                    snippets,
                    relevance_reason,
                }
            }
            RelevanceResult::MicroSummary { text } => {
                RelevanceExtraction::MicroSummary {
                    summary: format!("[Turn {}] {}", self.turn, text),
                }
            }
        })
    }

    /// Format a relevance extraction result as a string for context injection
    pub fn format_relevance_result(result: &RelevanceExtraction) -> String {
        match result {
            RelevanceExtraction::Full { content } => content.clone(),
            RelevanceExtraction::MicroSummary { summary } => summary.clone(),
            RelevanceExtraction::RelevantSnippets { snippets, relevance_reason } => {
                let mut output = format!("[RELEVANT HISTORICAL CONTENT - {}]:\n", relevance_reason);
                for snippet in snippets {
                    output.push_str(&format!("  \"{}\"\n", snippet));
                }
                output
            }
        }
    }
}

/// Configuration for context management
#[derive(Debug, Clone)]
pub struct ContextManagerConfig {
    /// Base probability for loading full content (at turn 0)
    pub base_full_probability: f64,
    /// Decay rate for exponential probability scaling
    pub decay_rate: f64,
    /// Ratio of microsummary vs re-summarization when not loading full
    pub microsummary_ratio: f64,
    /// Minimum probability for loading full content (floor)
    pub min_full_probability: f64,
    /// Maximum turn age to consider for summarization
    pub max_turn_age: usize,
}

impl Default for ContextManagerConfig {
    fn default() -> Self {
        Self {
            base_full_probability: 0.95, // Recent turns almost always loaded in full
            decay_rate: 0.3,             // Moderate decay rate
            microsummary_ratio: 0.7,     // 70% microsummary, 30% re-summarization when not full
            min_full_probability: 0.1,   // Even old turns have 10% chance of full load
            max_turn_age: 20,            // After 20 turns, use steady-state probabilities
        }
    }
}

/// Strategy selected for loading a message
#[derive(Debug, Clone, PartialEq)]
pub enum LoadStrategy {
    /// Load entire message content
    Full,
    /// Load existing microsummary (no new summarization)
    MicroSummary,
    /// Run compression agent to generate new summary
    Compress,
}

/// Context manager for handling long conversations
pub struct ContextManager {
    config: ContextManagerConfig,
    summary_cache: SummaryCache,
    current_turn: usize,
}

impl ContextManager {
    /// Create a new context manager with default configuration
    pub fn new(cache_dir: PathBuf) -> Self {
        Self {
            config: ContextManagerConfig::default(),
            summary_cache: SummaryCache::new(cache_dir),
            current_turn: 0,
        }
    }

    /// Create a new context manager with custom configuration
    pub fn with_config(cache_dir: PathBuf, config: ContextManagerConfig) -> Self {
        Self {
            config,
            summary_cache: SummaryCache::new(cache_dir),
            current_turn: 0,
        }
    }

    /// Increment the current turn counter
    pub fn next_turn(&mut self) {
        self.current_turn += 1;
    }

    /// Get the current turn number
    pub fn current_turn(&self) -> usize {
        self.current_turn
    }

    /// Set the current turn number (useful for testing)
    #[cfg(test)]
    pub fn set_current_turn(&mut self, turn: usize) {
        self.current_turn = turn;
    }

    /// Determine load strategy for a message based on turn age and message properties
    pub fn determine_load_strategy(
        &self,
        message_turn: usize,
        role: &str,
        has_existing_summary: bool,
    ) -> LoadStrategy {
        // System messages are never summarized
        if role == "system" {
            return LoadStrategy::Full;
        }

        let turn_age = self.current_turn.saturating_sub(message_turn);

        // Clamp turn age for probability calculation
        let effective_age = turn_age.min(self.config.max_turn_age) as f64;

        // Calculate probability of loading full content using exponential decay
        let p_full = self.config.min_full_probability
            + (self.config.base_full_probability - self.config.min_full_probability)
                * (-self.config.decay_rate * effective_age).exp();

        let roll: f64 = rand::random();

        if roll < p_full {
            LoadStrategy::Full
        } else if has_existing_summary {
            // When not loading full, use microsummary_ratio to decide between microsummary and re-summarization
            let conditional_roll: f64 = rand::random();
            if conditional_roll < self.config.microsummary_ratio {
                LoadStrategy::MicroSummary
            } else {
                LoadStrategy::Compress
            }
        } else {
            // No existing summary, must compress
            LoadStrategy::Compress
        }
    }

    /// Get the summary cache directory for a given file
    pub fn get_cache_dir_for_file(&self, file_path: &Path) -> PathBuf {
        self.summary_cache.get_cache_dir_for_file(file_path)
    }

    /// Check if a summary exists for a file
    pub fn has_summary(&self, file_path: &Path) -> bool {
        self.summary_cache.has_summary(file_path)
    }

    /// Load an existing summary for a file
    pub fn load_summary(&self, file_path: &Path) -> Option<ContextSummary> {
        self.summary_cache.load_summary(file_path)
    }

    /// Save a summary for a file
    pub fn save_summary(&self, file_path: &Path, summary: &ContextSummary) -> std::io::Result<()> {
        self.summary_cache.save_summary(file_path, summary)
    }

    /// Get the global cache directory
    pub fn cache_dir(&self) -> &Path {
        self.summary_cache.cache_dir()
    }
}

/// Helper function to calculate turn age from message metadata
pub fn calculate_turn_age(
    message_created: SystemTime,
    conversation_start: SystemTime,
    avg_turn_duration: std::time::Duration,
) -> usize {
    let elapsed = message_created
        .elapsed()
        .unwrap_or_else(|_| std::time::Duration::from_secs(0));
    let total_conversation_time = conversation_start
        .elapsed()
        .unwrap_or_else(|_| std::time::Duration::from_secs(0));

    if avg_turn_duration.as_secs() == 0 {
        return 0;
    }

    let estimated_turns = (total_conversation_time.as_secs_f64() / avg_turn_duration.as_secs_f64()) as usize;
    let message_turn = (elapsed.as_secs_f64() / avg_turn_duration.as_secs_f64()) as usize;

    estimated_turns.saturating_sub(message_turn)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_system_messages_never_summarized() {
        let temp_dir = TempDir::new().unwrap();
        let manager = ContextManager::new(temp_dir.path().to_path_buf());

        // System messages should always be loaded in full regardless of age
        assert_eq!(
            manager.determine_load_strategy(0, "system", false),
            LoadStrategy::Full
        );
        assert_eq!(
            manager.determine_load_strategy(100, "system", true),
            LoadStrategy::Full
        );
    }

    #[test]
    fn test_recent_messages_prefer_full() {
        let temp_dir = TempDir::new().unwrap();
        let manager = ContextManager::new(temp_dir.path().to_path_buf());

        // Recent messages (turn_age = 0) should have high probability of full load
        let mut full_count = 0;
        for _ in 0..100 {
            if manager.determine_load_strategy(10, "user", false) == LoadStrategy::Full {
                full_count += 1;
            }
        }
        // With base_full_probability = 0.95, we expect ~95 full loads
        assert!(full_count > 80, "Expected most recent messages to load full, got {} / 100", full_count);
    }

    #[test]
    fn test_old_messages_have_lower_full_probability() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = ContextManager::new(temp_dir.path().to_path_buf());
        
        // Advance the current turn to simulate a long conversation
        manager.set_current_turn(20);

        // Old messages (turn_age = 20, message from turn 0) should have lower probability of full load
        let mut full_count = 0;
        for _ in 0..100 {
            // Simulate message from turn 0, current turn is 20
            if manager.determine_load_strategy(0, "user", false) == LoadStrategy::Full {
                full_count += 1;
            }
        }
        // With min_full_probability = 0.1 and decay, we expect ~10-20% full loads
        assert!(full_count < 40, "Expected few old messages to load full, got {} / 100", full_count);
    }
}
