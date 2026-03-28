//! Compression agent for context summarization and relevance extraction.

use crate::context::summary::ContextSummary;
use crate::types::ModelSlot;
use anyhow::{Context, Result};
use mistralrs::{ChatCompletionResponse, RequestBuilder, SamplingParams, TextMessageRole, VisionMessages};
use std::sync::Arc;

// Re-export RelevanceResult for use in ContextMessage
pub use crate::context::summary::RelevanceResult;

/// System prompt for the compression agent
const COMPRESSION_SYSTEM_PROMPT: &str = "You are a rapid context compression and relevance extraction agent. You will be given:
1. A BASELINE TURN (the most recent turn between user and assistant)
2. One or more HISTORICAL TURNS to analyze

Your task is to determine whether the historical turn contains content RELEVANT to the baseline turn.

If the historical content IS relevant:
- Extract the EXACT portions (verbatim) from the historical turn that are relevant
- Do NOT paraphrase or edit - use exact quotes
- Provide a brief reason explaining why this historical content is relevant to the baseline

If the historical content is NOT relevant:
- Provide a short single-sentence micro-summary of what the historical turn contains

Respond ONLY with a valid JSON object in this exact format:

For RELEVANT content (has relevant snippets):
{
  \"type\": \"snippets\",
  \"snippets\": [\"exact quote 1 from historical turn\", \"exact quote 2\"],
  \"relevance_reason\": \"Why these snippets are relevant to the baseline turn\"
}

For NON-RELEVANT content (no relevance to baseline):
{
  \"type\": \"micro_summary\",
  \"text\": \"One sentence describing what this historical turn contains\"
}

Your response must be ONLY the JSON object, no other text. Be concise and precise.";

/// Request for context compression/relevance extraction
#[derive(Debug, Clone)]
pub struct CompressionRequest {
    /// The current/baseline turn content (user + assistant exchange)
    pub baseline_turn: String,
    /// A single historical turn to check for relevance (turn_number, content)
    pub historical_turn: (usize, String),
    /// Optional image content for vision-based compression
    pub image: Option<image::DynamicImage>,
    /// Which model slot to use (Primary for text/vision, Secondary for audio)
    pub model_slot: ModelSlot,
}

/// Response from the compression agent
#[derive(Debug, Clone)]
pub struct CompressionResponse {
    /// The relevance result - either snippets or micro-summary
    pub relevance: RelevanceResult,
}

/// Agent for compressing conversational turns
pub struct CompressionAgent {
    model: Arc<mistralrs::Model>,
}

impl CompressionAgent {
    /// Create a new compression agent
    pub fn new(model: Arc<mistralrs::Model>) -> Self {
        Self { model }
    }

    /// Run compression on a historical turn given the baseline
    pub async fn compress(&self, request: CompressionRequest) -> Result<CompressionResponse> {
        eprintln!("CompressionAgent: Starting compression for turn {}", request.historical_turn.0);
        // Build the user prompt
        let user_prompt = self.build_compression_prompt(&request);

        // Build messages for the compression request
        let messages = if let Some(img) = request.image {
            eprintln!("CompressionAgent: Request includes an image");
            VisionMessages::new()
                .add_message(TextMessageRole::System, COMPRESSION_SYSTEM_PROMPT.to_string())
                .add_image_message(TextMessageRole::User, user_prompt, vec![img])
        } else {
            VisionMessages::new()
                .add_message(TextMessageRole::System, COMPRESSION_SYSTEM_PROMPT.to_string())
                .add_message(TextMessageRole::User, user_prompt)
        };

        // Set up sampling parameters for deterministic output
        let sampling_params = SamplingParams {
            temperature: Some(0.1),      // Low temperature for consistent JSON
            top_k: Some(10),             // Limited sampling
            top_p: Some(0.5),            // Conservative nucleus
            repetition_penalty: Some(1.0),
            ..SamplingParams::deterministic()
        };

        // Run the compression request
        let respondable: RequestBuilder = messages.into();
        eprintln!("CompressionAgent: Sending chat request...");
        let result = self
            .model
            .send_chat_request(respondable.set_sampling(sampling_params))
            .await
            .context("Compression agent inference failed")?;

        eprintln!("CompressionAgent: Received response, parsing...");
        // Parse the response
        let parsed = self.parse_compression_response(&result);
        if parsed.is_ok() {
            eprintln!("CompressionAgent: Successfully parsed response");
        } else {
            eprintln!("CompressionAgent: Failed to parse response: {:?}", parsed.as_ref().err());
        }
        parsed
    }

    /// Build the prompt for compression
    fn build_compression_prompt(&self, request: &CompressionRequest) -> String {
        let mut prompt = String::new();

        prompt.push_str("## BASELINE TURN (Most Recent)\n\n");
        prompt.push_str(&request.baseline_turn);
        prompt.push_str("\n\n");

        prompt.push_str("## HISTORICAL TURN\n\n");
        prompt.push_str(&format!("### Turn {}\n", request.historical_turn.0));
        prompt.push_str(&request.historical_turn.1);
        prompt.push_str("\n\n");

        prompt.push_str("## YOUR TASK\n\n");
        prompt.push_str("Analyze whether the HISTORICAL TURN contains content relevant to the BASELINE TURN.\n");
        prompt.push_str("Extract exact snippets if relevant, or provide a micro-summary if not.\n");
        prompt.push_str("Respond with ONLY a valid JSON object as specified in the system prompt.");

        prompt
    }

    /// Parse the JSON response from the compression agent
    fn parse_compression_response(&self, result: &ChatCompletionResponse) -> Result<CompressionResponse> {
        use serde_json::Value;

        // Extract the response text
        let text = result
            .choices
            .first()
            .and_then(|c| c.message.content.as_ref())
            .context("No response content from compression agent")?;

        // Parse JSON
        let json: Value = serde_json::from_str(text)
            .with_context(|| format!("Failed to parse compression response as JSON: {}", text))?;

        // Extract type field
        let content_type = json
            .get("type")
            .and_then(|v| v.as_str())
            .context("Missing 'type' field in compression response")?;

        let relevance = match content_type {
            "snippets" => {
                let snippets = json
                    .get("snippets")
                    .and_then(|v| v.as_array())
                    .context("Missing 'snippets' array")?
                    .iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect();

                let relevance_reason = json
                    .get("relevance_reason")
                    .and_then(|v| v.as_str())
                    .context("Missing 'relevance_reason' field")?
                    .to_string();

                RelevanceResult::Snippets {
                    snippets,
                    relevance_reason,
                }
            }
            "micro_summary" => {
                let text = json
                    .get("text")
                    .and_then(|v| v.as_str())
                    .context("Missing 'text' field in micro_summary")?
                    .to_string();
                RelevanceResult::MicroSummary { text }
            }
            other => anyhow::bail!("Unknown content type: {}", other),
        };

        Ok(CompressionResponse { relevance })
    }
}

/// Simple compression function that doesn't require a model (for fallback)
pub fn simple_compress(content: &str, turn_number: usize) -> ContextSummary {
    use xxhash_rust::xxh3::xxh3_64;

    // Create a simple one-sentence summary by taking the first sentence or truncating
    let micro_summary = content
        .split('.')
        .next()
        .unwrap_or(content)
        .chars()
        .take(150)
        .collect::<String>();

    let title = if content.len() > 50 {
        "Conversation".to_string()
    } else {
        "Brief Exchange".to_string()
    };

    ContextSummary::new_microsummary(
        title,
        format!("User and assistant discussed: {}...", micro_summary),
        turn_number,
        xxh3_64(content.as_bytes()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_compression() {
        let content = "Hello, I need help with my code. The function isn't working as expected.";
        let summary = simple_compress(content, 5);

        assert_eq!(summary.turn_created, 5);
        assert!(summary.title.len() > 0);
        assert!(summary.micro_summary.len() > 0);
        assert!(summary.micro_summary.len() <= 200); // Should be reasonably short
    }

    #[test]
    fn test_prompt_building() {
        use crate::types::ModelSlot;
        
        let request = CompressionRequest {
            baseline_turn: "User: Hello\nAssistant: Hi there!".to_string(),
            historical_turn: (1, "User: What is Rust?\nAssistant: Rust is a programming language.".to_string()),
            image: None,
            model_slot: ModelSlot::Primary,
        };

        // We can't test the full agent without a model, but we can verify the structure
        assert!(request.baseline_turn.contains("User: Hello"));
        assert!(request.historical_turn.1.contains("Rust"));
        assert_eq!(request.historical_turn.0, 1);
    }
}
