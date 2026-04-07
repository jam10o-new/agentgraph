use anyhow::{Context, Result, anyhow};
use mistralrs::{
    ChatCompletionResponse, Model, RequestBuilder, SamplingParams, TextMessageRole, TextMessages,
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::fs;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RootSummary {
    pub title: String,
    pub micro_summary: String,
    #[serde(default)]
    pub domains: Vec<String>, // List of domain names for which specialized summaries exist
    #[serde(default)]
    pub included_turn_indices: Vec<usize>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SpecializedSummary {
    pub domain: String,
    pub extracted_information: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HistoryTurnRole {
    User,
    Assistant,
    System,
    Skill(String, String), // Name, Description
}

#[derive(Debug, Clone)]
pub struct HistoryTurn {
    pub role: HistoryTurnRole,
    pub content: String,
    pub turn_index: usize,
}

pub struct CompressionManager {
    pub base_path: PathBuf,
    pub threshold: f64,
    pub inverse_prob: f64,
    pub resummarize_prob: f64,
}

impl CompressionManager {
    pub fn new(
        base_path: PathBuf,
        threshold: f64,
        inverse_prob: f64,
        resummarize_prob: f64,
    ) -> Self {
        Self {
            base_path: base_path.join(".agent_context"),
            threshold,
            inverse_prob,
            resummarize_prob,
        }
    }

    fn turn_cache_dir(&self, turn_idx: usize) -> PathBuf {
        self.base_path.join(format!("turn_{}", turn_idx))
    }

    fn root_summary_path(&self, turn_idx: usize) -> PathBuf {
        self.turn_cache_dir(turn_idx).join("root.json")
    }

    fn specialized_summary_path(&self, turn_idx: usize, domain: &str) -> PathBuf {
        self.turn_cache_dir(turn_idx)
            .join(format!("domain_{}.json", domain))
    }

    pub async fn get_compressed_context(
        &self,
        model: Arc<Model>,
        history: &[HistoryTurn],
        latest_turn: &str,
        sampling: SamplingParams,
    ) -> Result<Vec<(TextMessageRole, String)>> {
        let mut context = Vec::new();
        let total_turns = history.len();

        for (i, turn) in history.iter().enumerate() {
            // depth is distance from the end (0 = latest turn in history)
            let depth = total_turns - 1 - i;
            
            // Probability formula: older (higher depth) -> higher prob of compression
            let prob = 1.0 - self.inverse_prob.powf(depth as f64);

            // Never compress standard system messages, only skills
            if let HistoryTurnRole::System = turn.role {
                context.push((TextMessageRole::System, turn.content.clone()));
                continue;
            }

            // If probability is above threshold, we compress
            if prob > self.threshold {
                let compressed = self
                    .process_turn_compression(
                        model.clone(),
                        history,
                        i,
                        latest_turn,
                        sampling.clone(),
                    )
                    .await?;
                context.push(compressed);
            } else {
                context.push(turn_to_mistral(turn));
            }
        }

        Ok(context)
    }

    async fn process_turn_compression(
        &self,
        model: Arc<Model>,
        history: &[HistoryTurn],
        turn_idx: usize,
        latest_turn: &str,
        sampling: SamplingParams,
    ) -> Result<(TextMessageRole, String)> {
        let turn = &history[turn_idx];
        let cache_dir = self.turn_cache_dir(turn.turn_index);
        fs::create_dir_all(&cache_dir).await?;

        let root_path = self.root_summary_path(turn.turn_index);

        let mut root = if root_path.exists() {
            let content = fs::read_to_string(&root_path).await?;
            let r: RootSummary = serde_json::from_str(&content)?;

            let should_resummarize = rand::random::<f64>() < self.resummarize_prob;

            if should_resummarize {
                self.generate_root_summary(
                    model.clone(),
                    history,
                    turn_idx,
                    sampling.clone(),
                    Some(r),
                )
                .await?
            } else {
                r
            }
        } else {
            self.generate_root_summary(model.clone(), history, turn_idx, sampling.clone(), None)
                .await?
        };

        // Save updated root
        fs::write(&root_path, serde_json::to_string_pretty(&root)?).await?;

        // 2.5) if root > actual turn, load only domains or actual turn
        if root.micro_summary.len() > turn.content.len() {
            if root.domains.is_empty() {
                return Ok(turn_to_mistral(turn));
            }
        }

        // 3) if only root exists
        if root.domains.is_empty() {
            if self
                .is_summary_sufficient(model.clone(), &root, latest_turn, sampling.clone())
                .await?
            {
                return Ok((role_to_mistral(&turn.role), root.micro_summary.clone()));
            } else {
                let spec = self
                    .generate_specialized_summary(
                        model.clone(),
                        turn,
                        latest_turn,
                        sampling.clone(),
                    )
                    .await?;
                root.domains.push(spec.domain.clone());
                fs::write(&root_path, serde_json::to_string_pretty(&root)?).await?;
                fs::write(
                    self.specialized_summary_path(turn.turn_index, &spec.domain),
                    serde_json::to_string_pretty(&spec)?,
                )
                .await?;
                return Ok((role_to_mistral(&turn.role), spec.extracted_information));
            }
        }

        // 4) selection logic for > 1 summary
        let selected_domains = self
            .select_relevant_domains(model.clone(), &root, latest_turn, sampling.clone())
            .await?;
        if selected_domains.is_empty()
            || (selected_domains.len() == 1 && selected_domains[0] == "root")
        {
            Ok((role_to_mistral(&turn.role), root.micro_summary.clone()))
        } else {
            let mut combined = String::new();
            for domain in selected_domains {
                if domain == "root" {
                    combined.push_str(&format!("General: {}\n", root.micro_summary));
                } else {
                    let p = self.specialized_summary_path(turn.turn_index, &domain);
                    if p.exists() {
                        let content = fs::read_to_string(p).await?;
                        let spec: SpecializedSummary = serde_json::from_str(&content)?;
                        combined.push_str(&format!(
                            "{}: {}\n",
                            spec.domain, spec.extracted_information
                        ));
                    }
                }
            }
            Ok((role_to_mistral(&turn.role), combined))
        }
    }

    async fn generate_root_summary(
        &self,
        model: Arc<Model>,
        history: &[HistoryTurn],
        turn_idx: usize,
        sampling: SamplingParams,
        existing_root: Option<RootSummary>,
    ) -> Result<RootSummary> {
        let turn = &history[turn_idx];

        // Sliding window of adjacent turns
        let start = turn_idx.saturating_sub(1);
        let end = (turn_idx + 2).min(history.len());
        let window = &history[start..end];
        let window_indices: Vec<usize> = window.iter().map(|t| t.turn_index).collect();

        let mut context_text = String::new();
        for t in window {
            context_text.push_str(&format!("{}: {}\n", role_to_mistral(&t.role), t.content));
        }

        let mut system_prompt = "You are a concise summarizer. Output ONLY valid JSON with 'title' and 'micro_summary' fields.".to_string();

        if let Some(er) = &existing_root {
            system_prompt.push_str(&format!(
                "\nRefine the existing summary considering these other domain summaries: {:?}",
                er.domains
            ));
        }

        let prompt = format!(
            "Summarize this turn within its local context.\n\nLocal Context:\n{}\n\nTarget Turn Content: {}",
            context_text, turn.content
        );

        let messages = TextMessages::new()
            .add_message(TextMessageRole::System, system_prompt)
            .add_message(TextMessageRole::User, prompt);

        let req = RequestBuilder::from(messages).set_sampling(sampling);
        let resp: ChatCompletionResponse = model.send_chat_request(req).await?;
        let content = resp.choices[0]
            .message
            .content
            .as_ref()
            .context("Empty summary")?;

        let root: RootSummary = if let Some(start_json) = content.find('{') {
            if let Some(end_json) = content.rfind('}') {
                let json_str = &content[start_json..=end_json];
                let mut r: RootSummary = serde_json::from_str(json_str)
                    .map_err(|e| anyhow!("JSON Parse Error: {}, content: {}", e, json_str))?;
                
                // Populate non-JSON fields
                r.included_turn_indices = window_indices;
                if let Some(er) = existing_root {
                    r.domains = er.domains;
                }
                r
            } else {
                anyhow::bail!("No JSON end found in model response")
            }
        } else {
            anyhow::bail!("No JSON start found in model response")
        };

        Ok(root)
    }

    async fn is_summary_sufficient(
        &self,
        model: Arc<Model>,
        root: &RootSummary,
        latest_turn: &str,
        sampling: SamplingParams,
    ) -> Result<bool> {
        let prompt = format!(
            "Evaluation Task: Is the general summary sufficient to understand the context for the latest turn?\n\nGeneral Summary: {}\n\nLatest Turn: {}\n\nRespond with exactly 'YES' or 'NO'.",
            root.micro_summary, latest_turn
        );
        let messages = TextMessages::new().add_message(TextMessageRole::User, prompt);
        let req = RequestBuilder::from(messages).set_sampling(sampling);
        let resp: ChatCompletionResponse = model.send_chat_request(req).await?;
        let content = resp.choices[0]
            .message
            .content
            .as_ref()
            .map(|s| s.trim().to_uppercase())
            .unwrap_or_default();
        Ok(content.contains("YES"))
    }

    async fn generate_specialized_summary(
        &self,
        model: Arc<Model>,
        turn: &HistoryTurn,
        latest_turn: &str,
        sampling: SamplingParams,
    ) -> Result<SpecializedSummary> {
        let prompt = format!(
            "Extraction Task: The general summary was insufficient. Extract information from the historical turn specifically relevant to the latest turn.\n\nHistorical Turn: {}\n\nLatest Turn: {}\n\nOutput ONLY valid JSON with 'domain' (single word topic) and 'extracted_information'.",
            turn.content, latest_turn
        );
        let messages = TextMessages::new().add_message(TextMessageRole::User, prompt);
        let req = RequestBuilder::from(messages).set_sampling(sampling);
        let resp: ChatCompletionResponse = model.send_chat_request(req).await?;
        let content = resp.choices[0]
            .message
            .content
            .as_ref()
            .context("Empty specialized summary")?;

        if let Some(start) = content.find('{') {
            if let Some(end) = content.rfind('}') {
                let json_str = &content[start..=end];
                let spec: SpecializedSummary = serde_json::from_str(json_str)?;
                return Ok(spec);
            }
        }
        anyhow::bail!("Failed to parse specialized summary from: {}", content)
    }

    async fn select_relevant_domains(
        &self,
        model: Arc<Model>,
        root: &RootSummary,
        latest_turn: &str,
        sampling: SamplingParams,
    ) -> Result<Vec<String>> {
        let domains = root.domains.join(", ");
        let prompt = format!(
            "Selection Task: Given the latest turn, which specialized summaries are relevant? \nAvailable: root (general), {}\n\nLatest Turn: {}\n\nRespond with a comma-separated list of relevant domains or 'root'.",
            domains, latest_turn
        );
        let messages = TextMessages::new().add_message(TextMessageRole::User, prompt);
        let req = RequestBuilder::from(messages).set_sampling(sampling);
        let resp: ChatCompletionResponse = model.send_chat_request(req).await?;
        let content = resp.choices[0]
            .message
            .content
            .as_ref()
            .unwrap_or(&"".to_string())
            .clone();

        let selected: Vec<String> = content
            .split(',')
            .map(|s| s.trim().to_lowercase())
            .filter(|s| s == "root" || root.domains.contains(s))
            .collect();
        Ok(selected)
    }
}

pub fn role_to_mistral(role: &HistoryTurnRole) -> TextMessageRole {
    match role {
        HistoryTurnRole::User => TextMessageRole::User,
        HistoryTurnRole::Assistant => TextMessageRole::Assistant,
        HistoryTurnRole::System | HistoryTurnRole::Skill(_, _) => TextMessageRole::System,
    }
}

pub fn turn_to_mistral(turn: &HistoryTurn) -> (TextMessageRole, String) {
    let role = role_to_mistral(&turn.role);
    let content = match &turn.role {
        HistoryTurnRole::Skill(name, desc) => {
            format!("## Skill: {}\n{}\n\n{}", name, desc, turn.content)
        }
        _ => turn.content.clone(),
    };
    (role, content)
}

pub fn extract_frontmatter(content: &str) -> Option<(String, String, String)> {
    if !content.starts_with("---\n") {
        return None;
    }
    let rest = &content[4..];
    if let Some(end_pos) = rest.find("\n---\n") {
        let fm_str = &rest[..end_pos];
        let remaining = rest[end_pos + 5..].trim_start().to_string();

        let mut name = None;
        let mut desc = None;
        for line in fm_str.lines() {
            if line.starts_with("name:") {
                name = Some(line["name:".len()..].trim().to_string());
            } else if line.starts_with("description:") {
                desc = Some(line["description:".len()..].trim().to_string());
            }
        }
        if let (Some(n), Some(d)) = (name, desc) {
            return Some((n, d, remaining));
        }
    }
    None
}
