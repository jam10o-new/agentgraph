use anyhow::{Context, Result, anyhow};
use mistralrs::{
    ChatCompletionResponse, Model, RequestBuilder, SamplingParams, TextMessageRole, TextMessages,
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
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

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MetaSummary {
    pub turn_index: usize,
    pub content: String,
    #[serde(default)]
    pub included_domains: Vec<String>,
    #[serde(default)]
    pub included_turn_indices: Vec<usize>,
    pub generated_at: u64,
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
    /// If true, this turn is never compressed or folded into a metasummary.
    pub excluded_from_compression: bool,
}

use ag_config::CompressionConfig;

pub struct CompressionManager {
    pub base_path: PathBuf,
    pub threshold: f64,
    pub inverse_prob: f64,
    pub resummarize_prob: f64,
}

impl CompressionManager {
    pub fn new(base_path: PathBuf, compression: &CompressionConfig) -> Self {
        Self {
            base_path: base_path.join(".agent_context"),
            threshold: compression.threshold,
            inverse_prob: compression.inverse_probability,
            resummarize_prob: compression.resummarize_probability,
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

    fn metasummary_dir(&self) -> PathBuf {
        self.base_path.join("metasummaries")
    }

    fn metasummary_path(&self, turn_idx: usize) -> PathBuf {
        self.metasummary_dir()
            .join(format!("metasummary_{}.json", turn_idx))
    }

    async fn load_latest_metasummary(&self) -> Result<Option<MetaSummary>> {
        let dir = self.metasummary_dir();
        if !dir.exists() {
            return Ok(None);
        }
        let mut latest: Option<MetaSummary> = None;
        let mut entries = fs::read_dir(&dir).await?;
        while let Some(entry) = entries.next_entry().await? {
            let p = entry.path();
            if p.extension().and_then(|e| e.to_str()) == Some("json") {
                let content = fs::read_to_string(&p).await?;
                let ms: MetaSummary = serde_json::from_str(&content)?;
                if latest
                    .as_ref()
                    .map(|l| ms.turn_index > l.turn_index)
                    .unwrap_or(true)
                {
                    latest = Some(ms);
                }
            }
        }
        Ok(latest)
    }

    async fn save_metasummary(&self, ms: &MetaSummary) -> Result<()> {
        fs::create_dir_all(self.metasummary_dir()).await?;
        let path = self.metasummary_path(ms.turn_index);
        fs::write(&path, serde_json::to_string_pretty(ms)?).await?;
        Ok(())
    }

    async fn generate_metasummary(
        &self,
        model: Arc<Model>,
        turns: &[HistoryTurn],
        latest_turn: &str,
        sampling: SamplingParams,
    ) -> Result<MetaSummary> {
        let mut summary_text = String::new();
        let mut included_domains = Vec::new();
        let mut included_turn_indices = Vec::new();

        for turn in turns {
            // Never fold excluded turns into a metasummary
            if turn.excluded_from_compression {
                continue;
            }
            included_turn_indices.push(turn.turn_index);
            summary_text.push_str(&format!(
                "\n--- Turn {} ({}) ---\n",
                turn.turn_index,
                role_to_mistral(&turn.role)
            ));

            let root_path = self.root_summary_path(turn.turn_index);
            if root_path.exists() {
                let content = fs::read_to_string(&root_path).await?;
                let root: RootSummary = serde_json::from_str(&content)?;
                summary_text.push_str(&format!("Summary: {}\n", root.micro_summary));
                summary_text.push_str(&format!("Domains: {:?}\n", root.domains));
                for domain in &root.domains {
                    let spec_path = self.specialized_summary_path(turn.turn_index, domain);
                    if spec_path.exists() {
                        let spec_content = fs::read_to_string(&spec_path).await?;
                        let spec: SpecializedSummary = serde_json::from_str(&spec_content)?;
                        summary_text.push_str(&format!(
                            "  [{}]: {}\n",
                            spec.domain, spec.extracted_information
                        ));
                        if !included_domains.contains(&spec.domain) {
                            included_domains.push(spec.domain.clone());
                        }
                    }
                }
            } else {
                let preview: String = turn.content.chars().take(300).collect();
                summary_text.push_str(&format!("Content (raw): {}\n", preview));
            }
        }

        let system_prompt = "You are a context archivist. Summarize a series of conversation summaries into a single metasummary. Deduplicate repeated information. Mention all domains/topics covered. Include the turn indices of important turns the model may want to reread. Output ONLY valid JSON with 'content' (string), 'included_domains' (array of strings), and 'included_turn_indices' (array of integers) fields.";
        let prompt = format!(
            "Latest turn context: {}\n\nHistorical summaries to collate:\n{}\n\nProduce a concise metasummary that captures all unique information.",
            latest_turn, summary_text
        );

        let messages = TextMessages::new()
            .add_message(TextMessageRole::System, system_prompt)
            .add_message(TextMessageRole::User, prompt);

        let req = RequestBuilder::from(messages).set_sampling(sampling);
        let resp: ChatCompletionResponse =
            send_chat_request_with_retry(&model, req, 3, Duration::from_millis(500)).await?;
        let content = resp.choices[0]
            .message
            .content
            .as_ref()
            .context("Empty metasummary")?;

        let mut ms: MetaSummary = if let Some(start_json) = content.find('{') {
            if let Some(end_json) = content.rfind('}') {
                let json_str = &content[start_json..=end_json];
                serde_json::from_str(json_str)
                    .map_err(|e| anyhow!("JSON Parse Error: {}, content: {}", e, json_str))?
            } else {
                anyhow::bail!("No JSON end found in model response")
            }
        } else {
            anyhow::bail!("No JSON start found in model response")
        };

        ms.generated_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs();
        ms.included_domains = included_domains;
        ms.included_turn_indices = included_turn_indices;
        Ok(ms)
    }

    pub async fn get_compressed_context(
        &self,
        model: Arc<Model>,
        history: &mut Vec<HistoryTurn>,
        latest_turn: &str,
        sampling: SamplingParams,
        checkpoint_limit: Option<usize>,
    ) -> Result<Vec<(TextMessageRole, String)>> {
        let mut attempt = 0;
        loop {
            // Phase 0: Apply existing metasummaries, but never filter excluded turns
            if let Some(ms) = self.load_latest_metasummary().await? {
                let cutoff = ms.turn_index;
                history.retain(|t| t.turn_index > cutoff || t.excluded_from_compression);
            }

            // Phase 1: Compress remaining history
            let mut context = Vec::new();
            let total_turns = history.len();
            for (i, turn) in history.iter().enumerate() {
                // Excluded turns are passed verbatim, never compressed
                if turn.excluded_from_compression {
                    context.push(turn_to_mistral(turn));
                    continue;
                }

                let depth = total_turns - 1 - i;
                let prob = 1.0 - self.inverse_prob.powf(depth as f64);

                if let HistoryTurnRole::System = turn.role {
                    context.push((TextMessageRole::System, turn.content.clone()));
                    continue;
                }

                if prob > self.threshold {
                    let maybe_compressed = self
                        .process_turn_compression(
                            model.clone(),
                            history,
                            i,
                            latest_turn,
                            sampling.clone(),
                        )
                        .await;
                    if let Ok(compressed) = maybe_compressed {
                        context.push(compressed);
                    } else {
                        context.push(turn_to_mistral(turn));
                    }
                } else {
                    context.push(turn_to_mistral(turn));
                }
            }

            // Phase 2: Post-compression checkpointing
            if let Some(limit) = checkpoint_limit {
                let model_max_seq_chars =
                    if let Some(model_max_seq_len) = model.clone().config().unwrap().max_seq_len {
                        model_max_seq_len * 3 //TOD0 make a const or calc max token length from config instead of hardcoding here
                    } else {
                        limit
                    };
                let total_chars: usize = context.iter().map(|(_, s)| s.len()).sum();
                if ((total_chars > limit && limit != 0) || total_chars > model_max_seq_chars)
                    && attempt == 0
                {
                    // Determine how many oldest non-system, non-excluded turns to fold
                    let mut fold_count = 0;
                    for turn in history.iter() {
                        if matches!(turn.role, HistoryTurnRole::System)
                            || turn.excluded_from_compression
                        {
                            continue;
                        }
                        fold_count += 1;
                        if fold_count >= history.len() / 2 {
                            break;
                        }
                    }
                    if fold_count > 0 {
                        let turns_to_fold: Vec<HistoryTurn> =
                            history.iter().take(fold_count).cloned().collect();
                        let ms = self
                            .generate_metasummary(
                                model.clone(),
                                &turns_to_fold,
                                latest_turn,
                                sampling.clone(),
                            )
                            .await?;
                        self.save_metasummary(&ms).await?;
                        attempt += 1;
                        continue;
                    }
                }
            }

            return Ok(context);
        }
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
        let resp: ChatCompletionResponse =
            send_chat_request_with_retry(&model, req, 3, Duration::from_millis(500)).await?;
        let content;
        let empty = String::new();
        if let Some(choice) = resp.choices.first() {
            content = choice.message.content.as_ref().unwrap_or(&empty);
        } else {
            content = &empty;
        }

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
        let resp: ChatCompletionResponse =
            send_chat_request_with_retry(&model, req, 3, Duration::from_millis(500)).await?;
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
        let resp: ChatCompletionResponse =
            send_chat_request_with_retry(&model, req, 3, Duration::from_millis(500)).await?;
        let empty = &String::new();
        let content = resp.choices[0].message.content.as_ref().unwrap_or(empty);

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
        let resp: ChatCompletionResponse =
            send_chat_request_with_retry(&model, req, 3, Duration::from_millis(500)).await?;
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

/// Send a non-streaming chat request with retry on recoverable errors
/// (OOMs, timeouts). Used for compression-related LLM calls in context
/// summarization, where losing a single summarization attempt is not
/// critical but a clean retry with backoff can recover from transient
/// resource exhaustion.
async fn send_chat_request_with_retry(
    model: &Model,
    req: RequestBuilder,
    max_retries: u32,
    delay: Duration,
) -> Result<ChatCompletionResponse> {
    let mut remaining = max_retries;
    loop {
        match model.send_chat_request(req.clone()).await {
            Ok(resp) => return Ok(resp),
            Err(_e) if remaining > 0 => {
                remaining -= 1;
                tokio::time::sleep(delay).await;
            }
            Err(e) => return Err(e.into()),
        }
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
