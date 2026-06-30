use crate::inference_provider::Role;
use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

// ── Data types (compatible with serialized forms) ──────────────────────

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RootSummary {
    pub title: String,
    pub micro_summary: String,
    #[serde(default)]
    pub domains: Vec<String>,
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
    Skill(String, String),
    Tool,
}

#[derive(Debug, Clone)]
pub struct HistoryTurn {
    pub role: HistoryTurnRole,
    pub content: String,
    pub turn_index: usize,
    pub excluded_from_compression: bool,
    /// Stable file identity: xxh3 of file path. All versions of the same
    /// file share this key so we can detect when a summary is stale.
    pub file_key: String,
    /// Human-readable file path for diagnostics and version preamble.
    pub file_path: String,
    /// xxh3 hash of the file content at the time this turn was built.
    /// Compared against the summary's stored content_hash to detect drift.
    pub content_hash: String,
}

use ag_config::CompressionConfig;

// ── Token / budget constants ───────────────────────────────────────────

const ROOT_SUMMARY_MAX_TOKENS: usize = 256;
const SPECIALIZED_SUMMARY_MAX_TOKENS: usize = 256;
const METASUMMARY_MAX_TOKENS: usize = 1024;
const SUFFICIENCY_CHECK_MAX_TOKENS: usize = 8;
/// Compression prompt per-turn char cap to stay within model context window.
const COMPRESSION_PROMPT_CHAR_LIMIT: usize = 64_000;

/// Default distance threshold for creating a new domain from an embedding.
const DOMAIN_CREATION_DISTANCE: f32 = 0.35;
/// When context pressure is this fraction of the token limit, retrieval
/// selectivity is at maximum (only the single closest summary is kept).
const MAX_PRESSURE_RETRIEVAL_K: usize = 1;
/// When context pressure is zero or negative (under limit), keep all domain
/// summaries for a turn.
const MIN_PRESSURE_RETRIEVAL_K: usize = 8;

// ── Helpers ────────────────────────────────────────────────────────────

/// Cosine similarity ∈ [-1, 1].  1 = identical direction, -1 = opposite.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let (dot, norm_a, norm_b) = a.iter().zip(b.iter()).fold(
        (0.0f32, 0.0f32, 0.0f32),
        |(d, na, nb), (&x, &y)| (d + x * y, na + x * x, nb + y * y),
    );
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a.sqrt() * norm_b.sqrt())
    }
}

/// Cosine distance ∈ [0, 2].  0 = identical, 2 = opposite.
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    (1.0 - cosine_similarity(a, b)).clamp(0.0, 2.0)
}

// ── CompressionManager ─────────────────────────────────────────────────

pub struct CompressionManager {
    db: Arc<Mutex<rusqlite::Connection>>,
    threshold: f64,
    inverse_prob: f64,
    resummarize_prob: f64,
}

impl CompressionManager {
    /// Create a new manager backed by a SQLite database at `db_path`.
    /// If tables don't exist yet they are created automatically.
    pub fn new(db_path: &Path, compression: &CompressionConfig) -> Result<Self> {
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)
                .context(format!("Failed to create db parent dir: {}", parent.display()))?;
        }

        let conn = rusqlite::Connection::open(db_path)
            .context(format!("Failed to open compression db: {}", db_path.display()))?;

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_key TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                file_path TEXT NOT NULL,
                role TEXT NOT NULL,
                turn_order INTEGER NOT NULL,
                root_title TEXT,
                root_micro_summary TEXT,
                generated_at INTEGER NOT NULL,
                content_len INTEGER NOT NULL DEFAULT 0,
                UNIQUE(file_key, content_hash)
            );

            CREATE TABLE IF NOT EXISTS domains (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                centroid_embedding BLOB,
                created_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS domain_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                summary_id INTEGER NOT NULL REFERENCES summaries(id),
                domain_id INTEGER NOT NULL REFERENCES domains(id),
                extracted_information TEXT NOT NULL,
                embedding BLOB NOT NULL,
                UNIQUE(summary_id, domain_id)
            );

            CREATE TABLE IF NOT EXISTS metasummaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                turn_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                generated_at INTEGER NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_summaries_file_key ON summaries(file_key);
            CREATE INDEX IF NOT EXISTS idx_summaries_turn ON summaries(turn_order);
            CREATE INDEX IF NOT EXISTS idx_domain_summaries_domain ON domain_summaries(domain_id);
            CREATE INDEX IF NOT EXISTS idx_metasummaries_turn ON metasummaries(turn_index);",
        )
        .context("Failed to create compression schema")?;

        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;

        // Schema version check: auto-migrate by dropping and recreating
        // when the schema layout changes. The DB is a cache — losing it
        // is acceptable (summaries are regenerated on demand).
        let current_version: i64 = conn.pragma_query_value(None, "user_version", |r| r.get(0))?;
        const SCHEMA_VERSION: i64 = 2;
        if current_version < SCHEMA_VERSION {
            conn.execute_batch(
                "DROP TABLE IF EXISTS domain_summaries;
                 DROP TABLE IF EXISTS summaries;
                 DROP TABLE IF EXISTS domains;
                 DROP TABLE IF EXISTS metasummaries;",
            )?;
            // Re-create with latest schema
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_key TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    role TEXT NOT NULL,
                    turn_order INTEGER NOT NULL,
                    root_title TEXT,
                    root_micro_summary TEXT,
                    generated_at INTEGER NOT NULL,
                    content_len INTEGER NOT NULL DEFAULT 0,
                    UNIQUE(file_key, content_hash)
                );
                CREATE TABLE IF NOT EXISTS domains (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    centroid_embedding BLOB,
                    created_at INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS domain_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    summary_id INTEGER NOT NULL REFERENCES summaries(id),
                    domain_id INTEGER NOT NULL REFERENCES domains(id),
                    extracted_information TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    UNIQUE(summary_id, domain_id)
                );
                CREATE TABLE IF NOT EXISTS metasummaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    turn_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    generated_at INTEGER NOT NULL
                );",
            )?;
            conn.pragma_update(None, "user_version", SCHEMA_VERSION)?;
        }

        // Recreate indices (idempotent)
        conn.execute_batch(
            "CREATE INDEX IF NOT EXISTS idx_summaries_file_key ON summaries(file_key);
             CREATE INDEX IF NOT EXISTS idx_summaries_turn ON summaries(turn_order);
             CREATE INDEX IF NOT EXISTS idx_domain_summaries_domain ON domain_summaries(domain_id);
             CREATE INDEX IF NOT EXISTS idx_metasummaries_turn ON metasummaries(turn_index);",
        )
        .context("Failed to create indices")?;
        Ok(Self {
            db: Arc::new(Mutex::new(conn)),
            threshold: compression.threshold,
            inverse_prob: compression.inverse_probability,
            resummarize_prob: compression.resummarize_probability,
        })
    }

    // ── Embedding helpers ──────────────────────────────────────────────

    /// Serialize a f32 slice to bytes for BLOB storage.
    fn embed_to_blob(embedding: &[f32]) -> Vec<u8> {
        bytemuck::cast_slice(embedding).to_vec()
    }

    /// Deserialize bytes back to f32 vector.
    fn blob_to_embed(blob: &[u8]) -> Vec<f32> {
        bytemuck::cast_slice(blob).to_vec()
    }

    // ── Pressure-scaled probability ────────────────────────────────────

    /// Compute the probability of compressing a turn based on how close we
    /// are to the context limit (pressure ∈ [0, ∞)).  When pressure is low
    /// we rarely compress (preserving turns verbatim).  When pressure ≥ 1.0
    /// (at or over the limit) we always compress.
    fn compression_probability(&self, pressure: f64, depth: usize) -> f64 {
        let base_prob = 1.0 - self.inverse_prob.powf(depth as f64);
        // Scale linearly with pressure: at pressure=0, multiply by 0.1;
        // at pressure≥1.0, multiply by 1.0.
        let scale = (0.1 + 0.9 * pressure).clamp(0.0, 1.0);
        base_prob * scale
    }

    /// Number of domain summaries to keep per turn based on pressure.
    /// Low pressure → keep many; high pressure → keep only the best.
    fn retrieval_k(pressure: f64) -> usize {
        if pressure >= 1.0 {
            MAX_PRESSURE_RETRIEVAL_K
        } else if pressure <= 0.3 {
            MIN_PRESSURE_RETRIEVAL_K
        } else {
            let t = (pressure - 0.3) / 0.7;
            let k = MIN_PRESSURE_RETRIEVAL_K as f64
                - t * (MIN_PRESSURE_RETRIEVAL_K - MAX_PRESSURE_RETRIEVAL_K) as f64;
            k.round() as usize
        }
    }

    fn prune_by_relevance(
        &self,
        _latest_emb: &[f32],
        k: usize,
        context: Vec<(Role, String)>,
    ) -> Vec<(Role, String)> {
        let total = context.len();
        if total <= k {
            return context;
        }
        let sys: Vec<_> = context
            .iter()
            .take_while(|(r, _)| matches!(r, Role::System))
            .cloned()
            .collect();
        let rest: Vec<_> = context.iter().skip(sys.len()).cloned().collect();
        let keep = k.min(rest.len());
        let start = rest.len().saturating_sub(keep);
        let mut out = sys;
        out.extend(rest[start..].iter().cloned());
        out
    }

    // ── SQLite helpers ─────────────────────────────────────────────────

    fn insert_summary(
        &self,
        turn: &HistoryTurn,
        title: &str,
        micro_summary: &str,
    ) -> Result<i64> {
        let db = self.db.lock().unwrap();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        db.execute(
            "INSERT OR REPLACE INTO summaries (file_key, content_hash, file_path, role, turn_order,
             root_title, root_micro_summary, generated_at, content_len)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            rusqlite::params![
                turn.file_key,
                turn.content_hash,
                turn.file_path,
                role_label(&turn.role),
                turn.turn_index as i64,
                title,
                micro_summary,
                now,
                turn.content.len() as i64,
            ],
        )
        .context("Failed to insert summary")?;
        Ok(db.last_insert_rowid())
    }

    // ── Domain assignment via vector similarity ────────────────────────

    /// Ensure the summary is assigned to a domain.  If no domain exists yet,
    /// create the "general" domain.  Local domains are created automatically
    /// when a specialized summary's embedding is far from all existing centroids.
    /// Find the nearest domain to `embedding` whose distance is ≤ `max_dist`.
    /// Returns `Some(domain_id)` or `None` if all domains are too far.
    fn find_nearest_domain(&self, embedding: &[f32], max_dist: f32) -> Option<i64> {
        let db = self.db.lock().unwrap();
        let mut stmt = db
            .prepare("SELECT id, centroid_embedding FROM domains WHERE centroid_embedding IS NOT NULL")
            .ok()?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?))
        })
        .ok()?;

        let mut best_id: Option<i64> = None;
        let mut best_dist = f32::MAX;

        for row in rows.flatten() {
            let (id, blob) = row;
            let other = Self::blob_to_embed(&blob);
            if other.len() == embedding.len() {
                let d = cosine_distance(embedding, &other);
                if d < best_dist {
                    best_dist = d;
                    best_id = Some(id);
                }
            }
        }

        if best_dist <= max_dist {
            best_id
        } else {
            None
        }
    }

    fn create_domain(&self, name: &str, embedding: &[f32]) -> Result<i64> {
        let db = self.db.lock().unwrap();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let blob = Self::embed_to_blob(embedding);
        db.execute(
            "INSERT OR IGNORE INTO domains (name, centroid_embedding, created_at) VALUES (?1, ?2, ?3)",
            rusqlite::params![name, blob, now],
        )
        .context("Failed to insert domain")?;

        let id: i64 = db.query_row(
            "SELECT id FROM domains WHERE name = ?1",
            [name],
            |r| r.get(0),
        )?;
        Ok(id)
    }

    fn store_domain_summary(
        &self,
        summary_id: i64,
        domain_id: i64,
        information: &str,
        embedding: &[f32],
    ) -> Result<()> {
        let db = self.db.lock().unwrap();
        let blob = Self::embed_to_blob(embedding);
        db.execute(
            "INSERT OR IGNORE INTO domain_summaries (summary_id, domain_id, extracted_information, embedding)
             VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![summary_id, domain_id, information, blob],
        )
        .context("Failed to store domain summary")?;

        // Update domain centroid: average of all embeddings in this domain
        self.update_domain_centroid(domain_id);
        Ok(())
    }

    /// Recompute the centroid of a domain as the mean of all its embeddings.
    fn update_domain_centroid(&self, domain_id: i64) {
        let db = self.db.lock().unwrap();
        let mut stmt = match db
            .prepare("SELECT embedding FROM domain_summaries WHERE domain_id = ?1")
        {
            Ok(s) => s,
            Err(_) => return,
        };
        let rows: Vec<Vec<u8>> = stmt
            .query_map([domain_id], |row| row.get(0))
            .ok()
            .map(|r| r.flatten().collect())
            .unwrap_or_default();

        if rows.is_empty() {
            return;
        }
        let dim = Self::blob_to_embed(&rows[0]).len();
        let mut sum = vec![0.0f32; dim];
        let n = rows.len() as f32;
        for blob in &rows {
            let emb = Self::blob_to_embed(blob);
            if emb.len() == dim {
                for (s, v) in sum.iter_mut().zip(emb) {
                    *s += v;
                }
            }
        }
        for s in sum.iter_mut() {
            *s /= n;
        }
        let new_blob = Self::embed_to_blob(&sum);
        let _ = db.execute(
            "UPDATE domains SET centroid_embedding = ?1 WHERE id = ?2",
            rusqlite::params![new_blob, domain_id],
        );
    }

    // ── Metasummary ────────────────────────────────────────────────────

    fn save_metasummary(&self, ms: &MetaSummary) -> Result<()> {
        let db = self.db.lock().unwrap();
        db.execute(
            "INSERT INTO metasummaries (turn_index, content, generated_at) VALUES (?1, ?2, ?3)",
            rusqlite::params![ms.turn_index as i64, ms.content, ms.generated_at],
        )
        .context("Failed to save metasummary")?;
        Ok(())
    }

    /// Load a summary from DB by file key. Returns (title, micro_summary).
    fn load_summary_from_db(&self, file_key: &str) -> Option<(String, String)> {
        let db = self.db.lock().unwrap();
        db.query_row(
            "SELECT root_title, root_micro_summary FROM summaries
             WHERE file_key = ?1 ORDER BY generated_at DESC LIMIT 1",
            [file_key],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
        )
        .ok()
    }

    /// Load domain entries for a summary by file key.
    fn load_domains_for_file(&self, file_key: &str) -> Option<Vec<(String, String)>> {
        let db = self.db.lock().unwrap();
        let summary_id: i64 = db
            .query_row(
                "SELECT id FROM summaries WHERE file_key = ?1 ORDER BY generated_at DESC LIMIT 1",
                [file_key],
                |row| row.get(0),
            )
            .ok()?;
        let mut stmt = db
            .prepare(
                "SELECT d.name, ds.extracted_information
                 FROM domain_summaries ds
                 JOIN domains d ON d.id = ds.domain_id
                 WHERE ds.summary_id = ?1",
            )
            .ok()?;
        let result: Vec<(String, String)> = stmt
            .query_map([summary_id], |row| Ok((row.get(0)?, row.get(1)?)))
            .ok()?
            .flatten()
            .collect();
        if result.is_empty() {
            None
        } else {
            Some(result)
        }
    }

    // ── Provider-based compression (parallel to get_compressed_context) ──

    /// Produce a compressed context from `history`, optionally checkpointing
    /// with metasummaries when the output exceeds `checkpoint_limit` tokens.
    /// `max_seq_len` comes from [`ProviderConfig::max_seq_len`] when available.
    pub async fn get_compressed_context(
        &self,
        provider: &dyn crate::InferenceProvider,
        history: &mut Vec<HistoryTurn>,
        latest_turn: &str,
        checkpoint_limit: Option<usize>,
        max_seq_len: Option<usize>,
    ) -> Result<Vec<(Role, String)>> {
        history.retain(|t| !t.file_key.is_empty());

        let mut attempt = 0;
        loop {
            // Phase 0: Apply existing metasummaries
            {
                let db = self.db.lock().unwrap();
                let cutoff: Option<usize> = db
                    .query_row(
                        "SELECT turn_index FROM metasummaries ORDER BY turn_index DESC LIMIT 1",
                        [],
                        |row| row.get(0),
                    )
                    .ok();
                if let Some(cut) = cutoff {
                    history.retain(|t| t.turn_index > cut || t.excluded_from_compression);
                }
            }

            // Phase 1: Build compressed context
            let mut context = Vec::new();
            let total_turns = history.len();

            let (pressure, _) = self.compute_pressure(provider, history, max_seq_len).await;

            for (i, turn) in history.iter().enumerate() {
                if turn.excluded_from_compression {
                    context.push(turn_to_tuple(turn));
                    continue;
                }
                if let HistoryTurnRole::System = turn.role {
                    context.push((Role::System, turn.content.clone()));
                    continue;
                }

                let depth = total_turns - 1 - i;
                let prob = self.compression_probability(pressure, depth);

                if prob > self.threshold {
                    match self
                        .process_turn_compression(provider, turn, latest_turn)
                        .await
                    {
                        Ok(compressed) => context.push(compressed),
                        Err(_) => context.push(turn_to_tuple(turn)),
                    }
                } else {
                    context.push(turn_to_tuple(turn));
                }
            }

            // Phase 2: Pressure-based retrieval pruning
            if pressure > 0.3 && !latest_turn.is_empty() {
                if let Ok(latest_emb) = Self::embed_text(provider, latest_turn).await {
                    let k = Self::retrieval_k(pressure);
                    context = self.prune_by_relevance(&latest_emb, k, context);
                }
            }

            // Phase 3: Token-limit checkpointing
            if let Some(limit) = checkpoint_limit {
                let token_limit = max_seq_len.unwrap_or(limit / 4);

                let context_text: String = context
                    .iter()
                    .map(|(_, s)| s.as_str())
                    .collect::<Vec<_>>()
                    .join("\n");
                let token_count = count_tokens(provider, &context_text).await;

                if token_count >= token_limit && attempt < 3 {
                    let slice = history.len().min((attempt + 1) * history.len() / 3);
                    let turns_to_fold: Vec<HistoryTurn> = history
                        .iter()
                        .take(slice)
                        .filter(|t| {
                            !matches!(t.role, HistoryTurnRole::System)
                                && !t.excluded_from_compression
                        })
                        .cloned()
                        .collect();

                    if !turns_to_fold.is_empty() {
                        let ms = self
                            .generate_metasummary(provider, &turns_to_fold, latest_turn)
                            .await?;
                        self.save_metasummary(&ms)?;
                        attempt += 1;
                        continue;
                    }
                }

                // Safety net: hard strip oldest non-system turns
                if token_count >= token_limit {
                    let mut running = 0usize;
                    let mut cutoff = 0usize;
                    for (i, (_, content)) in context.iter().enumerate().rev() {
                        running += content.len() / 4;
                        if running >= token_limit {
                            cutoff = i + 1;
                            break;
                        }
                    }
                    if cutoff > 0 {
                        let sys_count = context
                            .iter()
                            .take_while(|(r, _)| matches!(r, Role::System))
                            .count();
                        let final_cutoff = cutoff.max(sys_count);
                        context.drain(..final_cutoff);
                    }
                }
            }

            return Ok(context);
        }
    }

    async fn compute_pressure(
        &self,
        provider: &dyn crate::InferenceProvider,
        history: &[HistoryTurn],
        max_seq_len: Option<usize>,
    ) -> (f64, usize) {
        let max_tokens = max_seq_len.unwrap_or(32768);
        let text: String = history
            .iter()
            .map(|t| t.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        let current = count_tokens(provider, &text).await;
        (current as f64 / max_tokens as f64, max_tokens)
    }

    async fn process_turn_compression(
        &self,
        provider: &dyn crate::InferenceProvider,
        turn: &HistoryTurn,
        latest_turn: &str,
    ) -> Result<(Role, String)> {
        let file_key = &turn.file_key;
        let role_mistral = history_role_to_role(&turn.role);

        let (existing_title, existing_micro, existing_id) = {
            let db = self.db.lock().unwrap();
            db.query_row(
                "SELECT root_title, root_micro_summary, id, content_hash FROM summaries
                 WHERE file_key = ?1 ORDER BY generated_at DESC LIMIT 1",
                [file_key],
                |row| {
                    let title: String = row.get(0)?;
                    let micro: String = row.get(1)?;
                    let id: i64 = row.get(2)?;
                    let stored_hash: String = row.get(3)?;
                    let annotated = if stored_hash != turn.content_hash && !stored_hash.is_empty() {
                        format!(
                            "[⚠ Summary from older file version of {} (modified since). \
                             The current file content differs from what was summarized. \
                             Original summary follows:]\n{}",
                            turn.file_path, micro
                        )
                    } else {
                        micro
                    };
                    Ok((title, annotated, id))
                },
            )
            .map(|(t, m, id)| (t, m, id))
            .ok()
            .unwrap_or_else(|| (String::new(), String::new(), 0))
        };

        let (_root_title, root_micro, summary_id) = if existing_micro.is_empty()
            || rand::random::<f64>() < self.resummarize_prob
        {
            let root = self
                .generate_root_summary(provider, turn)
                .await?;
            let id = if existing_id > 0 {
                {
                    let db = self.db.lock().unwrap();
                    db.execute(
                        "UPDATE summaries SET root_title = ?1, root_micro_summary = ?2,
                         content_hash = ?3, generated_at = ?4, content_len = ?5 WHERE id = ?6",
                        rusqlite::params![
                            root.title,
                            root.micro_summary,
                            turn.content_hash,
                            std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap()
                                .as_secs(),
                            turn.content.len() as i64,
                            existing_id,
                        ],
                    )
                    .ok();
                }
                existing_id
            } else {
                match self.insert_summary(turn, &root.title, &root.micro_summary) {
                    Ok(id) => id,
                    Err(_) => existing_id,
                }
            };
            (root.title, root.micro_summary, id)
        } else {
            (existing_title, existing_micro, existing_id)
        };

        if root_micro.len() > turn.content.len() {
            return Ok(turn_to_tuple(turn));
        }

        if summary_id > 0 {
            self.ensure_domain_assignment(provider, summary_id, &root_micro, latest_turn)
                .await;
        }

        if self
            .is_summary_sufficient(provider, &root_micro, latest_turn)
            .await?
        {
            let result = self
                .retrieve_relevant_summaries(provider, summary_id, latest_turn)
                .await;
            if let Ok(combined) = result {
                if !combined.is_empty() && combined.len() <= turn.content.len() {
                    return Ok((role_mistral, combined));
                }
            }
            Ok((role_mistral, root_micro))
        } else {
            let spec = self
                .generate_specialized_summary(provider, turn, latest_turn)
                .await?;
            self.store_specialized_summary(provider, summary_id, &spec)
                .await;
            if spec.extracted_information.len() > turn.content.len() {
                Ok(turn_to_tuple(turn))
            } else {
                Ok((role_mistral, spec.extracted_information))
            }
        }
    }

    async fn ensure_domain_assignment(
        &self,
        provider: &dyn crate::InferenceProvider,
        summary_id: i64,
        micro_summary: &str,
        latest_turn: &str,
    ) {
        if summary_id == 0 {
            return;
        }
        {
            let db = self.db.lock().unwrap();
            let count: i64 = db
                .query_row(
                    "SELECT COUNT(*) FROM domain_summaries WHERE summary_id = ?1",
                    [summary_id],
                    |r| r.get(0),
                )
                .unwrap_or(0);
            if count > 0 {
                return;
            }
        }
        let emb = match Self::embed_text(provider, micro_summary).await {
            Ok(e) => e,
            Err(_) => return,
        };

        let nearest = self.find_nearest_domain(&emb, DOMAIN_CREATION_DISTANCE);

        match nearest {
            Some(domain_id) => {
                let _ = self.store_domain_summary(summary_id, domain_id, micro_summary, &emb);
            }
            None => {
                let domain_name = self
                    .generate_domain_name(provider, micro_summary)
                    .await
                    .unwrap_or_else(|_| format!("domain_{}", micro_summary.len() % 100));
                let domain_id = self.create_domain(&domain_name, &emb);
                if let Ok(did) = domain_id {
                    let _ = self.store_domain_summary(summary_id, did, micro_summary, &emb);
                }
            }
        }
    }

    async fn generate_domain_name(
        &self,
        provider: &dyn crate::InferenceProvider,
        summary: &str,
    ) -> Result<String> {
        use crate::inference_provider::{ChatMessage, ChatRequest, Role, SamplingConfig};

        let prompt = format!(
            "Name the single domain or topic (1-3 words, lowercase) that best categorizes this:\n\n{}",
            summary
        );
        let req = ChatRequest {
            messages: vec![ChatMessage {
                role: Role::User,
                content: prompt,
                tool_calls: None,
                tool_call_id: None,
            }],
            sampling: SamplingConfig::new(Some(16)),
            constraint: None,
            enable_thinking: false,
            ..Default::default()
        };
        let resp = send_chat_request_with_retry(provider, req, 2, Duration::from_millis(500)).await?;
        Ok(resp.trim().to_lowercase())
    }

    async fn retrieve_relevant_summaries(
        &self,
        provider: &dyn crate::InferenceProvider,
        summary_id: i64,
        latest_turn: &str,
    ) -> Result<String> {
        if summary_id == 0 || latest_turn.is_empty() {
            return Ok(String::new());
        }

        let latest_emb = Self::embed_text(provider, latest_turn).await?;

        let db = self.db.lock().unwrap();
        let mut stmt = db.prepare(
            "SELECT ds.domain_id, d.name, ds.extracted_information, ds.embedding
             FROM domain_summaries ds
             JOIN domains d ON d.id = ds.domain_id
             WHERE ds.summary_id = ?1",
        )?;
        let entries: Vec<(i64, String, String, Vec<u8>)> = stmt
            .query_map([summary_id], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
            })?
            .flatten()
            .collect();

        if entries.is_empty() {
            return Ok(String::new());
        }

        let mut scored: Vec<(f32, String)> = entries
            .iter()
            .map(|(_, name, info, blob)| {
                let emb = Self::blob_to_embed(blob);
                let dist = if emb.len() == latest_emb.len() {
                    cosine_distance(&latest_emb, &emb)
                } else {
                    1.0
                };
                (dist, format!("[{}] {}", name, info))
            })
            .collect();
        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let max_per_turn = MIN_PRESSURE_RETRIEVAL_K.min(scored.len());
        let combined: String = scored
            .iter()
            .skip(if scored.len() > 1 { 1 } else { 0 })
            .take(max_per_turn - 1)
            .map(|(_, s)| s.as_str())
            .collect::<Vec<_>>()
            .join("\n");

        if combined.is_empty() {
            Ok(scored.first().map(|(_, s)| s.clone()).unwrap_or_default())
        } else {
            Ok(combined)
        }
    }

    async fn generate_root_summary(
        &self,
        provider: &dyn crate::InferenceProvider,
        turn: &HistoryTurn,
    ) -> Result<RootSummary> {
        use crate::inference_provider::{ChatMessage, ChatRequest, GuidanceConstraint, Role, SamplingConfig};

        let preview: String = turn.content.chars().take(4096).collect();
        let truncated = if turn.content.len() > 4096 { " [truncated]" } else { "" };
        let prompt = format!("Content to summarize:{}{}\n\n{}", preview, truncated, turn.content);
        let system = "You are a concise summarizer. Output ONLY valid JSON with \
                      'title' (string) and 'micro_summary' (string, max 3 sentences).";

        let req = ChatRequest {
            messages: vec![
                ChatMessage {
                    role: Role::System,
                    content: system.to_string(),
                    tool_calls: None,
                    tool_call_id: None,
                },
                ChatMessage {
                    role: Role::User,
                    content: prompt,
                    tool_calls: None,
                    tool_call_id: None,
                },
            ],
            sampling: SamplingConfig::new(Some(ROOT_SUMMARY_MAX_TOKENS as u32)),
            constraint: Some(GuidanceConstraint::JsonSchema(serde_json::json!({
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "micro_summary": {"type": "string", "maxLength": 500}
                },
                "required": ["title", "micro_summary"]
            }))),
            enable_thinking: false,
            ..Default::default()
        };
        let content = send_chat_request_with_retry(provider, req, 3, Duration::from_millis(500)).await?;

        let root: RootSummary = if let Some(start) = content.find('{') {
            if let Some(end) = content.rfind('}') {
                let json = &content[start..=end];
                serde_json::from_str(json)
                    .map_err(|e| anyhow!("Root summary parse error: {} in {}", e, json))?
            } else {
                anyhow::bail!("No JSON end in root summary")
            }
        } else {
            anyhow::bail!("No JSON start in root summary")
        };

        Ok(root)
    }

    async fn is_summary_sufficient(
        &self,
        provider: &dyn crate::InferenceProvider,
        micro_summary: &str,
        latest_turn: &str,
    ) -> Result<bool> {
        use crate::inference_provider::{ChatMessage, ChatRequest, Role, SamplingConfig};

        let prompt = format!(
            "Is this summary sufficient to understand the latest turn?\n\n\
             Summary: {}\n\nLatest Turn: {}\n\nRespond ONLY with YES or NO.",
            micro_summary, latest_turn
        );
        let req = ChatRequest {
            messages: vec![ChatMessage {
                role: Role::User,
                content: prompt,
                tool_calls: None,
                tool_call_id: None,
            }],
            sampling: SamplingConfig::new(Some(SUFFICIENCY_CHECK_MAX_TOKENS as u32)),
            constraint: None,
            enable_thinking: false,
            ..Default::default()
        };
        let resp = send_chat_request_with_retry(provider, req, 3, Duration::from_millis(500)).await?;
        Ok(resp.trim().to_uppercase().contains("YES"))
    }

    async fn generate_specialized_summary(
        &self,
        provider: &dyn crate::InferenceProvider,
        turn: &HistoryTurn,
        latest_turn: &str,
    ) -> Result<SpecializedSummary> {
        use crate::inference_provider::{ChatMessage, ChatRequest, GuidanceConstraint, Role, SamplingConfig};

        let prompt = format!(
            "Extract information from this historical turn specifically relevant \
             to the latest turn.\n\nHistorical: {}\n\nLatest: {}\n\nOutput ONLY valid \
             JSON with 'domain' (1-2 word topic) and 'extracted_information' fields.",
            turn.content, latest_turn
        );
        let req = ChatRequest {
            messages: vec![ChatMessage {
                role: Role::User,
                content: prompt,
                tool_calls: None,
                tool_call_id: None,
            }],
            sampling: SamplingConfig::new(Some(SPECIALIZED_SUMMARY_MAX_TOKENS as u32)),
            constraint: Some(GuidanceConstraint::JsonSchema(serde_json::json!({
                "type": "object",
                "properties": {
                    "domain": {"type": "string"},
                    "extracted_information": {"type": "string"}
                },
                "required": ["domain", "extracted_information"]
            }))),
            enable_thinking: false,
            ..Default::default()
        };
        let content = send_chat_request_with_retry(provider, req, 3, Duration::from_millis(500)).await?;

        if let Some(start) = content.find('{') {
            if let Some(end) = content.rfind('}') {
                let json = &content[start..=end];
                let spec: SpecializedSummary = serde_json::from_str(json)?;
                return Ok(spec);
            }
        }
        anyhow::bail!("Failed to parse specialized summary")
    }

    async fn store_specialized_summary(
        &self,
        provider: &dyn crate::InferenceProvider,
        summary_id: i64,
        spec: &SpecializedSummary,
    ) {
        if summary_id == 0 {
            return;
        }
        let emb = match Self::embed_text(provider, &spec.extracted_information).await {
            Ok(e) => e,
            Err(_) => return,
        };
        let nearest = self.find_nearest_domain(&emb, DOMAIN_CREATION_DISTANCE);
        let domain_id = match nearest {
            Some(id) => id,
            None => {
                match self.create_domain(&spec.domain, &emb) {
                    Ok(id) => id,
                    Err(_) => return,
                }
            }
        };
        let _ = self.store_domain_summary(summary_id, domain_id, &spec.extracted_information, &emb);
    }

    async fn generate_metasummary(
        &self,
        provider: &dyn crate::InferenceProvider,
        turns: &[HistoryTurn],
        latest_turn: &str,
    ) -> Result<MetaSummary> {
        use crate::inference_provider::{ChatMessage, ChatRequest, GuidanceConstraint, Role, SamplingConfig};

        let mut summary_text = String::new();
        let mut included_domains = Vec::new();
        let mut included_turn_indices = Vec::new();
        let mut turns_truncated = false;

        for turn in turns {
            if summary_text.len() >= COMPRESSION_PROMPT_CHAR_LIMIT {
                turns_truncated = true;
                break;
            }
            if turn.excluded_from_compression {
                continue;
            }
            included_turn_indices.push(turn.turn_index);

            summary_text.push_str(&format!(
                "\n--- Turn {} ({}) ---\n",
                turn.turn_index,
                role_label(&turn.role),
            ));

            if let Some((title, micro)) = self.load_summary_from_db(&turn.file_key) {
                summary_text.push_str(&format!("Summary: {} ({})\n", micro, title));
                if let Some(domains) = self.load_domains_for_file(&turn.file_key) {
                    for (domain_name, info) in &domains {
                        summary_text.push_str(&format!("  [{}]: {}\n", domain_name, info));
                        if !included_domains.contains(domain_name) {
                            included_domains.push(domain_name.clone());
                        }
                    }
                }
            } else {
                let preview: String = turn.content.chars().take(300).collect();
                summary_text.push_str(&format!("Content (raw): {}\n", preview));
            }
        }

        let system_prompt = "You are a context archivist. Summarize conversation summaries into \
                             a single metasummary. Deduplicate repeated information. Mention all \
                             domains/topics covered. Include turn indices of important turns. \
                             Output ONLY valid JSON with 'content' (string), 'included_domains' \
                             (array of strings), and 'included_turn_indices' (array of integers).";
        let truncation_note = if turns_truncated {
            "\n(Note: older historical turns were truncated to stay within the context window.)\n"
        } else {
            ""
        };
        let prompt = format!(
            "Latest turn context: {}\n\nHistorical summaries to collate:{}\n{}\n\n\
             Produce a concise metasummary that captures all unique information.",
            latest_turn, truncation_note, summary_text
        );

        let req = ChatRequest {
            messages: vec![
                ChatMessage {
                    role: Role::System,
                    content: system_prompt.to_string(),
                    tool_calls: None,
                    tool_call_id: None,
                },
                ChatMessage {
                    role: Role::User,
                    content: prompt,
                    tool_calls: None,
                    tool_call_id: None,
                },
            ],
            sampling: SamplingConfig::new(Some(METASUMMARY_MAX_TOKENS as u32)),
            constraint: Some(GuidanceConstraint::JsonSchema(serde_json::json!({
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Concise metasummary capturing all unique information"
                    },
                    "included_domains": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "included_turn_indices": {
                        "type": "array",
                        "items": {"type": "integer"}
                    }
                },
                "required": ["content", "included_domains", "included_turn_indices"]
            }))),
            enable_thinking: false,
            ..Default::default()
        };
        let content = send_chat_request_with_retry(provider, req, 3, Duration::from_millis(500)).await?;

        let mut ms: MetaSummary = serde_json::from_str(&content).or_else(|_| {
            if let Some(start) = content.find('{') {
                if let Some(end) = content.rfind('}') {
                    serde_json::from_str(&content[start..=end])
                        .map_err(|e| anyhow!("JSON Parse Error: {}", e))
                } else {
                    anyhow::bail!("No JSON end in model response")
                }
            } else {
                anyhow::bail!("No JSON start in model response")
            }
        })?;

        ms.generated_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs();
        ms.included_domains = included_domains;
        ms.included_turn_indices = included_turn_indices;
        Ok(ms)
    }

    async fn embed_text(
        provider: &dyn crate::InferenceProvider,
        text: &str,
    ) -> Result<Vec<f32>> {
        provider.embed(text).await
    }
}

// ── Token / LLM helpers ────────────────────────────────────────────────

pub(crate) async fn count_tokens(
    provider: &dyn crate::InferenceProvider,
    text: &str,
) -> usize {
    match provider.count_tokens(text).await {
        Ok(n) => n,
        Err(_) => text.len() / 4,
    }
}

pub(crate) async fn send_chat_request_with_retry(
    provider: &dyn crate::InferenceProvider,
    request: crate::inference_provider::ChatRequest,
    max_retries: u32,
    base_delay: Duration,
) -> Result<String> {
    let mut remaining = max_retries;
    loop {
        match provider.chat(request.clone()).await {
            Ok(resp) => return Ok(resp),
            Err(e) if remaining > 0 => {
                let attempt = max_retries - remaining;
                let delay = base_delay * 2u32.pow(attempt);
                eprintln!(
                    "compression (provider): LLM request failed (attempt {}/{}, retrying in {:?}): {:?}",
                    attempt + 1,
                    max_retries + 1,
                    delay,
                    e
                );
                remaining -= 1;
                tokio::time::sleep(delay).await;
            }
            Err(e) => return Err(e),
        }
    }
}

// ── Role helpers ───────────────────────────────────────────────────────

pub fn history_role_to_role(role: &HistoryTurnRole) -> Role {
    match role {
        HistoryTurnRole::User => Role::User,
        HistoryTurnRole::Assistant => Role::Assistant,
        HistoryTurnRole::Tool => Role::Tool,
        HistoryTurnRole::System | HistoryTurnRole::Skill(_, _) => Role::System,
    }
}

pub fn turn_to_tuple(turn: &HistoryTurn) -> (Role, String) {
    let role = history_role_to_role(&turn.role);
    let content = match &turn.role {
        HistoryTurnRole::Skill(name, desc) => {
            format!("## Skill: {}\n{}\n\n{}", name, desc, turn.content)
        }
        _ => turn.content.clone(),
    };
    (role, content)
}

fn role_label(role: &HistoryTurnRole) -> &str {
    match role {
        HistoryTurnRole::User => "user",
        HistoryTurnRole::Assistant => "assistant",
        HistoryTurnRole::System => "system",
        HistoryTurnRole::Skill(_, _) => "skill",
        HistoryTurnRole::Tool => "tool",
    }
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
