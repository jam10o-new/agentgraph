//! Message building and file handling for agentgraph.

use crate::Args;
use crate::context::{CompressionAgent, ContextManager, LoadStrategy};
use crate::types::{ContentType, FileMessage, FileMetadata, MessageContent, ModelSlot};
use anyhow::{Context, Result};
use image;
use mistralrs::{TextMessageRole, VisionMessages};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use tokio::fs;
use walkdir::WalkDir;

/// YAML frontmatter for skill files
#[derive(Debug, Clone, Deserialize)]
pub struct SkillFrontmatter {
    pub name: String,
    pub description: String,
}

/// Extract YAML frontmatter from a text file
/// Returns (frontmatter, remaining_content) if frontmatter exists, None otherwise
pub fn extract_frontmatter(content: &str) -> Option<(SkillFrontmatter, String)> {
    if !content.starts_with("---\n") {
        return None;
    }

    // Find the closing ---
    let rest = &content[4..]; // Skip opening ---
    if let Some(end_pos) = rest.find("\n---\n") {
        let frontmatter_str = &rest[..end_pos];
        let remaining = rest[end_pos + 5..].trim_start().to_string();

        if let Ok(frontmatter) = serde_yaml::from_str::<SkillFrontmatter>(frontmatter_str) {
            return Some((frontmatter, remaining));
        }
    }

    None
}

/// Load skill context from a file - extracts only frontmatter if present
pub async fn load_skill_context(path: &Path) -> Result<String> {
    let content = fs::read_to_string(path).await?;

    // Check if file has YAML frontmatter
    if let Some((frontmatter, _remaining)) = extract_frontmatter(&content) {
        // Return only name and description for context injection
        Ok(format!(
            "## Skill: {}\n{}",
            frontmatter.name, frontmatter.description
        ))
    } else {
        // No frontmatter - return full content
        Ok(content)
    }
}

/// Load full skill content (for RRDS command)
pub async fn load_full_skill_content(path: &Path) -> Result<String> {
    fs::read_to_string(path)
        .await
        .map_err(|e| anyhow::anyhow!(e))
}

/// Detect content type from file extension
pub fn detect_content_type(path: &Path) -> ContentType {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase())
        .unwrap_or_default();

    match ext.as_str() {
        "wav" | "mp3" | "ogg" | "flac" | "m4a" | "aac" => ContentType::Audio,
        "mp4" | "avi" | "mkv" | "mov" | "webm" => ContentType::Video,
        "jpg" | "jpeg" | "png" | "gif" | "webp" | "bmp" => ContentType::Image,
        _ => ContentType::Text,
    }
}

/// Convert role string to mistral role
pub fn to_mistral_role(role: &str) -> TextMessageRole {
    match role {
        "system" => TextMessageRole::System,
        "assistant" => TextMessageRole::Assistant,
        "tool" => TextMessageRole::Tool,
        _ => TextMessageRole::User,
    }
}

impl FileMessage {
    pub fn format_for_model(
        &self,
        target_slot: ModelSlot,
    ) -> Option<(
        String,
        Option<Vec<image::DynamicImage>>,
        Option<Vec<mistralrs::AudioInput>>,
    )> {
        let include_metadata = self.role == "user";

        match &self.content {
            MessageContent::Text(text) => {
                let final_text = if include_metadata {
                    let metadata_str = format_metadata(&self.metadata);
                    format!("{}\n{}", metadata_str, text)
                } else {
                    text.clone()
                };
                Some((final_text, None, None))
            }
            MessageContent::Image(img) => {
                if target_slot != ModelSlot::Primary {
                    return None;
                }
                let text_prompt = if include_metadata {
                    format_metadata(&self.metadata)
                } else {
                    String::new()
                };
                Some((text_prompt, Some(vec![img.clone()]), None))
            }
            MessageContent::Audio(audio_bytes) => {
                if target_slot != ModelSlot::Secondary {
                    return None;
                }
                let text_prompt = if include_metadata {
                    format_metadata(&self.metadata)
                } else {
                    "".to_string()
                };

                let audio_input = mistralrs::AudioInput::from_bytes(audio_bytes).ok()?;

                Some((text_prompt, None, Some(vec![audio_input])))
            }
        }
    }
}

fn format_metadata(metadata: &FileMetadata) -> String {
    format!(
        "[File: {}; Created: {:?}; Modified: {:?}]",
        metadata.path.display(),
        humantime::format_rfc3339_seconds(metadata.created),
        humantime::format_rfc3339_seconds(metadata.modified)
    )
}

/// Process video file into frames and audio
pub async fn process_video_to_messages(
    video_path: &Path,
    metadata: &FileMetadata,
) -> Result<(Vec<FileMessage>, Vec<FileMessage>)> {
    let temp_dir = std::env::temp_dir().join(format!(
        "agentgraph_video_{}",
        metadata
            .created
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs()
    ));
    fs::create_dir_all(&temp_dir).await?;

    let frames_dir = temp_dir.join("frames");
    let audio_path = temp_dir.join("audio.wav");
    fs::create_dir_all(&frames_dir).await?;

    // Extract frames (1 per 2 seconds)
    let ffmpeg_frames = tokio::process::Command::new("ffmpeg")
        .args(&[
            "-i",
            video_path.to_str().unwrap(),
            "-vf",
            "fps=0.5",
            "-q:v",
            "2",
            frames_dir.join("frame_%04d.jpg").to_str().unwrap(),
        ])
        .output()
        .await?;

    if !ffmpeg_frames.status.success() {
        anyhow::bail!(
            "Failed to extract frames: {}",
            String::from_utf8_lossy(&ffmpeg_frames.stderr)
        );
    }

    // Extract audio
    let ffmpeg_audio = tokio::process::Command::new("ffmpeg")
        .args(&[
            "-i",
            video_path.to_str().unwrap(),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            audio_path.to_str().unwrap(),
        ])
        .output()
        .await?;

    if !ffmpeg_audio.status.success() {
        anyhow::bail!(
            "Failed to extract audio: {}",
            String::from_utf8_lossy(&ffmpeg_audio.stderr)
        );
    }

    // Load frames as image messages
    let mut frame_messages = Vec::new();
    let mut frame_paths = Vec::new();

    let mut read_dir = fs::read_dir(&frames_dir).await?;
    while let Some(entry) = read_dir.next_entry().await? {
        let path = entry.path();
        if path.extension().map(|e| e == "jpg").unwrap_or(false) {
            frame_paths.push(path);
        }
    }
    frame_paths.sort();

    for frame_path in frame_paths {
        let bytes = fs::read(&frame_path).await?;
        let img = image::load_from_memory(&bytes)?;
        frame_messages.push(FileMessage {
            role: "user".to_string(),
            content: MessageContent::Image(img),
            metadata: FileMetadata {
                path: frame_path,
                created: metadata.created,
                modified: metadata.modified,
            },
            content_type: ContentType::Image,
        });
    }

    // Load audio as audio message
    let audio_bytes = fs::read(&audio_path).await?;
    let audio_message = FileMessage {
        role: "user".to_string(),
        content: MessageContent::Audio(audio_bytes),
        metadata: FileMetadata {
            path: audio_path,
            created: metadata.created,
            modified: metadata.modified,
        },
        content_type: ContentType::Audio,
    };

    // Cleanup temp dir in background
    tokio::spawn(async move {
        let _ = fs::remove_dir_all(temp_dir).await;
    });

    Ok((frame_messages, vec![audio_message]))
}

/// Normalize path to directory
pub fn normalize_to_dir(path: &Path) -> Option<PathBuf> {
    if path.is_dir() {
        Some(path.to_path_buf())
    } else if path.is_file() {
        path.parent().map(|p| p.to_path_buf())
    } else {
        None
    }
}

/// Collect all directories to watch
pub async fn collect_viewer_dirs(args: &Args) -> Result<Vec<PathBuf>> {
    use std::collections::BTreeSet;

    let mut dirs = BTreeSet::new();

    let all_paths = args
        .input_final
        .iter()
        .chain(&args.input_cat)
        .chain(&args.system_final)
        .chain(&args.system_cat)
        .chain(&args.assistant_final)
        .chain(&args.assistant_cat)
        .chain(args.output_new.iter())
        .chain(args.output_overwrite.iter());

    for path in all_paths {
        if let Some(dir) = normalize_to_dir(path) {
            dirs.insert(dir.clone());
            fs::create_dir_all(dir).await?;
        }
    }

    Ok(dirs.into_iter().collect())
}

/// Load messages from a directory, optionally applying context management
pub async fn load_dir_messages(
    dir: &Path,
    role: &str,
    concat: bool,
    latest_n: usize,
    context_manager: Option<&ContextManager>,
    compression_agent: Option<&CompressionAgent>,
    message_turn_offset: usize,
) -> Result<Vec<FileMessage>> {
    let mut entries: Vec<_> = WalkDir::new(dir)
        .max_depth(1)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .filter_map(|e| {
            let meta = e.metadata().ok()?;
            let created = meta.created().ok()?;
            let modified = meta.modified().ok()?;
            Some((created, modified, e.path().to_path_buf()))
        })
        .collect();

    if entries.is_empty() {
        return Ok(Vec::new());
    }

    entries.sort_by_key(|(created, _, _)| *created);
    entries.reverse();

    if !concat {
        entries.truncate(latest_n);
    }

    let mut messages = Vec::new();

    for (idx, (created, modified, path)) in entries.iter().enumerate() {
        let content_type = detect_content_type(path);

        let content = if let Some(mgr) = context_manager {
            // Determine load strategy based on turn age
            let message_turn = message_turn_offset + idx;
            let has_summary = mgr.has_summary(path);
            let strategy = mgr.determine_load_strategy(message_turn, role, has_summary);

            // For Full strategy or non-text content, load full content
            if strategy == LoadStrategy::Full {
                match content_type {
                    ContentType::Text => MessageContent::Text(load_skill_context(path).await?),
                    ContentType::Image => {
                        let bytes = fs::read(path).await?;
                        let img = image::load_from_memory(&bytes)
                            .with_context(|| format!("Failed to load image: {}", path.display()))?;
                        MessageContent::Image(img)
                    }
                    ContentType::Audio => {
                        let bytes = fs::read(path).await?;
                        MessageContent::Audio(bytes)
                    }
                    ContentType::Video => {
                        MessageContent::Text(format!("[Video file: {}]", path.display()))
                    }
                }
            } else if strategy == LoadStrategy::Compress {
                if let Some(mgr) = context_manager {
                    if let Some(summary) = mgr.load_summary(path) {
                        MessageContent::Text(summary.as_context_string())
                    } else if let Some(agent) = compression_agent {
                        // Perform compression on the fly
                        let bytes = fs::read(path).await?;
                        let mut image_content = None;
                        
                        let content_str = match content_type {
                            ContentType::Text => String::from_utf8_lossy(&bytes).into_owned(),
                            ContentType::Image => {
                                let img = image::load_from_memory(&bytes)
                                    .with_context(|| format!("Failed to load image for compression: {}", path.display()))?;
                                image_content = Some(img);
                                "[IMAGE CONTENT]".to_string()
                            }
                            ContentType::Audio => "[AUDIO CONTENT]".to_string(),
                            ContentType::Video => "[VIDEO CONTENT]".to_string(),
                        };

                        let request = crate::context::CompressionRequest {
                            baseline_turn: "Summarize this historical context.".to_string(),
                            historical_turn: (message_turn, content_str),
                            image: image_content,
                            model_slot: ModelSlot::Primary,
                        };

                        match agent.compress(request).await {
                            Ok(resp) => {
                                use crate::context::RelevanceResult;
                                let summary_text = match resp.relevance {
                                    RelevanceResult::Snippets { snippets, relevance_reason } => {
                                        format!("[RELEVANT HISTORICAL CONTENT - {}]:\n{}", 
                                            relevance_reason, snippets.join("\n"))
                                    }
                                    RelevanceResult::MicroSummary { text } => {
                                        format!("[HISTORICAL SUMMARY]: {}", text)
                                    }
                                };
                                MessageContent::Text(summary_text)
                            }
                            Err(_) => {
                                // Fallback to full content on error
                                match content_type {
                                    ContentType::Text => MessageContent::Text(String::from_utf8_lossy(&bytes).into_owned()),
                                    ContentType::Image => {
                                        let img = image::load_from_memory(&bytes)
                                            .with_context(|| format!("Failed to load image: {}", path.display()))?;
                                        MessageContent::Image(img)
                                    }
                                    ContentType::Audio => MessageContent::Audio(bytes),
                                    ContentType::Video => MessageContent::Text(format!("[Video file: {}]", path.display()))
                                }
                            }
                        }
                    } else {
                        // No agent, fallback to full
                        match content_type {
                            ContentType::Text => MessageContent::Text(load_skill_context(path).await?),
                            ContentType::Image => {
                                let bytes = fs::read(path).await?;
                                let img = image::load_from_memory(&bytes)
                                    .with_context(|| format!("Failed to load image: {}", path.display()))?;
                                MessageContent::Image(img)
                            }
                            ContentType::Audio => {
                                let bytes = fs::read(path).await?;
                                MessageContent::Audio(bytes)
                            }
                            ContentType::Video => {
                                MessageContent::Text(format!("[Video file: {}]", path.display()))
                            }
                        }
                    }
                } else {
                    // Fallback
                    match content_type {
                        ContentType::Text => MessageContent::Text(load_skill_context(path).await?),
                        ContentType::Image => {
                            let bytes = fs::read(path).await?;
                            let img = image::load_from_memory(&bytes)
                                .with_context(|| format!("Failed to load image: {}", path.display()))?;
                            MessageContent::Image(img)
                        }
                        ContentType::Audio => {
                            let bytes = fs::read(path).await?;
                            MessageContent::Audio(bytes)
                        }
                        ContentType::Video => {
                            MessageContent::Text(format!("[Video file: {}]", path.display()))
                        }
                    }
                }
            } else {
                // For MicroSummary
                if let Some(summary) = mgr.load_summary(path) {
                    MessageContent::Text(summary.as_context_string())
                } else {
                    // No cached summary - load full content
                    match content_type {
                        ContentType::Text => MessageContent::Text(load_skill_context(path).await?),
                        ContentType::Image => {
                            let bytes = fs::read(path).await?;
                            let img = image::load_from_memory(&bytes)
                                .with_context(|| format!("Failed to load image: {}", path.display()))?;
                            MessageContent::Image(img)
                        }
                        ContentType::Audio => {
                            let bytes = fs::read(path).await?;
                            MessageContent::Audio(bytes)
                        }
                        ContentType::Video => {
                            MessageContent::Text(format!("[Video file: {}]", path.display()))
                        }
                    }
                }
            }
        } else {
            // Standard loading without context manager
            match content_type {
                ContentType::Text => MessageContent::Text(fs::read_to_string(path).await?),
                ContentType::Image => {
                    let bytes = fs::read(path).await?;
                    let img = image::load_from_memory(&bytes)
                        .with_context(|| format!("Failed to load image: {}", path.display()))?;
                    MessageContent::Image(img)
                }
                ContentType::Audio => {
                    let bytes = fs::read(path).await?;
                    MessageContent::Audio(bytes)
                }
                ContentType::Video => MessageContent::Text(format!("[Video file: {}]", path.display())),
            }
        };

        messages.push(FileMessage {
            role: role.to_string(),
            content,
            metadata: FileMetadata {
                path: path.clone(),
                created: *created,
                modified: *modified,
            },
            content_type,
        });
    }

    Ok(messages)
}

/// Get the latest file in a directory
pub fn latest_file(dir: &Path) -> Result<PathBuf> {
    let mut files: Vec<_> = WalkDir::new(dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .filter_map(|e| {
            let time = e.metadata().ok()?.created().ok()?;
            Some((time, e.path().to_path_buf()))
        })
        .collect();

    files.sort_by_key(|(t, _)| *t);

    files
        .last()
        .map(|(_, p)| p.clone())
        .context("no valid files found")
}

/// Concatenate all files in a directory, handling skill frontmatter
pub async fn concat_files(dir: &Path) -> Result<String> {
    let mut out = String::new();
    for entry in WalkDir::new(dir).into_iter().filter_map(|e| e.ok()) {
        if entry.file_type().is_file() {
            let content = load_skill_context(entry.path()).await?;
            out.push_str(&content);
            out.push('\n');
        }
    }
    Ok(out)
}

/// Build messages from all input sources
pub async fn build_messages(
    args: &Args,
    context_manager: Option<&ContextManager>,
    compression_agent: Option<&CompressionAgent>,
) -> Result<(VisionMessages, Option<VisionMessages>)> {
    let mut primary_messages = VisionMessages::new();
    let mut secondary_messages: Option<VisionMessages> = None;

    // System messages (combined into single message)
    let mut system_messages = Vec::new();
    for dir in &args.system_final {
        if let Ok(msgs) = load_dir_messages(dir, "system", false, 1, None, None, 0).await {
            system_messages.extend(msgs);
        }
    }
    for dir in &args.system_cat {
        if let Ok(msgs) = load_dir_messages(dir, "system", true, args.latest_n, None, None, 0).await {
            system_messages.extend(msgs);
        }
    }
    // Re-sort to chronological order (load_dir_messages returns latest first)
    system_messages.sort_by_key(|m| m.metadata.created);

    let mut system_text = String::new();
    let mut system_images = Vec::new();
    let mut system_audio_bytes = Vec::new();

    for msg in system_messages {
        match msg.content {
            MessageContent::Text(t) => {
                system_text.push_str(&t);
                system_text.push('\n');
            }
            MessageContent::Image(img) => {
                system_images.push(img);
            }
            MessageContent::Audio(audio) => {
                system_audio_bytes.push(audio);
            }
        }
    }

    // Tool system message
    if args.tools {
        if args.verbose {
            eprintln!("Tools enabled, adding tool instruction message.");
        }
        let tool_instructions = format!(
            "You have access to command execution tools. To spawn a process, provide the binary and arguments: {open_exec}command arg1 arg2 ...{close_exec} (returns index). For long-running processes (daemons, servers, agents), prefix with 'setsid' to detach: {open_exec}setsid command arg1 arg2 ...{close_exec}. To kill: {open_kill}idx{close_kill}. To read output: {open_read}idx{close_read}. To write stdin: {open_writ}idx input text{close_writ} (include \\n in text to send enter/newline). Commands execute immediately and return results. You will need to perform multiple Commands execution tool calls to execute and then read the outputs of commands you executed.",
            open_exec = crate::CMD_OPEN_EXEC,
            close_exec = crate::CMD_CLOSE_EXEC,
            open_kill = crate::CMD_OPEN_KILL,
            close_kill = crate::CMD_CLOSE_KILL,
            open_read = crate::CMD_OPEN_READ,
            close_read = crate::CMD_CLOSE_READ,
            open_writ = crate::CMD_OPEN_WRIT,
            close_writ = crate::CMD_CLOSE_WRIT
        );
        system_text.push_str("\n## Tool Instructions\n");
        system_text.push_str(&tool_instructions);
        system_text.push('\n');
    }

    // Skill reading instruction (always available, not gated by --tools flag)
    let skill_instructions = format!(
        "You can read skill files from your context directories using: {open_skill}skill_name{close_skill}. This searches all context directories for files with YAML frontmatter matching the skill name and loads the full content.",
        open_skill = crate::CMD_OPEN_READ_SKILL,
        close_skill = crate::CMD_CLOSE_READ_SKILL
    );
    system_text.push_str("\n## Skill Instructions\n");
    system_text.push_str(&skill_instructions);
    system_text.push('\n');

    // Handle multimodal system message
    if !system_images.is_empty() && !system_audio_bytes.is_empty() {
        if let Some(agent) = compression_agent {
            let combined_content = format!(
                "System prompt with {} images and {} audio clips.\nText: {}",
                system_images.len(),
                system_audio_bytes.len(),
                system_text
            );
            let request = crate::context::CompressionRequest {
                baseline_turn: "Summarize this multimodal system prompt.".to_string(),
                historical_turn: (0, combined_content),
                image: system_images.first().cloned(),
                model_slot: ModelSlot::Primary,
            };
            if let Ok(resp) = agent.compress(request).await {
                use crate::context::RelevanceResult;
                let summary_text = match resp.relevance {
                    RelevanceResult::Snippets {
                        snippets,
                        relevance_reason,
                    } => {
                        format!(
                            "[SYSTEM SUMMARY - {}]:\n{}",
                            relevance_reason,
                            snippets.join("\n")
                        )
                    }
                    RelevanceResult::MicroSummary { text } => {
                        format!("[SYSTEM SUMMARY]: {}", text)
                    }
                };
                primary_messages =
                    primary_messages.add_message(TextMessageRole::System, summary_text);
            } else {
                primary_messages =
                    primary_messages.add_message(TextMessageRole::System, system_text.clone());
            }
        } else {
            primary_messages =
                primary_messages.add_message(TextMessageRole::System, system_text.clone());
        }
    } else if !system_images.is_empty() {
        primary_messages = primary_messages.add_image_message(
            TextMessageRole::System,
            system_text.clone(),
            system_images,
        );
    } else if !system_audio_bytes.is_empty() {
        let mut audio_inputs = Vec::new();
        for bytes in system_audio_bytes {
            if let Ok(input) = mistralrs::AudioInput::from_bytes(&bytes) {
                audio_inputs.push(input);
            }
        }
        let sec_msgs = secondary_messages.get_or_insert_with(VisionMessages::new);
        *sec_msgs = sec_msgs.clone().add_multimodal_message(
            TextMessageRole::System,
            &system_text,
            vec![],
            audio_inputs,
        );
        primary_messages = primary_messages.add_message(TextMessageRole::System, system_text);
    } else {
        primary_messages = primary_messages.add_message(TextMessageRole::System, system_text);
    }

    // Collect all messages
    let mut timeline: Vec<FileMessage> = Vec::new();
    let mut video_tasks = Vec::new();

    let sources = [
        (&args.input_final, "user", false),
        (&args.input_cat, "user", true),
        (&args.assistant_final, "assistant", false),
        (&args.assistant_cat, "assistant", true),
    ];

    // Track turn offset for each source if using context manager
    let mut turn_offset = 0;

    for (paths, role, concat) in sources {
        for dir in paths {
            if args.verbose {
                if context_manager.is_some() {
                    eprintln!(
                        "adding messages from: {} (with context management)",
                        dir.display()
                    );
                } else {
                    eprintln!("adding messages from: {}", dir.display());
                }
            }
            let msgs = load_dir_messages(
                dir,
                role,
                concat,
                args.latest_n,
                context_manager,
                compression_agent,
                turn_offset,
            )
            .await?;

            turn_offset += msgs.len();

            for msg in msgs {
                if msg.content_type == ContentType::Video {
                    if let MessageContent::Text(path_str) = &msg.content {
                        let path = PathBuf::from(
                            path_str
                                .trim_start_matches("[Video file: ")
                                .trim_end_matches(']'),
                        );
                        let metadata = msg.metadata.clone();
                        if let Ok(tuple) = process_video_to_messages(&path, &metadata).await {
                            video_tasks.push(tuple);
                        }
                    }
                } else {
                    timeline.push(msg);
                }
            }
        }
    }

    // Process videos in parallel
    if !video_tasks.is_empty() {
        for result in video_tasks {
            let (frames, audio) = result;
            timeline.extend(frames);
            timeline.extend(audio);
        }
    }

    timeline.sort_by_key(|msg| msg.metadata.created);

    if args.verbose {
        if context_manager.is_some() {
            eprintln!(
                "Building {} messages with context management.",
                timeline.len()
            );
        } else {
            eprintln!("Building {} messages.", timeline.len());
        }
    }

    // Route messages to appropriate model slots
    let mut primary_count = 0;
    let mut primary_image_count = 0;
    let mut maybe_switchable = (false, false);
    for msg in timeline {
        if let Some((text, images, _)) = msg.format_for_model(ModelSlot::Primary) {
            primary_count += 1;
            if let Some(images) = images {
                primary_image_count += images.len();
                primary_messages =
                    primary_messages.add_image_message(to_mistral_role(&msg.role), text, images);
                maybe_switchable = (true, maybe_switchable.1 || false);
            } else {
                primary_messages = primary_messages.add_message(to_mistral_role(&msg.role), text);
            }
        }

        if let Some((text, _, audio)) = msg.format_for_model(ModelSlot::Secondary) {
            if let Some(audio) = audio {
                let sec_msgs = secondary_messages.get_or_insert_with(VisionMessages::new);
                *sec_msgs = sec_msgs.clone().add_multimodal_message(
                    to_mistral_role(&msg.role),
                    &text,
                    vec![],
                    audio,
                );
                maybe_switchable = (maybe_switchable.0 || false, true);
            } else {
                let sec_msgs = secondary_messages.get_or_insert_with(VisionMessages::new);
                *sec_msgs = sec_msgs
                    .clone()
                    .add_message(to_mistral_role(&msg.role), text);
            }
        }
    }

    if args.verbose {
        eprintln!(
            "Messages built. Primary: {} messages ({} images)",
            primary_count, primary_image_count
        );
        if context_manager.is_some() {
            eprintln!("Messages built with context management.");
        }
    }

    match maybe_switchable {
        (true, true) => Ok((primary_messages, secondary_messages)),
        (false, true) => Ok((primary_messages, secondary_messages)),
        (true, false) => Ok((primary_messages, None)),
        (false, false) => Ok((primary_messages, None)),
    }
}
