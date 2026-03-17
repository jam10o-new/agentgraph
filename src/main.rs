use anyhow::{Context, Result};
use clap::Parser;
use notify::{RecursiveMode, Watcher};
use std::fmt::Debug;
use std::path::{Path, PathBuf};

use mistralrs::{GgufModelBuilder, IsqBits, TextMessageRole, VisionMessages, VisionModelBuilder};
use std::sync::Arc;

use walkdir::WalkDir;

use futures::StreamExt;
use futures::stream::FuturesUnordered;
use rand::RngExt;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::time::Duration;
use tokio::fs::{self, File};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::process::Command;
use tokio::sync::mpsc::{Receiver, Sender, UnboundedSender, channel, unbounded_channel};
use tokio::sync::oneshot;
use tokio::sync::{Mutex, broadcast, watch};

// Add these constants near the top of your file
const CMD_OPEN_EXEC: &str = "【EXEC ";
const CMD_CLOSE_EXEC: &str = " CEXE】";
const CMD_OPEN_KILL: &str = "【KILL ";
const CMD_CLOSE_KILL: &str = " LLIK】";
const CMD_OPEN_READ: &str = "【READ ";
const CMD_CLOSE_READ: &str = " DAER】";
const CMD_OPEN_WRIT: &str = "【WRIT ";
const CMD_CLOSE_WRIT: &str = " TIRW】";
// Maximum length of any opener (for pre-active buffer sizing)
const MAX_OPENER_LEN: usize = 6; // "【EXEC " is 6 chars

// Command types
enum CommandType {
    Exec(String),        // full command string: "command arg arg"
    Kill(usize),         // index
    Read(usize),         // index
    Writ(usize, String), // index and input
}

// Helper struct to manage command state within the loop
struct CommandParser {
    buffer: String,
    active: bool,
    cmd_type: Option<CmdOpenType>,
}

impl CommandParser {
    fn new() -> Self {
        Self {
            buffer: String::new(),
            active: false,
            cmd_type: None,
        }
    }

    fn reset(&mut self) {
        self.buffer.clear();
        self.active = false;
        self.cmd_type = None;
    }

    // Check if combined (buffer + new content) contains a command opener
    // Returns (bytes_before_opener, opener_type, bytes_after_opener) if found
    fn find_opener(&self, combined: &str) -> Option<(usize, CmdOpenType, usize)> {
        // Check for each opener type, find earliest occurrence
        let mut earliest: Option<(usize, CmdOpenType, usize)> = None;

        if let Some(pos) = combined.find(CMD_OPEN_EXEC) {
            let after_pos = pos + CMD_OPEN_EXEC.len();
            if earliest.is_none() || pos < earliest.unwrap().0 {
                earliest = Some((pos, CmdOpenType::Exec, after_pos));
            }
        }
        if let Some(pos) = combined.find(CMD_OPEN_KILL) {
            let after_pos = pos + CMD_OPEN_KILL.len();
            if earliest.is_none() || pos < earliest.unwrap().0 {
                earliest = Some((pos, CmdOpenType::Kill, after_pos));
            }
        }
        if let Some(pos) = combined.find(CMD_OPEN_READ) {
            let after_pos = pos + CMD_OPEN_READ.len();
            if earliest.is_none() || pos < earliest.unwrap().0 {
                earliest = Some((pos, CmdOpenType::Read, after_pos));
            }
        }
        if let Some(pos) = combined.find(CMD_OPEN_WRIT) {
            let after_pos = pos + CMD_OPEN_WRIT.len();
            if earliest.is_none() || pos < earliest.unwrap().0 {
                earliest = Some((pos, CmdOpenType::Writ, after_pos));
            }
        }

        earliest
    }

    // Check if buffer contains a closer (now using contains/finds instead of ends_with)
    // Returns (cmd_content, remaining_after_closer) if found
    fn find_closer(&self, cmd_type: CmdOpenType) -> Option<(String, String)> {
        let closer = match cmd_type {
            CmdOpenType::Exec => CMD_CLOSE_EXEC,
            CmdOpenType::Kill => CMD_CLOSE_KILL,
            CmdOpenType::Read => CMD_CLOSE_READ,
            CmdOpenType::Writ => CMD_CLOSE_WRIT,
        };

        self.buffer.find(closer).map(|pos| {
            let content = self.buffer[..pos].to_string();
            let remaining = self.buffer[pos + closer.len()..].to_string();
            (content, remaining)
        })
    }

    // Process new content with sliding window handling
    // Returns (content_to_output, completed_command, leftover_for_next_iteration)
    fn process(&mut self, content: &str) -> (String, Option<CommandType>, String) {
        if !self.active {
            // PRE-ACTIVE MODE: Check if opener appears across buffer+content boundary

            let combined = format!("{}{}", self.buffer, content);
            if let Some((before_pos, cmd_type, after_pos)) = self.find_opener(&combined) {
                // Found opener! Transition to active mode

                // Content before opener goes to output
                let before_opener = &combined[..before_pos];

                // Content after opener goes into command buffer
                let after_opener = &combined[after_pos..];

                self.active = true;
                self.cmd_type = Some(cmd_type);
                self.buffer.clear();
                self.buffer.push_str(after_opener);

                // Check if closer is already in the remaining content
                if let Some((cmd_content, remaining)) = self.find_closer(cmd_type) {
                    let cmd = self.parse_command_content(cmd_type, &cmd_content);
                    self.reset();
                    return (before_opener.to_string(), cmd, remaining);
                }

                return (before_opener.to_string(), None, String::new());
            }

            // No opener found - update sliding window buffer
            let combined = format!("{}{}", self.buffer, content);
            if combined.len() <= MAX_OPENER_LEN {
                // Still building up to max opener length
                self.buffer = combined;
                return (String::new(), None, String::new());
            } else {
                // Find a valid char boundary to split at, rounding UP buffer size if needed
                // We want to keep at least MAX_OPENER_LEN bytes in buffer, but may keep more
                // to ensure we don't split a multi-byte character

                let split_point = combined.floor_char_boundary(combined.len() - MAX_OPENER_LEN);

                // Everything before split_point goes to output
                let output = combined[..split_point].to_string();
                // Everything from split_point onward stays in buffer (may be > MAX_OPENER_LEN)
                self.buffer = combined[split_point..].to_string();

                return (output, None, String::new());
            }
        }

        // ACTIVE MODE: Accumulating command content
        self.buffer.push_str(content);

        // Check for closer in the accumulated buffer
        if let Some(cmd_type) = self.cmd_type {
            if let Some((cmd_content, remaining)) = self.find_closer(cmd_type) {
                let cmd = self.parse_command_content(cmd_type, &cmd_content);
                self.reset();
                return (String::new(), cmd, remaining);
            }
        }

        // Still accumulating - no output, no command, no leftover
        (String::new(), None, String::new())
    }

    // Parse command content based on type
    fn parse_command_content(&self, cmd_type: CmdOpenType, content: &str) -> Option<CommandType> {
        match cmd_type {
            CmdOpenType::Exec => Some(CommandType::Exec(content.to_string())),
            CmdOpenType::Kill => content.trim().parse::<usize>().ok().map(CommandType::Kill),
            CmdOpenType::Read => content.trim().parse::<usize>().ok().map(CommandType::Read),
            CmdOpenType::Writ => {
                let trimmed = content.trim();
                if let Some(space_idx) = trimmed.find(' ') {
                    let idx = trimmed[..space_idx].parse::<usize>().ok()?;
                    let input = trimmed[space_idx + 1..].to_string();
                    Some(CommandType::Writ(idx, input))
                } else {
                    None
                }
            }
        }
    }

    /// Flush any characters held in the sliding-window buffer at end-of-stream.
    /// In pre-active mode the buffer contains up to MAX_OPENER_LEN characters that
    /// were being held back in case they formed a command opener — return them as
    /// plain output.  In active mode a command was opened but never closed; discard
    /// it and return an empty string.
    fn flush(&mut self) -> String {
        if self.active {
            self.reset();
            String::new()
        } else {
            std::mem::take(&mut self.buffer)
        }
    }
}

#[derive(Clone, Copy)]
enum CmdOpenType {
    Exec,
    Kill,
    Read,
    Writ,
}

// Hardcoded pipe directory
const PIPE_DIR: &str = "/tmp/agentgraph_pipes";
const PIPE_PREFIX: &str = "pipe_";

// Request format
#[derive(Serialize, Deserialize, Clone, Debug)]
struct InferenceRequest {
    args: Args,
    requesting_pid: u32,
}

pub struct CommandIO {
    pub stdin_tx: Sender<Vec<u8>>,
    pub stdout_rx: Receiver<Vec<u8>>,
    pub stderr_rx: Receiver<Vec<u8>>,
    kill_tx: Option<oneshot::Sender<()>>,
    pub exited_rx: Option<oneshot::Receiver<()>>,
}

impl Drop for CommandIO {
    fn drop(&mut self) {
        if let Some(tx) = self.kill_tx.take() {
            let _ = tx.send(());
        }
    }
}

impl CommandIO {
    pub fn kill(&mut self) {
        if let Some(tx) = self.kill_tx.take() {
            let _ = tx.send(());
        }
    }
}

/// Generic helper to pump data from an async reader to a channel.
async fn pump_stream<R>(mut reader: R, tx: Sender<Vec<u8>>)
where
    R: AsyncReadExt + Unpin,
{
    let mut buf = [0u8; 4096];
    loop {
        match reader.read(&mut buf).await {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                if tx.send(buf[..n].to_vec()).await.is_err() {
                    break;
                }
            }
        }
    }
}

pub async fn spawn_command_io<S, I>(program: S, args: I) -> Result<CommandIO>
where
    S: AsRef<str>,
    I: IntoIterator,
    I::Item: AsRef<str>,
{
    let mut child = Command::new(program.as_ref())
        .args(args.into_iter().map(|s| s.as_ref().to_owned()))
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true)
        .spawn()?;

    let mut child_stdin = child.stdin.take().expect("stdin missing");
    let child_stdout = child.stdout.take().expect("stdout missing");
    let child_stderr = child.stderr.take().expect("stderr missing");

    let (stdin_tx, mut stdin_rx) = channel::<Vec<u8>>(32);
    let (stdout_tx, stdout_rx) = channel::<Vec<u8>>(32);
    let (stderr_tx, stderr_rx) = channel::<Vec<u8>>(32);
    let (kill_tx, kill_rx) = oneshot::channel::<()>();
    let (exit_tx, exit_rx) = oneshot::channel();

    // stdin bridge
    tokio::spawn(async move {
        while let Some(buf) = stdin_rx.recv().await {
            if child_stdin.write_all(&buf).await.is_err() {
                break;
            }
        }
    });

    // stdout/stderr bridges - deduplicated via pump_stream
    tokio::spawn(pump_stream(child_stdout, stdout_tx));
    tokio::spawn(pump_stream(child_stderr, stderr_tx));

    // Child reaper
    tokio::spawn(async move {
        tokio::select! {
            _ = child.wait() => {
                let _ = exit_tx.send(());
            }
            _ = kill_rx => {
                let _ = child.kill().await;
                let _ = child.wait().await;
                // Note: exit_tx is not sent here, see LOGICAL ISSUE NOTE above.
            }
        }
    });

    Ok(CommandIO {
        stdin_tx,
        stdout_rx,
        stderr_rx,
        kill_tx: Some(kill_tx),
        exited_rx: Some(exit_rx),
    })
}

#[derive(Parser, Debug, Clone, Serialize, Deserialize)]
#[command(author, version)]
struct Args {
    /// Watch mode (trigger inference on any system prompt or user input modification or addition)
    #[arg(short = 'W')]
    watch: bool,

    /// Verbose logging
    #[arg(short, long)]
    verbose: bool,

    /// Enable tool use and system prompt addendum for tools.
    #[arg(short, long)]
    tools: bool,

    /// Realtime output
    /// recommended for final agent -> human nodes
    /// STRONGLY discouraged for agent -> agent nodes, or if the output is intended to be structured -
    #[arg(long)]
    stream_realtime: bool,

    /// GGUF file list (which files are we downloading and loading as our local model).
    /// This flag is only relevant when running as a leader - all inference happens using the model running on the current leader node
    #[arg(long, value_delimiter = ',')]
    gguf: Vec<String>,

    /// Model name (HF repo) - PRIMARY slot (vision model)
    #[arg(short = 'm', long, default_value = "Qwen/Qwen3-VL-8B-Instruct")]
    model: String,

    /// Secondary model name (HF repo) - SECONDARY slot (audio model, default: Voxtral)
    #[arg(
        short = 'M',
        long,
        default_value = "mistralai/Voxtral-Mini-4B-Realtime-2602"
    )]
    secondary_model: String,

    /// Number of latest files to include (for -I, -S, -A flags)
    #[arg(long, default_value_t = 1)]
    latest_n: usize,

    /// Realtime audio listener - RTSP/RTMP URL or "pipewire" for local input
    #[arg(long)]
    realtime_listener: Option<String>,

    /// Audio chunk min duration in seconds (for realtime listener)
    #[arg(long, default_value_t = 3.0)]
    audio_chunk_min_secs: f32,

    /// Audio chunk max duration in seconds (for realtime listener)
    #[arg(long, default_value_t = 8.0)]
    audio_chunk_max_secs: f32,

    /// Path to a local model file
    #[arg(long)]
    model_path: Option<String>,

    /// Sleep / debounce duration in milliseconds
    #[arg(long, default_value_t = 250)]
    sleep_ms: u64,

    /// Latest file in directory is treated as user input (you can use this flag multiple times)
    #[arg(short = 'I')]
    input_final: Vec<PathBuf>,

    /// All files in directory are treated as user input (you can use this flag multiple times)
    #[arg(short = 'i')]
    input_cat: Vec<PathBuf>,

    /// Latest file in directory is treated as system messages (you can use this flag multiple times)
    #[arg(short = 'S')]
    system_final: Vec<PathBuf>,

    /// All files in directory are treated as system messages (you can use this flag multiple times)
    #[arg(short = 's')]
    system_cat: Vec<PathBuf>,

    /// Latest file in directory is treated as assistant messages (you can use this flag multiple times)
    /// New assistant messages do not trigger inference!
    /// Output flags are not automatically stored in context, so at least one assistant flag is required for the agent to be aware of it's own responses
    #[arg(short = 'A')]
    assistant_final: Vec<PathBuf>,

    /// All files in directory are treated as assistant messages (you can use this flag multiple times)
    #[arg(short = 'a')]
    assistant_cat: Vec<PathBuf>,

    /// Each assistant response is stored in a new file in this directory
    #[arg(short = 'O')]
    output_new: Option<PathBuf>,

    /// Each assistant response clears and overwrites this file
    #[arg(short = 'o')]
    output_overwrite: Option<PathBuf>,
}

#[derive(Debug, Clone)]
enum MessageContent {
    Text(String),
    Image(image::DynamicImage),
    Audio(Vec<u8>), // Raw audio bytes
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum ContentType {
    Text,
    Image,
    Audio,
    Video, // Will be split into Image+Audio
}

fn detect_content_type(path: &Path) -> ContentType {
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

fn to_mistral_role(role: &str) -> TextMessageRole {
    match role {
        "system" => TextMessageRole::System,
        "assistant" => TextMessageRole::Assistant,
        "tool" => TextMessageRole::Tool,
        _ => TextMessageRole::User,
    }
}

#[derive(Debug, Clone)]
struct FileMetadata {
    path: PathBuf,
    created: std::time::SystemTime,
    modified: std::time::SystemTime,
}

#[derive(Debug, Clone)]
struct FileMessage {
    role: String,
    content: MessageContent,
    metadata: FileMetadata,
    content_type: ContentType,
}

impl FileMessage {
    fn format_for_model(
        &self,
        target_slot: ModelSlot,
    ) -> Option<(
        String,
        Option<Vec<image::DynamicImage>>,
        Option<Vec<mistralrs::AudioInput>>,
    )> {
        let include_metadata = self.role == "user";

        // Route content to appropriate model slot
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
                // Images go to primary (vision) slot only
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
                // Audio goes to secondary (audio) slot only
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
async fn process_video_to_messages(
    video_path: &Path,
    metadata: &FileMetadata,
) -> Result<(Vec<FileMessage>, Vec<FileMessage>)> {
    // Create temp directory for frames
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
    let ffmpeg_frames = Command::new("ffmpeg")
        .args(&[
            "-i",
            video_path.to_str().unwrap(),
            "-vf",
            "fps=0.5", // 1 frame per 2 seconds
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
    let ffmpeg_audio = Command::new("ffmpeg")
        .args(&[
            "-i",
            video_path.to_str().unwrap(),
            "-vn", // no video
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

    // Load frames as image messages – correct async directory iteration
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

fn normalize_to_dir(path: &Path) -> Option<PathBuf> {
    if path.is_dir() {
        Some(path.to_path_buf())
    } else if path.is_file() {
        path.parent().map(|p| p.to_path_buf())
    } else {
        None
    }
}

async fn collect_viewer_dirs(args: &Args) -> Result<Vec<PathBuf>> {
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

async fn load_dir_messages(
    dir: &Path,
    role: &str,
    concat: bool,
    latest_n: usize,
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

    for (created, modified, path) in entries {
        let content_type = detect_content_type(&path);

        let content = match content_type {
            ContentType::Image => {
                let bytes = fs::read(&path).await?;
                let img = image::load_from_memory(&bytes)
                    .with_context(|| format!("Failed to load image: {}", path.display()))?;
                MessageContent::Image(img)
            }
            ContentType::Audio => {
                let bytes = fs::read(&path).await?;
                MessageContent::Audio(bytes)
            }
            ContentType::Video => MessageContent::Text(format!("[Video file: {}]", path.display())),
            ContentType::Text => MessageContent::Text(fs::read_to_string(&path).await?),
        };

        messages.push(FileMessage {
            role: role.to_string(),
            content,
            metadata: FileMetadata {
                path,
                created,
                modified,
            },
            content_type,
        });
    }

    Ok(messages)
}

fn format_metadata(metadata: &FileMetadata) -> String {
    format!(
        "[File: {}; Created: {:?}; Modified: {:?}]",
        metadata.path.display(),
        humantime::format_rfc3339_seconds(metadata.created),
        humantime::format_rfc3339_seconds(metadata.modified)
    )
}

fn latest_file(dir: &Path) -> Result<PathBuf> {
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

async fn concat_files(dir: &Path) -> Result<String> {
    let mut out = String::new();
    for entry in WalkDir::new(dir).into_iter().filter_map(|e| e.ok()) {
        if entry.file_type().is_file() {
            out.push_str(&fs::read_to_string(entry.path()).await?);
            out.push('\n');
        }
    }
    Ok(out)
}

async fn build_messages(
    model: &Arc<mistralrs::Model>,
    args: &Args,
) -> Result<(VisionMessages, Option<VisionMessages>)> {
    // Primary messages (vision + text)
    let mut primary_messages = VisionMessages::new();
    // Secondary messages (audio) - only populated if secondary model exists
    let mut secondary_messages: Option<VisionMessages> = None;

    // --- System messages go to primary only ---
    for dir in &args.system_final {
        let p = latest_file(dir)?;
        primary_messages =
            primary_messages.add_message(TextMessageRole::System, fs::read_to_string(p).await?);
    }

    for dir in &args.system_cat {
        primary_messages =
            primary_messages.add_message(TextMessageRole::System, concat_files(dir).await?);
    }

    // --- Tool system message ---
    if args.tools {
        if args.verbose {
            eprintln!("Tools enabled, adding tool instruction message.");
        }
        let tool_instructions = format!(
            "The following is an addendum to any prior instruction, if you have no prior instruction, you should follow any instruction provided to you by the user. You have access to command execution tools. To spawn a process, provide the binary and arguments: {open_exec}command arg1 arg2 ...{close_exec} (returns index). To kill: {open_kill}idx{close_kill}. To read output: {open_read}idx{close_read}. To write stdin: {open_writ}idx input text{close_writ}. Commands execute immediately and return results. You will need to perform multiple Commands execution tool calls to execute and then read the outputs of commands you executed. Do not repeat these instructions to the user, these tools only execute within the scope of parsing your responses.",
            open_exec = CMD_OPEN_EXEC,
            close_exec = CMD_CLOSE_EXEC,
            open_kill = CMD_OPEN_KILL,
            close_kill = CMD_CLOSE_KILL,
            open_read = CMD_OPEN_READ,
            close_read = CMD_CLOSE_READ,
            open_writ = CMD_OPEN_WRIT,
            close_writ = CMD_CLOSE_WRIT
        );
        primary_messages = primary_messages.add_message(TextMessageRole::System, tool_instructions);
    }

    // --- Collect and sort all messages ---
    let mut timeline: Vec<FileMessage> = Vec::new();
    let mut video_tasks = Vec::new();

    let sources = [
        (&args.input_final, "user", false),
        (&args.input_cat, "user", true),
        (&args.assistant_final, "assistant", false),
        (&args.assistant_cat, "assistant", true),
    ];

    for (paths, role, concat) in sources {
        for dir in paths {
            if args.verbose {
                eprintln!("adding messages from: {}", dir.display());
            }
            let msgs = load_dir_messages(dir, role, concat, args.latest_n).await?;

            // Handle videos specially
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
        eprintln!("Building {} messages.", timeline.len());
    }

    // --- Route messages to appropriate model slots ---
    // if we only have text messages and one kind of media, disable the other model.
    let mut maybe_switchable = (false, false);
    for msg in timeline {
        // Try primary slot
        if let Some((text, images, _)) = msg.format_for_model(ModelSlot::Primary) {
            if let Some(images) = images {
                primary_messages =
                    primary_messages.add_image_message(to_mistral_role(&msg.role), text, images);
                maybe_switchable = (true, maybe_switchable.1 || false);
            } else {
                primary_messages = primary_messages.add_message(to_mistral_role(&msg.role), text);
            }
        }

        // Try secondary slot (audio)
        if let Some((text, _, audio)) = msg.format_for_model(ModelSlot::Secondary) {
            if let Some(audio) = audio {
                let sec_msgs = secondary_messages.get_or_insert_with(VisionMessages::new);
                *sec_msgs = sec_msgs.clone().add_multimodal_message(
                    to_mistral_role(&msg.role),
                    &text,
                    vec![], // no images
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
        eprintln!("Messages built.");
    }

    match maybe_switchable {
        (true, true) => Ok((primary_messages, secondary_messages)),
        (false, true) => Ok((primary_messages, secondary_messages)),
        (true, false) => Ok((primary_messages, None)),
        (false, false) => Ok((primary_messages, None)),
    }
}

// ============================================================================
// COLLABORATIVE MODEL SYSTEM MESSAGES (hardcoded per-turn injection)
// ============================================================================

const PRIMARY_BEFORE_SECONDARY: &str = "You are a Vision-capable model running collaboratively alongside an Audio-capable model. You are responding BEFORE your collaborator, and they will continue the response to the user. Provide your visual observations completely, as your collaborator will build upon them.";
const PRIMARY_AFTER_SECONDARY: &str = "You are a Vision-capable model running collaboratively alongside an Audio-capable model. You are responding AFTER your collaborator, and they have already begun the response to the user. You may provide a brief preamble acknowledging their observations before adding your visual analysis.";
const SECONDARY_BEFORE_PRIMARY: &str = "You are an Audio-capable model running collaboratively alongside a Vision-capable model. You are responding BEFORE your collaborator, and they will continue the response to the user. Provide your auditory observations completely, as your collaborator will build upon them.";
const SECONDARY_AFTER_PRIMARY: &str = "You are an Audio-capable model running collaboratively alongside a Vision-capable model. You are responding AFTER your collaborator, and they have already begun the response to the user. You may provide a brief preamble acknowledging their observations before adding your audio analysis.";
struct RealtimeListener {
    pub chunk_rx: tokio::sync::mpsc::Receiver<Vec<u8>>,
    pub shutdown_tx: oneshot::Sender<()>,
    pub speech_detected_rx: broadcast::Receiver<()>,
}

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use flume;

fn create_wav_header(data_len: usize) -> [u8; 44] {
    const SAMPLE_RATE: u32 = 48000;
    const BITS_PER_SAMPLE: u16 = 32;
    const CHANNELS: u16 = 1;
    const BYTES_PER_SECOND: u32 = SAMPLE_RATE * CHANNELS as u32 * (BITS_PER_SAMPLE as u32 / 8);
    const BLOCK_ALIGN: u16 = CHANNELS * (BITS_PER_SAMPLE / 8);
    const AUDIO_FORMAT: u16 = 3;

    let mut header = [0u8; 44];

    // RIFF chunk
    header[0..4].copy_from_slice(b"RIFF");
    header[4..8].copy_from_slice(&(36 + data_len as u32).to_le_bytes());
    header[8..12].copy_from_slice(b"WAVE");

    // fmt subchunk
    header[12..16].copy_from_slice(b"fmt ");
    header[16..20].copy_from_slice(&(16u32).to_le_bytes()); // fmt chunk size (16)
    header[20..22].copy_from_slice(&AUDIO_FORMAT.to_le_bytes()); // 3 = float
    header[22..24].copy_from_slice(&CHANNELS.to_le_bytes());
    header[24..28].copy_from_slice(&SAMPLE_RATE.to_le_bytes());
    header[28..32].copy_from_slice(&BYTES_PER_SECOND.to_le_bytes());
    header[32..34].copy_from_slice(&BLOCK_ALIGN.to_le_bytes());
    header[34..36].copy_from_slice(&BITS_PER_SAMPLE.to_le_bytes());

    // data subchunk
    header[36..40].copy_from_slice(b"data");
    header[40..44].copy_from_slice(&(data_len as u32).to_le_bytes());

    header
}

async fn spawn_realtime_listener(
    source: &str,
    args: &Args,
    model: Arc<mistralrs::Model>,
) -> Result<RealtimeListener> {
    // Only support pipewire (default PulseAudio input) for simplicity.
    if source != "pipewire" {
        anyhow::bail!("Only 'pipewire' source is supported");
    }

    let (chunk_tx, chunk_rx) = tokio::sync::mpsc::channel::<Vec<u8>>(32);
    let (shutdown_tx, mut shutdown_rx) = oneshot::channel();
    let (speech_detected_tx, speech_detected_rx) = broadcast::channel(16);

    let min_duration = Duration::from_secs_f32(args.audio_chunk_min_secs);
    let max_duration = Duration::from_secs_f32(args.audio_chunk_max_secs);

    let verbose = args.verbose;
    let model_clone = model.clone();
    let model_id = args.secondary_model.clone();
    tokio::spawn(async move {
        let (audio_tx, audio_rx) = flume::bounded::<Vec<u8>>(128);

        // Channel to signal the blocking audio thread to shut down.
        let (shutdown_tx_blocking, shutdown_rx_blocking) = std::sync::mpsc::channel();

        // Spawn a blocking thread for cpal audio capture.
        let audio_handle = tokio::task::spawn_blocking(move || {
            // Get default host and input device.
            let host = cpal::default_host();
            let device = match host.default_input_device() {
                Some(dev) => dev,
                None => {
                    eprintln!("No default input device found");
                    return;
                }
            };

            let config = {
                let mut best = None;
                for cfg in device.supported_input_configs().unwrap() {
                    if cfg.channels() == 1 && cfg.sample_format() == cpal::SampleFormat::F32 {
                        let rate = 48000;
                        if cfg.min_sample_rate() <= rate && rate <= cfg.max_sample_rate() {
                            best = Some(cfg.with_sample_rate(rate));
                            break;
                        }
                    }
                }
                match best {
                    Some(cfg) => cfg,
                    None => {
                        eprintln!("No supported 48kHz mono F32 input config");
                        return;
                    }
                }
            };

            // Build the input stream.
            let tx = audio_tx.clone();
            let stream = device.build_input_stream(
                config.into(),
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    let bytes: &[u8] = bytemuck::cast_slice(data); // f32 -> [u8; 4] per sample
                    let _ = tx.try_send(bytes.to_vec());
                },
                |err| eprintln!("Audio stream error: {}", err),
                None,
            );

            let stream = match stream {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("Failed to build input stream: {}", e);
                    return;
                }
            };

            // Start the stream.
            if let Err(e) = stream.play() {
                eprintln!("Failed to start audio stream: {}", e);
                return;
            }

            // Block until shutdown signal is received.
            let _ = shutdown_rx_blocking.recv();
            // Stream is dropped here, stopping capture.
        });

        // Main async loop: receive audio bytes from flume and perform chunking.
        let mut chunk_start = std::time::Instant::now();
        let mut current_chunk_duration = rand::rng().random_range(min_duration..=max_duration);
        let mut accumulated_buffer = Vec::new();

        loop {
            tokio::select! {
                _ = &mut shutdown_rx => {
                    // Signal the blocking thread to stop.
                    let _ = shutdown_tx_blocking.send(());
                    break;
                }
                result = audio_rx.recv_async() => {
                    match result {
                        Ok(bytes) => {
                            // Add new bytes to the current chunk.
                            accumulated_buffer.extend_from_slice(&bytes);

                            // Check if it's time to send a chunk.
                            if chunk_start.elapsed() >= current_chunk_duration {
                                if !accumulated_buffer.is_empty() {
                                    let model_clone = model_clone.clone();
                                    let speech_detected_tx = speech_detected_tx.clone();
                                    let chunk = accumulated_buffer.clone();
                                    let tx_clone = chunk_tx.clone();
                                    let model_id = model_id.clone();
                                    tokio::spawn(async move {
                                        let mut wav = create_wav_header(chunk.len()).to_vec();
                                        wav.append(&mut chunk.clone());
                                        match detect_speech(&wav, &model_clone, &model_id).await {
                                            Ok(true) => {
                                                let _ = speech_detected_tx.send(());
                                                let _ = tx_clone.send(wav.to_vec()).await;
                                            }
                                            Ok(false) => {}
                                            Err(e) => {
                                                eprintln!("Speech detection error: {}", e);
                                            }
                                        }
                                    });

                                    accumulated_buffer.clear();
                                    chunk_start = std::time::Instant::now();
                                    current_chunk_duration = rand::rng()
                                        .random_range(min_duration..=max_duration);
                                }
                            }
                        }
                        Err(flume::RecvError::Disconnected) => {
                            // Audio thread exited; end the loop.
                            break;
                        }
                    }
                }
            }
        }

        // Ensure the audio thread is stopped.
        let _ = shutdown_tx_blocking.send(());
        let _ = audio_handle.await;
    });

    Ok(RealtimeListener {
        chunk_rx,
        shutdown_tx,
        speech_detected_rx,
    })
}

async fn detect_speech(
    audio_chunk: &[u8],
    model: &mistralrs::Model,
    audio_model_id: &str,
) -> Result<bool> {
    if !model
        .is_model_loaded(audio_model_id)
        .context("Failed to check secondary model load state")?
    {
        model
            .reload_model(audio_model_id)
            .await
            .context("Failed to reload secondary model")?;
    }

    let audio = mistralrs::AudioInput::from_bytes(audio_chunk)
        .context("Failed to create AudioInput from bytes")?;

    let messages = mistralrs::VisionMessages::new().add_multimodal_message(
        TextMessageRole::User,
        "Respond with only 'true' if this audio contains intelligible speech, or 'false' if it does not. Be concise.",
        vec![],
        vec![audio],
    );

    let response = model
        .send_chat_request_with_model(messages, Some(audio_model_id))
        .await
        .map_err(|e| {
            eprintln!("Full error from mistralrs: {:?}", e);
            anyhow::anyhow!("Speech detection request failed: {}", e)
        })?;
    let content = response.choices[0]
        .message
        .content
        .as_ref()
        .map(|s| s.trim().to_lowercase())
        .unwrap_or_default();

    Ok(content.contains("true"))
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum ModelSlot {
    Primary,
    Secondary,
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct InterruptKind: u8 {
        const FS                          = 0b0001;
        const AUDIO                       = 0b0010;
        const PARALLEL_INFERENCE_COMPLETION  = 0b0100;
        const SYNTHESIS_INFERENCE_INITIATION = 0b1000;
    }
}

#[derive(Clone)]
struct ParallelInferenceParams {
    messages: VisionMessages,
    system_addendum: String,
    interrupted_by: InterruptKind,
    context_inputs: Vec<ParallelInferenceParams>,
    streaming: bool,
    model_slot: ModelSlot,
}

impl ParallelInferenceParams {
    fn synthesis_params(
        messages: VisionMessages,
        active_modes: Vec<ParallelInferenceParams>,
        model_slot: ModelSlot,
    ) -> Self {
        ParallelInferenceParams {
            messages,
            system_addendum: "You are an agent tasked with synthesizing the responses of specialist agents into a single synthesized response, making connections between responses, and respecting prior instructions about response formatting, if any exist. You will receive a set of responses that should be synthesized - try to incorporate all information provided by each response provided to you, discarding statements by specialists that indicate their lack of capability (ie, if a vision-enabled agent indicates it is incapable of processing audio input, ignore that statement for purposes of your synthesis, but incorporate all of the unique information they provide about what they see). The user does not see the responses of the specialists, and they are not persisted in your context, so if you do not incorporate information they provide, that information will be lost.".into(),
            interrupted_by: InterruptKind::all(),
            context_inputs: active_modes,
            streaming: true,
            model_slot,
        }
    }

    fn vision_params(messages: VisionMessages) -> Self {
        ParallelInferenceParams {
            messages,
            system_addendum: "You are a vision-enabled agent tasked with analyzing the visual content of images and videos. You share a historical context with other agents, and your responses are synthesized with theirs, but your task is to use your own unique insights given the visual content only. Your response will be synthesized with other agents' responses, so be concise and to the point.".into(),
            interrupted_by: InterruptKind::FS | InterruptKind::PARALLEL_INFERENCE_COMPLETION,
            context_inputs: Vec::new(),
            streaming: false,
            model_slot: ModelSlot::Primary,
        }
    }

    fn audio_params(messages: VisionMessages) -> Self {
        ParallelInferenceParams {
            messages,
            system_addendum: "You are an audio-enabled agent tasked with analyzing audio content. You share a historical context with other agents, and your responses are synthesized with theirs, but your task is to use your own unique insights given the audio content only. Your response will be synthesized with other agents' responses, so be concise and to the point.".into(),
            interrupted_by: InterruptKind::AUDIO | InterruptKind::PARALLEL_INFERENCE_COMPLETION,
            context_inputs: Vec::new(),
            streaming: false,
            model_slot: ModelSlot::Secondary,
        }
    }

    fn full_pipeline(
        vision_messages: VisionMessages,
        audio_messages: VisionMessages,
        primary: ModelSlot,
    ) -> Self {
        let input_context = vec![
            ParallelInferenceParams::vision_params(vision_messages.clone()),
            ParallelInferenceParams::audio_params(audio_messages.clone()),
        ];
        match primary {
            ModelSlot::Primary => ParallelInferenceParams::synthesis_params(
                vision_messages,
                input_context,
                ModelSlot::Primary,
            ),
            ModelSlot::Secondary => ParallelInferenceParams::synthesis_params(
                audio_messages,
                input_context,
                ModelSlot::Secondary,
            ),
        }
    }
}

// --- Coroutine infrastructure ---

enum CoroutineResponse {
    Complete(String),
    Interrupted(String),
    Error(anyhow::Error),
}

/// Runs a single inference coroutine for `params`, streaming tokens to `token_tx` if provided.
/// Cancels early and returns `Interrupted` when `cancel_rx` becomes true.
async fn run_inference_coroutine(
    model: Arc<mistralrs::Model>,
    params: ParallelInferenceParams,
    mut cancel_rx: watch::Receiver<bool>,
    token_tx: Option<UnboundedSender<String>>,
) -> CoroutineResponse {
    let stream_result = match params.model_slot {
        ModelSlot::Primary => model.stream_chat_request(params.messages).await,
        ModelSlot::Secondary => {
            model
                .stream_chat_request_with_model(params.messages, Some("secondary"))
                .await
        }
    };

    let mut stream = match stream_result {
        Ok(s) => s,
        Err(e) => return CoroutineResponse::Error(e.into()),
    };

    let mut buffer = String::new();
    loop {
        tokio::select! {
            chunk_opt = stream.next() => {
                match chunk_opt {
                    Some(chunk) => {
                        match extract_content(chunk) {
                            Ok(content) if !content.is_empty() => {
                                if let Some(ref tx) = token_tx {
                                    let _ = tx.send(content.clone());
                                }
                                buffer.push_str(&content);
                            }
                            Ok(_) => {}
                            Err(e) => return CoroutineResponse::Error(e),
                        }
                    }
                    None => return CoroutineResponse::Complete(buffer),
                }
            }
            result = cancel_rx.changed() => {
                if result.is_ok() && *cancel_rx.borrow() {
                    return CoroutineResponse::Interrupted(buffer);
                }
            }
        }
    }
}

/// Runs all `params_list` entries in parallel via FuturesUnordered.
/// The first coroutine to complete cancels all others (they return Interrupted).
/// Errors are retried up to `max_retries` times before being replaced with an error string.
type CoroutineFut = std::pin::Pin<
    Box<
        dyn std::future::Future<Output = (usize, ParallelInferenceParams, CoroutineResponse)>
            + Send,
    >,
>;

async fn run_parallel_stage(
    model: &Arc<mistralrs::Model>,
    params_list: Vec<ParallelInferenceParams>,
    max_retries: u32,
) -> Vec<(usize, String)> {
    let (cancel_tx, cancel_rx) = watch::channel(false);
    let mut pool: FuturesUnordered<CoroutineFut> = FuturesUnordered::new();
    let mut retry_counts: std::collections::HashMap<usize, u32> = std::collections::HashMap::new();

    let total = params_list.len();
    for (idx, params) in params_list.into_iter().enumerate() {
        let model = model.clone();
        let rx = cancel_rx.clone();
        pool.push(Box::pin(async move {
            let resp = run_inference_coroutine(model, params.clone(), rx, None).await;
            (idx, params, resp)
        }));
    }

    let mut results: Vec<Option<String>> = vec![None; total];

    while let Some((idx, params, resp)) = pool.next().await {
        match resp {
            CoroutineResponse::Complete(s) => {
                let _ = cancel_tx.send(true);
                results[idx] = Some(s);
            }
            CoroutineResponse::Interrupted(s) => {
                results[idx] = Some(s);
            }
            CoroutineResponse::Error(e) => {
                let retries = retry_counts.entry(idx).or_insert(0);
                if *retries < max_retries {
                    *retries += 1;
                    eprintln!("Coroutine {} error (retry {}): {:?}", idx, retries, e);
                    let model = model.clone();
                    let rx = cancel_rx.clone();
                    pool.push(Box::pin(async move {
                        let resp = run_inference_coroutine(model, params.clone(), rx, None).await;
                        (idx, params, resp)
                    }));
                } else {
                    eprintln!(
                        "Coroutine {} failed after {} retries: {:?}",
                        idx, max_retries, e
                    );
                    results[idx] = Some(format!("[Specialist error: {}]", e));
                }
            }
        }
    }

    results
        .into_iter()
        .enumerate()
        .map(|(i, s)| (i, s.unwrap_or_default()))
        .collect()
}

/// The outcome of a single call to `run_streaming_loop`.
enum StreamOutcome {
    /// Inference finished cleanly; contains the full accumulated response.
    Complete(String),
    /// A tool command was detected and executed; contains the response so far
    /// (including the command output). The caller should append this to
    /// `output_buffer` and restart inference.
    CommandExecuted(String),
    /// Inference was cancelled because a new audio chunk arrived.
    AudioInterrupted {
        response: String,
        audio: Option<Vec<u8>>,
    },
    /// Inference was cancelled by a filesystem event (or the coroutine returned
    /// `Interrupted` for another reason). If `event` is `Some`, the caller
    /// should call `handle_interrupt` before restarting.
    FsInterrupted {
        response: String,
        event: Option<notify::Event>,
    },
}

/// Drives a single streaming inference to completion (or interruption).
///
/// Handles token-by-token streaming, real-time file writes, command parsing /
/// execution, audio interrupts, and filesystem interrupts — returning a
/// `StreamOutcome` that tells the caller what happened and what was accumulated.
async fn run_streaming_loop(
    model: Arc<mistralrs::Model>,
    params: ParallelInferenceParams,
    realtime_file: &mut Option<File>,
    stream_realtime: bool,
    speech_detected_rx: &mut Option<broadcast::Receiver<()>>,
    interrupt_rx: &mut Option<broadcast::Receiver<Result<notify::Event, Arc<notify::Error>>>>,
    latest_audio_rx: &watch::Receiver<Option<Vec<u8>>>,
    subprocesses: &mut Vec<CommandIO>,
    args: &Args,
) -> Result<StreamOutcome> {
    let (cancel_tx, cancel_rx) = watch::channel(false);
    let (token_tx, mut token_rx) = unbounded_channel::<String>();
    let mut cmd_parser = CommandParser::new();
    let mut tool_context = String::new();
    let mut response = String::new();
    let mut last_modify_interrupt: Option<std::time::Instant> = None;
    let mut inference_fut = std::pin::pin!(run_inference_coroutine(
        model,
        params,
        cancel_rx,
        Some(token_tx),
    ));

    loop {
        tokio::select! {
            result = &mut inference_fut => {
                // Drain any tokens that arrived just before the future completed.
                while let Ok(tok) = token_rx.try_recv() {
                    let (out, cmd, _) = cmd_parser.process(&tok);
                    response.push_str(&out);
                    if stream_realtime {
                        if let Some(ref mut f) = *realtime_file {
                            let _ = f.write_all(out.as_bytes()).await;
                            let _ = f.flush().await;
                        }
                    }
                    if let Some(cmd) = cmd {
                        let res = execute_command(cmd, subprocesses, &mut tool_context, args).await;
                        response.push_str(&res);
                        if stream_realtime {
                            if let Some(ref mut f) = *realtime_file {
                                let _ = f.write_all(res.as_bytes()).await;
                                let _ = f.flush().await;
                            }
                        }
                        return Ok(StreamOutcome::CommandExecuted(response));
                    }
                }
                // Flush the parser's sliding-window buffer — this is the fix for the
                // "final token not written" bug that breaks structured-output use cases.
                let tail = cmd_parser.flush();
                if !tail.is_empty() {
                    response.push_str(&tail);
                    if stream_realtime {
                        if let Some(ref mut f) = *realtime_file {
                            let _ = f.write_all(tail.as_bytes()).await;
                            let _ = f.flush().await;
                        }
                    }
                }
                return match result {
                    CoroutineResponse::Complete(_) => Ok(StreamOutcome::Complete(response)),
                    CoroutineResponse::Interrupted(_) => Ok(StreamOutcome::FsInterrupted { response, event: None }),
                    CoroutineResponse::Error(e) => Err(e),
                };
            }
            Some(tok) = token_rx.recv() => {
                let (out, cmd, _) = cmd_parser.process(&tok);
                response.push_str(&out);
                if stream_realtime {
                    if let Some(ref mut f) = *realtime_file {
                        let _ = f.write_all(out.as_bytes()).await;
                        let _ = f.flush().await;
                    }
                }
                if let Some(cmd) = cmd {
                    let res = execute_command(cmd, subprocesses, &mut tool_context, args).await;
                    response.push_str(&res);
                    if stream_realtime {
                        if let Some(ref mut f) = *realtime_file {
                            let _ = f.write_all(res.as_bytes()).await;
                            let _ = f.flush().await;
                        }
                    }
                    let _ = cancel_tx.send(true);
                    return Ok(StreamOutcome::CommandExecuted(response));
                }
            }
            _ = async {
                if let Some(ref mut rx) = *speech_detected_rx {
                    rx.recv().await.ok()
                } else {
                    std::future::pending::<Option<()>>().await
                }
            } => {
                let _ = cancel_tx.send(true);
                let audio = latest_audio_rx.borrow().clone();
                return Ok(StreamOutcome::AudioInterrupted { response, audio });
            }
            event = async {
                if let Some(ref mut rx) = *interrupt_rx {
                    rx.recv().await.ok()
                } else {
                    std::future::pending::<Option<Result<notify::Event, Arc<notify::Error>>>>().await
                }
            } => {
                if let Some(Ok(ev)) = event {
                    if is_interrupt_event(&ev, &mut last_modify_interrupt, &args) {
                        let _ = cancel_tx.send(true);
                        return Ok(StreamOutcome::FsInterrupted { response, event: Some(ev) });
                    }
                }
            }

        }
    }
}

async fn run_once(
    model: &Arc<mistralrs::Model>,
    args: &Args,
    mut interrupt_rx: Option<broadcast::Receiver<Result<notify::Event, Arc<notify::Error>>>>,
    audio_channels: Option<(broadcast::Receiver<()>, watch::Receiver<Option<Vec<u8>>>)>,
    pending_audio: Vec<Vec<u8>>,
) -> Result<()> {
    if args.verbose {
        eprintln!("Building messages..");
    }

    let timestamp = chrono::Utc::now().timestamp_millis();
    let new_file_path = args
        .output_new
        .as_ref()
        .map(|dir| dir.join(format!("out-{}.txt", timestamp)));

    if let Some(overwrite_path) = &args.output_overwrite {
        if args.verbose {
            eprintln!(
                "Pre-clearing file for overwrite: {}",
                overwrite_path.display()
            );
        }
        if let Some(parent) = overwrite_path.parent() {
            fs::create_dir_all(parent).await?;
        }
        File::create(overwrite_path).await?;
    }
    if let Some(ref path) = new_file_path {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).await?;
        }
    }

    let stream_realtime = args.stream_realtime;
    let mut current_file_path: Option<std::path::PathBuf> = new_file_path.clone();

    let mut realtime_file: Option<File> = if stream_realtime {
        if let Some(ref path) = current_file_path {
            if args.verbose {
                eprintln!("File created: {}", path.display());
            }
            Some(File::create(path).await?)
        } else if let Some(ref path) = args.output_overwrite {
            Some(File::options().write(true).open(path).await?)
        } else {
            None
        }
    } else {
        None
    };

    // Use externally-owned audio channels when supplied (leader path).
    // handle_request (worker path) passes None for both.
    let (mut speech_detected_rx, latest_audio_rx): (
        Option<broadcast::Receiver<()>>,
        watch::Receiver<Option<Vec<u8>>>,
    ) = match audio_channels {
        Some((sdr, lar)) => (Some(sdr), lar),
        None => {
            let (_, rx) = watch::channel::<Option<Vec<u8>>>(None);
            (None, rx)
        }
    };

    let (_primary_messages, secondary_messages) = build_messages(model, args).await?;
    let has_secondary = secondary_messages.is_some() && args.secondary_model != "none";

    let mut output_buffer = String::new();
    let mut restart_count = 0;
    let mut audio_interrupt_pending = !pending_audio.is_empty();
    let mut pending_audio_queue: Vec<Vec<u8>> = pending_audio;
    let mut subprocesses: Vec<CommandIO> = Vec::new();

    'restart_loop: loop {
        restart_count += 1;
        if args.verbose && restart_count > 1 {
            eprintln!("Restart iteration {}", restart_count);
        }

        let audio_first = std::mem::take(&mut audio_interrupt_pending);

        // Rebuild messages on every restart so new file state is picked up
        let (current_primary, current_secondary) = build_messages(model, args).await?;
        let has_sec = current_secondary.is_some() && args.secondary_model != "none";

        // Decide which slot drives the synthesis (audio-first => secondary is primary)
        let primary_slot = if audio_first && has_sec {
            ModelSlot::Secondary
        } else {
            ModelSlot::Primary
        };

        // Build the synthesis params (parallel pipeline or single-model)
        let synth_params = if has_sec {
            let mut audio_msgs = current_secondary.clone().unwrap();
            for audio in &pending_audio_queue {
                audio_msgs = audio_msgs.add_multimodal_message(
                    TextMessageRole::User,
                    "",
                    vec![],
                    vec![
                        mistralrs::AudioInput::from_bytes(audio)
                            .context("Failed to create AudioInput from bytes")?,
                    ],
                );
            }
            pending_audio_queue.clear();
            ParallelInferenceParams::full_pipeline(
                current_primary.clone(),
                audio_msgs,
                primary_slot,
            )
        } else {
            ParallelInferenceParams {
                messages: current_primary.clone(),
                system_addendum: String::new(),
                interrupted_by: InterruptKind::all(),
                context_inputs: Vec::new(),
                streaming: true,
                model_slot: ModelSlot::Primary,
            }
        };

        // --- Parallel specialist stage ---
        if !synth_params.context_inputs.is_empty() {
            if args.verbose {
                eprintln!(
                    "Running {} parallel specialists...",
                    synth_params.context_inputs.len()
                );
            }
            let specialist_results =
                run_parallel_stage(model, synth_params.context_inputs.clone(), 2).await;
            if args.verbose {
                eprintln!("Parallel stage complete, building synthesis messages.");
            }

            // Inject specialist responses into synthesis messages
            let mut synth_msgs = synth_params.messages.clone();
            if !synth_params.system_addendum.is_empty() {
                synth_msgs = synth_msgs.add_message(
                    TextMessageRole::System,
                    synth_params.system_addendum.clone(),
                );
            }
            for (idx, text) in &specialist_results {
                let label = match synth_params.context_inputs[*idx].model_slot {
                    ModelSlot::Primary => "VISION SPECIALIST",
                    ModelSlot::Secondary => "AUDIO SPECIALIST",
                };
                synth_msgs = synth_msgs.add_message(
                    TextMessageRole::System,
                    format!("[SPECIALIST RESPONSE ({label}): {text}]"),
                );
            }

            let streaming_params = ParallelInferenceParams {
                messages: synth_msgs,
                system_addendum: String::new(),
                interrupted_by: InterruptKind::all(),
                context_inputs: Vec::new(),
                streaming: true,
                model_slot: synth_params.model_slot,
            };

            match run_streaming_loop(
                model.clone(),
                streaming_params,
                &mut realtime_file,
                stream_realtime,
                &mut speech_detected_rx,
                &mut interrupt_rx,
                &latest_audio_rx,
                &mut subprocesses,
                args,
            )
            .await
            .context("Synthesis model failed")?
            {
                StreamOutcome::Complete(resp) => {
                    if args.verbose {
                        eprintln!("Synthesis complete.");
                    }
                    output_buffer.push_str(&resp);
                    break 'restart_loop;
                }
                StreamOutcome::CommandExecuted(resp) => {
                    output_buffer.push_str(&resp);
                    continue 'restart_loop;
                }
                StreamOutcome::AudioInterrupted { response, audio } => {
                    if args.verbose {
                        eprintln!("Audio interrupt during synthesis.");
                    }
                    output_buffer.push_str(&response);
                    audio_interrupt_pending = true;
                    if let Some(audio) = audio {
                        pending_audio_queue.push(audio);
                    }
                }
                StreamOutcome::FsInterrupted { response, event } => {
                    if args.verbose {
                        eprintln!("FS interrupt during synthesis.");
                    }
                    output_buffer.push_str(&response);
                    if let Some(ev) = event {
                        handle_interrupt(ev, &mut current_file_path, &mut realtime_file, args)
                            .await?;
                    }
                }
            }
        } else {
            // Single-model path: no parallel stage, stream synthesis directly
            let mut msgs = synth_params.messages.clone();
            if !synth_params.system_addendum.is_empty() {
                msgs = msgs.add_message(
                    TextMessageRole::System,
                    synth_params.system_addendum.clone(),
                );
            }
            if !output_buffer.is_empty() {
                msgs = msgs.add_message(TextMessageRole::Assistant, format!("{}", output_buffer));
            }

            let streaming_params = ParallelInferenceParams {
                messages: msgs,
                system_addendum: String::new(),
                interrupted_by: InterruptKind::all(),
                context_inputs: Vec::new(),
                streaming: true,
                model_slot: ModelSlot::Primary,
            };

            match run_streaming_loop(
                model.clone(),
                streaming_params,
                &mut realtime_file,
                stream_realtime,
                &mut speech_detected_rx,
                &mut interrupt_rx,
                &latest_audio_rx,
                &mut subprocesses,
                args,
            )
            .await
            .context("Primary model failed")?
            {
                StreamOutcome::Complete(resp) => {
                    if args.verbose {
                        eprintln!("Primary response complete.");
                    }
                    output_buffer.push_str(&resp);
                    break 'restart_loop;
                }
                StreamOutcome::CommandExecuted(resp) => {
                    output_buffer.push_str(&resp);
                    continue 'restart_loop;
                }
                StreamOutcome::AudioInterrupted { response, audio } => {
                    if args.verbose {
                        eprintln!("Audio interrupt.");
                    }
                    output_buffer.push_str(&response);
                    audio_interrupt_pending = true;
                    if let Some(audio) = audio {
                        pending_audio_queue.push(audio);
                    }
                }
                StreamOutcome::FsInterrupted { response, event } => {
                    if args.verbose {
                        eprintln!("FS interrupt.");
                    }
                    output_buffer.push_str(&response);
                    if let Some(ev) = event {
                        handle_interrupt(ev, &mut current_file_path, &mut realtime_file, args)
                            .await?;
                    }
                }
            }
        }
    } // end 'restart_loop

    // Cleanup — listener is now owned by main, nothing to shut down here.

    // Explicitly close the realtime file handle so the OS flushes and fires
    // IN_CLOSE_WRITE before we return — downstream watchers rely on this event.
    if let Some(mut f) = realtime_file.take() {
        let _ = f.shutdown().await;
    }
    drop(subprocesses);

    if !stream_realtime {
        if let Some(ref path) = new_file_path {
            if args.verbose {
                eprintln!("Saving output to: {}", path.display());
            }
            fs::write(path, &output_buffer).await?;
        }
        if let Some(ref path) = args.output_overwrite {
            if args.verbose {
                eprintln!("Saving output (overwrite): {}", path.display());
            }
            fs::write(path, &output_buffer).await?;
        }
    } else if args.verbose {
        if let Some(ref path) = current_file_path {
            eprintln!("Realtime complete: {}", path.display());
        }
    }

    Ok(())
}

async fn execute_command(
    cmd: CommandType,
    subprocesses: &mut Vec<CommandIO>,
    tool_context: &mut String,
    daemon_args: &Args,
) -> String {
    if !daemon_args.tools {
        return "[TOOL USE FAILED: TOOLS DISABLED]".to_string();
    };
    match cmd {
        CommandType::Exec(command_str) => {
            let parts: Vec<&str> = command_str.split_whitespace().collect();
            if parts.is_empty() {
                return "[EXEC failed: empty command]\n".to_string();
            }

            let program = parts[0];
            let args = &parts[1..];

            match spawn_command_io(program, args.iter().map(|s| *s)).await {
                Ok(cmd_io) => {
                    let idx = subprocesses.len();
                    subprocesses.push(cmd_io);
                    if daemon_args.verbose {
                        eprintln!("[EXEC {}: spawned '{}']", idx, command_str);
                    }
                    format!("[EXEC {}: spawned '{}']\n", idx, command_str)
                }
                Err(e) => {
                    if daemon_args.verbose {
                        eprintln!("[EXEC failed: {}]", e);
                    }
                    format!("[EXEC failed: {}]\n", e)
                }
            }
        }

        CommandType::Kill(idx) => {
            if idx < subprocesses.len() {
                // Take ownership to drop/kill
                let mut cmd_io = std::mem::replace(
                    &mut subprocesses[idx],
                    // Placeholder - will be removed below
                    CommandIO {
                        stdin_tx: tokio::sync::mpsc::channel(1).0,
                        stdout_rx: tokio::sync::mpsc::channel(1).1,
                        stderr_rx: tokio::sync::mpsc::channel(1).1,
                        kill_tx: None,
                        exited_rx: None,
                    },
                );
                cmd_io.kill();
                // Remove the placeholder
                subprocesses.remove(idx);
                if daemon_args.verbose {
                    eprintln!("[KILL {}: terminated]", idx);
                }
                format!("[KILL {}: terminated]\n", idx)
            } else {
                format!("[KILL {}: invalid index]\n", idx)
            }
        }

        CommandType::Read(idx) => {
            if idx >= subprocesses.len() {
                return format!("[READ {}: invalid index]\n", idx);
            }

            let cmd_io = &mut subprocesses[idx];
            let mut output = String::new();

            // Drain available stdout
            while let Ok(chunk) = cmd_io.stdout_rx.try_recv() {
                output.push_str(&String::from_utf8_lossy(&chunk));
            }
            // Drain available stderr
            while let Ok(chunk) = cmd_io.stderr_rx.try_recv() {
                output.push_str(&String::from_utf8_lossy(&chunk));
            }

            if output.is_empty() {
                if daemon_args.verbose {
                    eprintln!("[READ {}: (no new output)]", idx);
                }
                format!("[READ {}: (no new output)]\n", idx)
            } else {
                // Add to tool_context for inclusion in messages
                let formatted = format!(
                    "=== Command {} Output ===\n{}\n=== End Output ===\n",
                    idx, output
                );
                tool_context.push_str(&formatted);
                if daemon_args.verbose {
                    eprintln!("[READ {}: {} bytes]", idx, output.len());
                }
                format!("[READ {}: {} bytes captured]\n", idx, output.len())
            }
        }

        CommandType::Writ(idx, input) => {
            if idx >= subprocesses.len() {
                return format!("[WRIT {}: invalid index]\n", idx);
            }

            let cmd_io = &subprocesses[idx];
            match cmd_io.stdin_tx.send(input.clone().into_bytes()).await {
                Ok(_) => {
                    if daemon_args.verbose {
                        eprintln!("[WRIT {}: {} bytes sent]", idx, input.len());
                    }
                    format!("[WRIT {}: {} bytes sent]\n", idx, input.len())
                }
                Err(_) => {
                    format!("[WRIT {}: failed - channel closed]\n", idx)
                }
            }
        }
    }
}

fn is_interrupt_event(
    event: &notify::Event,
    last_modify: &mut Option<std::time::Instant>,
    args: &Args,
) -> bool {
    let watching: Vec<&PathBuf> = args
        .input_final
        .iter()
        .chain(args.input_cat.iter())
        .chain(args.system_final.iter())
        .chain(args.system_cat.iter())
        .collect();
    if !event.paths.iter().any(|x| watching.contains(&x)) {
        return false;
    }
    match &event.kind {
        notify::EventKind::Create(_) => true,
        notify::EventKind::Access(notify::event::AccessKind::Close(
            notify::event::AccessMode::Write | notify::event::AccessMode::Any,
        )) => true,
        notify::EventKind::Modify(_) => {
            let now = std::time::Instant::now();
            let elapsed = last_modify
                .map(|t| now.duration_since(t).as_millis() as u64)
                .unwrap_or(u64::MAX);
            if elapsed >= args.sleep_ms {
                *last_modify = Some(now);
                true
            } else {
                false
            }
        }
        _ => false,
    }
}

async fn handle_interrupt(
    event: notify::Event,
    current_file_path: &mut Option<PathBuf>,
    realtime_file: &mut Option<File>,
    args: &Args,
) -> Result<()> {
    match event.kind {
        notify::EventKind::Create(_)
        | notify::EventKind::Access(notify::event::AccessKind::Close(
            notify::event::AccessMode::Write | notify::event::AccessMode::Any,
        )) => {
            if args.output_new.is_some() {
                let new_timestamp = chrono::Utc::now().timestamp_millis();
                let new_path = args
                    .output_new
                    .as_ref()
                    .unwrap()
                    .join(format!("out-{}.txt", new_timestamp));

                if args.verbose {
                    eprintln!("Creating new file: {}", new_path.display());
                }

                if let Some(parent) = new_path.parent() {
                    let _ = fs::create_dir_all(parent).await;
                }

                *current_file_path = Some(new_path.clone());
                *realtime_file = Some(File::create(&new_path).await?);
            } else if args.output_overwrite.is_some() {
                if let Some(ref path) = args.output_overwrite {
                    *realtime_file = Some(File::create(path).await?);
                }
            }
        }
        notify::EventKind::Modify(_) => {
            if args.verbose {
                eprintln!("Wiping current output and restarting...");
            }

            if args.stream_realtime {
                if args.output_new.is_some() {
                    if let Some(ref path) = *current_file_path {
                        *realtime_file = Some(File::create(path).await?);
                    }
                } else if let Some(ref path) = args.output_overwrite {
                    *realtime_file = Some(File::create(path).await?);
                }
            }
        }
        _ => {}
    }
    Ok(())
}

fn extract_content(chunk: mistralrs::Response) -> anyhow::Result<String> {
    use mistralrs::{ChatCompletionChunkResponse, ChunkChoice, Delta, ResponseOk};

    match chunk.as_result() {
        Ok(ResponseOk::Chunk(ChatCompletionChunkResponse { choices, .. })) => {
            if let Some(ChunkChoice {
                delta:
                    Delta {
                        content: Some(text),
                        ..
                    },
                ..
            }) = choices.first()
            {
                Ok(text.clone())
            } else {
                Ok(String::new()) // chunk with no content (e.g., finish reason)
            }
        }
        Ok(other) => Ok(format!("{:?}", other)),
        Err(e) => {
            // Convert error to anyhow
            Err(anyhow::anyhow!("{}", e))
        }
    }
}

fn has_any_input(args: &Args) -> bool {
    !args.input_final.is_empty()
        || !args.input_cat.is_empty()
        || !args.system_final.is_empty()
        || !args.system_cat.is_empty()
        || !args.assistant_final.is_empty()
        || !args.assistant_cat.is_empty()
}

fn has_any_output(args: &Args) -> bool {
    args.output_new.is_some() || args.output_overwrite.is_some()
}

fn pipe_path(pid: u32) -> PathBuf {
    PathBuf::from(PIPE_DIR).join(format!("{}{}", PIPE_PREFIX, pid))
}

async fn find_oldest_pipe(my_pid: u32) -> Option<PathBuf> {
    let mut dir = fs::read_dir(PIPE_DIR).await.ok()?;
    let mut oldest: Option<(u32, PathBuf)> = None;

    while let Ok(Some(entry)) = dir.next_entry().await {
        let name = entry.file_name().to_str()?.to_owned();
        if let Some(pid_str) = name.strip_prefix(PIPE_PREFIX) {
            let pid = pid_str.parse::<u32>().ok()?;
            if pid != my_pid && is_process_alive(pid) {
                if oldest.clone().map_or(true, |(old_pid, _)| pid < old_pid) {
                    oldest = Some((pid, entry.path()));
                }
            } else if pid != my_pid && !is_process_alive(pid) {
                let _ = fs::remove_file(entry.path());
            }
        }
    }
    oldest.map(|(_, p)| p)
}

fn is_process_alive(pid: u32) -> bool {
    std::path::Path::new(&format!("/proc/{}", pid)).exists()
}

async fn cleanup_my_pipe(pid: u32) {
    let _ = fs::remove_file(pipe_path(pid)).await;
}

#[tokio::main]
async fn main() -> Result<()> {
    let my_pid = std::process::id();

    fs::create_dir_all(PIPE_DIR).await?;
    let my_pipe_path = pipe_path(my_pid);
    let _ = fs::remove_file(&my_pipe_path);
    let listener = UnixListener::bind(&my_pipe_path)?;
    let inference_lock = Arc::new(Mutex::new(()));

    println!("PID {} listening on {:?}", my_pid, my_pipe_path);

    let cleanup_pid = my_pid;
    tokio::spawn(async move {
        let _ = tokio::signal::ctrl_c().await;
        cleanup_my_pipe(cleanup_pid).await;
        std::process::exit(0);
    });

    let args = Args::parse();

    let has_input = has_any_input(&args);
    let has_output = has_any_output(&args);

    if !has_input || !has_output {
        eprintln!("No input or output specified.");
        std::process::exit(0);
    }

    // SWITCHED TO BROADCAST CHANNEL
    // This allows multiple consumers (main loop + inference tasks) to receive events.
    let (fs_tx, mut fs_rx) = broadcast::channel(64);
    let tx_clone = fs_tx.clone();
    let mut watcher =
        notify::recommended_watcher(move |res: Result<notify::Event, notify::Error>| {
            let res = res.map_err(Arc::new);
            let _ = tx_clone.send(res);
        })?;

    let dirs = collect_viewer_dirs(&args).await?;
    let dir_args = dirs.iter().map(|p| p.to_string_lossy());

    for d in args
        .input_final
        .iter()
        .chain(args.input_cat.iter())
        .chain(args.system_final.iter())
        .chain(args.system_cat.iter())
    {
        fs::create_dir_all(d).await?;
        watcher.watch(d, RecursiveMode::NonRecursive)?;
        if args.verbose {
            eprintln!("Watching: {}", d.display());
        }
    }

    let mut io = spawn_command_io("psi-viewer", dir_args).await?;
    let mut editor_exit = io.exited_rx.take();

    tokio::spawn(async move {
        loop {
            tokio::select! {
                Some(chunk) = io.stdout_rx.recv() => {
                    print!("{}", String::from_utf8_lossy(&chunk));
                }

                Some(chunk) = io.stderr_rx.recv() => {
                    eprint!("{}", String::from_utf8_lossy(&chunk));
                }

                _ = async {
                    if let Some(rx) = &mut editor_exit {
                        let _ = rx.await;
                    }
                }, if editor_exit.is_some() => {
                    eprintln!("Editor exited — shutting down daemon.");
                    std::process::exit(0);
                }

                else => break,
            }
        }
    });

    let mut model: Option<Arc<mistralrs::Model>> = None;

    // Persistent realtime listener — lives for the full process lifetime.
    // Spawned lazily on first model load so we have a model handle for detect_speech.
    let mut persistent_listener: Option<(
        broadcast::Receiver<()>,
        watch::Receiver<Option<Vec<u8>>>,
    )> = None;
    let mut _listener_guard: Option<oneshot::Sender<()>> = None;
    // Audio chunks queued by the main loop to be handed to the next run_once call.
    let mut queued_audio: Vec<Vec<u8>> = Vec::new();

    loop {
        let upstream = find_oldest_pipe(my_pid).await;

        tokio::select! {
            event = fs_rx.recv() => {
                match event {
                    Ok(Ok(event)) => {
                        if args.verbose {
                            eprintln!("Change detected: {:?}", event);
                        }


                        // FIX FOR DOUBLE INFERENCE:
                        // If we are the leader (no upstream) and we are currently inferencing,
                        // we skip sending a new request. The running inference task owns a
                        // receiver (fs_tx.subscribe()) and will handle the interrupt internally.
                        let should_skip = upstream.is_none() && inference_lock.try_lock().is_err();

                        if should_skip {
                            if args.verbose {
                                eprintln!("Inference in progress; event delegated to running task.");
                            }
                            continue;
                        } else { while let Ok(ev) = fs_rx.try_recv() {
                            if args.verbose {
                                eprintln!("Additional: {:?}", ev);
                            }
                        }}

                        println!("Running inference after filesystem change");

                        if let Some(upstream_pipe) = upstream {
                            let req = InferenceRequest {
                                args: args.clone(),
                                requesting_pid: my_pid,
                            };
                            if let Err(e) = send_request(&upstream_pipe, req).await {
                                eprintln!("Failed to reach leader: {}", e);
                            }
                        } else {
                            // Leader path: Load model if needed, then send request to self
                            // to be picked up by the listener branch.
                            if model.is_none() {
                                println!("Loading model for first inference...");
                                model = Some(Arc::new(load_model(&args).await?));
                                if args.verbose {
                                    eprintln!("Model initialized");
                                }
                                // Spawn the persistent audio listener now that we have a model.
                                if args.realtime_listener.is_some() && persistent_listener.is_none() {
                                    let source = args.realtime_listener.as_ref().unwrap();
                                    match spawn_realtime_listener(source, &args, model.as_ref().unwrap().clone()).await {
                                        Ok(rl) => {
                                            let (latest_audio_tx, latest_audio_rx) = watch::channel::<Option<Vec<u8>>>(None);
                                            let speech_rx = rl.speech_detected_rx.resubscribe();
                                            let mut chunk_rx = rl.chunk_rx;
                                            tokio::spawn(async move {
                                                while let Some(chunk) = chunk_rx.recv().await {
                                                    let _ = latest_audio_tx.send(Some(chunk));
                                                }
                                            });
                                            persistent_listener = Some((speech_rx, latest_audio_rx));
                                            _listener_guard = Some(rl.shutdown_tx);
                                            if args.verbose {
                                                eprintln!("Persistent audio listener spawned.");
                                            }
                                        }
                                        Err(e) => eprintln!("Failed to spawn realtime listener: {}", e),
                                    }
                                }
                            }
                            let req = InferenceRequest {
                                args: args.clone(),
                                requesting_pid: my_pid,
                            };
                            let _ = send_request(&my_pipe_path, req).await;
                        }
                    }
                    Ok(Err(e)) => {
                        eprintln!("Watcher error: {}", e);
                        break;
                    }
                    Err(broadcast::error::RecvError::Closed) => {
                        break;
                    }
                    Err(broadcast::error::RecvError::Lagged(_)) => {
                        // If we lagged, we just continue; events are noisy anyway.
                        continue;
                    }
                }
            }

            result = listener.accept() => {
                if let Ok((stream, _)) = result {
                    if model.is_none() {
                        println!("Loading model for first request...");
                        model = Some(Arc::new(load_model(&args).await?));
                        if args.verbose {
                            eprintln!("Model initialized");
                        }
                        // Spawn persistent listener if not yet running.
                        if args.realtime_listener.is_some() && persistent_listener.is_none() {
                            let source = args.realtime_listener.as_ref().unwrap();
                            match spawn_realtime_listener(source, &args, model.as_ref().unwrap().clone()).await {
                                Ok(rl) => {
                                    let (latest_audio_tx, latest_audio_rx) = watch::channel::<Option<Vec<u8>>>(None);
                                    let speech_rx = rl.speech_detected_rx.resubscribe();
                                    let mut chunk_rx = rl.chunk_rx;
                                    tokio::spawn(async move {
                                        while let Some(chunk) = chunk_rx.recv().await {
                                            let _ = latest_audio_tx.send(Some(chunk));
                                        }
                                    });
                                    persistent_listener = Some((speech_rx, latest_audio_rx));
                                    _listener_guard = Some(rl.shutdown_tx);
                                    if args.verbose {
                                        eprintln!("Persistent audio listener spawned.");
                                    }
                                }
                                Err(e) => eprintln!("Failed to spawn realtime listener: {}", e),
                            }
                        }
                    }

                    let m = model.as_ref().unwrap().clone();
                    let lock = inference_lock.clone();
                    let interrupt_tx = fs_tx.clone();
                    // Pass a resubscribed speech receiver + current audio snapshot.
                    let audio_chans = persistent_listener.as_ref().map(|(sdr, lar)| {
                        (sdr.resubscribe(), lar.clone())
                    });
                    let audio_queue = std::mem::take(&mut queued_audio);

                    tokio::spawn(async move {
                        let _guard = lock.lock().await;
                        handle_request(stream, m, interrupt_tx, audio_chans, audio_queue).await;
                    });
                }
            }

            // Audio-triggered inference: speech detected by the persistent listener.
            _ = async {
                if let Some((ref mut sdr, _)) = persistent_listener {
                    sdr.recv().await.ok().map(|_| ())
                } else {
                    std::future::pending::<Option<()>>().await
                }
            }, if inference_lock.try_lock().is_ok() => {
                if args.verbose {
                    eprintln!("Speech detected — triggering audio-first inference.");
                }
                // Snapshot the latest audio chunk into the queue.
                if let Some((_, ref lar)) = persistent_listener {
                    if let Some(chunk) = lar.borrow().clone() {
                        queued_audio.push(chunk);
                    }
                }
                if model.is_none() {
                    println!("Loading model for audio-triggered inference...");
                    model = Some(Arc::new(load_model(&args).await?));
                }
                let req = InferenceRequest {
                    args: args.clone(),
                    requesting_pid: my_pid,
                };
                let _ = send_request(&my_pipe_path, req).await;
            }
        }

        if !args.watch {
            break;
        }
    }

    cleanup_my_pipe(my_pid).await;
    Ok(())
}

async fn load_model(args: &Args) -> Result<mistralrs::Model> {
    println!(
        "primary_model={}{}",
        args.model,
        args.model_path
            .as_ref()
            .map(|p| format!("(local: {})", p))
            .unwrap_or_default()
    );
    println!("secondary_model={}", args.secondary_model);

    let primary_alias = args.model.clone();
    let secondary_alias = args.secondary_model.clone();

    let mut builder = mistralrs::MultiModelBuilder::new().with_default_model(&primary_alias);

    // Add primary model
    if args.verbose {
        eprintln!("Loading Primary Model");
    }
    if let Some(path) = &args.model_path {
        if args.verbose {
            eprintln!("Using local model file for primary: {}", path);
        }
        let primary_builder = VisionModelBuilder::new(path)
            .with_auto_isq(IsqBits::Four)
            .with_logging();
        builder = builder.add_model_with_alias(&primary_alias, primary_builder);
    } else if !args.gguf.is_empty() {
        println!("Loading primary as GGUF!");
        let primary_builder = GgufModelBuilder::new(&args.model, args.gguf.clone());
        builder = builder.add_model_with_alias(&primary_alias, primary_builder);
    } else {
        let primary_builder = VisionModelBuilder::new(&args.model)
            .with_auto_isq(IsqBits::Four)
            .with_logging();
        builder = builder.add_model_with_alias(&primary_alias, primary_builder);
    };

    // Only load secondary (audio) model when it is actually needed.
    if args.secondary_model != "none" && args.realtime_listener.is_some() {
        if args.verbose {
            eprintln!("Loading Secondary Model");
        }
        let secondary_builder = mistralrs::VisionModelBuilder::new(&args.secondary_model)
            .with_auto_isq(IsqBits::Four)
            .with_dtype(mistralrs::ModelDType::F32)
            .with_logging();
        builder = builder.add_model_with_alias(&secondary_alias, secondary_builder);
    } else if args.verbose {
        eprintln!("Skipping secondary model (not needed).");
    }
    if args.verbose {
        eprintln!("Building Models");
    }
    let model = builder.build().await?;
    if args.verbose {
        eprintln!("Models Built");
    }
    Ok(model)
}

async fn send_request(pipe: &PathBuf, req: InferenceRequest) -> Result<()> {
    let mut stream = UnixStream::connect(pipe).await?;
    let data = serde_json::to_vec(&req)?;
    stream.write_all(&data).await?;
    stream.write_all(b"\n").await?;
    stream.flush().await?;
    Ok(())
}

// Updated to accept broadcast sender for interrupts
async fn handle_request(
    stream: UnixStream,
    model: Arc<mistralrs::Model>,
    interrupt_tx: broadcast::Sender<Result<notify::Event, Arc<notify::Error>>>,
    audio_channels: Option<(broadcast::Receiver<()>, watch::Receiver<Option<Vec<u8>>>)>,
    pending_audio: Vec<Vec<u8>>,
) {
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    if let Ok(_) = reader.read_line(&mut line).await {
        if let Ok(req) = serde_json::from_str::<InferenceRequest>(&line) {
            println!("Handling request from PID {}", req.requesting_pid);
            let interrupt_rx = interrupt_tx.subscribe();
            let _ = run_once(
                &model,
                &req.args,
                Some(interrupt_rx),
                audio_channels,
                pending_audio,
            )
            .await;
        }
    }
}
