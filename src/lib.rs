//! AgentGraph - Multi-agent daemon system for real-time vision, audio, and tool-based inference.

pub mod audio;
pub mod command_exec;
pub mod commands;
pub mod events;
pub mod inference;
pub mod ipc;
pub mod messages;
pub mod model;
pub mod types;
pub mod utils;

// Re-export commonly used types
pub use commands::CommandParser;
pub use types::{CMD_CLOSE_EXEC, CMD_CLOSE_KILL, CMD_CLOSE_READ, CMD_CLOSE_WRIT};
pub use types::{CMD_OPEN_EXEC, CMD_OPEN_KILL, CMD_OPEN_READ, CMD_OPEN_WRIT};
pub use types::{
    CmdOpenType, CommandIO, CommandType, ContentType, FileMessage, FileMetadata, InferenceRequest,
    InterruptKind, MessageContent, ModelSlot, ParallelInferenceParams, RealtimeListener,
    StreamOutcome,
};
pub use types::{MAX_OPENER_LEN, PIPE_DIR, PIPE_PREFIX};

use clap::Parser;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// AgentGraph daemon arguments
#[derive(Parser, Debug, Clone, Serialize, Deserialize)]
#[command(author, version)]
pub struct Args {
    /// Watch mode (trigger inference on any system prompt or user input modification or addition)
    #[arg(short = 'W')]
    pub watch: bool,

    /// Verbose logging
    #[arg(short, long)]
    pub verbose: bool,

    /// Enable tool use and system prompt addendum for tools.
    #[arg(short, long)]
    pub tools: bool,

    /// Realtime output
    /// recommended for final agent -> human nodes
    /// STRONGLY discouraged for agent -> agent nodes, or if the output is intended to be structured -
    #[arg(long)]
    pub stream_realtime: bool,

    /// GGUF file list (which files are we downloading and loading as our local model).
    /// This flag is only relevant when running as a leader - all inference happens using the model running on the current leader node
    #[arg(long, value_delimiter = ',')]
    pub gguf: Vec<String>,

    /// Model name (HF repo) - PRIMARY slot (vision model)
    #[arg(short = 'm', long, default_value = "Qwen/Qwen3-VL-8B-Instruct")]
    pub model: String,

    /// Secondary model name (HF repo) - SECONDARY slot (audio model, default: Voxtral)
    #[arg(
        short = 'M',
        long,
        default_value = "mistralai/Voxtral-Mini-4B-Realtime-2602"
    )]
    pub secondary_model: String,

    /// Number of latest files to include (for -I, -S, -A flags)
    #[arg(long, default_value_t = 1)]
    pub latest_n: usize,

    /// Realtime audio listener - RTSP/RTMP URL or "pipewire" for local input
    #[arg(long)]
    pub realtime_listener: Option<String>,

    /// Audio chunk min duration in seconds (for realtime listener)
    #[arg(long, default_value_t = 3.0)]
    pub audio_chunk_min_secs: f32,

    /// Audio chunk max duration in seconds (for realtime listener)
    #[arg(long, default_value_t = 8.0)]
    pub audio_chunk_max_secs: f32,

    /// Path to a local model file
    #[arg(long)]
    pub model_path: Option<String>,

    /// Sleep / debounce duration in milliseconds
    #[arg(long, default_value_t = 250)]
    pub sleep_ms: u64,

    /// Latest file in directory is treated as user input (you can use this flag multiple times)
    #[arg(short = 'I')]
    pub input_final: Vec<PathBuf>,

    /// All files in directory are treated as user input (you can use this flag multiple times)
    #[arg(short = 'i')]
    pub input_cat: Vec<PathBuf>,

    /// Latest file in directory is treated as system messages (you can use this flag multiple times)
    #[arg(short = 'S')]
    pub system_final: Vec<PathBuf>,

    /// All files in directory are treated as system messages (you can use this flag multiple times)
    #[arg(short = 's')]
    pub system_cat: Vec<PathBuf>,

    /// Latest file in directory is treated as assistant messages (you can use this flag multiple times)
    /// New assistant messages do not trigger inference!
    /// Output flags are not automatically stored in context, so at least one assistant flag is required for the agent to be aware of it's own responses
    #[arg(short = 'A')]
    pub assistant_final: Vec<PathBuf>,

    /// All files in directory are treated as assistant messages (you can use this flag multiple times)
    #[arg(short = 'a')]
    pub assistant_cat: Vec<PathBuf>,

    /// Each assistant response is stored in a new file in this directory
    #[arg(short = 'O')]
    pub output_new: Option<PathBuf>,

    /// Each assistant response clears and overwrites this file
    #[arg(short = 'o')]
    pub output_overwrite: Option<PathBuf>,
}
