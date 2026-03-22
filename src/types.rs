//! Core types and constants for agentgraph.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::SystemTime;

// ============================================================================
// Command constants
// ============================================================================

pub const CMD_OPEN_EXEC: &str = "EEXEC ";
pub const CMD_CLOSE_EXEC: &str = " CEXEE";
pub const CMD_OPEN_KILL: &str = "KKILL ";
pub const CMD_CLOSE_KILL: &str = " LLIKK";
pub const CMD_OPEN_READ: &str = "RREAD ";
pub const CMD_CLOSE_READ: &str = " DAERR";
pub const CMD_OPEN_WRIT: &str = "WWRIT ";
pub const CMD_CLOSE_WRIT: &str = " TIRWW";
pub const CMD_OPEN_READ_SKILL: &str = "RRDS ";
pub const CMD_CLOSE_READ_SKILL: &str = " SDRR";

/// Maximum length of any opener (for pre-active buffer sizing)
pub const MAX_OPENER_LEN: usize = 6; // "【EXEC " is 6 chars

// ============================================================================
// IPC constants
// ============================================================================

pub const PIPE_DIR: &str = "/tmp/agentgraph_pipes";
pub const PIPE_PREFIX: &str = "pipe_";

// ============================================================================
// Command types
// ============================================================================

/// Command type parsed from model output
pub enum CommandType {
    Exec(String),        // full command string: "command arg arg"
    Kill(usize),         // index
    Read(usize),         // index
    Writ(usize, String), // index and input
    ReadSkill(String),   // skill name/path
}

/// Command opener type for state machine parsing
#[derive(Clone, Copy)]
pub enum CmdOpenType {
    Exec,
    Kill,
    Read,
    Writ,
    ReadSkill,
}

// ============================================================================
// Model slot types
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ModelSlot {
    Primary,
    Secondary,
}

/// Model alias for primary (vision) model
pub const MODEL_PRIMARY: &str = "primary";
/// Model alias for secondary (audio) model
pub const MODEL_SECONDARY: &str = "secondary";

// ============================================================================
// Interrupt types
// ============================================================================

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct InterruptKind: u8 {
        const FS                          = 0b0001;
        const AUDIO                       = 0b0010;
        const PARALLEL_INFERENCE_COMPLETION  = 0b0100;
        const SYNTHESIS_INFERENCE_INITIATION = 0b1000;
    }
}

// ============================================================================
// Inference request/response types
// ============================================================================

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct InferenceRequest {
    pub args: crate::Args,
    pub requesting_pid: u32,
}

// ============================================================================
// File and message types
// ============================================================================

#[derive(Debug, Clone)]
pub enum MessageContent {
    Text(String),
    Image(image::DynamicImage),
    Audio(Vec<u8>),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ContentType {
    Text,
    Image,
    Audio,
    Video,
}

#[derive(Debug, Clone)]
pub struct FileMetadata {
    pub path: PathBuf,
    pub created: SystemTime,
    pub modified: SystemTime,
}

#[derive(Debug, Clone)]
pub struct FileMessage {
    pub role: String,
    pub content: MessageContent,
    pub metadata: FileMetadata,
    pub content_type: ContentType,
}

// ============================================================================
// Inference outcome types
// ============================================================================

/// The outcome of a single call to `run_streaming_loop`.
pub enum StreamOutcome {
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

// ============================================================================
// Parallel inference types
// ============================================================================

#[derive(Clone)]
pub struct ParallelInferenceParams {
    pub messages: mistralrs::VisionMessages,
    pub system_addendum: String,
    pub interrupted_by: InterruptKind,
    pub context_inputs: Vec<ParallelInferenceParams>,
    pub streaming: bool,
    pub model_slot: ModelSlot,
}

impl ParallelInferenceParams {
    pub fn synthesis_params(
        messages: mistralrs::VisionMessages,
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

    pub fn vision_params(messages: mistralrs::VisionMessages) -> Self {
        ParallelInferenceParams {
            messages,
            system_addendum: "You are a vision-enabled agent tasked with analyzing the visual content of images and videos. You share a historical context with other agents, and your responses are synthesized with theirs, but your task is to use your own unique insights given the visual content only. Your response will be synthesized with other agents' responses, so be concise and to the point.".into(),
            interrupted_by: InterruptKind::FS | InterruptKind::PARALLEL_INFERENCE_COMPLETION,
            context_inputs: Vec::new(),
            streaming: false,
            model_slot: ModelSlot::Primary,
        }
    }

    pub fn audio_params(messages: mistralrs::VisionMessages) -> Self {
        ParallelInferenceParams {
            messages,
            system_addendum: "You are an audio-enabled agent tasked with analyzing audio content. You share a historical context with other agents, and your responses are synthesized with theirs, but your task is to use your own unique insights given the audio content only. Your response will be synthesized with other agents' responses, so be concise and to the point.".into(),
            interrupted_by: InterruptKind::AUDIO | InterruptKind::PARALLEL_INFERENCE_COMPLETION,
            context_inputs: Vec::new(),
            streaming: false,
            model_slot: ModelSlot::Secondary,
        }
    }

    pub fn full_pipeline(
        vision_messages: mistralrs::VisionMessages,
        audio_messages: mistralrs::VisionMessages,
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

// ============================================================================
// Coroutine types
// ============================================================================

pub enum CoroutineResponse {
    Complete(String),
    Interrupted(String),
    Error(anyhow::Error),
}

// ============================================================================
// CommandIO for subprocess management
// ============================================================================

pub struct CommandIO {
    pub stdin_tx: tokio::sync::mpsc::Sender<Vec<u8>>,
    pub stdout_rx: tokio::sync::mpsc::Receiver<Vec<u8>>,
    pub stderr_rx: tokio::sync::mpsc::Receiver<Vec<u8>>,
    kill_tx: Option<tokio::sync::oneshot::Sender<()>>,
    pub exited_rx: Option<tokio::sync::oneshot::Receiver<()>>,
}

impl CommandIO {
    pub fn new(
        stdin_tx: tokio::sync::mpsc::Sender<Vec<u8>>,
        stdout_rx: tokio::sync::mpsc::Receiver<Vec<u8>>,
        stderr_rx: tokio::sync::mpsc::Receiver<Vec<u8>>,
        kill_tx: Option<tokio::sync::oneshot::Sender<()>>,
        exited_rx: Option<tokio::sync::oneshot::Receiver<()>>,
    ) -> Self {
        Self {
            stdin_tx,
            stdout_rx,
            stderr_rx,
            kill_tx,
            exited_rx,
        }
    }

    pub fn kill(&mut self) {
        if let Some(tx) = self.kill_tx.take() {
            let _ = tx.send(());
        }
    }
}

impl Drop for CommandIO {
    fn drop(&mut self) {
        if let Some(tx) = self.kill_tx.take() {
            let _ = tx.send(());
        }
    }
}

// ============================================================================
// RealtimeListener for audio streaming
// ============================================================================

pub struct RealtimeListener {
    pub chunk_rx: tokio::sync::mpsc::Receiver<Vec<u8>>,
    pub shutdown_tx: tokio::sync::oneshot::Sender<()>,
    pub speech_detected_rx: tokio::sync::broadcast::Receiver<()>,
}

impl RealtimeListener {
    pub fn new(
        chunk_rx: tokio::sync::mpsc::Receiver<Vec<u8>>,
        shutdown_tx: tokio::sync::oneshot::Sender<()>,
        speech_detected_rx: tokio::sync::broadcast::Receiver<()>,
    ) -> Self {
        Self {
            chunk_rx,
            shutdown_tx,
            speech_detected_rx,
        }
    }
}
