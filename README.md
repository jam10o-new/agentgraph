# AgentGraph

A multi-agent daemon system for real-time vision, audio, and tool-based inference using local LLMs. AgentGraph orchestrates multiple cooperating agents that communicate via Unix pipes, enabling distributed AI pipelines for desktop automation, audio transcription, and screen analysis.

## Features

- **Multi-modal inference**: Vision (Qwen3-VL), Audio (Voxtral), and text models running locally
- **Distributed agent architecture**: Agents communicate via Unix domain sockets with automatic leader election
- **Real-time streaming**: Support for PipeWire audio capture and live desktop screenshot analysis
- **Tool execution**: Built-in command execution system (`【EXEC ... CEXE】`) for agents to interact with the host system
- **Watch mode**: File system watchers trigger inference on new input (images, audio, text)
- **Quantization support**: ISQ (Integrated Supervisor Quantization) for efficient VRAM usage

## Architecture

### Agent Communication

Agents form a directed acyclic graph (DAG) where:
- **Leader node**: The oldest process (lowest PID) holds the model in VRAM and performs all inference
- **Follower nodes**: Forward requests to the leader via Unix domain sockets in `/tmp/agentgraph_pipes/`
- **Automatic failover**: If the leader exits, the next oldest process becomes leader

### Model Slots

| Slot | Model | Purpose |
|------|-------|---------|
| Primary | `Qwen/Qwen3-VL-8B-Instruct` (default) | Vision + text inference |
| Secondary | `mistralai/Voxtral-Mini-4B-Realtime-2602` (default) | Audio speech detection + transcription |

### Command System

Agents can execute host commands using a structured command syntax with reversed-text markers:

```
EEXEC command arg1 arg2 CEXEE       → Spawn process (returns index)
KKILL idx LLIKK                     → Kill process by index
RREAD idx DAERR                     → Read stdout/stderr
WWRIT idx input text TIRWW          → Write to stdin
```

The reversed-text closing markers (e.g., `CEXEE` = `EXEC` reversed + first letter repeated) are designed to be rare in natural language while remaining comprehensible to models. Commands execute immediately and return results. 

## Installation

### Requirements

- Linux with Wayland (for screenshot tools)
- NVIDIA GPU with CUDA support (24GB+ VRAM recommended for dual-model)
- PipeWire (for audio capture)
- Rust toolchain

### Build

```bash
# Release build (recommended for performance)
cargo build --release

# The binary will be at: target/release/agentgraph
```

### Dependencies

The test scripts require:
- `grim` - Wayland screenshot utility
- `ffmpeg` - Video/audio extraction
- `psi-viewer` - System monitor (default editor process)

## Usage

### Basic Command Structure

```bash
agentgraph [OPTIONS]

# Key flags:
-W, --watch              # Watch mode: trigger inference on file changes
--verbose                # Enable debug logging
--tools                  # Enable EXEC command system
--stream-realtime        # Stream output token-by-token (for human-facing agents)
--realtime-listener SRC  # Audio source ("pipewire" or RTSP/RTMP URL)

# Input/Output:
-I, --input-final DIR    # Latest file in directory = user input
-i, --input-cat DIR      # All files in directory = user input
-S, --system-final DIR   # Latest file = system prompt
-s, --system-cat DIR     # All files = system prompt
-A, --assistant-final DIR # Latest file = assistant message
-a, --assistant-cat DIR  # All files = assistant messages
-O, --output-new DIR     # Create new file per response
-o, --output-overwrite FILE  # Overwrite same file per response
```

### Example: Single Agent Vision Pipeline

```bash
# Watch a directory for screenshots and produce descriptions
agentgraph -W --verbose \
  -S system_prompts/ \
  -I input_screenshots/ \
  -O output_descriptions/
```

## Test Cases (Usage Examples)

The project includes four comprehensive test suites that demonstrate real-world usage patterns.

### 1. Vision Test: Screen Analysis Pipeline

**Location**: `test_vision/`

**Topology**:
```
[grim screenshot loop] → [Agent A: Vision] → [Agent B: Summary]
```

- **Agent A**: Watches for PNG screenshots, produces structured vision reports
- **Agent B**: Receives Agent A's output, maintains rolling activity summary

**Run**:
```bash
cd test_vision
bash run_test.sh
```

**Key concepts demonstrated**:
- Agent-to-agent communication via output directories
- Vision model (Qwen3-VL) for image analysis
- Debounced re-inference on new screenshots
- `--stream-realtime` for human-facing final agent

**Example system prompts**: See `test_vision/agent_a/system/prompt.txt` and `agent_b/system/prompt.txt`

### 2. Tools Test: Screenshot → Vision → Summary

**Location**: `test_tools/`

**Topology**:
```
[Agent Screenshot] --EXEC grim→ [Agent Vision] → [Agent Summary]
```

- **Agent Screenshot**: Uses `【EXEC grim ... CEXE】` to capture screenshots
- **Agent Vision**: Describes screenshots using vision model
- **Agent Summary**: Maintains rolling narrative of screen activity

**Run**:
```bash
cd test_tools
bash run_test.sh
```

**Key concepts demonstrated**:
- `--tools` flag for command execution
- Agents triggering external tools (grim)
- Multi-hop agent pipelines
- Structured output parsing

**Example output**:
```
SCREENSHOT_READY: /path/to/screenshot_1234567890.png
[LAYOUT]: Two windows visible, terminal on left, browser on right
[APPS]: Kitty terminal, Firefox
[FOCUS]: Browser showing GitHub repository
...
```

### 3. Audio Test: Real-time Transcription + Summary

**Location**: `test_audio/`

**Topology**:
```
[PipeWire mic] → [Agent A: Voxtral] → [Agent B: Summary]
```

- **Agent A**: Listens to PipeWire audio, detects speech, transcribes chunks
- **Agent B**: Receives transcriptions, produces summaries

**Run**:
```bash
cd test_audio
bash run_test.sh
```

**Key concepts demonstrated**:
- `--realtime-listener pipewire` for live audio capture
- Secondary model (Voxtral) for speech detection
- Audio-first inference triggering
- Dual-model concurrent operation

**Configuration**:
```bash
--audio-chunk-min-secs 3.0   # Minimum audio chunk duration
--audio-chunk-max-secs 8.0   # Maximum audio chunk duration
```

### 4. Subagents Test: Autonomous Subagent Spawning & Coordination

**Location**: `test_subagents/`

**Topology**:
```
                    [ SUPERVISOR AGENT ]
                           |
        +------------------+------------------+
        |                  |                  |
        v                  v                  v
[ watcher_a ]      [ watcher_b ]      [ oneshot_c ]      [ oneshot_d ]
(watch mode)       (watch mode)       (system state)     (recommendations)
```

- **Supervisor Agent**: Orchestrates subagent spawning using `【EXEC cargo run ... CEXE】` commands
- **Watcher A**: Continuous monitoring of shared screenshot directory, general screen state analysis
- **Watcher B**: Continuous monitoring focused on UI elements and anomaly detection
- **Oneshot C**: Single-pass system state assessment from text input
- **Oneshot D**: Single-pass recommendation generation from watcher outputs

**Run**:
```bash
cd test_subagents
bash run_test.sh
```

**Key concepts demonstrated**:
- Autonomous subagent spawning via `--tools` EXEC command system
- Supervisor coordinates multiple agentgraph daemon instances
- Mix of watch-mode (continuous) and oneshot (single-pass) inference subagents
- Multi-agent orchestration patterns for complex task decomposition
- Subagent output collection via READ commands

**Example supervisor output**:
```
[SUBAGENTS_SPAWNED]: watcher_a (idx=0), watcher_b (idx=1), oneshot_c (idx=2), oneshot_d (idx=3)
[WATCHER_A_STATUS]: Active, monitoring shared_screenshots
[WATCHER_B_STATUS]: Active, monitoring shared_screenshots
[ONESHOT_C_RESULT]: System components operational, no anomalies detected
[ONESHOT_D_RESULT]: Continue monitoring, risk level: low
[COORDINATION_SUMMARY]: All 4 subagents spawned successfully and reporting...
```

## Known Issues

See [KNOWN_ISSUES.md](KNOWN_ISSUES.md) for detailed tracking.

### Current Issues

| ID | Status | Severity | Description |
|----|--------|----------|-------------|
| KI-001 | Open | High | CUDA OOM during speech detection (dual-model on 24GB GPU) |
| KI-005 | Open | Low | Duplicate outputs in tools test pipeline |
| KI-006 | Unknown | High | Potential audio parsing failures |

### Resolved Issues

- **KI-002** (Fixed 2026-03-18): Vision test re-inference loop
- **KI-003** (Fixed): Agent-to-agent feedback storm from `--stream-realtime`
- **KI-004** (Fixed): Realtime audio listener tied to single inference lifetime

## Configuration Examples

### Desktop Monitoring Agent

```bash
# System prompt in system/prompts.txt:
# "You are a desktop monitoring agent. Analyze screenshots and report changes."

agentgraph -W --verbose \
  -S system/ \
  -I screenshots/ \
  -O analysis/ \
  -m "Qwen/Qwen3-VL-8B-Instruct"
```

### Audio Transcription Agent

```bash
agentgraph -W --verbose \
  --realtime-listener pipewire \
  --audio-chunk-min-secs 3.0 \
  --audio-chunk-max-secs 5.0 \
  -S system/ \
  -I input/ \
  -O transcripts/ \
  -M "mistralai/Voxtral-Mini-4B-Realtime-2602"
```

### Tool-Using Agent (Screenshot Taker)

```bash
agentgraph -W --tools \
  -S system/ \
  -I triggers/ \
  -O output/
```

## Troubleshooting

### CUDA Out of Memory

- Reduce quantization level (edit `load_model` to use `IsqBits::Two` or `IsqBits::Three`)
- Disable secondary model if not needed: `-M none`
- Ensure no other GPU processes are running

### Audio Not Detected

- Verify PipeWire is running: `pactl info`
- Check audio source: `pactl list sources short`
- Test with `--verbose` to see speech detection logs

### Screenshot Failures

- Ensure `WAYLAND_DISPLAY` is set
- Test grim manually: `grim -s 0.5 test.png`
- Check Wayland compositor supports screenshot protocols

### Agent Communication Issues

- Check `/tmp/agentgraph_pipes/` for pipe files
- Verify leader election: oldest process (lowest PID) should be leader
- Look for "Failed to reach leader" errors in logs

## Project Structure

```
agentgraph/
├── src/
│   ├── main.rs          # CLI entrypoint and agent loop
│   ├── lib.rs           # Library root, re-exports modules
│   ├── audio.rs         # PipeWire audio capture and streaming
│   ├── command_exec.rs  # EXEC/READ/KILL/WRIT command system
│   ├── commands.rs      # Command parsing and dispatch
│   ├── events.rs        # Event types for agent communication
│   ├── inference.rs     # Model loading and inference logic
│   ├── ipc.rs           # Unix domain socket communication
│   ├── messages.rs      # Message types for agent protocols
│   ├── model.rs         # Model slot management (primary/secondary)
│   ├── types.rs         # Core type definitions
│   └── utils.rs         # Utility functions
├── test_audio/          # Real-time audio pipeline test
├── test_subagents/      # Autonomous subagent spawning test
├── test_tools/          # Tool execution pipeline test
├── test_vision/         # Vision-only pipeline test
├── KNOWN_ISSUES.md      # Issue tracking
└── Cargo.toml
```

## License

MIT License (see Cargo.toml for author information)

## Acknowledgments

- [mistral.rs](https://github.com/EricLBuehler/mistral.rs) - High-performance LLM inference
- [cpal](https://github.com/RustAudio/cpal) - Cross-platform audio capture
- [notify](https://github.com/notify-rs/notify) - File system watching
