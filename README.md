# AgentGraph

An async-first multi-agent daemon for real-time vision, audio, and text inference using `mistral.rs`. AgentGraph orchestrates multiple independent agents that communicate via the filesystem, utilizing a leader/follower pattern to maximize VRAM efficiency.

The usecases this was built for:
- realtime chat summarization for streaming
- digital assistant/coworker agents
- "second-brain"/digital twin organization and management
- fun little hobby agent colonies using tiny models

What I haven't tried and never intend to support:
- large models
- remote inference (no support at present, feel free to fork if you want this)

Help welcome:
- testing models that have support in mistral.rs to see if they work in this harness
- testing weird graphs and cycles/breaking stuff (and fixing it with PRs)

What I am actively working on/roadmap:
- multimodal models, support for realtime multimodal inputs (ie, mic, camera, rtmp, etc.), and "routing" input based on type between models with a shared context.
- multimodal outputs: TTS models, image gen, video gen. ideally realtime.
- running into issues that warrant upstream contribution.
- testing tool use and colony orchestration.

tl;dr: this is basically an opinionated, **vibecoded** (DO NOT USE FOR CRITICAL USECASES OR WITH UNTRUSTED INPUT! WE HAVE COMMAND EXECUTION AND NO SECURITY GUARDRAILS WHATSOEVER), wrapper and orchastration layer around [`mistral.rs`]("https://github.com/EricLBuehler/mistral.rs") where the filesystem is the primary medium for context engineering and management.

Protip: Use and spawn with [`psi-cli`](https://github.com/jam10o-new/psi-cli) - a minimal AF chat harness.

---

## Core Features

- **Async-First Architecture**: Every agent runs in its own coroutine with dedicated filesystem watchers and interrupt logic.
- **Leader/Follower Pattern**: A single leader process manages model loading and inference, while followers (CLI subcommands) communicate via Unix Domain Sockets. This ensures VRAM is never duplicated across processes.
- **Idiomatic Tool Calling**: Supports native `mistralrs` tool calling for models like Qwen or Llama. Current built-in tools include:
  - `execute_command`: Run shell commands on the host.
  - `spawn_new_agent`: Dynamically create new agents with custom configs.
  - `load_into_context`: Load arbitrary files into the model's volatile context.
- **Multimodal Support (Vision)**: Automatically detects images (`jpg`, `jpeg`, `png`, `webp`) in an agent's input directory and attaches them to the inference request.
- **Context Compression**: Just-in-time summarization that compresses older conversation turns into contextually relevant "domains" to save context window space over time.
- **Configuration-Driven**: All models, agents, and sampling parameters are managed via a simple YAML configuration.

## Installation

### Requirements
- Linux (CUDA recommended)
- Rust toolchain

### Build
By default, `agentgraph` compiles with support for CUDA, cuDNN, and Flash Attention. You can override these using cargo features:
```bash
# Default build (CUDA/cuDNN/Flash Attention)
cargo build --release

# CPU-only build (MKL)
cargo build --release --no-default-features --features mkl

# Apple Metal build
cargo build --release --no-default-features --features metal
```

## Usage

### Starting the Leader
The leader loads the models and starts the agent coroutines defined in your config. **Only one leader can run at a time.**
```bash
ag leader --config config.yaml
```

### Managing Agents
You can interact with the running leader using subcommands:
```bash
ag status                       # Get leader and active agent list
ag run <agent> [message]        # Trigger an agent turn, optionally injecting a message
ag stop <agent>                 # Abort an agent's current inference
ag reload                       # Reload leader configuration
ag spawn <name> <path> [args]   # Dynamically spawn a new agent
ag shutdown                     # Gracefully shut down the leader
```

## Configuration

AgentGraph is configured via `config.yaml`. The configuration defines:
- **Models**: Registry of models (path, builder type, ISQ quantization, etc.).
- **Agents**: Input/output directories, system prompts, history limits, and model assignment.
- **Sampling**: Global parameters like temperature, top_p, etc.
- **Compression**: Thresholds and probabilities for the summarization logic.

## Architecture

### Filesystem-as-IPC
Each agent watches a specific directory structure:
- `input/`: Any new file triggers a "turn". Text files are read as user input; images are attached as multimodal data.
- `system/`: Contains `.md` or `.txt` files used as system prompts. Supports YAML frontmatter for "Skills".
- `output/`: Assistant responses are streamed here in real-time.
- `interrupt`: Creating this file immediately aborts the agent's current inference.

### Tool Execution
When a model issues a tool call, the agent loop handles the execution and feeds the results back into the context for a follow-up turn. This allows for complex multi-step reasoning and environment interaction.

## Roadmap

- [ ] **Real-time Audio Input**: Integrating `cpal` for microphone support to enable voice-to-text turns (see `examples/audio_diag.rs`).
- [ ] **Multimodal Outputs**: Support for TTS (Text-to-Speech), image generation (Stable Diffusion/Flux), and video generation.
- [ ] **Input Routing**: Intelligently routing different media types to specialized models within a shared context.
- [ ] **Upstream Contributions**: Formalizing issues found during heavy use and contributing fixes back to `mistral.rs`.
- [ ] **Distributed Colonies**: Investigating ways for colonies to span multiple machines while maintaining filesystem-level synchronization.
