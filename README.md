# AgentGraph

A high-performance, async-first multi-agent daemon for real-time vision, audio, and text inference using `mistral.rs`. AgentGraph orchestrates multiple independent agents that communicate via the filesystem, utilizing a leader/follower pattern to maximize VRAM efficiency.

## Features

- **Async-First Architecture**: Every agent runs in its own coroutine with dedicated filesystem watchers and interrupt logic.
- **Leader/Follower Pattern**: A single leader process manages model loading and inference, while followers communicate via Unix Domain Sockets.
- **Advanced Context Compression**: Probabilistic, hierarchical summarization that compresses older conversation turns into contextually relevant "domains" to save context window space.
- **Multimodal Support**: Built-in support for Vision (Qwen-VL) and Audio (Voxtral) models.
- **Configuration-Driven**: All models, agents, and sampling parameters are managed via a simple YAML configuration.
- **Local Fork Integration**: Designed to work with a local fork of `mistral.rs` for rapid prototyping and upstream contributions.

## Installation

### Requirements
- Linux (CUDA recommended)
- Rust toolchain
- `mistralrs-fork` directory present in project root.

### Build
```bash
cargo build --release
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
ag spawn <name> <path> [args]   # Dynamically spawn a new agent
ag shutdown                     # Gracefully shut down the leader
```

## Configuration

AgentGraph is configured via `config.yaml`. See the provided template for a full list of options, including model definitions and compression thresholds.

## Architecture

### Filesystem Interface
Each agent watches a directory structure:
- `input/`: New files (images/text) trigger inference.
- `system/`: System prompts and "Skills" (YAML frontmatter).
- `output/`: Results are streamed here.
- `interrupt`: Creating this file aborts current inference.

### Context Management
As history grows, the system calculates a compression probability for each turn based on its depth. If triggered, turns are summarized. If a general summary is insufficient for the latest input, the model generates a "Specialized Summary" for that specific domain.

---

## Tool Calling (Sketch)

We are moving away from manual `EEXEC ... CEXE` parsing in favor of idiomatic `mistralrs` tool calling.

### Proposed Tool Implementation
Agents will be registered with a set of `mistralrs::Tool` definitions. Instead of the model outputting text markers, it will output tool calls that the leader executes via callbacks.

```rust
// Example Tool Definition
let tool = Tool {
    name: "execute_command".to_string(),
    description: "Execute a shell command on the host".to_string(),
    parameters: Some(json!({
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "args": {"type": "array", "items": {"type": "string"}}
        }
    })),
};

// Callback Implementation
let callback = Arc::new(move |args: String| {
    let parsed: CommandArgs = serde_json::from_str(&args)?;
    let output = std::process::Command::new(parsed.command)
        .args(parsed.args)
        .output()?;
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
});
```

Benefits:
1. **Model Native**: Models trained for tool use (like Qwen) handle this much more reliably than regex markers.
2. **Structured**: Arguments are validated against a schema before the callback is even reached.
3. **Async Integration**: Callbacks can be hooked directly into our async agent loops.
