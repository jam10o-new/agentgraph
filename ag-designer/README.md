# AgentGraph Designer

A visual graph-based configuration designer for AgentGraph. It provides an interface to define models, agents, and global parameters, and to orchestrate agentic workflows by wiring their inputs and outputs.

---

## Features

- **Visual Workflow Orchestration**: Wire `Model` outputs to `Agent` model pins, and chain `Agent` outputs to other `Agent` input pins.
- **Node Configuration**:
  - **Models**: Define model IDs, builders, GGUF paths, and chat templates.
  - **Agents**: Set input, output, and system prompt directories, history limits, and addendum prompts.
  - **Global Config**: Manage sampling (temperature, top_p, etc.) and context compression parameters.
- **Dynamic Input Mapping**: Connecting an output to an agent's "New Link" pin automatically generates a new input field and pin.
- **Filesystem Integration**: Includes directory pickers for input, output, and system prompt paths.
- **Persistence**: Save graphs to AgentGraph-compatible YAML files and reload existing configurations into the visual environment.
- **Daemon Spawning**: Launch the `ag` leader process using the current graph state directly from the interface.

## Installation

### Requirements
- Linux
- Rust toolchain

### Build
```bash
cargo build --release
```

### Dependency Note
The **Spawn** feature requires the `ag` binary to be available in your system `PATH`. You can install it from the project root:
```bash
cargo install --path ./
```

## Usage

- **Add Nodes**: Use the toolbar to insert Model, Agent, or Global Config nodes.
- **Wire Connections**: Drag wires between pins to define model assignment and agent chaining.
- **Configure**: Enter parameters directly into node bodies or use the directory pickers.
- **Save/Load**: Use the **Save** and **Load** buttons to manage YAML configuration files.
- **Spawn**: Click **Spawn** to write the current state to a temporary file and invoke the `ag` leader.
