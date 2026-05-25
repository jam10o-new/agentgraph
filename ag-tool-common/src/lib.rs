//! Shared utilities for AgentGraph tool binaries.
//!
//! Tool binaries are simple command-line programs that:
//!   1. `--describe` — print the tool's JSON Function schema (for agent discovery).
//!   2. `--help`     — print LLM guidance text (appended to the system prompt).
//!   3. Default      — read JSON arguments from stdin, execute, write result to stdout.
//!
//! Convention: binary name `ag-tool-<name>` maps to config entry `ag-tool-<name>`.

use serde_json::Value;

/// Check whether a flag is present in the process arguments.
pub fn has_flag(flag: &str) -> bool {
    std::env::args().any(|a| a == flag)
}

/// Print the tool's JSON Function schema to stdout (for `--describe`).
/// The output matches mistralrs's `Function` struct so it can be deserialized
/// directly into tool definitions.
pub fn describe(name: &str, description: &str, parameters: Value) {
    let schema = serde_json::json!({
        "name": name,
        "description": description,
        "parameters": parameters,
    });
    println!(
        "{}",
        serde_json::to_string(&schema).expect("serialize tool schema")
    );
}

/// Print LLM usage guidance text to stdout (for `--help`).
/// This text is appended to the agent's system prompt so the model knows
/// how to use the tool effectively.
pub fn guidance(text: &str) {
    println!("{}", text);
}

/// Read JSON arguments from stdin. Blocks until EOF.
pub fn read_args_string() -> Result<String, String> {
    let mut input = String::new();
    std::io::Read::read_to_string(&mut std::io::stdin(), &mut input)
        .map_err(|e| format!("Failed to read stdin: {e}"))?;
    Ok(input)
}

/// Read and deserialize JSON arguments from stdin.
pub fn read_args<T: serde::de::DeserializeOwned>() -> Result<T, String> {
    let input = read_args_string()?;
    serde_json::from_str(&input).map_err(|e| format!("Failed to parse arguments: {e}"))
}