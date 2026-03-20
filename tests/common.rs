//! Common test utilities for agentgraph integration tests.
//!
//! This module provides:
//! - Process management (spawn, kill, cleanup)
//! - Pattern matching for success/failure detection
//! - Directory and file management
//! - Output monitoring with inotify-like functionality

use anyhow::{Context, Result};
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

/// Configuration for a test agent process
#[derive(Debug, Clone)]
pub struct AgentConfig {
    pub label: String,
    pub system_dir: PathBuf,
    pub input_dir: PathBuf,
    pub output_dir: PathBuf,
    pub log_file: PathBuf,
    pub watch_mode: bool,
    pub verbose: bool,
    pub tools: bool,
    pub stream_realtime: bool,
    pub model: Option<String>,
    pub secondary_model: Option<String>,
    pub realtime_listener: Option<String>,
    pub audio_chunk_min_secs: Option<f32>,
    pub audio_chunk_max_secs: Option<f32>,
    pub input_from_output: Option<PathBuf>, // If set, -I points to another agent's output
    pub additional_inputs: Vec<PathBuf>,    // Additional -I input directories
    pub additional_args: Vec<String>,       // Any extra CLI arguments
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            label: String::new(),
            system_dir: PathBuf::new(),
            input_dir: PathBuf::new(),
            output_dir: PathBuf::new(),
            log_file: PathBuf::new(),
            watch_mode: true,
            verbose: true,
            tools: false,
            stream_realtime: false,
            model: None,
            secondary_model: None,
            realtime_listener: None,
            audio_chunk_min_secs: None,
            audio_chunk_max_secs: None,
            input_from_output: None,
            additional_inputs: Vec::new(),
            additional_args: Vec::new(),
        }
    }
}

/// Manages a running agent process
pub struct RunningAgent {
    pub config: AgentConfig,
    pub child: Child,
}

impl RunningAgent {
    pub fn spawn(binary: &Path, config: &AgentConfig) -> Result<Self> {
        let mut cmd = Command::new(binary);

        // Base arguments
        if config.watch_mode {
            cmd.arg("-W");
        }
        if config.verbose {
            cmd.arg("--verbose");
        }
        if config.tools {
            cmd.arg("--tools");
        }
        if config.stream_realtime {
            cmd.arg("--stream-realtime");
        }

        // Model arguments
        if let Some(ref model) = config.model {
            cmd.arg("-m").arg(model);
        }
        if let Some(ref secondary) = config.secondary_model {
            cmd.arg("-M").arg(secondary);
        }

        // Realtime listener arguments
        if let Some(ref listener) = config.realtime_listener {
            cmd.arg("--realtime-listener").arg(listener);
        }
        if let Some(min_secs) = config.audio_chunk_min_secs {
            cmd.arg("--audio-chunk-min-secs").arg(min_secs.to_string());
        }
        if let Some(max_secs) = config.audio_chunk_max_secs {
            cmd.arg("--audio-chunk-max-secs").arg(max_secs.to_string());
        }

        // Directory arguments
        if config.system_dir.as_os_str().is_empty() {
            // Skip if empty (for warmup agent)
        } else {
            cmd.arg("-S").arg(&config.system_dir);
        }

        // Primary input directory
        if let Some(ref input_from) = config.input_from_output {
            cmd.arg("-I").arg(input_from);
        } else if !config.input_dir.as_os_str().is_empty() {
            cmd.arg("-I").arg(&config.input_dir);
        }

        // Additional input directories (clap supports multiple -I flags)
        for additional_input in &config.additional_inputs {
            cmd.arg("-I").arg(additional_input);
        }

        if !config.output_dir.as_os_str().is_empty() {
            cmd.arg("-O").arg(&config.output_dir);
        }

        // Additional arguments
        for arg in &config.additional_args {
            cmd.arg(arg);
        }

        // Redirect stdout/stderr to log file
        let log_file = File::create(&config.log_file)
            .with_context(|| format!("Failed to create log file: {:?}", config.log_file))?;

        cmd.stdout(Stdio::from(log_file.try_clone()?))
            .stderr(Stdio::from(log_file));

        let child = cmd
            .spawn()
            .with_context(|| format!("Failed to spawn agent: {}", config.label))?;

        let pid = child.id();

        println!("    {} PID: {}", config.label, pid);

        Ok(Self {
            config: config.clone(),
            child,
        })
    }

    pub fn kill(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Test runner that manages multiple agents and monitors their output
pub struct TestRunner {
    pub base_dir: PathBuf,
    pub binary: PathBuf,
    pub agents: Vec<RunningAgent>,
    pub success_patterns: Vec<String>,
    pub fail_patterns: Vec<String>,
    pub success_file: PathBuf,
    pub fail_file: PathBuf,
    pub verbose: bool,
    pub max_runtime_secs: u64,
    pub fail_fast: bool,
    pub max_output_length_checks: Vec<(String, usize, String)>, // (output_dir_rel, max_len, fail_message)
}

impl TestRunner {
    pub fn new(base_dir: &Path, binary: &Path) -> Result<Self> {
        Ok(Self {
            base_dir: base_dir.to_path_buf(),
            binary: binary.to_path_buf(),
            agents: Vec::new(),
            success_patterns: Vec::new(),
            fail_patterns: Vec::new(),
            success_file: base_dir.join(".test_succeeded"),
            fail_file: base_dir.join(".test_failed"),
            verbose: false,
            max_runtime_secs: 300,
            fail_fast: true,
            max_output_length_checks: Vec::new(),
        })
    }

    pub fn set_max_output_length_check(
        &mut self,
        output_dir_rel: &str,
        max_len: usize,
        fail_message: &str,
    ) {
        self.max_output_length_checks.push((
            output_dir_rel.to_string(),
            max_len,
            fail_message.to_string(),
        ));
    }

    pub fn add_success_pattern(&mut self, pattern: &str) {
        self.success_patterns.push(pattern.to_string());
    }

    pub fn add_fail_pattern(&mut self, pattern: &str) {
        self.fail_patterns.push(pattern.to_string());
    }

    pub fn spawn_agent(&mut self, config: &AgentConfig) -> Result<()> {
        let agent = RunningAgent::spawn(&self.binary, config)?;
        self.agents.push(agent);
        Ok(())
    }

    pub fn cleanup(&mut self) {
        println!("\n=== Cleaning up ===");
        for agent in &mut self.agents {
            agent.kill();
        }

        // Kill any stale processes
        let _ = Command::new("pkill")
            .arg("-f")
            .arg(format!("agentgraph.*{}", self.base_dir.display()))
            .output();

        println!("=== Cleanup complete ===");
    }

    pub fn pre_run_cleanup(&mut self, dirs_to_clean: &[&str]) -> Result<()> {
        // Remove state files
        let _ = fs::remove_file(&self.success_file);
        let _ = fs::remove_file(&self.fail_file);

        // Clean specified directories (remove all files within them)
        for dir_rel in dirs_to_clean {
            let dir_path = self.base_dir.join(dir_rel);
            if dir_path.exists() && dir_path.is_dir() {
                // Read directory and remove each file
                if let Ok(entries) = fs::read_dir(&dir_path) {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        if path.is_file() {
                            let _ = fs::remove_file(&path);
                        }
                    }
                }
            }
        }

        println!("    Output directories cleared.");
        Ok(())
    }

    /// Monitor output directories for new files and check for patterns
    pub fn check_outputs(&self) -> Result<TestStatus> {
        let mut matched_success_count = 0;
        let mut matched_success_patterns = std::collections::HashSet::new();

        // Check max output length constraints
        for (output_dir_rel, max_len, fail_message) in &self.max_output_length_checks {
            let output_dir = self.base_dir.join(output_dir_rel);
            if output_dir.exists() {
                if let Ok(entries) = fs::read_dir(&output_dir) {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        if !path.is_file() {
                            continue;
                        }
                        if let Ok(content) = fs::read_to_string(&path) {
                            if content.len() > *max_len {
                                println!(
                                    "\n!!! FAIL: {} ({} bytes > {} bytes)",
                                    fail_message,
                                    content.len(),
                                    max_len
                                );
                                File::create(&self.fail_file)?;
                                return Ok(TestStatus::Failed);
                            }
                        }
                    }
                }
            }
        }

        for agent in &self.agents {
            let output_dir = &agent.config.output_dir;
            if !output_dir.exists() {
                continue;
            }

            if let Ok(entries) = fs::read_dir(output_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if !path.is_file() {
                        continue;
                    }

                    if let Ok(content) = fs::read_to_string(&path) {
                        if self.verbose {
                            println!(
                                "[{}] <<< {}: {}",
                                agent.config.label,
                                path.display(),
                                content.trim()
                            );
                        }

                        // Check fail patterns
                        for pattern in &self.fail_patterns {
                            if self.matches_pattern(&content, pattern) {
                                println!(
                                    "\n!!! FAIL [{}]: pattern '{}' matched in {}",
                                    agent.config.label,
                                    pattern,
                                    path.display()
                                );
                                File::create(&self.fail_file)?;
                                return Ok(TestStatus::Failed);
                            }
                        }

                        // Check success patterns
                        for pattern in &self.success_patterns {
                            if self.matches_pattern(&content, pattern) {
                                if !matched_success_patterns.contains(pattern) {
                                    println!(
                                        "\n>>> SUCCESS [{}]: pattern '{}' in {} <<<",
                                        agent.config.label,
                                        pattern,
                                        path.display()
                                    );
                                    matched_success_patterns.insert(pattern.clone());
                                    matched_success_count += 1;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Check if ALL success patterns matched
        if matched_success_count >= self.success_patterns.len() && !self.success_patterns.is_empty()
        {
            File::create(&self.success_file)?;
            return Ok(TestStatus::Success);
        }

        Ok(TestStatus::Running)
    }

    fn matches_pattern(&self, content: &str, pattern: &str) -> bool {
        // Handle anchored patterns (starting with ^)
        if pattern.starts_with('^') {
            let pattern = &pattern[1..];
            for line in content.lines() {
                if line.starts_with(pattern) {
                    return true;
                }
            }
            false
        } else {
            content.contains(pattern)
        }
    }

    /// Get output file counts for all agents
    pub fn get_output_counts(&self) -> Vec<(String, usize)> {
        self.agents
            .iter()
            .map(|agent| {
                let count = fs::read_dir(&agent.config.output_dir)
                    .map(|entries| {
                        entries
                            .filter(|e| e.as_ref().map(|e| e.path().is_file()).unwrap_or(false))
                            .count()
                    })
                    .unwrap_or(0);
                (agent.config.label.clone(), count)
            })
            .collect()
    }

    /// Run the test with a monitoring loop
    pub fn run(&mut self) -> TestResult {
        let start = Instant::now();
        let check_interval = Duration::from_secs(2);
        let heartbeat_interval = Duration::from_secs(30);
        let mut last_heartbeat = Instant::now();

        loop {
            // Check for failure file (from other sources)
            if self.fail_file.exists() {
                println!("\n=========================================");
                println!("  TEST RESULT: FAIL (fail file detected)");
                println!("=========================================");
                return TestResult::Fail;
            }

            // Check outputs
            match self.check_outputs() {
                Ok(TestStatus::Success) => {
                    println!("\n======================================================");
                    println!("  Test confirmed working — success conditions met.");
                    println!("======================================================");
                    if self.fail_fast {
                        return TestResult::Pass;
                    }
                }
                Ok(TestStatus::Failed) => {
                    println!("\n=========================================");
                    println!("  TEST RESULT: FAIL");
                    println!("=========================================");
                    return TestResult::Fail;
                }
                Ok(TestStatus::Running) => {}
                Err(e) => {
                    eprintln!("Error checking outputs: {}", e);
                }
            }

            let elapsed = start.elapsed().as_secs();

            // Heartbeat
            if last_heartbeat.elapsed() >= heartbeat_interval {
                let counts = self.get_output_counts();
                let heartbeat: Vec<String> = counts
                    .iter()
                    .map(|(label, count)| format!("{}: {}", label, count))
                    .collect();
                println!("[heartbeat {}s] {}", elapsed, heartbeat.join("  "));
                last_heartbeat = Instant::now();
            }

            // Timeout
            if elapsed >= self.max_runtime_secs {
                println!("\n=== Max runtime reached ({}) ===", self.max_runtime_secs);
                if self.success_file.exists() {
                    println!("TEST RESULT: PASS");
                    return TestResult::Pass;
                } else {
                    println!("TEST RESULT: INCONCLUSIVE");
                    let counts = self.get_output_counts();
                    for (label, count) in counts {
                        println!("  {} outputs: {}", label, count);
                    }
                    return TestResult::Inconclusive;
                }
            }

            std::thread::sleep(check_interval);
        }
    }
}

impl Drop for TestRunner {
    fn drop(&mut self) {
        self.cleanup();
    }
}

/// Test status returned by output checks
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TestStatus {
    Running,
    Success,
    Failed,
}

/// Final test result
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TestResult {
    Pass,
    Fail,
    Inconclusive,
}

/// Helper to create directory structure
pub fn create_dirs(base: &Path, dirs: &[&str]) -> Result<()> {
    for dir in dirs {
        let path = base.join(dir);
        fs::create_dir_all(&path)
            .with_context(|| format!("Failed to create directory: {:?}", path))?;
    }
    Ok(())
}

/// Get the project root directory
pub fn get_project_root() -> Result<PathBuf> {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    Ok(PathBuf::from(manifest_dir))
}

/// Get the binary path (prefers release build, falls back to debug)
pub fn get_binary_path() -> Result<PathBuf> {
    let project_root = get_project_root()?;
    let release_binary = project_root.join("target/release/agentgraph");
    let debug_binary = project_root.join("target/debug/agentgraph");

    if release_binary.exists() {
        Ok(release_binary)
    } else if debug_binary.exists() {
        Ok(debug_binary)
    } else {
        anyhow::bail!("No agentgraph binary found. Run `cargo build --release` first.")
    }
}
