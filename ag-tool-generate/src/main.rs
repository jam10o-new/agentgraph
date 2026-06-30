use ag_tool_common::{describe, guidance, has_flag};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;

// ─── Defaults ───

fn default_width() -> u32 { 1024 }
fn default_height() -> u32 { 768 }
fn default_video_width() -> u32 { 704 }
fn default_video_height() -> u32 { 480 }
fn default_one() -> u32 { 1 }
fn default_steps() -> u32 { 28 }
fn default_turbo_steps() -> u32 { 8 }
fn default_edit_steps() -> u32 { 50 }
fn default_animate_steps() -> u32 { 30 }
fn default_guidance() -> f64 { 3.5 }
fn default_turbo_guidance() -> f64 { 0.0 }
fn default_variant() -> String { "turbo".into() }
fn default_edit_guidance() -> f64 { 5.0 }
fn default_lora_scale() -> f64 { 0.8 }
fn default_frames() -> u32 { 25 }
fn default_output_dir() -> String { "./output".into() }
fn default_krea_model() -> String { "Comfy-Org/Krea-2".into() }
fn default_qwen_model() -> String { "Qwen/Qwen-Image-Edit".into() }
fn default_ltx_model() -> String { "Lightricks/LTX-Video".into() }
fn default_i2v_model() -> String { "Lightricks/LTX-Video".into() }

// ─── Template resolution ───

fn resolve_template(s: &str, ctx: &HashMap<String, Value>) -> String {
    if s.starts_with('{') && s.ends_with('}') && s.len() > 2 {
        let inner = &s[1..s.len()-1];
        // Check for dotted path like "key.outputs" or "key.output"
        if let Some(dot) = inner.find('.') {
            let key = &inner[..dot];
            let path = &inner[dot+1..];
            if let Some(val) = ctx.get(key) {
                if let Some(resolved) = val.get(path).and_then(|v| v.as_str()) {
                    return resolved.to_string();
                }
                // Array path: if the value is an array, return first element's field
                if let Some(arr) = val.get(path).and_then(|v| v.as_array()) {
                    let paths: Vec<String> = arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect();
                    if !paths.is_empty() {
                        return paths.join(",");
                    }
                }
            }
        }
        if let Some(val) = ctx.get(inner) {
            // Single string output (like edit step's "output")
            if let Some(s) = val.as_str() {
                return s.to_string();
            }
            // Array of outputs (like generate step's "outputs")
            if let Some(arr) = val.get("outputs").and_then(|v| v.as_array()) {
                let paths: Vec<String> = arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect();
                return paths.join(",");
            }
            if let Some(s) = val.get("output").and_then(|v| v.as_str()) {
                return s.to_string();
            }
        }
    }
    s.to_string()
}

fn resolve_template_list(list: &[String], ctx: &HashMap<String, Value>) -> Vec<String> {
    let mut out = Vec::new();
    for item in list {
        let resolved = resolve_template(item, ctx);
        // If the resolved value contains commas, split into multiple paths
        if resolved.contains(',') {
            for part in resolved.split(',') {
                let part = part.trim().to_string();
                if !part.is_empty() {
                    out.push(part);
                }
            }
        } else {
            out.push(resolved);
        }
    }
    out
}

// ─── Step types ───

#[derive(Deserialize, Serialize)]
struct GenerateStep {
    #[serde(default)]
    skip: bool,
    #[serde(default)]
    key: Option<String>,
    prompt: String,
    #[serde(default)]
    negative_prompt: String,
    #[serde(default = "default_width")]
    width: u32,
    #[serde(default = "default_height")]
    height: u32,
    #[serde(default = "default_one")]
    num_images: u32,
    #[serde(default = "default_turbo_steps")]
    num_inference_steps: u32,
    #[serde(default = "default_turbo_guidance")]
    guidance_scale: f64,
    #[serde(default)]
    lora: Option<String>,
    #[serde(default = "default_lora_scale")]
    lora_scale: f64,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default = "default_krea_model")]
    model: String,
    #[serde(default)]
    input_dir: Option<String>,
    #[serde(default = "default_variant")]
    model_variant: String,
}

#[derive(Deserialize, Serialize)]
struct EditStep {
    #[serde(default)]
    skip: bool,
    #[serde(default)]
    key: Option<String>,
    prompt: String,
    #[serde(default)]
    source_image: Option<String>,
    #[serde(default)]
    source_images: Option<Vec<String>>,
    #[serde(default = "default_edit_steps")]
    num_inference_steps: u32,
    #[serde(default)]
    true_cfg_scale: Option<f64>,
    #[serde(default)]
    negative_prompt: Option<String>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default = "default_qwen_model")]
    model: String,
}

#[derive(Deserialize, Serialize)]
struct AnimateStep {
    #[serde(default)]
    skip: bool,
    #[serde(default)]
    key: Option<String>,
    prompt: String,
    #[serde(default)]
    start_image: Option<String>,
    #[serde(default)]
    end_image: Option<String>,
    #[serde(default = "default_video_width")]
    width: u32,
    #[serde(default = "default_video_height")]
    height: u32,
    #[serde(default = "default_frames")]
    num_frames: u32,
    #[serde(default = "default_animate_steps")]
    num_inference_steps: u32,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default = "default_i2v_model")]
    model: String,
}

#[derive(Deserialize, Serialize)]
#[serde(tag = "type")]
enum Step {
    Generate(GenerateStep),
    Edit(EditStep),
    Animate(AnimateStep),
}

#[derive(Deserialize, Serialize)]
struct Args {
    steps: Vec<Step>,
    #[serde(default = "default_output_dir")]
    output_dir: String,
    #[serde(default)]
    daemonize: bool,
}

// ─── Script discovery ───

fn find_venv_python() -> Option<String> {
    // Check for venv in standard locations relative to binary
    let self_exe = std::env::current_exe().ok()?;
    let bin_dir = self_exe.parent()?;
    let venv_python = bin_dir.join("..").join(".venv").join("bin").join("python3");
    if venv_python.exists() {
        return Some(venv_python.to_string_lossy().to_string());
    }
    let venv_python2 = bin_dir.join("..").join("..").join(".venv").join("bin").join("python3");
    if venv_python2.exists() {
        return Some(venv_python2.to_string_lossy().to_string());
    }
    None
}

fn find_script(name: &str) -> PathBuf {
    let self_exe = std::env::current_exe().ok();
    if let Some(exe) = &self_exe {
        if let Some(dir) = exe.parent() {
            // sibling dir: <binary_dir>/scripts/<name>
            let sibling = dir.join("scripts").join(name);
            if sibling.exists() {
                return sibling;
            }
            // grandparent: <binary_dir>/../scripts/<name>
            if let Some(parent) = dir.parent() {
                let grand = parent.join("scripts").join(name);
                if grand.exists() {
                    return grand;
                }
            }
        }
    }
    let cwd = std::env::current_dir().unwrap_or_default();
    let cwd_scripts = cwd.join("scripts").join(name);
    if cwd_scripts.exists() {
        return cwd_scripts;
    }
    PathBuf::from(name)
}

// ─── Run a Python script with JSON stdin ───

fn run_python_script(script: &str, input: &Value) -> Result<Value, String> {
    let script_path = find_script(script);
    let input_json = serde_json::to_string(input).map_err(|e| e.to_string())?;

    // Use venv python if available
    let python = find_venv_python().unwrap_or_else(|| "python3".to_string());

    let child = std::process::Command::new(&python)
        .arg(&script_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to spawn {script}: {e}"))?;

    let write_result = {
        let mut stdin = child.stdin.as_ref().unwrap();
        stdin.write_all(input_json.as_bytes())
    };
    if let Err(e) = write_result {
        return Err(format!("Failed to write to {script} stdin: {e}"));
    }

    let output = child.wait_with_output().map_err(|e| format!("Failed to wait on {script}: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("{script} failed:\n{stderr}"));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    serde_json::from_str(&stdout).map_err(|e| format!("Failed to parse {script} output: {e}\nOutput: {stdout}"))
}

// ─── Pipeline execution ───

fn execute_pipeline(args: &Args) -> Result<Value, String> {
    let output_dir = &args.output_dir;
    std::fs::create_dir_all(output_dir).map_err(|e| format!("Cannot create output dir: {e}"))?;

    let mut ctx: HashMap<String, Value> = HashMap::new();

    for (i, step) in args.steps.iter().enumerate() {
        match step {
            Step::Generate(s) if !s.skip => {
                eprintln!("[pipeline] Step {}: generate (Krea-2, variant={})", i + 1, s.model_variant);
                let mut input = json!({
                    "prompt": s.prompt,
                    "negative_prompt": s.negative_prompt,
                    "width": s.width,
                    "height": s.height,
                    "num_images": s.num_images,
                    "num_inference_steps": s.num_inference_steps,
                    "guidance_scale": s.guidance_scale,
                    "model_variant": s.model_variant,
                    "model": s.model,
                    "output_dir": output_dir,
                });
                if let Some(ref lora) = s.lora {
                    input["lora"] = json!(lora);
                    input["lora_scale"] = json!(s.lora_scale);
                }
                if let Some(seed) = s.seed {
                    input["seed"] = json!(seed);
                }
                if let Some(ref input_dir) = s.input_dir {
                    input["input_dir"] = json!(input_dir);
                }
                let result = run_python_script("krea2_gen.py", &input)?;
                let step_key = s.key.clone().unwrap_or_else(|| format!("step_{}", i));
                ctx.insert(step_key, result.clone());
            }

            Step::Edit(s) if !s.skip => {
                eprintln!("[pipeline] Step {}: edit (Qwen-Image-Edit)", i + 1);

                // Resolve source images from explicit list, explicit single, or context
                let resolved_images: Vec<String> = if let Some(ref imgs) = s.source_images {
                    resolve_template_list(imgs, &ctx)
                } else if let Some(ref single) = s.source_image {
                    let resolved = resolve_template(single, &ctx);
                    if resolved.contains(',') {
                        resolved.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect()
                    } else {
                        vec![resolved]
                    }
                } else {
                    // Auto-from-context: use any known outputs
                    let mut auto = Vec::new();
                    let ctx_ordered: Vec<_> = ctx.iter().collect();
                    for (_, val) in ctx_ordered.into_iter().rev() {
                        if let Some(arr) = val.get("outputs").and_then(|v| v.as_array()) {
                            for item in arr {
                                if let Some(p) = item.as_str() {
                                    auto.push(p.to_string());
                                }
                            }
                            if !auto.is_empty() { break; }
                        }
                        if let Some(s) = val.get("output").and_then(|v| v.as_str()) {
                            auto.push(s.to_string());
                            break;
                        }
                    }
                    if auto.is_empty() {
                        return Err("Edit step has no source image available".into());
                    }
                    auto
                };

                if resolved_images.is_empty() {
                    return Err("Edit step has no source image available".into());
                }

                let mut input = json!({
                    "prompt": s.prompt,
                    "source_images": resolved_images,
                    "num_inference_steps": s.num_inference_steps,
                    "model": s.model,
                    "output_dir": output_dir,
                });
                if let Some(cfg) = s.true_cfg_scale {
                    input["true_cfg_scale"] = json!(cfg);
                }
                if let Some(ref neg) = s.negative_prompt {
                    input["negative_prompt"] = json!(neg);
                }
                if let Some(seed) = s.seed {
                    input["seed"] = json!(seed);
                }
                let result = run_python_script("qwen_image_edit.py", &input)?;
                let step_key = s.key.clone().unwrap_or_else(|| format!("step_{}", i));
                ctx.insert(step_key, result.clone());
            }

            Step::Animate(s) if !s.skip => {
                eprintln!("[pipeline] Step {}: animate (LTX-Video)", i + 1);
                let start = if let Some(ref explicit) = s.start_image {
                    let resolved = resolve_template(explicit, &ctx);
                    if resolved.is_empty() || resolved == *explicit { None }
                    else { Some(resolved) }
                } else {
                    None
                };

                let mut input = json!({
                    "prompt": s.prompt,
                    "width": s.width,
                    "height": s.height,
                    "num_frames": s.num_frames,
                    "num_inference_steps": s.num_inference_steps,
                    "model": s.model,
                    "output_dir": output_dir,
                });
                if let Some(ref start_img) = start {
                    input["start_image"] = json!(start_img);
                }
                if let Some(ref end_img) = s.end_image {
                    let resolved = resolve_template(end_img, &ctx);
                    input["end_image"] = json!(resolved);
                }
                if let Some(seed) = s.seed {
                    input["seed"] = json!(seed);
                }
                let result = run_python_script("ltx_video.py", &input)?;
                let step_key = s.key.clone().unwrap_or_else(|| format!("step_{}", i));
                ctx.insert(step_key, result.clone());
            }

            _ => {
                eprintln!("[pipeline] Step {}: skip", i + 1);
            }
        }
    }

    // Collect all outputs
    let mut all_outputs = json!({});
    for (key, val) in &ctx {
        all_outputs[key] = val.clone();
    }
    Ok(json!({
        "outputs": all_outputs,
        "num_steps": args.steps.len(),
    }))
}

// ─── Daemonize ───

fn daemonize_pipeline(args: &Args) -> Result<Value, String> {
    let job_id = uuid::Uuid::new_v4();
    let output_dir = &args.output_dir;
    std::fs::create_dir_all(output_dir).map_err(|e| format!("Cannot create output dir: {e}"))?;

    let job_file = PathBuf::from(output_dir).join(format!("job_{job_id}.json"));

    let args_json = serde_json::to_value(&args).map_err(|e| e.to_string())?;

    // Write a "pending" job file
    let job_meta = json!({
        "job_id": job_id.to_string(),
        "status": "pending",
        "steps": args.steps.len(),
        "output_dir": output_dir,
        "predicted_outputs": predict_outputs(args),
    });
    std::fs::write(&job_file, serde_json::to_string_pretty(&job_meta).unwrap())
        .map_err(|e| format!("Cannot write job file: {e}"))?;

    let child = std::process::Command::new(std::env::current_exe().unwrap())
        .arg("--execute")
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("Failed to spawn background pipeline: {e}"))?;

    let pid = child.id();

    Ok(json!({
        "job_id": job_id.to_string(),
        "status": "running",
        "job_file": job_file.to_string_lossy().to_string(),
        "pid": pid,
    }))
}

fn predict_outputs(args: &Args) -> Value {
    let mut preds = Vec::new();
    for step in &args.steps {
        match step {
            Step::Generate(s) => {
                preds.push(json!({"type": "generate", "outputs": vec!["*.png"; s.num_images as usize]}));
            }
            Step::Edit(_) => {
                preds.push(json!({"type": "edit", "output": "*.png"}));
            }
            Step::Animate(_) => {
                preds.push(json!({"type": "animate", "video": "*.mp4", "frames": "*.png"}));
            }
        }
    }
    json!(preds)
}

// ─── Main ───

fn main() {
    #[cfg(feature = "tui")]
    if has_flag("--tui") {
        tui::run();
        return;
    }

    if has_flag("--describe") {
        describe(
            "generate",
            "Generate, edit, and animate images using Krea-2, Qwen-Image-Edit, and LTX-Video.",
            json!({
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "items": { "$ref": "#/$defs/Step" }
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Output directory for generated files"
                    }
                },
                "required": ["steps", "output_dir"]
            }),
        );
        return;
    }

    if has_flag("--help") {
        guidance("Steps run in order. Each step can `skip`. Outputs are keyed and available to later steps via `{key}` template syntax.");
        return;
    }

    if has_flag("--execute") {
        let raw = std::io::read_to_string(std::io::stdin()).unwrap_or_default();
        let args: Args = serde_json::from_str(&raw).unwrap_or_else(|e| {
            eprintln!("Failed to parse pipeline args: {e}");
            std::process::exit(1);
        });

        if args.daemonize {
            match daemonize_pipeline(&args) {
                Ok(v) => { println!("{}", serde_json::to_string(&v).unwrap()); }
                Err(e) => { eprintln!("Daemonize failed: {e}"); std::process::exit(1); }
            }
        } else {
            match execute_pipeline(&args) {
                Ok(v) => { println!("{}", serde_json::to_string(&v).unwrap()); }
                Err(e) => { eprintln!("Pipeline failed: {e}"); std::process::exit(1); }
            }
        }
        return;
    }

    // Interactive TUI mode (or simple demo via --tui)
    let sample = json!({
        "steps": [
            {
                "type": "generate",
                "key": "gen1",
                "prompt": "a cat wearing a wizard hat in a magical forest, detailed, high quality",
                "num_images": 2,
                "model_variant": "turbo"
            },
            {
                "type": "edit",
                "key": "edit1",
                "prompt": "make the wizard hat purple with gold stars",
                "source_images": ["{gen1}"],
                "num_inference_steps": 10,
                "true_cfg_scale": 4.0
            }
        ],
        "output_dir": "./output"
    });
    println!("{}", serde_json::to_string_pretty(&sample).unwrap());
}

#[cfg(feature = "tui")]
mod tui {
    use super::*;

    pub fn run() {
        println!("ag-tool-generate TUI — coming soon");
    }
}
