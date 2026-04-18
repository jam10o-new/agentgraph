use ag_config::{AgentConfig, CompressionConfig, Config, ModelConfig, SamplingConfig};
use eframe::egui;
use egui_snarl::{
    ui::{PinInfo, SnarlPin, SnarlStyle, SnarlViewer},
    InPin, InPinId, NodeId, OutPin, OutPinId, Snarl,
};
use std::collections::HashMap;

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
enum MyNode {
    Model {
        name: String,
        model_id: String,
        path: Option<String>,
        gguf: Option<String>,
        isq: Option<String>,
        dtype: Option<String>,
        builder: String,
        chat_template: Option<String>,
    },
    Agent {
        name: String,
        inputs: Vec<String>,
        output_dir: String,
        system: Vec<String>,
        history_limit: Option<usize>,
        stream: bool,
        allowed_extensions: Vec<String>,
        prompt: Option<String>,
    },
}

struct DemoViewer {
    hf_models: Vec<String>,
}

impl SnarlViewer<MyNode> for DemoViewer {
    fn title(&mut self, node: &MyNode) -> String {
        match node {
            MyNode::Model { name, .. } => {
                if name.is_empty() {
                    "Model (unnamed)".to_string()
                } else {
                    format!("Model: {}", name)
                }
            }
            MyNode::Agent { name, .. } => {
                if name.is_empty() {
                    "Agent (unnamed)".to_string()
                } else {
                    format!("Agent: {}", name)
                }
            }
        }
    }

    fn outputs(&mut self, node: &MyNode) -> usize {
        match node {
            MyNode::Model { .. } => 1,
            MyNode::Agent { .. } => 1,
        }
    }

    fn inputs(&mut self, node: &MyNode) -> usize {
        match node {
            MyNode::Model { .. } => 0,
            MyNode::Agent { inputs, .. } => {
                // Pin 0: Model
                // Pins 1..=N: Inputs
                // Pin N+1: Empty input for new connections
                inputs.len() + 2
            }
        }
    }

    fn has_body(&mut self, _node: &MyNode) -> bool {
        true
    }

    fn show_body(
        &mut self,
        node_id: egui_snarl::NodeId,
        _inputs: &[InPin],
        _outputs: &[OutPin],
        ui: &mut egui::Ui,
        snarl: &mut Snarl<MyNode>,
    ) {
        ui.set_min_width(200.0);
        let node = &mut snarl[node_id];
        match node {
            MyNode::Model {
                name,
                model_id,
                path,
                gguf,
                isq,
                dtype,
                builder,
                chat_template,
            } => {
                ui.horizontal(|ui| {
                    ui.label("Name:");
                    ui.text_edit_singleline(name);
                });
                ui.horizontal(|ui| {
                    ui.label("Model ID:");
                    egui::ComboBox::from_id_salt(format!("model-id-{:?}", node_id))
                        .selected_text(&*model_id)
                        .show_ui(ui, |ui| {
                            for m in &self.hf_models {
                                ui.selectable_value(model_id, m.clone(), m);
                            }
                        });
                });
                ui.horizontal(|ui| {
                    ui.label("Builder:");
                    ui.text_edit_singleline(builder);
                });

                ui.collapsing("Advanced", |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Path:");
                        let mut p = path.clone().unwrap_or_default();
                        if ui.text_edit_singleline(&mut p).changed() {
                            *path = if p.is_empty() { None } else { Some(p) };
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("GGUF:");
                        let mut g = gguf.clone().unwrap_or_default();
                        if ui.text_edit_singleline(&mut g).changed() {
                            *gguf = if g.is_empty() { None } else { Some(g) };
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("ISQ:");
                        let mut i = isq.clone().unwrap_or_default();
                        if ui.text_edit_singleline(&mut i).changed() {
                            *isq = if i.is_empty() { None } else { Some(i) };
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("DType:");
                        let mut d = dtype.clone().unwrap_or_default();
                        if ui.text_edit_singleline(&mut d).changed() {
                            *dtype = if d.is_empty() { None } else { Some(d) };
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("Template:");
                        let mut t = chat_template.clone().unwrap_or_default();
                        if ui.text_edit_singleline(&mut t).changed() {
                            *chat_template = if t.is_empty() { None } else { Some(t) };
                        }
                    });
                });
            }
            MyNode::Agent {
                name,
                inputs,
                output_dir,
                system,
                history_limit,
                stream,
                allowed_extensions,
                prompt,
            } => {
                ui.horizontal(|ui| {
                    ui.label("Name:");
                    ui.text_edit_singleline(name);
                });

                ui.group(|ui| {
                    ui.label("Inputs (Paths):");
                    let mut to_remove = None;
                    for (i, input) in inputs.iter_mut().enumerate() {
                        ui.horizontal(|ui| {
                            ui.text_edit_singleline(input);
                            if ui.button("x").clicked() {
                                to_remove = Some(i);
                            }
                        });
                    }
                    if let Some(i) = to_remove {
                        inputs.remove(i);
                    }
                    if ui.button("+ Add Input").clicked() {
                        inputs.push(String::new());
                    }
                });

                ui.horizontal(|ui| {
                    ui.label("Output Dir:");
                    ui.text_edit_singleline(output_dir);
                });

                ui.collapsing("Agent Config", |ui| {
                    ui.checkbox(stream, "Stream");
                    ui.horizontal(|ui| {
                        ui.label("History Limit:");
                        let mut hl = history_limit.unwrap_or(0);
                        if ui.add(egui::DragValue::new(&mut hl)).changed() {
                            *history_limit = if hl == 0 { None } else { Some(hl) };
                        }
                    });

                    ui.group(|ui| {
                        ui.label("System Prompts:");
                        let mut to_remove = None;
                        for (i, s) in system.iter_mut().enumerate() {
                            ui.horizontal(|ui| {
                                ui.text_edit_singleline(s);
                                if ui.button("x").clicked() {
                                    to_remove = Some(i);
                                }
                            });
                        }
                        if let Some(i) = to_remove {
                            system.remove(i);
                        }
                        if ui.button("+ Add System Prompt").clicked() {
                            system.push(String::new());
                        }
                    });

                    ui.group(|ui| {
                        ui.label("Allowed Extensions:");
                        let mut to_remove = None;
                        for (i, ext) in allowed_extensions.iter_mut().enumerate() {
                            ui.horizontal(|ui| {
                                ui.text_edit_singleline(ext);
                                if ui.button("x").clicked() {
                                    to_remove = Some(i);
                                }
                            });
                        }
                        if let Some(i) = to_remove {
                            allowed_extensions.remove(i);
                        }
                        if ui.button("+ Add Extension").clicked() {
                            allowed_extensions.push(".txt".to_string());
                        }
                    });

                    ui.horizontal(|ui| {
                        ui.label("Override Prompt:");
                        let mut p = prompt.clone().unwrap_or_default();
                        if ui.text_edit_singleline(&mut p).changed() {
                            *prompt = if p.is_empty() { None } else { Some(p) };
                        }
                    });
                });
            }
        }
    }

    fn show_input(
        &mut self,
        pin: &InPin,
        ui: &mut egui::Ui,
        snarl: &mut Snarl<MyNode>,
    ) -> impl SnarlPin + 'static {
        let node = &snarl[pin.id.node];
        let mut pin_info = PinInfo::circle();

        match node {
            MyNode::Model { .. } => {}
            MyNode::Agent { inputs, .. } => {
                if pin.id.input == 0 {
                    ui.label("Model");
                    pin_info.fill = Some(egui::Color32::from_rgb(238, 207, 109)); // Model yellow
                } else if pin.id.input <= inputs.len() {
                    ui.label(format!("Input {}", pin.id.input));
                    pin_info.fill = Some(egui::Color32::from_rgb(38, 109, 211)); // Input blue
                } else {
                    ui.label("New Input");
                    pin_info.fill = Some(egui::Color32::from_rgb(100, 100, 100)); // Grey
                }
            }
        }
        pin_info
    }

    fn show_output(
        &mut self,
        pin: &OutPin,
        ui: &mut egui::Ui,
        snarl: &mut Snarl<MyNode>,
    ) -> impl SnarlPin + 'static {
        let node = &snarl[pin.id.node];
        let mut pin_info = PinInfo::circle();

        match node {
            MyNode::Model { .. } => {
                ui.label("Model");
                pin_info.fill = Some(egui::Color32::from_rgb(238, 207, 109));
            }
            MyNode::Agent { .. } => {
                ui.label("Output Dir");
                pin_info.fill = Some(egui::Color32::from_rgb(38, 211, 109)); // Output green
            }
        }
        pin_info
    }

    fn connect(&mut self, from: &OutPin, to: &InPin, snarl: &mut Snarl<MyNode>) {
        let source_node = &snarl[from.id.node];
        let dest_node = &snarl[to.id.node];

        match (source_node, dest_node) {
            (MyNode::Model { .. }, MyNode::Agent { .. }) => {
                if to.id.input == 0 {
                    snarl.connect(from.id, to.id);
                }
            }
            (MyNode::Agent { output_dir, .. }, MyNode::Agent { .. }) => {
                if to.id.input > 0 {
                    // If connecting to the "New Input" pin, add it to the inputs vector
                    let output_path = output_dir.clone();
                    let dest_node_mut = &mut snarl[to.id.node];
                    if let MyNode::Agent { inputs, .. } = dest_node_mut {
                        if to.id.input > inputs.len() {
                            inputs.push(output_path);
                            // The actual connection will be to the newly created pin
                            let new_in_pin = InPinId {
                                node: to.id.node,
                                input: inputs.len(),
                            };
                            snarl.connect(from.id, new_in_pin);
                        } else {
                            // Connecting to an existing input pin
                            snarl.connect(from.id, to.id);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn disconnect(&mut self, from: &OutPin, to: &InPin, snarl: &mut Snarl<MyNode>) {
        snarl.disconnect(from.id, to.id);
    }
}

struct SnarlApp {
    snarl: Snarl<MyNode>,
    style: SnarlStyle,
    hf_models: Vec<String>,
    status: String,
}

impl SnarlApp {
    fn new() -> Self {
        Self {
            snarl: Snarl::new(),
            style: SnarlStyle::new(),
            hf_models: scan_hf_cache(),
            status: "Ready".to_string(),
        }
    }

    fn export_config(&self) -> Config {
        let mut config = Config {
            models: HashMap::new(),
            sampling: SamplingConfig {
                temperature: Some(0.7),
                ..Default::default()
            },
            agents: HashMap::new(),
            compression: CompressionConfig {
                threshold: 0.5,
                inverse_probability: 0.9,
                resummarize_probability: 0.1,
            },
            shutdown_on_idle: false,
        };

        let mut node_to_model_alias: HashMap<NodeId, String> = HashMap::new();

        // Pass 1: Gather Models
        for (node_id, node) in self.snarl.node_ids() {
            if let MyNode::Model {
                name,
                model_id,
                path,
                gguf,
                isq,
                dtype,
                builder,
                chat_template,
            } = node
            {
                let alias = if name.is_empty() {
                    format!("model_{:?}", node_id)
                } else {
                    name.clone()
                };
                config.models.insert(
                    alias.clone(),
                    ModelConfig {
                        id: model_id.clone(),
                        path: path.clone(),
                        gguf: gguf.clone(),
                        isq: isq.clone(),
                        dtype: dtype.clone(),
                        builder: builder.clone(),
                        chat_template: chat_template.clone(),
                    },
                );
                node_to_model_alias.insert(node_id, alias);
            }
        }

        // Pass 2: Gather Agents and their connections
        for (node_id, node) in self.snarl.node_ids() {
            if let MyNode::Agent {
                name,
                inputs,
                output_dir,
                system,
                history_limit,
                stream,
                allowed_extensions,
                prompt,
            } = node
            {
                let agent_name = if name.is_empty() {
                    format!("agent_{:?}", node_id)
                } else {
                    name.clone()
                };

                let mut model_alias = String::new();

                // Get model connection from input pin 0
                let in_pin_0 = self.snarl.in_pin(InPinId {
                    node: node_id,
                    input: 0,
                });
                for remote in in_pin_0.remotes {
                    if let Some(alias) = node_to_model_alias.get(&remote.node) {
                        model_alias = alias.clone();
                        break;
                    }
                }

                // Gather inputs: prefer connected agent's output_dir, fallback to manual path
                let mut resolved_inputs = Vec::new();
                for (i, manual_path) in inputs.iter().enumerate() {
                    let pin_idx = i + 1;
                    let in_pin = self.snarl.in_pin(InPinId {
                        node: node_id,
                        input: pin_idx,
                    });

                    let mut found_remote = false;
                    for remote in in_pin.remotes {
                        let remote_node = &self.snarl[remote.node];
                        if let MyNode::Agent {
                            output_dir: remote_output_dir,
                            ..
                        } = remote_node
                        {
                            resolved_inputs.push(remote_output_dir.clone());
                            found_remote = true;
                            break;
                        }
                    }

                    if !found_remote {
                        resolved_inputs.push(manual_path.clone());
                    }
                }

                config.agents.insert(
                    agent_name,
                    AgentConfig {
                        inputs: resolved_inputs,
                        output: output_dir.clone(),
                        system: system.clone(),
                        model: model_alias,
                        history_limit: *history_limit,
                        stream: *stream,
                        allowed_extensions: allowed_extensions.clone(),
                        prompt: prompt.clone(),
                    },
                );
            }
        }

        config
    }
}

impl eframe::App for SnarlApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.button("Add Model").clicked() {
                    self.snarl.insert_node(
                        egui::Pos2::ZERO,
                        MyNode::Model {
                            name: String::new(),
                            model_id: String::new(),
                            path: None,
                            gguf: None,
                            isq: None,
                            dtype: None,
                            builder: "mistralrs".to_string(),
                            chat_template: None,
                        },
                    );
                }
                if ui.button("Add Agent").clicked() {
                    self.snarl.insert_node(
                        egui::Pos2::ZERO,
                        MyNode::Agent {
                            name: String::new(),
                            inputs: Vec::new(),
                            output_dir: String::new(),
                            system: Vec::new(),
                            history_limit: None,
                            stream: true,
                            allowed_extensions: Vec::new(),
                            prompt: None,
                        },
                    );
                }
                ui.separator();
                if ui.button("Spawn Leader").clicked() {
                    let config = self.export_config();
                    match spawn_leader(config) {
                        Ok(_) => self.status = "Leader Spawned".to_string(),
                        Err(e) => self.status = format!("Error: {}", e),
                    }
                }
                ui.label(&self.status);
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            self.snarl.show(
                &mut DemoViewer {
                    hf_models: self.hf_models.clone(),
                },
                &self.style,
                "snarl",
                ui,
            );
        });
    }
}

fn spawn_leader(config: Config) -> anyhow::Result<()> {
    let config_path = std::env::temp_dir().join("ag-config.yaml");
    let yaml = serde_yaml::to_string(&config)?;
    std::fs::write(&config_path, yaml)?;

    std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let status = tokio::process::Command::new("ag")
                .arg("--config")
                .arg(&config_path)
                .status()
                .await;

            match status {
                Ok(s) => println!("Leader exited with status: {:?}", s),
                Err(e) => eprintln!("Failed to spawn leader: {:?}", e),
            }
        });
    });
    Ok(())
}

fn scan_hf_cache() -> Vec<String> {
    let mut models = Vec::new();
    if let Ok(home) = std::env::var("HOME") {
        let cache_base = std::path::Path::new(&home).join(".cache/huggingface/hub");
        if let Ok(entries) = std::fs::read_dir(cache_base) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with("models--") {
                    let hf_id = name.trim_start_matches("models--").replace("--", "/");
                    models.push(hf_id);
                }
            }
        }
    }
    models
}

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "AgentGraph Designer",
        options,
        Box::new(|_cc| Ok(Box::new(SnarlApp::new()))),
    )
}
