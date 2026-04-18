use ag_config::{AgentConfig, CompressionConfig, Config, ModelConfig, SamplingConfig};
use eframe::egui;
use egui_snarl::{
    InPin, InPinId, NodeId, OutPin, OutPinId, Snarl,
    ui::{PinInfo, SnarlPin, SnarlStyle, SnarlViewer},
};
use std::collections::HashMap;

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
enum MyNode {
    Model {
        name: String,
        config: ModelConfig,
    },
    Agent {
        name: String,
        config: AgentConfig,
    },
    GlobalConfig {
        sampling: SamplingConfig,
        compression: CompressionConfig,
        shutdown_on_idle: bool,
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
            MyNode::GlobalConfig { .. } => "Global Config".to_string(),
        }
    }

    fn outputs(&mut self, node: &MyNode) -> usize {
        match node {
            MyNode::Model { .. } => 1,
            MyNode::Agent { .. } => 1,
            MyNode::GlobalConfig { .. } => 0,
        }
    }

    fn inputs(&mut self, node: &MyNode) -> usize {
        match node {
            MyNode::Model { .. } => 0,
            MyNode::Agent { config, .. } => config.inputs.len() + 2,
            MyNode::GlobalConfig { .. } => 0,
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
        let node = &mut snarl[node_id];
        match node {
            MyNode::Model {
                name,
                config:
                    ModelConfig {
                        id,
                        path,
                        gguf,
                        isq,
                        dtype,
                        builder,
                        chat_template,
                    },
            } => {
                ui.set_width(300.0);
                ui.vertical(|ui| {
                    ui.horizontal(|ui| {
                        ui.label("Name:");
                        ui.text_edit_singleline(name);
                    });
                    ui.horizontal(|ui| {
                        ui.label("ID:");
                        egui::ComboBox::from_id_salt(format!("model-id-{:?}", node_id))
                            .selected_text(&*id)
                            .show_ui(ui, |ui| {
                                for m in &self.hf_models {
                                    ui.selectable_value(id, m.clone(), m);
                                }
                            });
                    });
                    ui.horizontal(|ui| {
                        ui.label("Builder:");
                        ui.text_edit_singleline(builder);
                    });

                    ui.separator();

                    ui.collapsing("Advanced Fields", |ui| {
                        egui::Grid::new(format!("model_grid_{:?}", node_id))
                            .num_columns(2)
                            .spacing([10.0, 4.0])
                            .show(ui, |ui| {
                                ui.label("Path:");
                                let mut p = path.clone().unwrap_or_default();
                                if ui.text_edit_singleline(&mut p).changed() {
                                    *path = if p.is_empty() { None } else { Some(p) };
                                }
                                ui.end_row();

                                ui.label("GGUF:");
                                let mut g = gguf.clone().unwrap_or_default();
                                if ui.text_edit_singleline(&mut g).changed() {
                                    *gguf = if g.is_empty() { None } else { Some(g) };
                                }
                                ui.end_row();

                                ui.label("ISQ:");
                                let mut i = isq.clone().unwrap_or_default();
                                if ui.text_edit_singleline(&mut i).changed() {
                                    *isq = if i.is_empty() { None } else { Some(i) };
                                }
                                ui.end_row();

                                ui.label("DType:");
                                let mut d = dtype.clone().unwrap_or_default();
                                if ui.text_edit_singleline(&mut d).changed() {
                                    *dtype = if d.is_empty() { None } else { Some(d) };
                                }
                                ui.end_row();

                                ui.label("Template:");
                                let mut t = chat_template.clone().unwrap_or_default();
                                if ui.text_edit_singleline(&mut t).changed() {
                                    *chat_template = if t.is_empty() { None } else { Some(t) };
                                }
                                ui.end_row();
                            });
                    });
                });
            }
            MyNode::Agent {
                name,
                config:
                    AgentConfig {
                        inputs,
                        output,
                        system,
                        model: _, // Model is driven by wire
                        history_limit,
                        stream,
                        allowed_extensions,
                        prompt,
                    },
            } => {
                ui.set_width(500.0);
                ui.vertical(|ui| {
                    ui.horizontal(|ui| {
                        ui.label("Name:");
                        ui.text_edit_singleline(name);
                    });

                    ui.separator();

                    ui.columns(2, |columns| {
                        columns[0].vertical(|ui| {
                            ui.label(egui::RichText::new("Inputs").strong());
                            let mut to_remove = None;
                            for (i, input) in inputs.iter_mut().enumerate() {
                                ui.horizontal(|ui| {
                                    ui.text_edit_singleline(input);
                                    if ui.button("📁").clicked() {
                                        if let Some(path) = rfd::FileDialog::new().pick_folder() {
                                            *input = path.display().to_string();
                                        }
                                    }
                                    if ui.button("❌").clicked() {
                                        to_remove = Some(i);
                                    }
                                });
                            }
                            if let Some(i) = to_remove {
                                inputs.remove(i);
                            }
                            if ui.button("➕ Add Input").clicked() {
                                inputs.push(String::new());
                            }
                        });

                        columns[1].vertical(|ui| {
                            ui.label(egui::RichText::new("Output").strong());
                            ui.horizontal(|ui| {
                                ui.text_edit_singleline(output);
                                if ui.button("📁").clicked() {
                                    if let Some(path) = rfd::FileDialog::new().pick_folder() {
                                        *output = path.display().to_string();
                                    }
                                }
                            });

                            ui.add_space(8.0);
                            ui.label(egui::RichText::new("System Prompt Dirs").strong());
                            let mut to_remove_sys = None;
                            for (i, s) in system.iter_mut().enumerate() {
                                ui.horizontal(|ui| {
                                    ui.text_edit_singleline(s);
                                    if ui.button("📁").clicked() {
                                        if let Some(path) = rfd::FileDialog::new().pick_folder() {
                                            *s = path.display().to_string();
                                        }
                                    }
                                    if ui.button("❌").clicked() {
                                        to_remove_sys = Some(i);
                                    }
                                });
                            }
                            if let Some(i) = to_remove_sys {
                                system.remove(i);
                            }
                            if ui.button("➕ Add System Dir").clicked() {
                                system.push(String::new());
                            }
                        });
                    });

                    ui.separator();

                    ui.collapsing("Advanced Configuration", |ui| {
                        ui.horizontal(|ui| {
                            ui.checkbox(stream, "Stream");
                            ui.label("History:");
                            let mut hl = history_limit.unwrap_or(0);
                            if ui.add(egui::DragValue::new(&mut hl)).changed() {
                                *history_limit = if hl == 0 { None } else { Some(hl) };
                            }
                        });

                        ui.add_space(4.0);
                        ui.label("Allowed Extensions (comma separated):");
                        let mut ext_str = allowed_extensions.join(", ");
                        if ui.text_edit_singleline(&mut ext_str).changed() {
                            *allowed_extensions = ext_str
                                .split(',')
                                .map(|s| s.trim().to_string())
                                .filter(|s| !s.is_empty())
                                .collect();
                        }

                        ui.add_space(8.0);
                        ui.label("Addendum / Inline Prompt:");
                        let mut p = prompt.clone().unwrap_or_default();
                        if ui
                            .add(
                                egui::TextEdit::multiline(&mut p)
                                    .hint_text("Enter additional prompt context here...")
                                    .desired_rows(4)
                                    .desired_width(480.0),
                            )
                            .changed()
                        {
                            *prompt = if p.is_empty() { None } else { Some(p) };
                        }
                    });
                });
            }
            MyNode::GlobalConfig {
                sampling,
                compression,
                shutdown_on_idle,
            } => {
                ui.set_width(300.0);
                ui.vertical(|ui| {
                    ui.checkbox(shutdown_on_idle, "Shutdown on Idle");

                    ui.separator();
                    ui.label(egui::RichText::new("Sampling").strong());
                    egui::Grid::new(format!("sampling_grid_{:?}", node_id))
                        .num_columns(2)
                        .show(ui, |ui| {
                            ui.label("Temp:");
                            let mut t = sampling.temperature.unwrap_or(0.7);
                            if ui
                                .add(egui::DragValue::new(&mut t).speed(0.1).range(0.0..=2.0))
                                .changed()
                            {
                                sampling.temperature = Some(t);
                            }
                            ui.end_row();

                            ui.label("Top P:");
                            let mut p = sampling.top_p.unwrap_or(1.0);
                            if ui
                                .add(egui::DragValue::new(&mut p).speed(0.01).range(0.0..=1.0))
                                .changed()
                            {
                                sampling.top_p = Some(p);
                            }
                            ui.end_row();

                            ui.label("Max Len:");
                            let mut ml = sampling.max_len.unwrap_or(0);
                            if ui.add(egui::DragValue::new(&mut ml)).changed() {
                                sampling.max_len = if ml == 0 { None } else { Some(ml) };
                            }
                            ui.end_row();
                        });

                    ui.separator();
                    ui.label(egui::RichText::new("Compression").strong());
                    egui::Grid::new(format!("compression_grid_{:?}", node_id))
                        .num_columns(2)
                        .show(ui, |ui| {
                            ui.label("Threshold:");
                            ui.add(
                                egui::DragValue::new(&mut compression.threshold)
                                    .speed(0.05)
                                    .range(0.0..=1.0),
                            );
                            ui.end_row();

                            ui.label("Inv Prob:");
                            ui.add(
                                egui::DragValue::new(&mut compression.inverse_probability)
                                    .speed(0.05)
                                    .range(0.0..=1.0),
                            );
                            ui.end_row();
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
            MyNode::Agent { config, .. } => {
                if pin.id.input == 0 {
                    ui.label("Model");
                    pin_info.fill = Some(egui::Color32::from_rgb(238, 207, 109));
                } else if pin.id.input <= config.inputs.len() {
                    ui.label(format!("In {}", pin.id.input));
                    pin_info.fill = Some(egui::Color32::from_rgb(38, 109, 211));
                } else {
                    ui.label("New Link");
                    pin_info.fill = Some(egui::Color32::from_rgb(100, 100, 100));
                }
            }
            MyNode::GlobalConfig { .. } => {}
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
                ui.label("Model Out");
                pin_info.fill = Some(egui::Color32::from_rgb(238, 207, 109));
            }
            MyNode::Agent { .. } => {
                ui.label("Out Dir");
                pin_info.fill = Some(egui::Color32::from_rgb(38, 211, 109));
            }
            MyNode::GlobalConfig { .. } => {}
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
            (
                MyNode::Agent {
                    config: source_config,
                    ..
                },
                MyNode::Agent { .. },
            ) => {
                if to.id.input > 0 {
                    let output_path = source_config.output.clone();
                    let dest_node_mut = &mut snarl[to.id.node];
                    if let MyNode::Agent {
                        config: dest_config,
                        ..
                    } = dest_node_mut
                    {
                        if to.id.input > dest_config.inputs.len() {
                            dest_config.inputs.push(output_path);
                            let new_in_pin = InPinId {
                                node: to.id.node,
                                input: dest_config.inputs.len(),
                            };
                            snarl.connect(from.id, new_in_pin);
                        } else {
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
            sampling: SamplingConfig::default(),
            agents: HashMap::new(),
            compression: CompressionConfig {
                threshold: 0.5,
                inverse_probability: 0.9,
                resummarize_probability: 0.1,
            },
            shutdown_on_idle: false,
        };

        let mut node_to_model_alias: HashMap<NodeId, String> = HashMap::new();

        // Pass 0: Global Config
        for (_, node) in self.snarl.node_ids() {
            if let MyNode::GlobalConfig {
                sampling,
                compression,
                shutdown_on_idle,
            } = node
            {
                config.sampling = sampling.clone();
                config.compression = compression.clone();
                config.shutdown_on_idle = *shutdown_on_idle;
                break;
            }
        }

        // Pass 1: Models
        for (node_id, node) in self.snarl.node_ids() {
            if let MyNode::Model {
                name,
                config: m_config,
            } = node
            {
                let alias = if name.is_empty() {
                    format!("model_{:?}", node_id)
                } else {
                    name.clone()
                };
                config.models.insert(alias.clone(), m_config.clone());
                node_to_model_alias.insert(node_id, alias);
            }
        }

        // Pass 2: Agents
        for (node_id, node) in self.snarl.node_ids() {
            if let MyNode::Agent {
                name,
                config: a_config,
            } = node
            {
                let agent_name = if name.is_empty() {
                    format!("agent_{:?}", node_id)
                } else {
                    name.clone()
                };

                let mut model_alias = String::new();
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

                let mut resolved_inputs = Vec::new();
                for (i, manual_path) in a_config.inputs.iter().enumerate() {
                    let pin_idx = i + 1;
                    let in_pin = self.snarl.in_pin(InPinId {
                        node: node_id,
                        input: pin_idx,
                    });

                    let mut found_remote = false;
                    for remote in in_pin.remotes {
                        let remote_node = &self.snarl[remote.node];
                        if let MyNode::Agent {
                            config: remote_config,
                            ..
                        } = remote_node
                        {
                            resolved_inputs.push(remote_config.output.clone());
                            found_remote = true;
                            break;
                        }
                    }
                    if !found_remote {
                        resolved_inputs.push(manual_path.clone());
                    }
                }

                let mut final_agent_config = a_config.clone();
                final_agent_config.inputs = resolved_inputs;
                final_agent_config.model = model_alias;

                config.agents.insert(agent_name, final_agent_config);
            }
        }
        config
    }

    fn save_to_file(&self) -> anyhow::Result<()> {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("YAML", &["yaml", "yml"])
            .save_file()
        {
            let config = self.export_config();
            let yaml = serde_yaml::to_string(&config)?;
            std::fs::write(path, yaml)?;
            Ok(())
        } else {
            Err(anyhow::anyhow!("Save cancelled"))
        }
    }

    fn load_from_file(&mut self) -> anyhow::Result<()> {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("YAML", &["yaml", "yml"])
            .pick_file()
        {
            let config = Config::load(&path)?;
            self.snarl = Snarl::new();

            let mut model_alias_to_id = HashMap::new();
            let mut agent_name_to_id = HashMap::new();

            // 1. Global Config
            self.snarl.insert_node(
                egui::Pos2::new(0.0, -200.0),
                MyNode::GlobalConfig {
                    sampling: config.sampling.clone(),
                    compression: config.compression.clone(),
                    shutdown_on_idle: config.shutdown_on_idle,
                },
            );

            // 2. Models
            let mut y_offset = 0.0;
            for (alias, m) in &config.models {
                let id = self.snarl.insert_node(
                    egui::Pos2::new(0.0, y_offset),
                    MyNode::Model {
                        name: alias.clone(),
                        config: m.clone(),
                    },
                );
                model_alias_to_id.insert(alias.clone(), id);
                y_offset += 300.0;
            }

            // 3. Agents
            let mut x_offset = 500.0;
            y_offset = 0.0;
            for (name, a) in &config.agents {
                let id = self.snarl.insert_node(
                    egui::Pos2::new(x_offset, y_offset),
                    MyNode::Agent {
                        name: name.clone(),
                        config: a.clone(),
                    },
                );
                agent_name_to_id.insert(name.clone(), id);
                y_offset += 500.0;
                if y_offset > 1500.0 {
                    y_offset = 0.0;
                    x_offset += 600.0;
                }
            }

            // 4. Connections
            for (name, a) in &config.agents {
                let agent_id = agent_name_to_id[name];

                // Model connection
                if let Some(&model_id) = model_alias_to_id.get(&a.model) {
                    self.snarl.connect(
                        OutPinId {
                            node: model_id,
                            output: 0,
                        },
                        InPinId {
                            node: agent_id,
                            input: 0,
                        },
                    );
                }

                // Input connections
                for (i, input_path) in a.inputs.iter().enumerate() {
                    for (other_name, other_a) in &config.agents {
                        if other_a.output == *input_path {
                            let other_id = agent_name_to_id[other_name];
                            self.snarl.connect(
                                OutPinId {
                                    node: other_id,
                                    output: 0,
                                },
                                InPinId {
                                    node: agent_id,
                                    input: i + 1,
                                },
                            );
                            break;
                        }
                    }
                }
            }
            Ok(())
        } else {
            Err(anyhow::anyhow!("Load cancelled"))
        }
    }
}

impl eframe::App for SnarlApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.button("📁 Load").clicked() {
                    if let Err(e) = self.load_from_file() {
                        self.status = format!("Load Error: {}", e);
                    } else {
                        self.status = "Config Loaded".to_string();
                    }
                }
                if ui.button("💾 Save").clicked() {
                    if let Err(e) = self.save_to_file() {
                        self.status = format!("Save Error: {}", e);
                    } else {
                        self.status = "Config Saved".to_string();
                    }
                }
                ui.separator();
                if ui.button("➕ Model").clicked() {
                    self.snarl.insert_node(
                        egui::Pos2::ZERO,
                        MyNode::Model {
                            name: String::new(),
                            config: ModelConfig {
                                id: String::new(),
                                path: None,
                                gguf: None,
                                isq: None,
                                dtype: None,
                                builder: "vision".to_string(),
                                chat_template: None,
                            },
                        },
                    );
                }
                if ui.button("➕ Agent").clicked() {
                    self.snarl.insert_node(
                        egui::Pos2::ZERO,
                        MyNode::Agent {
                            name: String::new(),
                            config: AgentConfig {
                                inputs: Vec::new(),
                                output: String::new(),
                                system: Vec::new(),
                                model: String::new(),
                                history_limit: None,
                                stream: true,
                                allowed_extensions: Vec::new(),
                                prompt: None,
                            },
                        },
                    );
                }

                let has_global = self
                    .snarl
                    .node_ids()
                    .any(|(_, node)| matches!(node, MyNode::GlobalConfig { .. }));

                ui.add_enabled_ui(!has_global, |ui| {
                    if ui
                        .button("➕ Global")
                        .on_disabled_hover_text("Global config already exists")
                        .clicked()
                    {
                        self.snarl.insert_node(
                            egui::Pos2::ZERO,
                            MyNode::GlobalConfig {
                                sampling: SamplingConfig::default(),
                                compression: CompressionConfig {
                                    threshold: 0.5,
                                    inverse_probability: 0.9,
                                    resummarize_probability: 0.1,
                                },
                                shutdown_on_idle: false,
                            },
                        );
                    }
                });
                ui.separator();
                if ui.button("🚀 Spawn").clicked() {
                    let config = self.export_config();
                    match spawn_leader(config) {
                        Ok(_) => self.status = "Leader Spawned".to_string(),
                        Err(e) => self.status = format!("Spawn Error: {}", e),
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
