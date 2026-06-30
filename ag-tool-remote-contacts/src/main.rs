use ag_tool_common::{describe, guidance, has_flag, read_args};
use serde::Deserialize;
use serde_json::json;
use std::collections::HashMap;
use std::path::PathBuf;

const CONTACTS_DIR: &str = ".agentgraph/contacts";
const CONTACTS_FILE: &str = "contacts.json";

#[derive(Deserialize)]
struct Args {
    operation: String,
    /// Contact name (for add_contact, initiate_session, send_message)
    name: Option<String>,
    /// Remote identifier (for add_contact)
    remote_id: Option<String>,
    /// Free-form metadata (for add_contact)
    metadata: Option<String>,
    /// Message to send (for initiate_session, send_message)
    message: Option<String>,
    /// Comma-separated file paths to media files (for send_message)
    media: Option<String>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct ContactEntry {
    name: String,
    remote_id: String,
    metadata: Option<String>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct ContactBook {
    contacts: HashMap<String, ContactEntry>,
}

fn home_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp"))
}

fn contacts_path() -> PathBuf {
    home_dir().join(CONTACTS_DIR).join(CONTACTS_FILE)
}

fn load_contacts() -> ContactBook {
    let path = contacts_path();
    match std::fs::read_to_string(&path) {
        Ok(json) => serde_json::from_str(&json).unwrap_or(ContactBook {
            contacts: HashMap::new(),
        }),
        Err(_) => ContactBook {
            contacts: HashMap::new(),
        },
    }
}

fn save_contacts(book: &ContactBook) {
    let path = contacts_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(json) = serde_json::to_string_pretty(book) {
        let _ = std::fs::write(&path, &json);
    }
}

/// Send an IPC command to the leader via Unix socket and return the response.
fn send_ipc(cmd: &str) -> Result<String, String> {
    use std::io::Read;
    use std::io::Write;
    use std::os::unix::net::UnixStream;

    // Find the leader socket.
    let sock_dir = PathBuf::from("/tmp/agentgraph");
    let entries = std::fs::read_dir(&sock_dir).map_err(|e| format!("read_dir: {e}"))?;
    let mut socket_path: Option<PathBuf> = None;
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with("ag-") && name.ends_with(".sock") {
            let pid_str = name
                .strip_prefix("ag-")
                .and_then(|s| s.strip_suffix(".sock"))
                .unwrap_or("");
            if let Ok(pid) = pid_str.parse::<u32>() {
                let status_path = format!("/proc/{}/status", pid);
                if std::path::Path::new(&status_path).exists() {
                    socket_path = Some(entry.path());
                    break;
                }
            }
        }
    }

    let sock = socket_path.ok_or_else(|| "Leader socket not found".to_string())?;
    let mut stream = UnixStream::connect(&sock).map_err(|e| format!("connect: {e}"))?;
    stream.write_all(cmd.as_bytes()).map_err(|e| format!("write: {e}"))?;
    stream.shutdown(std::net::Shutdown::Write).map_err(|e| format!("shutdown: {e}"))?;

    let mut resp = String::new();
    stream.read_to_string(&mut resp).map_err(|e| format!("read: {e}"))?;
    Ok(resp)
}

fn main() {
    if has_flag("--describe") {
        describe(
            "remote_contacts",
            "Manage the agent's contact book and initiate remote sessions with contacts. \
             Use 'list_contacts' to see available contacts, 'add_contact' to add a new one, \
             'send_message' to send a message (with optional media) to a contact's session, \
             and 'initiate_session' to start a remote session with a contact.",
            json!({
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["list_contacts", "add_contact", "initiate_session", "send_message"],
                        "description": "Which operation to perform."
                    },
                    "name": {
                        "type": "string",
                        "description": "Contact name (required for add_contact, initiate_session, send_message)."
                    },
                    "remote_id": {
                        "type": "string",
                        "description": "Remote identifier (required for add_contact)."
                    },
                    "metadata": {
                        "type": "string",
                        "description": "Optional free-form metadata for the contact."
                    },
                    "message": {
                        "type": "string",
                        "description": "Message text to send (for initiate_session, send_message)."
                    },
                    "media": {
                        "type": "string",
                        "description": "Comma-separated file paths to attach as media (for send_message)."
                    }
                },
                "required": ["operation"]
            }),
        );
        return;
    }

    if has_flag("--help") {
        guidance(
            "The remote_contacts tool manages your contact book and lets you initiate \
             remote sessions with known contacts.\n\n\
             Available operations:\n\
             - list_contacts: Show all contacts (name, remote_id, metadata).\n\
             - add_contact: Add a new contact. Requires 'name' and 'remote_id'.\n\
             - send_message: Send a message (with optional media) to a contact's session. \
               Requires 'name'. Optionally 'message' for text and 'media' for file paths.\n\
             - initiate_session: Start a remote session with an existing contact. \
               Requires 'name' and optionally 'message'.\n\n\
             Contacts are persistent and shared across all agents.\n\
             Use send_message to proactively message a contact with text and/or media. \
             Use initiate_session to begin chatting with a remote contact without \
             waiting for them to initiate.",
        );
        return;
    }

    let args: Args = read_args().unwrap_or_else(|e| {
        eprintln!("Error: {e}");
        std::process::exit(1);
    });

    match args.operation.as_str() {
        "list_contacts" => {
            let book = load_contacts();
            let mut entries: Vec<&ContactEntry> = book.contacts.values().collect();
            entries.sort_by(|a, b| a.name.cmp(&b.name));
            if entries.is_empty() {
                println!("No contacts in contact book.");
            } else {
                let result: Vec<serde_json::Value> = entries
                    .iter()
                    .map(|e| {
                        json!({
                            "name": e.name,
                            "remote_id": e.remote_id,
                            "metadata": e.metadata,
                        })
                    })
                    .collect();
                println!("{}", serde_json::to_string_pretty(&result).unwrap());
            }
        }

        "add_contact" => {
            let name = args.name.as_deref().unwrap_or("");
            let remote_id = args.remote_id.as_deref().unwrap_or("");
            if name.is_empty() || remote_id.is_empty() {
                println!("Error: 'name' and 'remote_id' are required for add_contact.");
                std::process::exit(1);
            }
            let mut book = load_contacts();
            book.contacts.insert(
                name.to_string(),
                ContactEntry {
                    name: name.to_string(),
                    remote_id: remote_id.to_string(),
                    metadata: args.metadata.clone(),
                },
            );
            save_contacts(&book);
            println!("Contact '{}' added with remote_id '{}'.", name, remote_id);
        }

        "send_message" => {
            let name = args.name.as_deref().unwrap_or("");
            if name.is_empty() {
                println!("Error: 'name' is required for send_message.");
                std::process::exit(1);
            }

            let book = load_contacts();
            let contact = match book.contacts.get(name) {
                Some(c) => c,
                None => {
                    println!("Error: contact '{}' not found. Use add_contact first.", name);
                    std::process::exit(1);
                }
            };

            let message = args.message.as_deref().unwrap_or("");
            let media: Vec<String> = args
                .media
                .as_deref()
                .unwrap_or("")
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();

            let send_cmd = serde_json::json!({
                "SessionContactSendMessage": {
                    "contact": name,
                    "message": message,
                    "media": media,
                }
            });

            match send_ipc(&serde_json::to_string(&send_cmd).unwrap()) {
                Ok(resp) => {
                    let media_info = if media.is_empty() {
                        String::new()
                    } else {
                        format!(" with {} media file(s)", media.len())
                    };
                    println!(
                        "Message sent to '{}' (remote_id: {}){}. Response: {}",
                        name, contact.remote_id, media_info, resp
                    );
                }
                Err(e) => {
                    println!("Error sending message to '{}': {}", name, e);
                }
            }
        }

        "initiate_session" => {
            let name = args.name.as_deref().unwrap_or("");
            if name.is_empty() {
                println!("Error: 'name' is required for initiate_session.");
                std::process::exit(1);
            }

            let book = load_contacts();
            let contact = match book.contacts.get(name) {
                Some(c) => c,
                None => {
                    println!("Error: contact '{}' not found. Use add_contact first.", name);
                    std::process::exit(1);
                }
            };

            let message = args.message.as_deref().unwrap_or("Hello! Starting a new session.");
            let initiate_cmd = serde_json::json!({
                "SessionContactInitiate": {
                    "contact": name,
                    "message": message,
                }
            });

            match send_ipc(&serde_json::to_string(&initiate_cmd).unwrap()) {
                Ok(resp) => {
                    println!(
                        "Initiated session with '{}' (remote_id: {}). Response: {}",
                        name, contact.remote_id, resp
                    );
                }
                Err(e) => {
                    println!("Error initiating session with '{}': {}", name, e);
                }
            }
        }

        other => {
            println!("Error: Unknown operation '{}'. Use list_contacts, add_contact, send_message, or initiate_session.", other);
            std::process::exit(1);
        }
    }
}
