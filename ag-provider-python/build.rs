use std::env;
use std::fs;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=python_backend.py");

    let script = "python_backend.py";
    if !Path::new(script).exists() {
        return;
    }

    // Copy the Python script to the output directory alongside the binary.
    // resolve_script() in main.rs looks for the script next to the binary.
    let out_dir = env::var("OUT_DIR").unwrap();
    let dest = Path::new(&out_dir).join(script);
    fs::copy(script, &dest).expect("copy python_backend.py to OUT_DIR");
}
