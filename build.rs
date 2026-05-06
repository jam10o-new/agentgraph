fn main() {
    // Embed the short git commit hash so version strings include it.
    if let Ok(output) = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
    {
        if output.status.success() {
            let hash = String::from_utf8_lossy(&output.stdout).trim().to_string();
            println!("cargo:rustc-env=GIT_COMMIT_HASH={}", hash);
        }
    }

    // Detect dirty working tree.
    if let Ok(status) = std::process::Command::new("git")
        .args(["status", "--porcelain"])
        .output()
    {
        if status.status.success() && !status.stdout.is_empty() {
            println!("cargo:rustc-env=GIT_DIRTY=1");
        }
    }

    // Re-run the build script whenever the git index changes
    // (commits, adds, etc.) so the embedded hash stays current.
    println!("cargo:rerun-if-changed=.git/index");
}
