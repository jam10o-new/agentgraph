import os
import subprocess
import time
import signal
import shutil

def run_repro():
    print("--- Cleaning up existing agentgraph processes ---")
    subprocess.run(["pkill", "-9", "agentgraph"], stderr=subprocess.DEVNULL)
    
    project_root = os.getcwd()
    input_dir = os.path.join(project_root, "test_vision/agent_a/input")
    system_dir = os.path.join(project_root, "test_vision/agent_a/system")
    output_dir = os.path.join(project_root, "test_vision/agent_a/output")
    log_file = os.path.join(project_root, "repro.log")

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    if os.path.exists(log_file):
        os.remove(log_file)

    print("--- Starting AgentGraph Leader ---")
    binary = os.path.join(project_root, "target/debug/agentgraph")
    
    if not os.path.exists(binary):
        print(f"Binary not found at {binary}. Please build the project first.")
        return

    cmd = [
        binary,
        "-I", input_dir,
        "-S", system_dir,
        "-O", output_dir,
        "--latest-n", "1",
        "--verbose",
        "--no-ui",
        "-W"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    with open(log_file, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, preexec_fn=os.setsid)

    print(f"AgentGraph started with PID {proc.pid}")
    
    print("Waiting 15s for initial startup...")
    time.sleep(15)

    print("--- Triggering Inference (Touch) ---")
    screenshots = [f for f in os.listdir(input_dir) if f.endswith(".png")]
    if not screenshots:
        print(f"No screenshots found in {input_dir} to touch.")
        # Create a dummy one if needed? No, let's assume they exist as seen before.
    else:
        target_file = os.path.join(input_dir, screenshots[0])
        print(f"Touching {target_file}")
        subprocess.run(["touch", target_file])

    print("--- Polling for output (max 300s) ---")
    start_time = time.time()
    found_output = False
    
    while time.time() - start_time < 300:
        outputs = [f for f in os.listdir(output_dir) if f.endswith(".txt")]
        for out in outputs:
            p = os.path.join(output_dir, out)
            if os.path.getsize(p) > 0:
                print(f"Found non-empty output: {out}")
                found_output = True
                break
        if found_output:
            break
        
        # Check if process is still alive
        if proc.poll() is not None:
            print("Process exited unexpectedly.")
            break
            
        time.sleep(2)

    if not found_output:
        print("Timed out waiting for non-empty output.")

    print("--- Killing AgentGraph ---")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        time.sleep(2)
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass
    
    print("--- Analyzing Results ---")
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            log_content = f.read()
            # print(log_content) # Optional: print full log if needed
            
            checks = [
                ("Running inference after filesystem change", "Inference Triggered"),
                ("Models Built and Ready", "Model Loaded"),
                ("Messages built successfully", "Messages Formatted"),
                ("Primary model request created", "Request Initiated"),
                ("Stream created successfully", "Inference Started"),
                ("Primary response complete", "Inference Completed")
            ]
            
            print("\nLOG SUMMARY:")
            for pattern, label in checks:
                if pattern in log_content:
                    print(f"[PASS] {label}")
                else:
                    print(f"[FAIL] {label}")

            if "Err:" in log_content or "Stream creation failed" in log_content:
                print("[ERROR] Detected error in logs:")
                for line in log_content.splitlines():
                    if "Err:" in line or "Stream creation failed" in line:
                        print(f"  >> {line}")

    if os.path.exists(output_dir):
        outputs = os.listdir(output_dir)
        print(f"\nOutputs found in {output_dir}: {len(outputs)}")
        for out in outputs:
            p = os.path.join(output_dir, out)
            size = os.path.getsize(p)
            print(f"  {out}: {size} bytes")
            if size > 0:
                with open(p, "r") as f:
                    content = f.read()
                    print(f"  Content: {content[:200]}...")

if __name__ == "__main__":
    run_repro()
