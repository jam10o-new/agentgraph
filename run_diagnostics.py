import os
import subprocess
import time
import signal
import shutil

def run_test(model_name, test_dir, is_image=False):
    print(f"\n=== Testing {model_name} in {test_dir} (is_image={is_image}) ===")
    
    # Ensure clean slate for agentgraph
    subprocess.run(["pkill", "-9", "agentgraph"], stderr=subprocess.DEVNULL)
    time.sleep(1)
    if os.path.exists("/tmp/agentgraph_pipes"):
        try:
            shutil.rmtree("/tmp/agentgraph_pipes")
        except:
            pass

    project_root = os.getcwd()
    input_dir = os.path.join(project_root, test_dir, "input")
    system_dir = os.path.join(project_root, test_dir, "system")
    output_dir = os.path.join(project_root, test_dir, "output")
    log_file = os.path.join(project_root, f"test_{test_dir}.log")

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(system_dir, exist_ok=True)

    if os.path.exists(log_file):
        os.remove(log_file)

    # Setup files
    with open(os.path.join(system_dir, "prompt.txt"), "w") as f:
        f.write("You are a helpful assistant. Describe the image if provided, otherwise say 'TEXT ONLY'.")
    
    if is_image:
        src_screenshot = os.path.join(project_root, "test_vision/agent_a/input/screenshot_20260327_072041.png")
        if os.path.exists(src_screenshot):
            shutil.copy(src_screenshot, os.path.join(input_dir, "image.png"))
        else:
            print("ERROR: No screenshot found for image test.")
            return
    else:
        with open(os.path.join(input_dir, "user.txt"), "w") as f:
            f.write("Respond with exactly the word 'TEXT ONLY' and nothing else.")

    binary = os.path.join(project_root, "target/debug/agentgraph")
    cmd = [
        binary,
        "-I", input_dir,
        "-S", system_dir,
        "-O", output_dir,
        "--latest-n", "2",
        "--verbose",
        "--no-ui",
        "-m", model_name,
        "-W"
    ]
    
    with open(log_file, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, preexec_fn=os.setsid)

    print(f"AgentGraph started (PID {proc.pid}). Triggering first inference...")
    time.sleep(5)
    
    trigger_file = os.path.join(input_dir, "image.png" if is_image else "user.txt")
    subprocess.run(["touch", trigger_file])

    print("Model is loading and running (max 480s)...")
    
    start_time = time.time()
    found_output = False
    
    while time.time() - start_time < 480:
        if os.path.exists(output_dir):
            outputs = [f for f in os.listdir(output_dir) if f.endswith(".txt")]
            for out in outputs:
                p = os.path.join(output_dir, out)
                if os.path.getsize(p) > 0:
                    with open(p, "r") as f:
                        content = f.read().strip()
                        if content:
                            print(f"SUCCESS: Received output ({len(content)} chars): {content[:100]}...")
                            found_output = True
                            break
        if found_output:
            break
            
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                log = f.read()
                if "Err:" in log or "Stream creation failed" in log:
                    print("FAILURE: Error detected in logs.")
                    print(log[-500:])
                    break
        
        time.sleep(5)

    if not found_output:
        print("FAILED: Timed out or error occurred.")

    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except:
        pass
    
    return found_output

if __name__ == "__main__":
    # Test 1: Image with Qwen3.5-9B
    run_test("Qwen/Qwen3.5-9B", "test_vision_35", is_image=True)
    # Test 2: Image with Qwen3-VL-8B-Instruct (Baseline)
    run_test("Qwen/Qwen3-VL-8B-Instruct", "test_vision_3", is_image=True)
