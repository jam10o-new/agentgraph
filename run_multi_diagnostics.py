import os
import subprocess
import time
import signal
import shutil

def run_test(model_name, test_dir, is_image=False, latest_n=2):
    print(f"\n=== Testing {model_name} in {test_dir} (is_image={is_image}, latest_n={latest_n}) ===")
    
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

    with open(os.path.join(system_dir, "prompt.txt"), "w") as f:
        f.write("Describe the most recent image and mention if you see any previous ones.")
    
    if is_image:
        src_screenshot = os.path.join(project_root, "test_vision/agent_a/input/screenshot_20260327_072041.png")
        if os.path.exists(src_screenshot):
            shutil.copy(src_screenshot, os.path.join(input_dir, "image1.png"))
            time.sleep(1.1) # Ensure different timestamps
            shutil.copy(src_screenshot, os.path.join(input_dir, "image2.png"))
        else:
            print("ERROR: No screenshot found.")
            return

    binary = os.path.join(project_root, "target/debug/agentgraph")
    cmd = [
        binary,
        "-I", input_dir,
        "-S", system_dir,
        "-O", output_dir,
        "--latest-n", str(latest_n),
        "--compress-context",
        "--verbose",
        "--no-ui",
        "-m", model_name,
        "-W"
    ]
    
    with open(log_file, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, preexec_fn=os.setsid)

    print(f"AgentGraph started. Triggering first inference...")
    time.sleep(5)
    
    trigger_file = os.path.join(input_dir, "image2.png" if is_image else "user.txt")
    subprocess.run(["touch", trigger_file])

    print("Running (max 480s)...")
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
                            print(f"SUCCESS: Received output.")
                            found_output = True
                            break
        if found_output:
            break
        time.sleep(5)

    if not found_output:
        print("FAILED.")

    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except:
        pass
    
    return found_output

if __name__ == "__main__":
    run_test("Qwen/Qwen3.5-9B", "test_vision_multi", is_image=True, latest_n=2)
