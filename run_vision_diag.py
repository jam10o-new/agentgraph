import os
import subprocess
import time
import signal
import shutil

def run_test(model_name, test_label, is_image=True, latest_n=1, compress=False):
    print(f"\n>>> TEST: {test_label} (Model: {model_name}, images: {latest_n}, compress: {compress}) <<<")
    
    subprocess.run(["pkill", "-9", "agentgraph"], stderr=subprocess.DEVNULL)
    time.sleep(1)
    if os.path.exists("/tmp/agentgraph_pipes"):
        try:
            shutil.rmtree("/tmp/agentgraph_pipes")
        except:
            pass

    project_root = os.getcwd()
    test_dir = f"diag_{test_label}"
    input_dir = os.path.join(project_root, test_dir, "input")
    system_dir = os.path.join(project_root, test_dir, "system")
    output_dir = os.path.join(project_root, test_dir, "output")
    log_file = os.path.join(project_root, f"diag_{test_label}.log")

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(system_dir, exist_ok=True)

    if os.path.exists(log_file):
        os.remove(log_file)

    with open(os.path.join(system_dir, "prompt.txt"), "w") as f:
        f.write("You are a vision assistant. Describe the image briefly.")
    
    if is_image:
        src_screenshot = os.path.join(project_root, "test_vision/agent_a/input/screenshot_20260328_172303.png")
        if os.path.exists(src_screenshot):
            for i in range(latest_n):
                shutil.copy(src_screenshot, os.path.join(input_dir, f"image{i}.png"))
                time.sleep(0.1)
        else:
            print("ERROR: No screenshot found.")
            return

    binary = os.path.join(project_root, "target/release/agentgraph")
    cmd = [
        binary,
        "-I", input_dir,
        "-S", system_dir,
        "-O", output_dir,
        "--latest-n", str(latest_n),
        "--verbose",
        "--no-ui",
        "-m", model_name,
        "-W"
    ]
    if compress:
        cmd.append("--compress-context")
    
    with open(log_file, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, preexec_fn=os.setsid)

    print(f"Started. Triggering inference...")
    time.sleep(5)
    
    trigger_file = os.path.join(input_dir, f"image{latest_n-1}.png" if is_image else "user.txt")
    subprocess.run(["touch", trigger_file])

    print("Monitoring (max 600s)...")
    start_time = time.time()
    found_output = False
    is_looping = False
    
    while time.time() - start_time < 600:
        # Check for output
        if os.path.exists(output_dir):
            outputs = [f for f in os.listdir(output_dir) if f.endswith(".txt")]
            for out in outputs:
                p = os.path.join(output_dir, out)
                if os.path.getsize(p) > 0:
                    print(f"SUCCESS: Output found.")
                    found_output = True
                    break
        
        # Check for empty loop in logs
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                log_tail = f.read()[-2000:]
                if log_tail.count("Extracted empty content") > 50:
                    print("FAILURE: Detected infinite empty response loop.")
                    is_looping = True
                    break
        
        if found_output:
            break
        time.sleep(5)

    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except:
        pass
    
    return "LOOP" if is_looping else ("PASS" if found_output else "TIMEOUT")

if __name__ == "__main__":
    results = {}
    # 1. Baseline: Qwen3.5-9B with exactly one image
    results["Qwen3.5-9B-1img"] = run_test("Qwen/Qwen3.5-9B", "q35_9b_1img", latest_n=1)
    
    # 2. Known Good Baseline: Qwen3-VL-8B with two images
    results["Qwen3-VL-2img"] = run_test("Qwen/Qwen3-VL-8B-Instruct", "q3_vl_2img", latest_n=2)
    
    # 3. Scale Test: Qwen3.5-27B with one image
    results["Qwen3.5-27B-1img"] = run_test("Qwen/Qwen3.5-27B", "q35_27b_1img", latest_n=1)

    print("\n\n=== FINAL RESULTS ===")
    for test, res in results.items():
        print(f"{test}: {res}")
