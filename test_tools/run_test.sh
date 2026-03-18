#!/usr/bin/env bash
# =============================================================================
# agentgraph tools pipeline test
#
# Graph topology:
#
#     [ Agent SCREENSHOT ]  -- tool-calling agent, uses EXEC to run grim
#                              triggered by an initial prompt in its input dir
#                              writes "SCREENSHOT_READY: <path>" to output
#              |
#     [ Agent VISION ]      -- vision describer (Qwen3-VL primary)
#                              watches agent_vision/input/ for new PNGs
#                              (the screenshot agent writes directly there)
#                              writes structured [LAYOUT]/[FOCUS]/etc to output
#              |
#     [ Agent SUMMARY ]     -- rolling screen summariser (stream-realtime)
#                              watches agent_vision/output/ as user input
#                              writes [CURRENT]/[TIMELINE]/etc to output
#
# Key difference from vision test:
#   Screenshots are taken by agent_screenshot via the --tools EXEC mechanism,
#   not by an external bash loop. This tests the full tool-use pipeline.
#
# Pass conditions:
#   - agent_screenshot output contains "SCREENSHOT_READY:"
#   - agent_vision output contains "[FOCUS]:"
#   - agent_summary output contains "[CURRENT]:"
#
# Failure patterns:
#   - "SCREENSHOT_FAILED:" in screenshot agent output
#   - "VISION_PARSE_FAILED" in vision agent output
#   - CUDA OOM or other hard errors in any agent stderr
#
# Usage:
#   bash run_test.sh              # run until MAX_RUNTIME_SECS or first result
#   VERBOSE=1 bash run_test.sh    # echo all agent log lines live
#
# Stop with Ctrl-C — cleanup is automatic via EXIT trap.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$SCRIPT_DIR/../target/release/agentgraph"
BASE="$SCRIPT_DIR"

# --------------------------------------------------------------------------- #
# Tunable parameters
MAX_RUNTIME_SECS=300
# --------------------------------------------------------------------------- #

# ---- Error / success patterns --------------------------------------------- #
# Note: Patterns are anchored to start of line (^) to avoid matching quoted text
FAIL_PATTERNS=(
    "^SCREENSHOT_FAILED:"
    "^VISION_PARSE_FAILED"
    "^TOOL USE FAILED"
    "^EXEC failed"
    "^out of memory"
    "^CUDA_ERROR"
    "^thread '.*' panicked"
)

SUCCESS_PATTERNS=(
    "SCREENSHOT_READY:"    # screenshot agent used tool correctly
    "[FOCUS]:"             # vision agent fired on the image
    "[CURRENT]:"           # summary agent produced end-to-end output
)

# ---- Directories ----------------------------------------------------------- #
mkdir -p \
    "$BASE/agent_screenshot/system" \
    "$BASE/agent_screenshot/input" \
    "$BASE/agent_screenshot/output" \
    "$BASE/agent_vision/system" \
    "$BASE/agent_vision/input" \
    "$BASE/agent_vision/output" \
    "$BASE/agent_summary/system" \
    "$BASE/agent_summary/input" \
    "$BASE/agent_summary/output" \
    "$BASE/logs"

# ---- Log files ------------------------------------------------------------- #
LOG_SCREENSHOT="$BASE/logs/agent_screenshot.log"
LOG_VISION="$BASE/logs/agent_vision.log"
LOG_SUMMARY="$BASE/logs/agent_summary.log"

# ---- State files ----------------------------------------------------------- #
FAIL_FILE="$BASE/.test_failed"
SUCCESS_FILE="$BASE/.test_succeeded"

# ---- PID tracking ---------------------------------------------------------- #
AGENT_SCREENSHOT_PID=""
AGENT_VISION_PID=""
AGENT_SUMMARY_PID=""

cleanup() {
    echo ""
    echo "=== Cleaning up ==="
    [[ -n "$AGENT_SCREENSHOT_PID" ]] && kill "$AGENT_SCREENSHOT_PID" 2>/dev/null || true
    [[ -n "$AGENT_VISION_PID"     ]] && kill "$AGENT_VISION_PID"     2>/dev/null || true
    [[ -n "$AGENT_SUMMARY_PID"    ]] && kill "$AGENT_SUMMARY_PID"    2>/dev/null || true
    pkill -f "agentgraph.*test_tools" 2>/dev/null || true
    wait "$AGENT_SCREENSHOT_PID" "$AGENT_VISION_PID" "$AGENT_SUMMARY_PID" 2>/dev/null || true
    echo "=== Cleanup complete ==="
}
trap cleanup EXIT INT TERM

# ============================================================================
# PRE-RUN CLEANUP — clear output from any previous run
# ============================================================================
echo "=== Pre-run cleanup ==="
rm -f \
    "$BASE/agent_screenshot/input"/*.txt \
    "$BASE/agent_screenshot/output"/* \
    "$BASE/agent_vision/input"/*.png \
    "$BASE/agent_vision/output"/* \
    "$BASE/agent_summary/output"/* \
    "$BASE/logs"/*.log \
    "$FAIL_FILE" "$SUCCESS_FILE"
> "$LOG_SCREENSHOT"
> "$LOG_VISION"
> "$LOG_SUMMARY"
echo "    Output directories cleared."

# ============================================================================
# Monitor function — watches output directory for new files and checks content
# ============================================================================
monitor_output() {
    local label="$1"
    local outdir="$2"

    # Use inotifywait to watch for new/close_write events
    inotifywait -m -e close_write --format '%f' "$outdir" 2>/dev/null | while IFS= read -r filename; do
        filepath="$outdir/$filename"
        [[ ! -f "$filepath" ]] && continue

        content="$(cat "$filepath" 2>/dev/null)" || continue

        [[ "${VERBOSE:-0}" == "1" ]] && echo "[$label] <<< $filename: $content"

        for pat in "${FAIL_PATTERNS[@]}"; do
            if echo "$content" | grep -qE "$pat"; then
                echo ""
                echo "!!! FAIL [$label]: pattern '$pat' matched in $filename"
                touch "$FAIL_FILE"
            fi
        done

        for pat in "${SUCCESS_PATTERNS[@]}"; do
            if echo "$content" | grep -qF "$pat"; then
                echo ""
                echo ">>> SUCCESS [$label]: '$pat' in $filename <<<"
                touch "$SUCCESS_FILE"
            fi
        done
    done
}

# ============================================================================
# Launch Agent SCREENSHOT — tool-calling agent that runs grim
# --tools enables the EXEC/READ/KILL/WRIT command system
# No --stream-realtime (agent-to-agent node)
# No secondary model needed
# ============================================================================
echo ""
echo "=== Launching Agent SCREENSHOT (tool-calling, runs grim via EXEC) ==="

"$BIN" \
    -W \
    --verbose \
    --tools \
    -M none \
    -S "$BASE/agent_screenshot/system" \
    -I "$BASE/agent_screenshot/input" \
    -O "$BASE/agent_screenshot/output" \
    > >(tee "$LOG_SCREENSHOT") 2>&1 &
AGENT_SCREENSHOT_PID=$!
echo "    Agent SCREENSHOT PID: $AGENT_SCREENSHOT_PID"
sleep 1

# ============================================================================
# Launch Agent VISION — vision describer, watches agent_vision/input/ for PNGs
# No --stream-realtime (agent-to-agent node)
# No secondary model needed
# ============================================================================
echo ""
echo "=== Launching Agent VISION (Qwen3-VL screen observer) ==="

"$BIN" \
    -W \
    --verbose \
    -M none \
    -S "$BASE/agent_vision/system" \
    -I "$BASE/agent_vision/input" \
    -O "$BASE/agent_vision/output" \
    > >(tee "$LOG_VISION") 2>&1 &
AGENT_VISION_PID=$!
echo "    Agent VISION PID: $AGENT_VISION_PID"
sleep 1

# ============================================================================
# Launch Agent SUMMARY — stream-realtime human-facing summariser
# Watches agent_vision/output/ for completed vision reports
# ============================================================================
echo ""
echo "=== Launching Agent SUMMARY (stream-realtime summariser) ==="

"$BIN" \
    -W \
    --verbose \
    --stream-realtime \
    -M none \
    -S "$BASE/agent_summary/system" \
    -I "$BASE/agent_vision/output" \
    -O "$BASE/agent_summary/output" \
    > >(tee "$LOG_SUMMARY") 2>&1 &
AGENT_SUMMARY_PID=$!
echo "    Agent SUMMARY PID: $AGENT_SUMMARY_PID"

# ============================================================================
# Start output monitors in background
# ============================================================================
monitor_output "SCREENSHOT" "$BASE/agent_screenshot/output" &
monitor_output "VISION"     "$BASE/agent_vision/output"     &
monitor_output "SUMMARY"    "$BASE/agent_summary/output"    &

# ============================================================================
# Write initial trigger prompt to agent_screenshot/input/
# This is the only external stimulus — from here the agents drive themselves.
# ============================================================================
echo ""
echo "=== Writing trigger prompt to agent_screenshot/input/ ==="
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
cat > "$BASE/agent_screenshot/input/trigger_${TIMESTAMP}.txt" << 'PROMPT'
Take a screenshot of the current desktop now. Use grim with the -s 0.5 scale flag and save it to the designated screenshots directory. Report the full path using the SCREENSHOT_READY: prefix as instructed.
PROMPT
echo "    Trigger written. Models will load on first inference (~60s)."

echo ""
echo "--- Pipeline: screenshot agent → grim → vision agent → summary agent ---"
echo "--- Watching for SCREENSHOT_READY:, [FOCUS]:, [CURRENT]: ---"
echo ""

# ============================================================================
# Main wait loop
# ============================================================================
ELAPSED=0
INTERVAL=5
REPORTED_SUCCESS=false

while (( ELAPSED < MAX_RUNTIME_SECS )); do
    sleep $INTERVAL
    ELAPSED=$(( ELAPSED + INTERVAL ))

    if [[ -f "$FAIL_FILE" ]]; then
        echo ""
        echo "========================================="
        echo "  TEST RESULT: FAIL"
        echo "========================================="
        echo "  Check output files in $BASE/agent_*/output/"
        exit 1
    fi

    if [[ -f "$SUCCESS_FILE" ]] && ! $REPORTED_SUCCESS; then
        echo ""
        echo "======================================================"
        echo "  Tools pipeline confirmed working — monitoring on..."
        echo "======================================================"
        REPORTED_SUCCESS=true
    fi

    if (( ELAPSED % 30 == 0 )); then
        SS_OUT="$(ls "$BASE/agent_screenshot/output/" 2>/dev/null | wc -l)"
        V_IN="$(ls  "$BASE/agent_vision/input/"      2>/dev/null | wc -l)"
        V_OUT="$(ls "$BASE/agent_vision/output/"     2>/dev/null | wc -l)"
        S_OUT="$(ls "$BASE/agent_summary/output/"    2>/dev/null | wc -l)"
        echo "[heartbeat ${ELAPSED}s] screenshot_out: $SS_OUT  vision_in(PNGs): $V_IN  vision_out: $V_OUT  summary_out: $S_OUT"
    fi
done

echo ""
echo "=== Max runtime reached (${MAX_RUNTIME_SECS}s) ==="
if [[ -f "$SUCCESS_FILE" ]]; then
    echo "TEST RESULT: PASS"
    exit 0
elif [[ -f "$FAIL_FILE" ]]; then
    echo "TEST RESULT: FAIL"
    exit 1
else
    echo "TEST RESULT: INCONCLUSIVE"
    SS_OUT="$(ls "$BASE/agent_screenshot/output/" 2>/dev/null | wc -l)"
    V_IN="$(ls  "$BASE/agent_vision/input/"       2>/dev/null | wc -l)"
    V_OUT="$(ls "$BASE/agent_vision/output/"      2>/dev/null | wc -l)"
    S_OUT="$(ls "$BASE/agent_summary/output/"     2>/dev/null | wc -l)"
    echo "  screenshot agent outputs : $SS_OUT"
    echo "  PNGs in vision input     : $V_IN"
    echo "  vision agent outputs     : $V_OUT"
    echo "  summary agent outputs    : $S_OUT"
    echo ""
    echo "  Possible causes:"
    echo "   - Models haven't finished loading yet (check output files)"
    echo "   - Tool system didn't parse EXEC correctly (check --tools flag)"
    echo "   - grim couldn't connect to Wayland display"
    echo "   - Screenshot written to wrong path (check agent_screenshot output)"
    exit 2
fi
