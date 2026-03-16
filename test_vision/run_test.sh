#!/usr/bin/env bash
# =============================================================================
# agentgraph vision pipeline test
#
# Graph topology:
#
#   [grim screenshot loop]
#          | (writes PNG to agent_a/input/ every SCREENSHOT_INTERVAL_SECS)
#          |
#     [ Agent A ]  -- screen observer / vision describer (Qwen3-VL primary)
#                     watches agent_a/input/ for new PNGs
#                     writes structured reports to agent_a/output/
#          |
#     [ Agent B ]  -- downstream screen-activity summariser
#                     watches agent_a/output/ as user input (-I)
#                     writes rolling summary to agent_b/output/ (stream-realtime)
#
# Agent A: no --stream-realtime (agent-to-agent node)
#          triggers on Create/Close(Write) of each new screenshot PNG
#
# Agent B: --stream-realtime (human-facing final node)
#          triggers on Close(Write) of each completed Agent A report
#
# Pass condition  : Agent B produces output containing [CURRENT]:
# Failure patterns: VISION_PARSE_FAILED in either agent's output,
#                   or known model error strings in stderr
#
# Usage:
#   bash run_test.sh              # run until MAX_RUNTIME_SECS or first result
#   VERBOSE=1 bash run_test.sh    # echo all agent log lines
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
SCREENSHOT_COUNT=2            # total screenshots to take (one triggers model load, second tests re-inference)
SCREENSHOT_WAIT_SECS=60       # seconds to wait between shot 1 and shot 2 (gives model time to finish first inference)
# --------------------------------------------------------------------------- #

# ---- Error / success patterns --------------------------------------------- #
FAIL_PATTERNS=(
    "VISION_PARSE_FAILED"
    "Failed to create AudioInput from bytes"
    "out of memory"
    "CUDA_ERROR"
    "thread '.*' panicked"
)

SUCCESS_PATTERNS=(
    "[CURRENT]:"        # Agent B structured output — pipeline end-to-end worked
    "[FOCUS]:"          # Agent A structured output — vision model fired
)

# ---- Directories ----------------------------------------------------------- #
mkdir -p \
    "$BASE/agent_a/system" "$BASE/agent_a/input" "$BASE/agent_a/output" \
    "$BASE/agent_b/system" "$BASE/agent_b/input" "$BASE/agent_b/output" \
    "$BASE/logs"

# ---- Log files ------------------------------------------------------------- #
LOG_A="$BASE/logs/agent_a.log"
LOG_B="$BASE/logs/agent_b.log"

# ---- State files ----------------------------------------------------------- #
FAIL_FILE="$BASE/.test_failed"
SUCCESS_FILE="$BASE/.test_succeeded"

# ============================================================================
# CLEANUP — runs on EXIT (normal, Ctrl-C, error)
# Kills both agents, the screenshot loop, and removes leftover state files.
# ============================================================================
AGENT_A_PID=""
AGENT_B_PID=""
SCREENSHOT_PID=""

cleanup() {
    echo ""
    echo "=== Cleaning up ==="
    [[ -n "$SCREENSHOT_PID" ]] && kill "$SCREENSHOT_PID" 2>/dev/null || true
    [[ -n "$AGENT_A_PID"    ]] && kill "$AGENT_A_PID"    2>/dev/null || true
    [[ -n "$AGENT_B_PID"    ]] && kill "$AGENT_B_PID"    2>/dev/null || true
    # Kill any stale agentgraph processes from previous runs
    pkill -f "agentgraph.*test_vision" 2>/dev/null || true
    wait "$SCREENSHOT_PID" "$AGENT_A_PID" "$AGENT_B_PID" 2>/dev/null || true
    echo "=== Cleanup complete ==="
}
trap cleanup EXIT INT TERM

# ============================================================================
# PRE-RUN CLEANUP — clear output from any previous run
# ============================================================================
echo "=== Pre-run cleanup ==="
rm -f \
    "$BASE/agent_a/input"/*.png \
    "$BASE/agent_a/input"/*.txt \
    "$BASE/agent_a/output"/* \
    "$BASE/agent_b/output"/* \
    "$BASE/logs"/*.log \
    "$FAIL_FILE" "$SUCCESS_FILE"
> "$LOG_A"
> "$LOG_B"
echo "    Output directories cleared."

# ============================================================================
# Monitor function — tails a log file and checks patterns
# Writes to $FAIL_FILE / $SUCCESS_FILE; also prints matching lines.
# ============================================================================
monitor_log() {
    local label="$1"
    local logfile="$2"

    tail -F "$logfile" 2>/dev/null | while IFS= read -r line; do
        [[ "${VERBOSE:-0}" == "1" ]] && echo "[$label] $line"

        for pat in "${FAIL_PATTERNS[@]}"; do
            if echo "$line" | grep -qE "$pat"; then
                echo ""
                echo "!!! FAIL [$label]: pattern '$pat' matched: $line"
                touch "$FAIL_FILE"
            fi
        done

        for pat in "${SUCCESS_PATTERNS[@]}"; do
            if echo "$line" | grep -qF "$pat"; then
                echo ""
                echo ">>> SUCCESS [$label]: pattern '$pat' seen <<<"
                touch "$SUCCESS_FILE"
            fi
        done
    done
}

# ============================================================================
# Launch Agent A — vision screen observer (no --stream-realtime)
# ============================================================================
echo ""
echo "=== Launching Agent A (vision observer) ==="

"$BIN" \
    -W \
    --verbose \
    -S "$BASE/agent_a/system" \
    -I "$BASE/agent_a/input" \
    -M none \
    -O "$BASE/agent_a/output" \
    > >(tee "$LOG_A") 2>&1 &
AGENT_A_PID=$!
echo "    Agent A PID: $AGENT_A_PID"
sleep 1

# ============================================================================
# Launch Agent B — screen activity summariser (--stream-realtime, human-facing)
# ============================================================================
echo ""
echo "=== Launching Agent B (screen summariser, stream-realtime) ==="

"$BIN" \
    -W \
    --verbose \
    --stream-realtime \
    -S "$BASE/agent_b/system" \
    -I "$BASE/agent_a/output" \
    -M none \
    -O "$BASE/agent_b/output" \
    > >(tee "$LOG_B") 2>&1 &
AGENT_B_PID=$!
echo "    Agent B PID: $AGENT_B_PID"

# ============================================================================
# Start log monitors in background
# ============================================================================
monitor_log "AGENT_A" "$LOG_A" &
monitor_log "AGENT_B" "$LOG_B" &

# ============================================================================
# Screenshot sequence — takes SCREENSHOT_COUNT shots total.
# Shot 1 fires immediately to trigger model load + first inference.
# Shot 2 fires after SCREENSHOT_WAIT_SECS to let the first inference finish
# before adding a second image to context.
# ============================================================================
(
    for i in $(seq 1 "$SCREENSHOT_COUNT"); do
        TS="$(date +%Y%m%d_%H%M%S)"
        OUTFILE="$BASE/agent_a/input/screenshot_${TS}.png"
        if grim -s 0.5 "$OUTFILE" 2>/dev/null; then
            echo "[screenshot $i/$SCREENSHOT_COUNT] Captured: $OUTFILE"
        else
            echo "[screenshot $i/$SCREENSHOT_COUNT] grim failed — is WAYLAND_DISPLAY set?"
        fi
        # Wait between shots, but not after the last one
        if (( i < SCREENSHOT_COUNT )); then
            echo "[screenshot] Waiting ${SCREENSHOT_WAIT_SECS}s before next shot..."
            sleep "$SCREENSHOT_WAIT_SECS"
        fi
    done
    echo "[screenshot] Sequence complete."
) &
SCREENSHOT_PID=$!

echo ""
echo "--- Shot 1 fires immediately to load models (~60s to first inference). ---"
echo "--- Shot 2 fires after ${SCREENSHOT_WAIT_SECS}s to test re-inference. ---"
echo "--- Watching for [FOCUS]: (Agent A) and [CURRENT]: (Agent B) ---"
echo ""

# ============================================================================
# Main wait loop — poll for pass/fail, enforce MAX_RUNTIME_SECS ceiling
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
        echo "  Check $LOG_A and $LOG_B for details."
        exit 1
    fi

    if [[ -f "$SUCCESS_FILE" ]] && ! $REPORTED_SUCCESS; then
        echo ""
        echo "======================================================"
        echo "  Vision pipeline confirmed working — monitoring on..."
        echo "======================================================"
        REPORTED_SUCCESS=true
    fi

    if (( ELAPSED % 30 == 0 )); then
        A_OUT="$(ls "$BASE/agent_a/output/" 2>/dev/null | wc -l)"
        B_OUT="$(ls "$BASE/agent_b/output/" 2>/dev/null | wc -l)"
        SHOTS="$(ls "$BASE/agent_a/input/"*.png 2>/dev/null | wc -l)"
        echo "[heartbeat ${ELAPSED}s] screenshots: $SHOTS  agent_a outputs: $A_OUT  agent_b outputs: $B_OUT"
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
    echo "  Agent A output files : $(ls "$BASE/agent_a/output/" 2>/dev/null | wc -l)"
    echo "  Agent B output files : $(ls "$BASE/agent_b/output/" 2>/dev/null | wc -l)"
    echo "  Screenshots taken    : $(ls "$BASE/agent_a/input/"*.png 2>/dev/null | wc -l)"
    echo ""
    echo "  Possible causes:"
    echo "   - Models haven't finished loading yet (check logs/agent_a.log)"
    echo "   - grim couldn't connect to Wayland (check WAYLAND_DISPLAY)"
    echo "   - Vision inference slower than timeout"
    exit 2
fi
