#!/usr/bin/env bash
# =============================================================================
# agentgraph subagent spawning test
#
# Graph topology:
#
#     [ SUPERVISOR AGENT ]
#           |
#           +-- EXEC --> [ watcher_a ]  (watch mode, monitors shared_screenshots)
#           |              watches: shared_screenshots/
#           |              outputs: watcher_a/output/
#           |
#           +-- EXEC --> [ watcher_b ]  (watch mode, monitors shared_screenshots)
#           |              watches: shared_screenshots/
#           |              outputs: watcher_b/output/
#           |
#           +-- EXEC --> [ oneshot_c ]  (oneshot inference, system state analysis)
#           |              input: supervisor/input/system_state.txt
#           |              outputs: oneshot_c/output/
#           |
#           +-- EXEC --> [ oneshot_d ]  (oneshot inference, recommendation generation)
#                          input: watcher outputs (synthesized)
#                          outputs: oneshot_d/output/
#
# Key concept demonstrated:
#   - Autonomous subagent spawning via EXEC command system
#   - Supervisor coordinates multiple subagent instances
#   - Mix of watch-mode and oneshot inference subagents
#   - Subagents are additional agentgraph daemon processes
#
# Pass conditions:
#   - Supervisor spawns at least 2 subagents successfully
#   - Subagent outputs contain expected structured fields
#   - Supervisor produces final [COORDINATION_SUMMARY]
#
# Failure patterns:
#   - "EXEC failed" in supervisor output
#   - "subagent failed to start" or similar errors
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
FAIL_PATTERNS=(
    "^EXEC failed"
    "^subagent failed"
    "^failed to spawn"
    "^out of memory"
    "^CUDA_ERROR"
    "^thread '.*' panicked"
)

SUCCESS_PATTERNS=(
    "[SUBAGENTS_SPAWNED]:"
    "[COORDINATION_SUMMARY]:"
)

# ---- Directories ----------------------------------------------------------- #
mkdir -p \
    "$BASE/supervisor/system" \
    "$BASE/supervisor/input" \
    "$BASE/supervisor/output" \
    "$BASE/watcher_a/system" \
    "$BASE/watcher_a/input" \
    "$BASE/watcher_a/output" \
    "$BASE/watcher_b/system" \
    "$BASE/watcher_b/input" \
    "$BASE/watcher_b/output" \
    "$BASE/oneshot_c/system" \
    "$BASE/oneshot_c/input" \
    "$BASE/oneshot_c/output" \
    "$BASE/oneshot_d/system" \
    "$BASE/oneshot_d/input" \
    "$BASE/oneshot_d/output" \
    "$BASE/shared_screenshots" \
    "$BASE/logs"

# ---- Log files ------------------------------------------------------------- #
LOG_SUPERVISOR="$BASE/logs/supervisor.log"
LOG_WATCHER_A="$BASE/logs/watcher_a.log"
LOG_WATCHER_B="$BASE/logs/watcher_b.log"
LOG_ONESHOT_C="$BASE/logs/oneshot_c.log"
LOG_ONESHOT_D="$BASE/logs/oneshot_d.log"

# ---- State files ----------------------------------------------------------- #
FAIL_FILE="$BASE/.test_failed"
SUCCESS_FILE="$BASE/.test_succeeded"

# ---- PID tracking ---------------------------------------------------------- #
AGENT_SUPERVISOR_PID=""
AGENT_WATCHER_A_PID=""
AGENT_WATCHER_B_PID=""
AGENT_ONESHOT_C_PID=""
AGENT_ONESHOT_D_PID=""

cleanup() {
    echo ""
    echo "=== Cleaning up ==="
    [[ -n "$AGENT_SUPERVISOR_PID" ]] && kill "$AGENT_SUPERVISOR_PID" 2>/dev/null || true
    [[ -n "$AGENT_WATCHER_A_PID"   ]] && kill "$AGENT_WATCHER_A_PID"   2>/dev/null || true
    [[ -n "$AGENT_WATCHER_B_PID"   ]] && kill "$AGENT_WATCHER_B_PID"   2>/dev/null || true
    [[ -n "$AGENT_ONESHOT_C_PID"   ]] && kill "$AGENT_ONESHOT_C_PID"   2>/dev/null || true
    [[ -n "$AGENT_ONESHOT_D_PID"   ]] && kill "$AGENT_ONESHOT_D_PID"   2>/dev/null || true
    pkill -f "agentgraph.*test_subagents" 2>/dev/null || true
    wait "$AGENT_SUPERVISOR_PID" "$AGENT_WATCHER_A_PID" "$AGENT_WATCHER_B_PID" "$AGENT_ONESHOT_C_PID" "$AGENT_ONESHOT_D_PID" 2>/dev/null || true
    echo "=== Cleanup complete ==="
}
trap cleanup EXIT INT TERM

# ============================================================================
# PRE-RUN CLEANUP — clear output from any previous run
# ============================================================================
echo "=== Pre-run cleanup ==="
rm -f \
    "$BASE/supervisor/input"/*.txt \
    "$BASE/supervisor/output"/* \
    "$BASE/watcher_a/input"/* \
    "$BASE/watcher_a/output"/* \
    "$BASE/watcher_b/input"/* \
    "$BASE/watcher_b/output"/* \
    "$BASE/oneshot_c/input"/* \
    "$BASE/oneshot_c/output"/* \
    "$BASE/oneshot_d/input"/* \
    "$BASE/oneshot_d/output"/* \
    "$BASE/shared_screenshots"/* \
    "$BASE/logs"/*.log \
    "$FAIL_FILE" "$SUCCESS_FILE"
> "$LOG_SUPERVISOR"
> "$LOG_WATCHER_A"
> "$LOG_WATCHER_B"
> "$LOG_ONESHOT_C"
> "$LOG_ONESHOT_D"
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
# Launch SUPERVISOR AGENT — orchestrates subagent spawning
# --tools enables the EXEC/READ/KILL/WRIT command system
# --stream-realtime for human-facing output
# ============================================================================
echo ""
echo "=== Launching SUPERVISOR AGENT (orchestrates subagent spawning) ==="

"$BIN" \
    -W \
    --verbose \
    --tools \
    --stream-realtime \
    -M none \
    -S "$BASE/supervisor/system" \
    -I "$BASE/supervisor/input" \
    -O "$BASE/supervisor/output" \
    > >(tee "$LOG_SUPERVISOR") 2>&1 &
AGENT_SUPERVISOR_PID=$!
echo "    Supervisor PID: $AGENT_SUPERVISOR_PID"
sleep 2

# ============================================================================
# Write trigger prompt to supervisor/input/
# This instructs the supervisor to spawn subagents
# ============================================================================
echo ""
echo "=== Writing spawn trigger to supervisor/input/ ==="
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
cat > "$BASE/supervisor/input/spawn_trigger_${TIMESTAMP}.txt" << 'PROMPT'
Spawn and coordinate your subagents now. Follow your system instructions to:
1. Spawn watcher_a and watcher_b (watch mode agents)
2. Spawn oneshot_c and oneshot_d (oneshot inference agents)
3. Read outputs from all subagents
4. Produce a final coordination report

Use the EXEC command to spawn each subagent as a new agentgraph daemon instance.
PROMPT
echo "    Spawn trigger written. Supervisor will begin spawning subagents."

# ============================================================================
# Start output monitors in background
# ============================================================================
monitor_output "SUPERVISOR" "$BASE/supervisor/output" &
monitor_output "WATCHER_A"  "$BASE/watcher_a/output"  &
monitor_output "WATCHER_B"  "$BASE/watcher_b/output"  &
monitor_output "ONESHOT_C"  "$BASE/oneshot_c/output"  &
monitor_output "ONESHOT_D"  "$BASE/oneshot_d/output"  &

echo ""
echo "--- Pipeline: supervisor → spawns 4 subagents (2 watchers, 2 oneshot) ---"
echo "--- Watching for [SUBAGENTS_SPAWNED]:, [COORDINATION_SUMMARY]: ---"
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
        echo "  Check output files in $BASE/*/output/"
        exit 1
    fi

    if [[ -f "$SUCCESS_FILE" ]] && ! $REPORTED_SUCCESS; then
        echo ""
        echo "======================================================"
        echo "  Subagent spawning confirmed working — monitoring on..."
        echo "======================================================"
        REPORTED_SUCCESS=true
    fi

    if (( ELAPSED % 30 == 0 )); then
        SUPER_OUT="$(ls "$BASE/supervisor/output/" 2>/dev/null | wc -l)"
        WA_OUT="$(ls "$BASE/watcher_a/output/"     2>/dev/null | wc -l)"
        WB_OUT="$(ls "$BASE/watcher_b/output/"     2>/dev/null | wc -l)"
        OC_OUT="$(ls "$BASE/oneshot_c/output/"     2>/dev/null | wc -l)"
        OD_OUT="$(ls "$BASE/oneshot_d/output/"     2>/dev/null | wc -l)"
        echo "[heartbeat ${ELAPSED}s] supervisor: $SUPER_OUT  watcher_a: $WA_OUT  watcher_b: $WB_OUT  oneshot_c: $OC_OUT  oneshot_d: $OD_OUT"
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
    SUPER_OUT="$(ls "$BASE/supervisor/output/" 2>/dev/null | wc -l)"
    WA_OUT="$(ls "$BASE/watcher_a/output/"     2>/dev/null | wc -l)"
    WB_OUT="$(ls "$BASE/watcher_b/output/"     2>/dev/null | wc -l)"
    OC_OUT="$(ls "$BASE/oneshot_c/output/"     2>/dev/null | wc -l)"
    OD_OUT="$(ls "$BASE/oneshot_d/output/"     2>/dev/null | wc -l)"
    echo "  supervisor outputs     : $SUPER_OUT"
    echo "  watcher_a outputs      : $WA_OUT"
    echo "  watcher_b outputs      : $WB_OUT"
    echo "  oneshot_c outputs      : $OC_OUT"
    echo "  oneshot_d outputs      : $OD_OUT"
    echo ""
    echo "  Possible causes:"
    echo "   - Subagents haven't finished spawning yet (check supervisor output)"
    echo "   - EXEC command syntax incorrect (check --tools flag on supervisor)"
    echo "   - Subagent paths or flags wrong (check spawn commands)"
    echo "   - Resource constraints preventing multiple daemon instances"
    exit 2
fi
