#!/usr/bin/env bash
# =============================================================================
# agentgraph audio streaming pipeline test
#
# Graph topology:
#
#   [PipeWire mic/loopback]
#          |
#     [ Agent A ]  -- realtime audio listener / transcriber (Voxtral secondary model)
#          |  output written to agent_a/output/ (new file per inference)
#          |
#     [ Agent B ]  -- downstream summariser, watches agent_a/output as user input
#                     writes running summary to agent_b/output/
#
# Agent A: --realtime-listener pipewire  --stream-realtime
#          triggers on audio speech events; uses secondary (Voxtral) model for
#          detect_speech + audio transcription, primary (Qwen3-VL) for synthesis.
#
# Agent B: -W watch mode, -I agent_a/output  (latest output of A is user input)
#          pure text path, primary model only.
#
# The test monitors stderr of both agents for known error strings and prints
# structured pass/fail results.  It also watches agent_b/output for actual
# content, which confirms the full pipeline fired end-to-end.
#
# Stop with:  Ctrl-C  (or after MAX_RUNTIME_SECS seconds)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$SCRIPT_DIR/../target/release/agentgraph"
BASE="$SCRIPT_DIR"

# --------------------------------------------------------------------------- #
# Configurable timeouts
MAX_RUNTIME_SECS=300          # hard ceiling before we forcibly stop
AUDIO_TRIGGER_WAIT_SECS=120   # how long to wait for a first audio interrupt
# --------------------------------------------------------------------------- #

# Known error strings we watch for (any of these → test FAIL)
AUDIO_FAIL_PATTERNS=(
    "Speech detection error"
    "Failed to create AudioInput from bytes"
    "Speech detection request failed"
    "Audio stream error"
    "Failed to build input stream"
    "Failed to start audio stream"
    "No default input device found"
    "No supported 48kHz mono F32 input config"
    "Failed to check secondary model load state"
    "Failed to reload secondary model"
    "Full error from mistralrs"
    "AUDIO_PARSE_FAILED"   # agent A output — audio model couldn't parse audio
)

# Strings that confirm the pipeline succeeded
SUCCESS_PATTERNS=(
    "Audio interrupt"               # agentgraph verbose log: audio interrupt triggered
    "[TYPE]:"                       # agent A structured output
    "[SITUATION]:"                  # agent B structured output
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

# ---- PID tracking ---------------------------------------------------------- #
AGENT_A_PID=""
AGENT_B_PID=""

cleanup() {
    echo ""
    echo "=== Cleaning up ==="
    [[ -n "$AGENT_A_PID" ]] && kill "$AGENT_A_PID" 2>/dev/null || true
    [[ -n "$AGENT_B_PID" ]] && kill "$AGENT_B_PID" 2>/dev/null || true
    pkill -f "agentgraph.*test_audio" 2>/dev/null || true
    wait "$AGENT_A_PID" "$AGENT_B_PID" 2>/dev/null || true
    echo "=== Cleanup complete ==="
}
trap cleanup EXIT INT TERM

# ============================================================================
# PRE-RUN CLEANUP — clear output from any previous run
# ============================================================================
echo "=== Pre-run cleanup ==="
rm -f \
    "$BASE/agent_a/input"/*.txt \
    "$BASE/agent_a/output"/* \
    "$BASE/agent_b/output"/* \
    "$BASE/logs"/*.log \
    "$FAIL_FILE" "$SUCCESS_FILE"
> "$LOG_A"
> "$LOG_B"
echo "    Output directories cleared."

# ============================================================================
# Monitor function — run in background, watches a log file for patterns
# ============================================================================
monitor_log() {
    local label="$1"
    local logfile="$2"
    shift 2
    local fail_patterns=("${FAIL_PATTERNS[@]}")   # global from outer scope
    local success_patterns=("${SUCCESS_PATTERNS[@]}")

    tail -F "$logfile" 2>/dev/null | while IFS= read -r line; do
        [[ "${VERBOSE:-0}" == "1" ]] && echo "[$label] $line"

        # Check failure patterns
        for pat in "${AUDIO_FAIL_PATTERNS[@]}"; do
            if echo "$line" | grep -qF "$pat"; then
                echo ""
                echo "!!! FAIL: pattern '$pat' detected in [$label] output !!!"
                touch "$FAIL_FILE"
                break
            fi
        done

        # Check success patterns
        for pat in "${SUCCESS_PATTERNS[@]}"; do
            if echo "$line" | grep -qF "$pat"; then
                echo ""
                echo ">>> SUCCESS pattern '$pat' detected in [$label] output <<<"
                touch "$SUCCESS_FILE"
            fi
        done
    done
}

# ============================================================================
# Launch Agent A — the realtime audio listener / transcriber
# ============================================================================
echo "=== Launching Agent A (realtime audio listener + transcriber) ==="
echo "    Model: Qwen3-VL-8B-Instruct (primary) + Voxtral-Mini (secondary)"
echo "    Audio: pipewire  chunk: 3–8 s"
echo ""

"$BIN" \
    -W \
    --verbose \
    --realtime-listener pipewire \
    --audio-chunk-min-secs 3.0 \
    --audio-chunk-max-secs 8.0 \
    -S "$BASE/agent_a/system" \
    -I "$BASE/agent_a/input" \
    -O "$BASE/agent_a/output" \
    > >(tee "$LOG_A") 2>&1 &
AGENT_A_PID=$!

echo "Agent A PID: $AGENT_A_PID"
sleep 2

# ============================================================================
# Launch Agent B — the downstream summariser
# ============================================================================
echo ""
echo "=== Launching Agent B (summariser, watches Agent A output) ==="
echo ""

"$BIN" \
    -W \
    --verbose \
    --stream-realtime \
    -S "$BASE/agent_b/system" \
    -I "$BASE/agent_a/output" \
    -O "$BASE/agent_b/output" \
    > >(tee "$LOG_B") 2>&1 &
AGENT_B_PID=$!

echo "Agent B PID: $AGENT_B_PID"
echo ""

# ============================================================================
# Start background monitors
# ============================================================================
monitor_log "AGENT_A" "$LOG_A" &
MONITOR_A_PID=$!
monitor_log "AGENT_B" "$LOG_B" &
MONITOR_B_PID=$!

# ============================================================================
# Give agent A an initial prompt so it knows to start listening
# (agent A watches agent_a/input for user messages; we write a trigger file)
# ============================================================================
echo "=== Writing initial user prompt to Agent A input ==="
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
cat > "$BASE/agent_a/input/prompt_${TIMESTAMP}.txt" << 'PROMPT'
Begin monitoring the ambient audio environment. Each time you detect and process an audio chunk, output your structured report. Stay in watch mode — you will receive new audio chunks as the environment changes.
PROMPT

echo "Prompt written. Waiting up to ${AUDIO_TRIGGER_WAIT_SECS}s for audio interrupt to fire..."
echo ""
echo "--- Watching speech from speakers should trigger audio detection ---"
echo "--- (podcast / voice audio should be detected as speech) ---"
echo ""

# ============================================================================
# Wait loop — check for success/failure, enforce hard ceiling
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
        echo "  TEST RESULT: FAIL (error pattern seen)"
        echo "========================================="
        exit 1
    fi

    if [[ -f "$SUCCESS_FILE" ]] && ! $REPORTED_SUCCESS; then
        echo ""
        echo "======================================================"
        echo "  Audio pipeline confirmed working — monitoring on..."
        echo "======================================================"
        REPORTED_SUCCESS=true
    fi

    # Print a heartbeat every 30s
    if (( ELAPSED % 30 == 0 )); then
        echo "[heartbeat] ${ELAPSED}s elapsed — agents running. Agent A output files: $(ls "$BASE/agent_a/output/" 2>/dev/null | wc -l), Agent B output files: $(ls "$BASE/agent_b/output/" 2>/dev/null | wc -l)"
    fi
done

echo ""
echo "=== Max runtime reached (${MAX_RUNTIME_SECS}s) ==="
if [[ -f "$SUCCESS_FILE" ]]; then
    echo "TEST RESULT: PASS (audio pipeline triggered and worked correctly)"
    exit 0
else
    echo "TEST RESULT: INCONCLUSIVE (no audio interrupt triggered within timeout)"
    echo "  This could mean:"
    echo "   - The podcast/speech audio was not captured by PipeWire"
    echo "   - The speech detection model deemed no chunk speech-bearing"
    echo "   - The models have not yet loaded (check logs above)"
    exit 2
fi
