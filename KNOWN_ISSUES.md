# Known Issues

## KI-001 — Audio test: CUDA OOM during speech detection

**Test:** `test_audio/`
**Status:** Open
**Severity:** High — audio pipeline non-functional

### Description

When the audio test runs with both Qwen3-VL-8B (primary) and Voxtral-Mini-4B
(secondary) loaded on a single 24 GB GPU, `detect_speech` OOMs during the
Voxtral forward pass. The error appears as:

```
CUDA_ERROR_OUT_OF_MEMORY: out of memory
Speech detection error: Speech detection request failed: model error: ...
```

### Root cause

`spawn_realtime_listener` launches `detect_speech` as a background
`tokio::spawn` task, which runs Voxtral concurrently with any ongoing primary
(Qwen3-VL) inference. Both models are resident in VRAM simultaneously and the
combined weight footprint exhausts the 24 GB budget under concurrent load.

### Preferred resolution path

1. **Reduce Voxtral quantisation** — try Q2K or Q3K for the secondary model
   instead of Q4K (currently forced in `load_model`). This is the first thing
   to attempt before any serialisation changes.
2. **Serialise as fallback** — if quantisation alone is insufficient, gate
   `detect_speech` calls behind the same `inference_lock` used by the main
   loop, so Voxtral only runs when the primary model is idle. The `.unload()`
   / `.reload()` methods exist on the `Model` struct but are intentionally
   reserved for failure-recovery paths due to their latency cost.

### Related code

- `async fn detect_speech` — `src/main.rs`
- `async fn spawn_realtime_listener` — `src/main.rs`
- `async fn load_model` — secondary model ISQ bits hardcoded to `IsqBits::Four`

---

## KI-002 — Vision test: Agent A re-inference loop driven by Agent B

**Test:** `test_vision/`
**Status:** **Fixed** (2026-03-18)
**Severity:** ~~Medium~~ — Resolved

### Description

~~After Agent B produces its first output (stream-realtime mode, `-O
agent_b/output/`), the `Close(Write)` event on the completed output file
triggers Agent A to run inference again even though no new screenshot has
arrived. This repeats on every Agent B completion, creating a sustained
inference loop between the two agents.~~

The feedback loop between agents has been resolved. Agent B's output files no
longer trigger spurious re-inference in Agent A.

### Root cause

The root cause was twofold:

1. **Leader election bug**: `find_oldest_pipe` was considering ANY alive process
   as a potential leader, including newer processes with higher PIDs. This caused
   non-leader agents to incorrectly forward requests and create circular event flows.

2. **Path matching bug**: `is_interrupt_event` used exact path matching (`contains()`)
   instead of prefix matching (`starts_with()`), causing all events to pass the filter
   and trigger spurious inferences.

3. **Missing event filtering in main loop**: The main event loop was triggering
   inference for ALL filesystem events without checking relevance via
   `is_interrupt_event`.

### Resolution

Fixed in 2026-03-18 session:

1. **Fixed leader election**: `find_oldest_pipe` now only considers processes with
   LOWER PIDs (older processes) as leader candidates.

2. **Fixed path matching**: Changed from `contains()` to `starts_with()` to properly
   check if event paths are children of watched directories.

3. **Added main loop filtering**: Main loop now calls `is_interrupt_event` before
   triggering inference.

4. **Added Modify event debouncing**: `Create` and `Close(Write)` events now update
   the debounce timer, preventing subsequent `Modify` events from triggering.

5. **Filtered Modify events during streaming**: Running inference now only interrupts
   on `Create` and `Close(Write)` events, not `Modify` events.

### Known non-cause

This is **not** the token-per-write `Modify` storm from KI-003 (which was
fixed). The events are `Close(Write)` on distinct files, one per inference
completion.

### Related code

- `fn is_interrupt_event` — `src/main.rs` (lines ~1988-2030)
- `async fn find_oldest_pipe` — `src/main.rs` (lines ~2135-2155)
- `async fn main` — watcher subscription loop — `src/main.rs` (lines ~2265-2320)

---

## KI-005 — Tools test: Duplicate vision agent outputs

**Test:** `test_tools/`
**Status:** Open
**Severity:** Low — produces extra outputs but pipeline completes correctly

### Description

The vision agent produces 2-3 output files per screenshot instead of 1. The
leader (screenshot agent) correctly produces 1 output, but the vision agent
outputs are duplicated.

### Observed behavior

```
Screenshot outputs: 1
Vision outputs: 2-3
Summary outputs: 2-3
```

### Root cause (suspected)

The vision agent receives the screenshot file creation event and triggers
inference. Both the `Create` event and the subsequent `Close(Write)` event
pass `is_interrupt_event` filtering, causing two separate inference requests
to be sent to the leader. The 250ms debounce window may not be sufficient if
the file write completes quickly.

### Preferred resolution path

1. Only trigger inference on `Close(Write)` events for file completion, not on
   `Create` events (which fire before file content is written).
2. Alternatively, increase the debounce window or implement per-file debounce
   tracking.

### Related code

- `fn is_interrupt_event` — `src/main.rs`
- Main event loop — `src/main.rs`

---

## KI-006 — Audio test: Potential audio parsing failures

**Test:** `test_audio/`
**Status:** Unknown — needs testing
**Severity:** High — may block audio pipeline functionality

### Description

The audio test has not been successfully run to completion. Potential issues
include:

1. CUDA OOM during speech detection (KI-001)
2. Audio format parsing failures
3. PipeWire audio capture issues

### Preferred resolution path

1. Run `test_audio/run_test.sh` with verbose logging
2. Check for `AUDIO_PARSE_FAILED` or speech detection errors in logs
3. If OOM persists, reduce Voxtral quantization from Q4K to Q2K/Q3K in
   `load_model` function

### Related code

- `async fn detect_speech` — `src/main.rs`
- `async fn load_model` — secondary model ISQ configuration
- `async fn spawn_realtime_listener` — `src/main.rs`

---

## KI-003 — *(Fixed)* Agent-to-agent `--stream-realtime` feedback storm

**Status:** Fixed in this session
**Fix:** Removed `--stream-realtime` from agent-to-agent nodes in test scripts;
changed `is_interrupt_event` to debounce `Modify` events via `sleep_ms`
cooldown; added `Close(Write)` as the primary clean-completion trigger.

---

## KI-004 — *(Fixed)* Realtime audio listener tied to single inference lifetime

**Status:** Fixed in this session
**Fix:** Lifted `spawn_realtime_listener` from inside `run_once` to the `main`
loop, giving the PipeWire connection process-lifetime. Added a
speech-detection arm to the main `tokio::select!` that enqueues audio chunks
and triggers inference independently of filesystem events.