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
**Status:** Open
**Severity:** Medium — wastes GPU cycles, does not affect correctness

### Description

After Agent B produces its first output (stream-realtime mode, `-O
agent_b/output/`), the `Close(Write)` event on the completed output file
triggers Agent A to run inference again even though no new screenshot has
arrived. This repeats on every Agent B completion, creating a sustained
inference loop between the two agents.

### Root cause

Agent A watches `agent_a/input/` for screenshots. Agent B watches
`agent_a/output/` for Agent A reports. However, Agent B is also watching its
own system prompt directory (`agent_b/system/`), and the inotify watcher on
`agent_a/output/` fires a `Close(Write)` event when Agent A's output file is
finalised.

The deeper issue: `is_interrupt_event` now correctly fires on `Create` and
`Close(Write)`, but Agent B's own completed output file (`agent_b/output/`) is
in a directory that — via the `psi-viewer` viewer path or an upstream watcher
subscription — somehow re-enters Agent A's event stream. Needs further
tracing to identify the exact subscription path responsible.

### Known non-cause

This is **not** the token-per-write `Modify` storm from KI-003 (which was
fixed). The events are `Close(Write)` on distinct files, one per inference
completion.

### Preferred resolution path

Investigate which watcher subscription is forwarding `agent_b/output/` events
upstream to Agent A. Likely fix: tighten `is_interrupt_event` to only accept
`Create` events on files matching the expected input media types (`.txt`,
`.png`, `.wav`, etc.), ignoring `Close(Write)` on files that were not in the
original watched directory set.

### Related code

- `fn is_interrupt_event` — `src/main.rs`
- `async fn handle_interrupt` — `src/main.rs`
- `async fn main` — watcher subscription loop

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