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

The feedback loop between agents has been resolved. Agent B's output files no
longer trigger spurious re-inference in Agent A.

---

## KI-005 — Tools test: Duplicate vision agent outputs

**Test:** `test_tools/`
**Status:** Open
**Severity:** Low — produces extra outputs but pipeline completes correctly

### Description

The vision agent produces 2-3 output files per screenshot instead of 1. The
leader (screenshot agent) correctly produces 1 output, but the vision agent
outputs are duplicated.

### Root cause (suspected)

The vision agent receives the screenshot file creation event and triggers
inference. Both the `Create` event and the subsequent `Close(Write)` event
pass `is_interrupt_event` filtering, causing two separate inference requests
to be sent to the leader.

### Preferred resolution path

1. Only trigger inference on `Close(Write)` events for file completion, not on
   `Create` events.

---

## KI-006 — Audio test: Potential audio parsing failures

**Test:** `test_audio/`
**Status:** Open
**Severity:** High — audio pipeline fails with NO_AUDIO_DETECTED

### Description

The audio test fails even when audio should be present. Agent A outputs
`[AUDIO_STATUS]: NO_AUDIO_DETECTED`.

---

## KI-007 — Vision pipeline: Infinite loop of empty outputs

**Test:** `test_vision/`
**Status:** **New Regression** (2026-03-28)
**Severity:** High — blocks vision pipeline

### Description

During the vision test, the model (specifically Qwen3.5-9B) enters an infinite
loop of emitting empty content chunks. This differs from filesystem-driven
loops; the inference coroutine itself keeps receiving `Some(chunk)` from the
stream where `delta.content` is `None` or empty, preventing the stream from
closing and blocking the agent.

### Root cause

Likely related to how Qwen3.5-9B handles multimodal inputs when its internal
limits (sequence length or image count) are approached or when the prompt
formatting (e.g. empty text turns with images) causes the model to stall.

### Preferred resolution path

1. Implement a "patience" counter or max-empty-chunks limit in
   `run_inference_coroutine` to forcefully terminate streams that are not
   producing tokens.
2. Investigate if `mistral-rs` device mapping for this model can be tuned to
   avoid this state.

### Related code

- `async fn run_inference_coroutine` — `src/inference.rs`
- `pub fn extract_content` — `src/inference.rs`

---

## KI-008 — Primary model: Image history regression

**Test:** `run_multi_diagnostics.py`
**Status:** **Open** (2026-03-28)
**Severity:** High — prevents use of image history with default model

### Description

The default model (Qwen3.5-9B) fails to produce output when multiple images are
present in the history. Diagnostics show that while Qwen3-VL-8B can handle 2+
images, Qwen3.5-9B (under current `mistral-rs` loading) defaults to
`max_num_images: 1`.

### Resolution Attempt

Context compression was implemented to convert older images to text summaries
using the primary model as a specialist agent. However, this has not yet
resolved the stall/infinite loop described in KI-007.

### Related code

- `src/messages.rs` — `load_dir_messages` compression logic
- `src/context/compression.rs` — `CompressionAgent`
- `src/model.rs` — `load_model` (missing `with_max_num_images` configuration)

---

## KI-003 — *(Fixed)* Agent-to-agent `--stream-realtime` feedback storm

**Status:** Fixed

---

## KI-004 — *(Fixed)* Realtime audio listener tied to single inference lifetime

**Status:** Fixed
