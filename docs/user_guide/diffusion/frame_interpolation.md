# Frame Interpolation

## Overview

vLLM-Omni supports post-generation frame interpolation for supported video
diffusion pipelines. This feature inserts synthesized intermediate frames
between adjacent generated frames to improve temporal smoothness without
rerunning the diffusion denoising loop.

Frame interpolation is coordinated by the diffusion engine and runs on
diffusion workers instead of the API server encoding path. This keeps the
FastAPI event loop free from heavy synchronous PyTorch work and lets
multi-worker deployments split adjacent-frame pairs across ranks.

For an input video with `N` generated frames and interpolation exponent `exp`,
the output frame count is:

```text
(N - 1) * 2**exp + 1
```

The output FPS is multiplied by `2**exp` so the clip duration remains close to
the original generated video.

## Supported Pipelines

Frame interpolation is currently supported for:

- `WanPipeline` (Wan2.2 text-to-video)
- `WanImageToVideoPipeline`

## Request Parameters

The video APIs `/v1/videos` and `/v1/videos/sync` accept:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_frame_interpolation` | bool | `false` | Enable post-generation frame interpolation |
| `frame_interpolation_exp` | int | `1` | Interpolation exponent. `1=2x`, `2=4x`, etc. |
| `frame_interpolation_scale` | float | `1.0` | RIFE inference scale |

## Server Parameters

RIFE weights are configured by the server operator:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--frame-interpolation-model-path` | str | `None` | Local directory or Hugging Face repo ID containing `flownet.pkl`; when omitted, the built-in RIFE repo is used |
| `--preload-frame-interpolation-model` | flag | `false` | Load the RIFE model during diffusion engine startup |

`--preload-frame-interpolation-model` moves RIFE weight loading and a small
worker-side warmup run into service startup. Startup takes longer and allocates
the RIFE model on worker devices earlier, but the first interpolated request
does not pay that one-time loading cost.

## Execution Flow

For supported Wan2.2 pipelines, the execution order is:

1. Diffusion worker finishes denoising and decodes the raw video tensor on NPU.
2. Inside `DiffusionWorker.generate()`, the pipeline-registered
   `worker_postprocess_func` runs on the still-on-NPU tensor:
   1. If frame interpolation is enabled, RIFE interpolates inline (no engine
      round-trip) and records the FPS multiplier in `custom_output`.
   2. The bf16 video tensor is converted to uint8 `(B, T, H, W, C)` on NPU
      and the `video_format=uint8_bthwc` marker is set.
3. The worker returns; `return_result` triggers the single SHM pack (one
   NPU→CPU transfer of the now-uint8 tensor).
4. The diffusion engine detects the markers in `custom_output` and bypasses
   the model-specific `post_process_func` (no CPU `bf16→fp32→uint8`
   conversion).
5. The API server receives the already-formatted uint8 frames and performs
   MP4 export.

Frame interpolation only runs for pipelines that register a
`worker_postprocess_func` in `_DIFFUSION_WORKER_POSTPROCESS_FUNCS`. If a
request enables `enable_frame_interpolation` against a pipeline without a
registered hook, the engine logs a warning and returns the original
(un-interpolated) video.

This design keeps the entire post-generation pipeline on the worker's NPU
until the last possible moment, avoiding any engine ↔ worker round-trip
and the ~600 ms CPU dtype/permute the engine would otherwise spend.

## Example

Start the server:

```bash
vllm serve Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --omni \
  --port 8091 \
  --frame-interpolation-model-path /path/to/rife-4.22.lite \
  --preload-frame-interpolation-model
```

Run a sync request with interpolation enabled:

```bash
curl -X POST http://localhost:8091/v1/videos/sync \
  -F "prompt=A dog running through a park" \
  -F "num_frames=81" \
  -F "width=832" \
  -F "height=480" \
  -F "fps=16" \
  -F "num_inference_steps=40" \
  -F "guidance_scale=1.0" \
  -F "guidance_scale_2=1.0" \
  -F "enable_frame_interpolation=true" \
  -F "frame_interpolation_exp=1" \
  -F "frame_interpolation_scale=1.0" \
  -F "seed=42" \
  -o sync_t2v_interpolated.mp4
```

## Notes

- This is a post-processing feature. It does not modify the diffusion denoising
  schedule.
- Higher interpolation exponents increase post-processing time and memory usage.
- `--preload-frame-interpolation-model` shifts RIFE load time and device memory
  allocation from the first interpolated request to service startup. Use it
  when predictable first-request latency is more important than the extra
  startup time.
- If the interpolation model weights are not available locally,
  `--frame-interpolation-model-path` may point to a Hugging Face repo containing
  `flownet.pkl`. If this option is omitted, vLLM-Omni resolves the default
  `elfgum/RIFE-4.22.lite` repo at load time.
