# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side, NPU-resident video post-processing helpers.

These are generic building blocks that any video pipeline can compose into
its registered ``worker_postprocess_func`` (see
``vllm_omni.diffusion.registry._DIFFUSION_WORKER_POSTPROCESS_FUNCS``).
They run inside ``DiffusionWorker.generate()`` right after ``execute_model``
returns and before ``return_result`` triggers the SHM pack, so the video
tensor stays on accelerator and no d2h is needed between diffusion and
these steps.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig

logger = init_logger(__name__)


# --- Inline-RIFE metadata broadcast encoding -------------------------------
# Layout of the int64 meta tensor that rank 0 broadcasts to peers:
#   [has_video, dtype_code, ndim, s0, s1, ..., s_{_MAX_NDIM-1}]
# Keeping it as a fixed-size NPU tensor avoids ``dist.broadcast_object_list``
# which falls back to gloo + pickle and costs ~100 ms even for a tiny dict.
_MAX_NDIM = 5
_META_LEN = 3 + _MAX_NDIM
_DTYPE_TO_CODE: dict[torch.dtype, int] = {
    torch.bfloat16: 1,
    torch.float16: 2,
    torch.float32: 3,
    torch.float64: 4,
    torch.uint8: 5,
    torch.int8: 6,
    torch.int16: 7,
    torch.int32: 8,
    torch.int64: 9,
}
_CODE_TO_DTYPE: dict[int, torch.dtype] = {v: k for k, v in _DTYPE_TO_CODE.items()}


def maybe_interpolate_video_inline(
    output: DiffusionOutput,
    sampling_params: Any,
    od_config: OmniDiffusionConfig,
    rank: int,
    group: dist.ProcessGroup | None,
) -> DiffusionOutput:
    """Run distributed RIFE on ``output.output`` while it is still on NPU.

    Multi-rank deployments commonly leave the VAE decoded tensor only on
    rank 0 (e.g. ``vae_patch_parallel`` / ``distributed_vae_executor`` both
    return ``torch.empty(0)`` on rank > 0). To run the all_gather-based
    distributed RIFE, we first HCCL-broadcast the rank-0 tensor so every
    rank holds an identical NPU-resident copy, then dispatch the
    distributed kernel. Only rank 0's reply reaches the engine via
    ``return_result``; non-rank-0 outputs are discarded by the executor,
    which is why we don't bother writing ``custom_output`` on them.

    Calling this from the pipeline's worker_postprocess_func avoids the
    engine round-trip that an explicit collective_rpc dispatch would
    force. The result stays on NPU; ``return_result``'s SHM pack is the
    single unavoidable d2h.
    """
    if not getattr(sampling_params, "enable_frame_interpolation", False):
        return output

    from vllm_omni.diffusion.postprocess.rife_interpolator import (
        _select_torch_device,
        interpolate_video_tensor,
        interpolate_video_tensor_distributed,
    )

    is_distributed = (
        dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size(group=group) > 1
    )

    # --- Single-rank fast path -----------------------------------------
    if not is_distributed:
        if rank != 0 or not isinstance(output.output, torch.Tensor) or output.output.numel() == 0:
            return output
        with torch.inference_mode():
            interpolated, multiplier = interpolate_video_tensor(
                output.output,
                exp=sampling_params.frame_interpolation_exp,
                scale=sampling_params.frame_interpolation_scale,
                model_path=od_config.frame_interpolation_model_path,
            )
        output.output = interpolated
        if not isinstance(output.custom_output, dict):
            output.custom_output = {}
        output.custom_output["video_fps_multiplier"] = multiplier
        return output

    # --- Distributed path ----------------------------------------------
    target_device = _select_torch_device()

    if rank == 0:
        has_video = (
            isinstance(output.output, torch.Tensor)
            and output.output.numel() > 0
            and 4 <= output.output.dim() <= _MAX_NDIM
        )
        if has_video:
            shape = tuple(int(x) for x in output.output.shape)
            ndim = len(shape)
            dtype_code = _DTYPE_TO_CODE.get(output.output.dtype, 0)
            meta_list = [1, dtype_code, ndim] + list(shape) + [0] * (_MAX_NDIM - ndim)
        else:
            meta_list = [0] * _META_LEN
    else:
        meta_list = [0] * _META_LEN

    # Step 1: HCCL-broadcast the metadata tensor.
    meta_tensor = torch.tensor(meta_list, dtype=torch.int64, device=target_device)
    dist.broadcast(meta_tensor, src=0, group=group)
    meta_cpu = meta_tensor.cpu().tolist()  # single sync to read back

    if not bool(meta_cpu[0]):
        return output

    dtype_code = int(meta_cpu[1])
    ndim = int(meta_cpu[2])
    shape = tuple(int(meta_cpu[3 + i]) for i in range(ndim))
    target_dtype = _CODE_TO_DTYPE.get(dtype_code)
    if target_dtype is None:
        logger.warning("unknown dtype_code=%d in inline RIFE meta; aborting", dtype_code)
        return output

    # Step 2: HCCL-broadcast the actual bytes; every rank ends up with an
    # identically-shaped NPU-resident copy. No d2h/h2d here.
    if rank == 0:
        video_tensor = output.output
        if video_tensor.device != target_device:
            video_tensor = video_tensor.to(target_device)
        if not video_tensor.is_contiguous():
            video_tensor = video_tensor.contiguous()
    else:
        video_tensor = torch.empty(shape, dtype=target_dtype, device=target_device)
    dist.broadcast(video_tensor, src=0, group=group)

    # Step 3: distributed RIFE; every rank participates in all_gather.
    with torch.inference_mode():
        interpolated, multiplier = interpolate_video_tensor_distributed(
            video_tensor,
            exp=sampling_params.frame_interpolation_exp,
            scale=sampling_params.frame_interpolation_scale,
            model_path=od_config.frame_interpolation_model_path,
            group=group,
        )

    if rank == 0:
        output.output = interpolated
        if not isinstance(output.custom_output, dict):
            output.custom_output = {}
        output.custom_output["video_fps_multiplier"] = multiplier
    return output


def make_video_worker_postprocess_func(od_config: OmniDiffusionConfig):
    """Default video pipeline worker_postprocess_func factory.

    Returned function runs distributed RIFE inline on NPU. Pipelines whose
    output matches the standard ``(B, C, T, H, W)`` layout can simply
    ``return make_video_worker_postprocess_func(od_config)`` from their
    ``get_xxx_worker_postprocess_func`` factory. Pipelines that need
    different NPU-side steps can compose the individual helpers in this
    module directly.
    """

    def worker_postprocess_func(output, *, sampling_params, rank, group):
        return maybe_interpolate_video_inline(output, sampling_params, od_config, rank, group)

    return worker_postprocess_func
