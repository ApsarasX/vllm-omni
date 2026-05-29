# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
RIFE 4.22.lite frame interpolation for vLLM-Omni video generation.

RIFE model code is vendored and adapted from:
  - https://github.com/hzwer/ECCV2022-RIFE  (MIT License)
  - https://github.com/hzwer/Practical-RIFE  (MIT License)
  Copyright (c) 2021 Zhewei Huang

The FrameInterpolator wrapper and vLLM-Omni integration code are original work.
"""

from __future__ import annotations

import os
import threading
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from vllm.logger import init_logger

logger = init_logger(__name__)

_DEFAULT_RIFE_HF_REPO = "elfgum/RIFE-4.22.lite"
_MODEL_CACHE: dict[tuple[str, str], Model] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def warp(ten_input: torch.Tensor, ten_flow: torch.Tensor) -> torch.Tensor:
    """Warp input tensor by optical flow using grid_sample."""
    ten_horizontal = (
        torch.linspace(-1.0, 1.0, ten_flow.shape[3], device=ten_flow.device)
        .view(1, 1, 1, ten_flow.shape[3])
        .expand(ten_flow.shape[0], -1, ten_flow.shape[2], -1)
    )
    ten_vertical = (
        torch.linspace(-1.0, 1.0, ten_flow.shape[2], device=ten_flow.device)
        .view(1, 1, ten_flow.shape[2], 1)
        .expand(ten_flow.shape[0], -1, -1, ten_flow.shape[3])
    )
    ten_grid = torch.cat([ten_horizontal, ten_vertical], dim=1)

    ten_flow = torch.cat(
        [
            ten_flow[:, 0:1, :, :] / ((ten_input.shape[3] - 1.0) / 2.0),
            ten_flow[:, 1:2, :, :] / ((ten_input.shape[2] - 1.0) / 2.0),
        ],
        dim=1,
    )
    grid = (ten_grid + ten_flow).permute(0, 2, 3, 1)
    return F.grid_sample(
        input=ten_input,
        grid=grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


def _conv(
    in_planes: int,
    out_planes: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    dilation: int = 1,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        ),
        nn.LeakyReLU(0.2, True),
    )


class ResConv(nn.Module):
    """Residual convolution block with learnable beta scaling."""

    def __init__(self, c: int, dilation: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) * self.beta + x)


class IFBlock(nn.Module):
    """Single-scale optical flow, mask, and feature block."""

    def __init__(self, in_planes: int, c: int = 64):
        super().__init__()
        self.conv0 = nn.Sequential(
            _conv(in_planes, c // 2, 3, 2, 1),
            _conv(c // 2, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
        )
        self.lastconv = nn.Sequential(
            nn.ConvTranspose2d(c, 4 * 13, 4, 2, 1),
            nn.PixelShuffle(2),
        )

    def forward(
        self,
        x: torch.Tensor,
        flow: torch.Tensor | None = None,
        scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False)
        if flow is not None:
            flow = (
                F.interpolate(
                    flow,
                    scale_factor=1.0 / scale,
                    mode="bilinear",
                    align_corners=False,
                )
                * 1.0
                / scale
            )
            x = torch.cat((x, flow), 1)
        feat = self.conv0(x)
        feat = self.convblock(feat)
        tmp = self.lastconv(feat)
        tmp = F.interpolate(tmp, scale_factor=scale, mode="bilinear", align_corners=False)
        flow = tmp[:, :4] * scale
        mask = tmp[:, 4:5]
        feat = tmp[:, 5:]
        return flow, mask, feat


class Head(nn.Module):
    """Feature encoder producing four-channel features at full resolution."""

    def __init__(self):
        super().__init__()
        self.cnn0 = nn.Conv2d(3, 16, 3, 2, 1)
        self.cnn1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(16, 4, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.cnn0(x)
        x = self.relu(x0)
        x1 = self.cnn1(x)
        x = self.relu(x1)
        x2 = self.cnn2(x)
        x = self.relu(x2)
        x3 = self.cnn3(x)
        return x3


class IFNet(nn.Module):
    """Four-scale IFNet optical flow network."""

    def __init__(self):
        super().__init__()
        self.block0 = IFBlock(7 + 8, c=192)
        self.block1 = IFBlock(8 + 4 + 8 + 8, c=128)
        self.block2 = IFBlock(8 + 4 + 8 + 8, c=64)
        self.block3 = IFBlock(8 + 4 + 8 + 8, c=32)
        self.encode = Head()

    def forward(
        self,
        x: torch.Tensor,
        timestep: float = 0.5,
        scale_list: list[float] | None = None,
    ) -> tuple[list[torch.Tensor], torch.Tensor, list[tuple[torch.Tensor, torch.Tensor] | torch.Tensor]]:
        if scale_list is None:
            scale_list = [8, 4, 2, 1]

        channel = x.shape[1] // 2
        img0 = x[:, :channel]
        img1 = x[:, channel:]

        if not torch.is_tensor(timestep):
            timestep = (x[:, :1].clone() * 0 + 1) * timestep
        else:
            timestep = timestep.repeat(1, 1, img0.shape[2], img0.shape[3])

        f0 = self.encode(img0[:, :3])
        f1 = self.encode(img1[:, :3])

        flow_list: list[torch.Tensor] = []
        merged: list[tuple[torch.Tensor, torch.Tensor] | torch.Tensor] = []
        mask_list: list[torch.Tensor] = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        mask = None

        for i, block in enumerate([self.block0, self.block1, self.block2, self.block3]):
            if flow is None:
                flow, mask, feat = block(
                    torch.cat((img0[:, :3], img1[:, :3], f0, f1, timestep), 1),
                    None,
                    scale=scale_list[i],
                )
            else:
                wf0 = warp(f0, flow[:, :2])
                wf1 = warp(f1, flow[:, 2:4])
                fd, m0, feat = block(
                    torch.cat(
                        (
                            warped_img0[:, :3],
                            warped_img1[:, :3],
                            wf0,
                            wf1,
                            timestep,
                            mask,
                            feat,
                        ),
                        1,
                    ),
                    flow,
                    scale=scale_list[i],
                )
                mask = m0
                flow = flow + fd

            mask_list.append(mask)
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged.append((warped_img0, warped_img1))

        mask = torch.sigmoid(mask)
        merged[3] = warped_img0 * mask + warped_img1 * (1 - mask)
        return flow_list, mask_list[3], merged


class Model:
    """Wraps IFNet and exposes RIFE-compatible load/inference helpers."""

    def __init__(self):
        self.flownet = IFNet()

    def eval(self) -> Model:
        self.flownet.eval()
        return self

    def device(self) -> torch.device:
        return next(self.flownet.parameters()).device

    def load_model(self, path: str) -> None:
        flownet_path = os.path.join(path, "flownet.pkl")
        if not os.path.isfile(flownet_path):
            raise FileNotFoundError(
                f"RIFE weight file not found: {flownet_path}. Expected layout: <model_path>/flownet.pkl"
            )

        state = torch.load(flownet_path, map_location="cpu", weights_only=False)
        state = {k.removeprefix("module."): v for k, v in state.items()}
        self.flownet.load_state_dict(state, strict=False)
        logger.info("Loaded RIFE weights from %s", flownet_path)

    def inference(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
        scale: float = 1.0,
        timestep: float = 0.5,
    ) -> torch.Tensor:
        _n, _c, h, w = img0.shape
        ph = ((h - 1) // 32 + 1) * 32
        pw = ((w - 1) // 32 + 1) * 32
        pad = (0, pw - w, 0, ph - h)
        img0 = F.pad(img0, pad)
        img1 = F.pad(img1, pad)

        imgs = torch.cat((img0, img1), 1)
        scale_list = [8 / scale, 4 / scale, 2 / scale, 1 / scale]
        with torch.no_grad():
            _flow_list, _mask, merged = self.flownet(
                imgs,
                timestep=timestep,
                scale_list=scale_list,
            )
        return merged[3][:, :, :h, :w]


def _resolve_rife_model_path(model_path: str | None) -> str:
    model_path = model_path or _DEFAULT_RIFE_HF_REPO
    if os.path.isdir(model_path):
        return model_path
    from vllm_omni.model_executor.model_loader.weight_utils import (
        download_weights_from_hf_specific,
    )

    return download_weights_from_hf_specific(
        model_path,
        cache_dir=None,
        allow_patterns=["flownet.pkl"],
        require_all=True,
    )


def _select_torch_device() -> torch.device:
    try:
        from vllm_omni.platforms import current_omni_platform

        return current_omni_platform.get_torch_device()
    except Exception as exc:
        logger.warning("Failed to resolve current vLLM-Omni torch device: %s", exc)

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _normalize_video_tensor_layout(video: torch.Tensor) -> tuple[torch.Tensor, Any]:
    if video.ndim == 5:
        if video.shape[1] in (3, 4):
            return video, lambda out: out
        if video.shape[2] in (3, 4):
            return video.permute(0, 2, 1, 3, 4), lambda out: out.permute(0, 2, 1, 3, 4)
    elif video.ndim == 4:
        if video.shape[0] in (3, 4):
            return video.unsqueeze(0), lambda out: out.squeeze(0)
        if video.shape[1] in (3, 4):
            return video.permute(1, 0, 2, 3).unsqueeze(0), lambda out: out.squeeze(0).permute(1, 0, 2, 3)
    raise ValueError(f"Unsupported video tensor shape for interpolation: {tuple(video.shape)}")


def get_video_frame_count(video: torch.Tensor) -> int:
    """Return the number of frames using RIFE's supported tensor layouts."""
    normalized_video, _ = _normalize_video_tensor_layout(video)
    return int(normalized_video.shape[2])


def get_frame_interpolation_pair_range(num_pairs: int, world_size: int, rank: int) -> tuple[int, int]:
    """Return the contiguous adjacent-frame pair range assigned to a rank.

    All ranks together cover ``[0, num_pairs)``; the remainder (when
    ``num_pairs`` is not divisible by ``world_size``) is distributed to the
    low-rank ranks, so neighbouring shards still join cleanly without
    overlap.
    """
    if num_pairs < 0:
        raise ValueError(f"num_pairs must be >= 0, got {num_pairs}")
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")

    base = num_pairs // world_size
    remainder = num_pairs % world_size
    start = rank * base + min(rank, remainder)
    end = start + base + (1 if rank < remainder else 0)
    return start, end


def _normalize_video_tensor_range(video: torch.Tensor) -> tuple[torch.Tensor, Any]:
    original_dtype = video.dtype
    video = video.detach()
    if video.is_floating_point():
        video = video.to(torch.float32)
        if torch.amin(video) < 0.0 or torch.amax(video) > 1.0:
            return video.clamp(-1.0, 1.0) * 0.5 + 0.5, lambda out: (out * 2.0 - 1.0).to(original_dtype)
        return video.clamp(0.0, 1.0), lambda out: out.to(original_dtype)
    return video.to(torch.float32) / 255.0, lambda out: (out * 255.0).round().clamp(0, 255).to(original_dtype)


class FrameInterpolator:
    """Lazy-loaded RIFE 4.22.lite frame interpolator."""

    def __init__(self, model_path: str | None = None):
        self._model_path = model_path
        self._resolved_path: str | None = None

    def _ensure_model_loaded(self, preferred_device: torch.device | None = None) -> Model:
        resolved_path = _resolve_rife_model_path(self._model_path)
        self._resolved_path = resolved_path
        device = preferred_device or _select_torch_device()
        cache_key = (resolved_path, str(device))

        with _MODEL_CACHE_LOCK:
            if cache_key in _MODEL_CACHE:
                return _MODEL_CACHE[cache_key]

            model = Model()
            model.load_model(resolved_path)
            model.eval()
            model.flownet = model.flownet.to(device)
            _MODEL_CACHE[cache_key] = model
            logger.info("RIFE model loaded on device: %s", device)
            return model

    def _make_inference(
        self,
        model: Model,
        img0: torch.Tensor,
        img1: torch.Tensor,
        n: int,
        scale: float,
    ) -> list[torch.Tensor]:
        if n == 1:
            return [model.inference(img0, img1, scale=scale)]
        mid = model.inference(img0, img1, scale=scale)
        return (
            self._make_inference(model, img0, mid, n // 2, scale)
            + [mid]
            + self._make_inference(model, mid, img1, n // 2, scale)
        )

    def interpolate_tensor(
        self,
        video: torch.Tensor,
        exp: int = 1,
        scale: float = 1.0,
    ) -> tuple[torch.Tensor, int]:
        if exp < 1:
            raise ValueError(f"frame interpolation exp must be >= 1, got {exp}")
        if scale <= 0:
            raise ValueError(f"frame interpolation scale must be > 0, got {scale}")

        video, restore_layout = _normalize_video_tensor_layout(video)
        if video.shape[2] < 2:
            return restore_layout(video), 1

        video, restore_range = _normalize_video_tensor_range(video)
        # A CPU tensor may be transport/offload state rather than an execution
        # choice, so only trust it when it is already on an accelerator.
        preferred_device = video.device
        if preferred_device.type == "cpu":
            preferred_device = _select_torch_device()
        model = self._ensure_model_loaded(preferred_device=preferred_device)
        video = video.to(model.device())
        intermediates_per_pair = 2**exp // 2

        result_frames: list[torch.Tensor] = []
        for idx in range(video.shape[2] - 1):
            img0 = video[:, :, idx, :, :]
            img1 = video[:, :, idx + 1, :, :]
            result_frames.append(img0)
            result_frames.extend(self._make_inference(model, img0, img1, intermediates_per_pair, scale))
        result_frames.append(video[:, :, -1, :, :])
        result = torch.stack(result_frames, dim=2)
        return restore_layout(restore_range(result)), 2**exp

    def interpolate_tensor_pair_range(
        self,
        video: torch.Tensor,
        start_pair: int,
        end_pair: int,
        exp: int = 1,
        scale: float = 1.0,
    ) -> tuple[torch.Tensor, int]:
        """Run RIFE on a contiguous adjacent-frame pair range [start_pair, end_pair).

        Used by the distributed entry point to split work across ranks. The
        input must already be in (B, C, T, H, W) layout and value range
        [0, 1] (i.e. the caller has run ``_normalize_video_tensor_layout``
        and ``_normalize_video_tensor_range``).
        """
        if exp < 1:
            raise ValueError(f"frame interpolation exp must be >= 1, got {exp}")
        if scale <= 0:
            raise ValueError(f"frame interpolation scale must be > 0, got {scale}")

        num_pairs = video.shape[2] - 1
        if start_pair < 0 or end_pair < start_pair or end_pair > num_pairs:
            raise ValueError(
                f"Invalid pair range [{start_pair}, {end_pair}) for {num_pairs} adjacent frame pairs"
            )

        preferred_device = video.device
        if preferred_device.type == "cpu":
            preferred_device = _select_torch_device()
        model = self._ensure_model_loaded(preferred_device=preferred_device)
        video = video.to(model.device())
        if start_pair == end_pair:
            # Empty shard still reports the canonical FPS multiplier (2**exp)
            # so callers don't need to special-case it during gather: the
            # multiplier is a property of the requested interpolation, not of
            # how many pairs this rank happened to be assigned.
            return video[:, :, :0, :, :], 2**exp
        intermediates_per_pair = 2**exp // 2

        result_frames: list[torch.Tensor] = []
        for idx in range(start_pair, end_pair):
            img0 = video[:, :, idx, :, :]
            img1 = video[:, :, idx + 1, :, :]
            result_frames.append(img0)
            result_frames.extend(self._make_inference(model, img0, img1, intermediates_per_pair, scale))
        # Only the last rank in the partition appends the final frame, so
        # concatenated shards reproduce the full interpolated sequence exactly.
        if end_pair == num_pairs:
            result_frames.append(video[:, :, -1, :, :])
        return torch.stack(result_frames, dim=2), 2**exp


def interpolate_video_tensor(
    video: torch.Tensor,
    exp: int = 1,
    scale: float = 1.0,
    model_path: str | None = None,
) -> tuple[torch.Tensor, int]:
    """Interpolate a video tensor and return the FPS multiplier."""
    interpolator = FrameInterpolator(model_path=model_path)
    return interpolator.interpolate_tensor(video, exp=exp, scale=scale)


def _gather_interpolated_shards_to_rank0(
    local_video: torch.Tensor, group: dist.ProcessGroup | None
) -> torch.Tensor:
    """Gather per-rank interpolated shards back to rank 0 on the accelerator.

    Shapes can differ across ranks (the last rank carries one extra frame),
    so each rank first all_gathers its shape, pads to ``max_shape``, then
    all_gathers the padded data. HCCL's ``gather`` path is slow on NPU, so
    we use ``all_gather`` and discard the broadcast on non-zero ranks.
    """
    device = local_video.device
    rank = dist.get_rank(group=group)
    world_size = dist.get_world_size(group=group)
    shape_tensor = torch.tensor(tuple(local_video.shape), device=device, dtype=torch.int64)
    shape_tensors = [torch.empty_like(shape_tensor) for _ in range(world_size)]
    dist.all_gather(shape_tensors, shape_tensor, group=group)

    shapes = [tuple(int(v) for v in shape.cpu().tolist()) for shape in shape_tensors]
    max_shape = tuple(max(shape[dim] for shape in shapes) for dim in range(local_video.ndim))
    padded = torch.zeros(max_shape, device=device, dtype=local_video.dtype)
    if local_video.numel() > 0:
        slices = tuple(slice(0, size) for size in local_video.shape)
        padded[slices] = local_video

    gathered = [torch.empty_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded, group=group)

    if rank != 0:
        return local_video[:, :, :0, :, :]

    # Keep shards on accelerator: slice = NPU view (zero-copy), cat = single
    # NPU kernel. Avoid per-shard .cpu() which serialises N PCIe transfers.
    # Caller runs restore_range / restore_layout on NPU; only the SHM pack
    # at the end of the worker pipeline triggers a single d2h.
    shards = []
    for shard, shape in zip(gathered, shapes):
        if shape[2] == 0:
            continue
        slices = tuple(slice(0, size) for size in shape)
        shards.append(shard[slices])
    if not shards:
        return local_video[:, :, :0, :, :]
    return torch.cat(shards, dim=2)


def interpolate_video_tensor_distributed(
    video: torch.Tensor,
    exp: int = 1,
    scale: float = 1.0,
    model_path: str | None = None,
    group: dist.ProcessGroup | None = None,
) -> tuple[torch.Tensor, int]:
    """Interpolate a video tensor by assigning contiguous time ranges to ranks.

    All ranks must call this function with the same input. Rank 0 returns
    the assembled video. Non-zero ranks return an empty sentinel shard used
    only to complete collective execution; callers must not consume
    non-rank-0 results.

    The result is kept on accelerator (no .cpu() here); the caller decides
    when to incur the single d2h required for downstream IPC / serialization.
    """
    if not dist.is_available() or not dist.is_initialized():
        return interpolate_video_tensor(video, exp=exp, scale=scale, model_path=model_path)

    rank = dist.get_rank(group=group)
    world_size = dist.get_world_size(group=group)
    video, restore_layout = _normalize_video_tensor_layout(video)
    if video.shape[2] < 2:
        return restore_layout(video), 1

    # Upload to accelerator BEFORE the dtype conversion so the fp32 promotion
    # in _normalize_video_tensor_range runs on NPU (~10x faster than CPU on
    # large tensors), and we ship the smaller bf16 input over PCIe instead of
    # the fp32 promoted tensor.
    target_device = _select_torch_device() if video.device.type == "cpu" else video.device
    video = video.to(target_device, non_blocking=True)

    video, restore_range = _normalize_video_tensor_range(video)

    num_pairs = video.shape[2] - 1
    start_pair, end_pair = get_frame_interpolation_pair_range(num_pairs, world_size, rank)
    if rank == 0:
        rank_ranges = [
            get_frame_interpolation_pair_range(num_pairs, world_size, rank_id) for rank_id in range(world_size)
        ]
        logger.info(
            "RIFE distributed interpolation across %d ranks: "
            "num_frames=%d num_adjacent_pairs=%d rank_pair_ranges=%s",
            world_size,
            int(video.shape[2]),
            num_pairs,
            rank_ranges,
        )

    interpolator = FrameInterpolator(model_path=model_path)
    local_video, multiplier = interpolator.interpolate_tensor_pair_range(
        video,
        start_pair=start_pair,
        end_pair=end_pair,
        exp=exp,
        scale=scale,
    )
    merged_video = _gather_interpolated_shards_to_rank0(local_video, group)

    if rank == 0:
        merged_video = restore_layout(restore_range(merged_video))
    return merged_video, multiplier
