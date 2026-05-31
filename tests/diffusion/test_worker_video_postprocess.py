# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for worker-side NPU video post-processing helpers."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.postprocess.worker_video_postprocess import (
    VIDEO_FORMAT_KEY,
    VIDEO_FORMAT_UINT8_BTHWC,
    make_video_worker_postprocess_func,
    maybe_convert_video_to_uint8_on_npu,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _fake_npu_tensor_5d(shape=(1, 3, 5, 8, 8), dtype=torch.bfloat16) -> torch.Tensor:
    """The helper checks ``device.type != "cpu"`` to decide whether to act,
    so a real CPU tensor would be ignored. We can't allocate NPU tensors in
    CPU CI, so we monkey-patch ``device.type`` via subclass override.
    """

    class _NpuLikeTensor(torch.Tensor):
        @property
        def device(self):
            return SimpleNamespace(type="npu")

    base = torch.zeros(shape, dtype=dtype)
    return base.as_subclass(_NpuLikeTensor)


def test_uint8_conversion_produces_canonical_layout_and_marker():
    """``(B, C, T, H, W)`` bf16 in [-1, 1] should become
    ``(B, T, H, W, C)`` uint8 in [0, 255] with the bypass marker set, so
    the engine knows to skip ``post_process_func``.
    """
    video = _fake_npu_tensor_5d(shape=(1, 3, 4, 6, 8))
    # The value 0.0 in [-1, 1] should map to 128 in uint8 (round of 127.5 = 128).
    output = DiffusionOutput(output=video)

    result = maybe_convert_video_to_uint8_on_npu(output, rank=0)

    assert result.output.dtype == torch.uint8
    assert result.output.shape == (1, 4, 6, 8, 3)  # (B, T, H, W, C)
    # ((0 / 2) + 0.5) * 255 = 127.5 → round → 128
    assert torch.all(result.output == 128)
    assert isinstance(result.custom_output, dict)
    assert result.custom_output[VIDEO_FORMAT_KEY] == VIDEO_FORMAT_UINT8_BTHWC


def test_uint8_conversion_preserves_clipping_range():
    """Values outside [-1, 1] must be clamped before scaling so we never
    overflow uint8."""
    raw = torch.tensor([[[[[ -2.0, -1.0, 0.0, 1.0, 2.0]]]]])  # (1,1,1,1,5)
    # Reshape to a valid (B, C, T, H, W) with C=3 (broadcast manually).
    base = raw.expand(1, 3, 1, 1, 5).clone().to(torch.float32)

    class _NpuLikeTensor(torch.Tensor):
        @property
        def device(self):
            return SimpleNamespace(type="npu")

    video = base.as_subclass(_NpuLikeTensor)
    output = DiffusionOutput(output=video)

    result = maybe_convert_video_to_uint8_on_npu(output, rank=0)

    # After clamp(-1, 1) -> [-1, -1, 0, 1, 1]
    # → (x/2 + 0.5) = [0, 0, 0.5, 1, 1]
    # → *255 round = [0, 0, 128, 255, 255]
    assert result.output[0, 0, 0, :, 0].tolist() == [0, 0, 128, 255, 255]


def test_uint8_conversion_skips_non_rank0():
    """The output is only meaningful on rank 0; non-rank-0 must early-return
    to avoid touching their (empty) sentinel tensor."""
    sentinel = torch.empty(0)
    output = DiffusionOutput(output=sentinel)

    result = maybe_convert_video_to_uint8_on_npu(output, rank=1)

    assert result.output is sentinel
    assert not isinstance(result.custom_output, dict) or VIDEO_FORMAT_KEY not in result.custom_output


def test_uint8_conversion_skips_cpu_tensors():
    """A CPU tensor (e.g. from a pipeline that doesn't keep VAE output on
    NPU) should fall through so the engine's existing ``post_process_func``
    can handle it. Avoids forcing every pipeline to be NPU-resident."""
    cpu_tensor = torch.zeros(1, 3, 4, 8, 8, dtype=torch.bfloat16)  # device=cpu
    output = DiffusionOutput(output=cpu_tensor)

    result = maybe_convert_video_to_uint8_on_npu(output, rank=0)

    assert result.output is cpu_tensor
    assert result.output.dtype == torch.bfloat16
    assert not isinstance(result.custom_output, dict) or VIDEO_FORMAT_KEY not in result.custom_output


def test_uint8_conversion_skips_non_video_shapes():
    """A 4D image tensor or a 5D tensor with non-(3,4) channel dim isn't
    a recognized video layout; let the engine handle it."""
    img = _fake_npu_tensor_5d(shape=(1, 3, 5, 8, 8))
    # Modify the channel dim to something unexpected.
    weird = _fake_npu_tensor_5d(shape=(1, 7, 5, 8, 8))  # C=7
    for tensor in (weird,):
        output = DiffusionOutput(output=tensor)
        result = maybe_convert_video_to_uint8_on_npu(output, rank=0)
        assert result.output is tensor

    # 4D tensors are also unsupported by the helper's narrow contract.
    flat = _fake_npu_tensor_5d(shape=(1, 3, 5, 8, 8)).squeeze(0)  # 4D
    output = DiffusionOutput(output=flat)
    result = maybe_convert_video_to_uint8_on_npu(output, rank=0)
    assert result.output is flat


def test_uint8_conversion_idempotent_on_uint8_input():
    """If something already converted to uint8, calling again should be a
    no-op (integer dtype short-circuits the float-only path)."""
    already_uint8 = _fake_npu_tensor_5d(shape=(1, 3, 4, 8, 8), dtype=torch.uint8)
    output = DiffusionOutput(output=already_uint8)

    result = maybe_convert_video_to_uint8_on_npu(output, rank=0)

    assert result.output is already_uint8


def test_make_video_worker_postprocess_func_chains_helpers(monkeypatch):
    """The default factory must compose RIFE inline + uint8 conversion in
    that order — RIFE produces the post-interpolation tensor, then uint8
    formats it. Reversing the order would feed uint8 frames to RIFE which
    expects floats."""
    seen = []

    def _fake_interpolate(output, sampling_params, od_config, rank, group):
        seen.append("interpolate")
        return output

    def _fake_uint8(output, rank):
        seen.append("uint8")
        return output

    monkeypatch.setattr(
        "vllm_omni.diffusion.postprocess.worker_video_postprocess.maybe_interpolate_video_inline",
        _fake_interpolate,
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.postprocess.worker_video_postprocess.maybe_convert_video_to_uint8_on_npu",
        _fake_uint8,
    )

    od_config = SimpleNamespace()
    hook = make_video_worker_postprocess_func(od_config)

    hook(DiffusionOutput(output=None), sampling_params=None, rank=0, group=None)

    assert seen == ["interpolate", "uint8"]
