# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for OpenAI-compatible video API encoding helpers."""

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.postprocess import rife_interpolator
from vllm_omni.entrypoints.openai import video_api_utils

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _install_fake_video_mux(monkeypatch, mux_calls):
    def _fake_mux_video_audio_bytes(frames, audio, fps, audio_sample_rate, video_codec_options=None):
        mux_calls.append(
            {
                "frames": frames,
                "audio": audio,
                "fps": fps,
                "audio_sample_rate": audio_sample_rate,
                "video_codec_options": video_codec_options,
            }
        )
        return b"fake-video"

    monkeypatch.setattr(
        "vllm_omni.diffusion.utils.media_utils.mux_video_audio_bytes",
        _fake_mux_video_audio_bytes,
    )


def test_encode_video_bytes_exports_frames_without_interpolation(monkeypatch):
    mux_calls = []
    _install_fake_video_mux(monkeypatch, mux_calls)

    frames = [np.full((2, 2, 3), fill_value=i / 5, dtype=np.float32) for i in range(5)]
    video_bytes = video_api_utils._encode_video_bytes(
        frames,
        fps=8,
    )

    assert video_bytes == b"fake-video"
    assert mux_calls[0]["frames"].shape == (5, 2, 2, 3)
    assert mux_calls[0]["frames"].dtype == np.uint8
    assert mux_calls[0]["fps"] == 8.0
    assert mux_calls[0]["audio"] is None


def test_rife_model_inference_runs_on_dummy_tensors():
    model = rife_interpolator.Model().eval()
    img0 = torch.rand(1, 3, 32, 32)
    img1 = torch.rand(1, 3, 32, 32)

    output = model.inference(img0, img1, scale=1.0)

    assert output.shape == (1, 3, 32, 32)
    assert torch.isfinite(output).all()


def test_frame_interpolator_runs_actual_torch_tensor_path(monkeypatch):
    model = rife_interpolator.Model().eval()
    interpolator = rife_interpolator.FrameInterpolator()
    monkeypatch.setattr(interpolator, "_ensure_model_loaded", lambda preferred_device=None: model)

    video = torch.zeros(1, 3, 2, 32, 32)
    output_video, multiplier = interpolator.interpolate_tensor(video, exp=1, scale=1.0)

    assert multiplier == 2
    assert output_video.shape == (1, 3, 3, 32, 32)
    assert torch.isfinite(output_video).all()


def test_frame_interpolator_uses_platform_device_when_tensor_is_cpu(monkeypatch):
    chosen_devices = []
    model = rife_interpolator.Model().eval()

    def _fake_ensure_model_loaded(*, preferred_device=None):
        chosen_devices.append(preferred_device)
        return model

    interpolator = rife_interpolator.FrameInterpolator()
    monkeypatch.setattr(interpolator, "_ensure_model_loaded", _fake_ensure_model_loaded)
    monkeypatch.setattr(model.flownet, "to", lambda device: model.flownet)
    monkeypatch.setattr(rife_interpolator, "_select_torch_device", lambda: torch.device("cuda"))

    video = torch.zeros(1, 3, 2, 32, 32)
    output_video, multiplier = interpolator.interpolate_tensor(video, exp=1, scale=1.0)

    assert chosen_devices == [torch.device("cuda")]
    assert multiplier == 2
    assert output_video.shape == (1, 3, 3, 32, 32)


# --- Distributed RIFE primitives ------------------------------------------


class _FakeInterpolationModel:
    def device(self):
        return torch.device("cpu")

    def inference(self, img0, img1, scale=1.0, timestep=0.5):
        del scale, timestep
        return (img0 + img1) / 2


def test_frame_interpolation_pair_ranges_are_contiguous():
    ranges = [
        rife_interpolator.get_frame_interpolation_pair_range(num_pairs=10, world_size=2, rank=rank)
        for rank in range(2)
    ]

    assert ranges == [(0, 5), (5, 10)]


def test_frame_interpolation_pair_ranges_distribute_remainder():
    ranges = [
        rife_interpolator.get_frame_interpolation_pair_range(num_pairs=10, world_size=3, rank=rank)
        for rank in range(3)
    ]

    assert ranges == [(0, 4), (4, 7), (7, 10)]


@pytest.mark.parametrize(
    ("num_pairs", "world_size", "expected"),
    [
        (0, 4, [(0, 0), (0, 0), (0, 0), (0, 0)]),
        (1, 4, [(0, 1), (1, 1), (1, 1), (1, 1)]),
    ],
)
def test_frame_interpolation_pair_ranges_handle_degenerate_workload(num_pairs, world_size, expected):
    ranges = [
        rife_interpolator.get_frame_interpolation_pair_range(
            num_pairs=num_pairs, world_size=world_size, rank=rank
        )
        for rank in range(world_size)
    ]

    assert ranges == expected


@pytest.mark.parametrize(
    ("num_pairs", "world_size", "rank", "message"),
    [
        (-1, 2, 0, "num_pairs"),
        (5, 0, 0, "world_size"),
        (5, 2, 5, "rank"),
    ],
)
def test_frame_interpolation_pair_range_validates_inputs(num_pairs, world_size, rank, message):
    with pytest.raises(ValueError, match=message):
        rife_interpolator.get_frame_interpolation_pair_range(
            num_pairs=num_pairs,
            world_size=world_size,
            rank=rank,
        )


def test_get_video_frame_count_uses_supported_layouts():
    channels_first_5d = torch.zeros(1, 3, 7, 2, 2)
    time_first_5d = torch.zeros(1, 7, 3, 2, 2)
    channels_first_4d = torch.zeros(3, 7, 2, 2)
    time_first_4d = torch.zeros(7, 3, 2, 2)

    assert rife_interpolator.get_video_frame_count(channels_first_5d) == 7
    assert rife_interpolator.get_video_frame_count(time_first_5d) == 7
    assert rife_interpolator.get_video_frame_count(channels_first_4d) == 7
    assert rife_interpolator.get_video_frame_count(time_first_4d) == 7


def test_get_video_frame_count_prefers_channel_axis_when_ambiguous():
    """Document the known ambiguity when T and C are both in {3, 4}."""
    ambiguous = torch.zeros(1, 4, 3, 2, 2)

    assert rife_interpolator.get_video_frame_count(ambiguous) == 3


def test_frame_interpolator_pair_range_outputs_non_overlapping_segments(monkeypatch):
    interpolator = rife_interpolator.FrameInterpolator()
    monkeypatch.setattr(
        interpolator, "_ensure_model_loaded", lambda preferred_device=None: _FakeInterpolationModel()
    )

    video = torch.arange(4, dtype=torch.float32).view(1, 1, 4, 1, 1).expand(1, 3, 4, 1, 1)
    first, multiplier = interpolator.interpolate_tensor_pair_range(video, start_pair=0, end_pair=2, exp=1)
    second, _ = interpolator.interpolate_tensor_pair_range(video, start_pair=2, end_pair=3, exp=1)
    merged = torch.cat([first, second], dim=2)

    assert multiplier == 2
    assert merged[:, 0, :, 0, 0].tolist() == [[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]]


def test_frame_interpolator_pair_range_validates_input(monkeypatch):
    interpolator = rife_interpolator.FrameInterpolator()
    monkeypatch.setattr(
        interpolator, "_ensure_model_loaded", lambda preferred_device=None: _FakeInterpolationModel()
    )
    video = torch.zeros(1, 3, 4, 1, 1)

    empty, multiplier = interpolator.interpolate_tensor_pair_range(video, start_pair=2, end_pair=2, exp=1)

    assert empty.shape == (1, 3, 0, 1, 1)
    assert multiplier == 2
    with pytest.raises(ValueError, match="Invalid pair range"):
        interpolator.interpolate_tensor_pair_range(video, start_pair=-1, end_pair=2)
    with pytest.raises(ValueError, match="Invalid pair range"):
        interpolator.interpolate_tensor_pair_range(video, start_pair=0, end_pair=100)
