# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wan2.2 post-process regression tests."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import get_wan22_post_process_func
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_i2v import get_wan22_i2v_post_process_func

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize(
    "factory",
    [
        get_wan22_post_process_func,
        get_wan22_i2v_post_process_func,
    ],
)
def test_wan22_postprocess_ignores_frame_interpolation_sampling_params(monkeypatch, factory):
    """Wan22 post_process_func should NOT run RIFE itself any more — the
    pipeline-registered worker_postprocess_func does that on-NPU. This is a
    regression test: even when sampling_params has
    enable_frame_interpolation=True, the engine-side post_process must
    only run the diffusers VideoProcessor.postprocess_video step.
    """
    calls = []

    class _FakeVideoProcessor:
        def __init__(self, vae_scale_factor):
            self.vae_scale_factor = vae_scale_factor

        def postprocess_video(self, video, output_type="np"):
            calls.append((self.vae_scale_factor, video, output_type))
            return ["processed-video"]

    monkeypatch.setattr("diffusers.video_processor.VideoProcessor", _FakeVideoProcessor)

    post_process = factory(SimpleNamespace())
    video = torch.zeros(1, 3, 3, 4, 4)
    sampling_params = SimpleNamespace(
        enable_frame_interpolation=True,
        frame_interpolation_exp=1,
        frame_interpolation_scale=1.0,
    )

    result = post_process(video, sampling_params=sampling_params)

    assert result == {"video": ["processed-video"], "custom_output": {}}
    assert len(calls) == 1
    assert calls[0][0] == 8
    assert calls[0][1] is video
    assert calls[0][2] == "np"
    assert sampling_params.enable_frame_interpolation is True
