# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the worker-NPU frame interpolation integration on DiffusionEngine.

These tests exercise the inline hook contract:
- ``step()`` reads ``custom_output["video_fps_multiplier"]`` and skips
  re-running RIFE in post_process_func.
- ``step()`` warns when the user requested FI but the worker hook didn't
  produce the marker (e.g. pipeline not registered).
- ``step()`` short-circuits ``post_process_func`` when worker has already
  converted to uint8 (B,T,H,W,C).
- ``_dummy_run()`` honors preload_frame_interpolation_model:
  registered hook  → request enables FI;
  unregistered     → warn and skip FI;
  registered + hook silently drops FI → RuntimeError at startup.
"""

import asyncio
from types import SimpleNamespace

import pytest
import torch
from pytest_mock import MockerFixture

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _make_engine_for_step_test(mocker: MockerFixture, *, worker_output: DiffusionOutput):
    engine = DiffusionEngine.__new__(DiffusionEngine)
    engine.od_config = SimpleNamespace(
        enable_cpu_offload=False,
        model_class_name="WanPipeline",
    )
    engine.pre_process_func = None
    engine.post_process_func = mocker.Mock(side_effect=AssertionError("post_process_func should be bypassed"))
    engine._post_process_accepts_sampling_params = True
    # async_add_req_and_wait_for_response is the awaited entry inside step()
    async def _fake_wait(_request):
        return worker_output

    engine.async_add_req_and_wait_for_response = _fake_wait
    return engine


def test_step_bypasses_post_process_when_worker_did_uint8(mocker: MockerFixture) -> None:
    """When worker emits ``video_format=uint8_bthwc`` marker, engine
    short-circuits post_process_func and uses ``.numpy()`` directly."""
    from vllm_omni.diffusion.postprocess.worker_video_postprocess import (
        VIDEO_FORMAT_KEY,
        VIDEO_FORMAT_UINT8_BTHWC,
    )
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    uint8_video = torch.zeros(1, 5, 8, 8, 3, dtype=torch.uint8)
    worker_output = DiffusionOutput(
        output=uint8_video,
        custom_output={"video_fps_multiplier": 2, VIDEO_FORMAT_KEY: VIDEO_FORMAT_UINT8_BTHWC},
    )
    engine = _make_engine_for_step_test(mocker, worker_output=worker_output)

    mocker.patch("vllm_omni.diffusion.diffusion_engine.supports_audio_output", return_value=False)
    mocker.patch.object(engine, "_check_and_start_background_loop", new=lambda: asyncio.sleep(0))

    request = OmniDiffusionRequest(
        prompts=["p"],
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1, enable_frame_interpolation=True),
        request_id="r",
    )

    outputs = asyncio.run(engine.step(request))

    assert engine.post_process_func.call_count == 0
    assert len(outputs) == 1
    assert outputs[0].images and outputs[0].images[0].shape == (1, 5, 8, 8, 3)


def test_step_warns_when_fi_requested_but_worker_silently_skipped(
    mocker: MockerFixture, caplog
) -> None:
    """If user enables FI but worker did NOT emit ``video_fps_multiplier``,
    engine must log a warning (no silent fall-through)."""
    import logging

    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    worker_output = DiffusionOutput(output=torch.zeros(1, 3, 5, 8, 8), custom_output={})
    engine = DiffusionEngine.__new__(DiffusionEngine)
    engine.od_config = SimpleNamespace(
        enable_cpu_offload=False,
        model_class_name="WanPipeline",
    )
    engine.pre_process_func = None
    engine.post_process_func = None
    engine._post_process_accepts_sampling_params = False

    async def _fake_wait(_request):
        return worker_output

    engine.async_add_req_and_wait_for_response = _fake_wait

    mocker.patch("vllm_omni.diffusion.diffusion_engine.supports_audio_output", return_value=False)
    mocker.patch.object(engine, "_check_and_start_background_loop", new=lambda: asyncio.sleep(0))

    request = OmniDiffusionRequest(
        prompts=["p"],
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1, enable_frame_interpolation=True),
        request_id="r",
    )

    with caplog.at_level(logging.WARNING):
        asyncio.run(engine.step(request))

    assert any(
        "worker did not produce video_fps_multiplier" in rec.getMessage() for rec in caplog.records
    )


def _make_dummy_run_engine(mocker: MockerFixture, *, od_config: SimpleNamespace, output: DiffusionOutput):
    engine = DiffusionEngine.__new__(DiffusionEngine)
    engine.od_config = od_config
    engine.pre_process_func = None
    engine.add_req_and_wait_for_response = mocker.Mock(return_value=output)
    mocker.patch(
        "vllm_omni.diffusion.diffusion_engine.supports_multimodal_input", return_value=(False, False)
    )
    mocker.patch(
        "vllm_omni.diffusion.diffusion_engine.get_dummy_run_num_frames", return_value=1
    )
    return engine


def test_dummy_run_warms_frame_interpolation_when_worker_hook_registered(mocker: MockerFixture) -> None:
    """With ``WanPipeline`` registered in _DIFFUSION_WORKER_POSTPROCESS_FUNCS,
    a preload request should enable FI and number-of-frames should be >= 5."""
    od_config = SimpleNamespace(
        model_class_name="WanPipeline",
        preload_frame_interpolation_model=True,
        diffusion_load_format="default",
        frame_interpolation_model_path="/models/rife",
    )
    warmup_output = DiffusionOutput(
        output=torch.zeros(1, 3, 5, 8, 8),
        custom_output={"video_fps_multiplier": 2},
    )
    engine = _make_dummy_run_engine(mocker, od_config=od_config, output=warmup_output)

    engine._dummy_run()

    warmup_request = engine.add_req_and_wait_for_response.call_args.args[0]
    assert warmup_request.sampling_params.enable_frame_interpolation is True
    assert warmup_request.sampling_params.num_frames >= 5
    assert warmup_request.sampling_params.fps == 16


def test_dummy_run_warns_when_worker_hook_not_registered(mocker: MockerFixture, caplog) -> None:
    """Preload requested but pipeline NOT in _DIFFUSION_WORKER_POSTPROCESS_FUNCS:
    skip FI in warmup and log a warning."""
    import logging

    od_config = SimpleNamespace(
        model_class_name="QwenImagePipeline",  # not registered for worker hook
        preload_frame_interpolation_model=True,
        diffusion_load_format="default",
        frame_interpolation_model_path="/models/rife",
    )
    output = DiffusionOutput(output=torch.zeros(1, 3, 1, 8, 8))
    engine = _make_dummy_run_engine(mocker, od_config=od_config, output=output)

    with caplog.at_level(logging.WARNING):
        engine._dummy_run()

    assert any(
        "no registered worker_postprocess_func" in rec.getMessage() for rec in caplog.records
    )
    warmup_request = engine.add_req_and_wait_for_response.call_args.args[0]
    assert warmup_request.sampling_params.enable_frame_interpolation is False


def test_dummy_run_raises_when_registered_hook_silently_drops_fi(mocker: MockerFixture) -> None:
    """Hook registered but didn't emit video_fps_multiplier: fail-loud at
    startup so the bug surfaces immediately."""
    od_config = SimpleNamespace(
        model_class_name="WanPipeline",
        preload_frame_interpolation_model=True,
        diffusion_load_format="default",
        frame_interpolation_model_path="/models/rife",
    )
    # No marker in custom_output: simulate a buggy hook
    output = DiffusionOutput(output=torch.zeros(1, 3, 5, 8, 8))
    engine = _make_dummy_run_engine(mocker, od_config=od_config, output=output)

    with pytest.raises(RuntimeError, match=r"frame_interpolation_model_path='/models/rife'"):
        engine._dummy_run()
