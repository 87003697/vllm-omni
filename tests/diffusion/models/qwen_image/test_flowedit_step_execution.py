# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical equivalence: FlowEdit step execution vs forward().

This test verifies that the step execution path (prepare_encode → denoise_step
→ step_scheduler → post_decode) produces bit-exact results compared to the
monolithic forward() path.

Run on Koala GPU pod:
    uv run python -m pytest tests/diffusion/models/qwen_image/test_flowedit_step_execution.py -xvs
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.fixture
def mock_pipeline():
    """Create a QwenImageFlowEditPipeline with fully mocked components.

    The mock transformer is deterministic (seeded by timestep+input shape)
    to allow exact comparison between forward() and step execution paths.
    """
    from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image_flowedit import (
        QwenImageFlowEditPipeline,
    )

    SEED = 42
    HIDDEN_DIM = 16
    SEQ_SRC = 4
    SEQ_COND = 2
    PROMPT_LEN = 5
    VAE_SCALE = 8

    def deterministic_forward(hidden_states, timestep, **kwargs):
        """Transformer mock: deterministic function of shape + timestep."""
        B, S, D = hidden_states.shape
        t_val = timestep[0].item() if timestep.ndim > 0 else timestep.item()
        gen = torch.Generator().manual_seed(int(abs(t_val) * 1e6) % (2**31) + B * 100 + S)
        noise = torch.randn(B, S, D, generator=gen)
        return (noise + hidden_states * 0.05,)

    with patch.object(QwenImageFlowEditPipeline, "__init__", lambda self, **kw: None):
        pipe = QwenImageFlowEditPipeline()

    # Core attributes
    pipe.vae_scale_factor = VAE_SCALE
    pipe._attention_kwargs = {}
    pipe._guidance_scale = 1.0
    pipe._current_timestep = None
    pipe._interrupt = False
    pipe.device = torch.device("cpu")
    pipe.tokenizer_max_length = 1024
    pipe.latent_channels = HIDDEN_DIM

    # Transformer mock
    pipe.transformer = MagicMock()
    pipe.transformer.in_channels = HIDDEN_DIM * 4
    pipe.transformer.guidance_embeds = True
    pipe.transformer.do_true_cfg = True
    pipe.predict_noise = lambda **kwargs: deterministic_forward(**kwargs)[0]

    # Scheduler mock
    pipe.scheduler = MagicMock()
    pipe.scheduler.config = {
        "base_image_seq_len": 256,
        "max_image_seq_len": 4096,
        "base_shift": 0.5,
        "max_shift": 1.15,
    }

    # VAE mock
    pipe.vae = MagicMock()
    pipe.vae.dtype = torch.float32
    pipe.vae.config = SimpleNamespace(
        z_dim=HIDDEN_DIM // 2,
        latents_mean=[0.0] * (HIDDEN_DIM // 2),
        latents_std=[1.0] * (HIDDEN_DIM // 2),
    )
    pipe.vae.decode = lambda latents, return_dict=False: (latents.unsqueeze(2),)

    # Encode prompt mock
    def _encode_prompt(prompt=None, image=None, num_images_per_prompt=1,
                       max_sequence_length=1024, prompt_name=None, **kwargs):
        s = SEED + 3000 if prompt_name == "negative_prompt" else SEED + 2000
        embeds = torch.randn(1, PROMPT_LEN, HIDDEN_DIM, generator=torch.Generator().manual_seed(s))
        mask = torch.ones(1, PROMPT_LEN, dtype=torch.bool)
        return embeds, mask

    pipe.encode_prompt = _encode_prompt

    # prepare_latents mock: returns (noise_latents_unused, image_latents)
    def _prepare_latents(images, batch_size, num_channels, height, width,
                         dtype, device, generator, latents):
        total_seq = SEQ_SRC + SEQ_COND
        img_lat = torch.randn(1, total_seq, HIDDEN_DIM,
                              generator=torch.Generator().manual_seed(SEED + 5000))
        return None, img_lat

    pipe.prepare_latents = _prepare_latents

    # prepare_timesteps mock
    def _prepare_timesteps(num_steps, sigmas, seq_len):
        ts = torch.linspace(1000, 1000 / num_steps, num_steps)
        return ts, num_steps

    pipe.prepare_timesteps = _prepare_timesteps

    # _unpack_latents stub (identity for testing)
    pipe._unpack_latents = lambda lat, h, w, s: lat

    # progress_bar (no-op)
    from contextlib import contextmanager

    @contextmanager
    def _noop_pbar(total=None):
        class PB:
            def update(self): pass
        yield PB()

    pipe.progress_bar = _noop_pbar

    # Provide the fixture data dimensions for test functions
    pipe._test_dims = SimpleNamespace(
        seq_src=SEQ_SRC, seq_cond=SEQ_COND,
        hidden_dim=HIDDEN_DIM, vae_scale=VAE_SCALE,
        seed=SEED, prompt_len=PROMPT_LEN,
    )

    return pipe


def _make_request(pipe, cfg_scale_tgt=7.5, cfg_scale_src=None, n_max=None, num_steps=4, seed=42):
    """Build a request dict and matching sampling params."""
    dims = pipe._test_dims
    # vae_image_sizes = (width, height) for each image
    vh0 = dims.seq_src * (dims.vae_scale * 2)
    vw0 = dims.vae_scale * 2
    vh1 = dims.seq_cond * (dims.vae_scale * 2)
    vw1 = dims.vae_scale * 2

    prompt_dict = {
        "prompt": "edit the object",
        "negative_prompt": "bad quality",
        "additional_information": {
            "condition_images": [None, None],
            "vae_images": [
                torch.randn(1, 3, vh0, vw0),
                torch.randn(1, 3, vh1, vw1),
            ],
            "vae_image_sizes": [(vw0, vh0), (vw1, vh1)],
        },
    }

    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    sampling = OmniDiffusionSamplingParams(
        height=vh0,
        width=vw0,
        num_inference_steps=num_steps,
        true_cfg_scale=cfg_scale_tgt,
        true_cfg_scale_src=cfg_scale_src,
        n_max=n_max,
        generator=torch.Generator().manual_seed(seed),
        guidance_scale=1.0,
    )

    req = SimpleNamespace(
        prompts=[prompt_dict],
        sampling_params=sampling,
        is_dummy_run=lambda: False,
    )

    return req, prompt_dict, sampling


def _run_forward(pipe, req):
    """Run the forward() path and extract the output (z_edit latent)."""
    result = pipe.forward(req)
    return result.output


def _run_step_execution(pipe, prompt_dict, sampling, seed=42):
    """Run the step execution path and extract the output (z_edit latent)."""
    from vllm_omni.diffusion.worker.utils import DiffusionRequestState
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    # Create fresh sampling params (same seed)
    fresh_sampling = OmniDiffusionSamplingParams(
        height=sampling.height,
        width=sampling.width,
        num_inference_steps=sampling.num_inference_steps,
        true_cfg_scale=sampling.true_cfg_scale,
        true_cfg_scale_src=sampling.true_cfg_scale_src,
        n_max=sampling.n_max,
        generator=torch.Generator().manual_seed(seed),
        guidance_scale=sampling.guidance_scale if sampling.guidance_scale_provided else 1.0,
    )

    state = DiffusionRequestState(
        request_id="test-0",
        sampling=fresh_sampling,
        prompts=[prompt_dict],
    )

    state = pipe.prepare_encode(state)

    for _ in range(state.total_steps):
        input_batch = SimpleNamespace(
            latents=state.latents,
            image_latents=state.sampling.image_latent,
            timesteps=state.current_timestep,
            prompt_embeds=state.prompt_embeds,
            prompt_embeds_mask=state.prompt_embeds_mask,
            negative_prompt_embeds=state.negative_prompt_embeds,
            negative_prompt_embeds_mask=state.negative_prompt_embeds_mask,
            guidance=state.guidance,
            do_true_cfg=state.do_true_cfg,
            true_cfg_scale=state.sampling.true_cfg_scale,
            cfg_normalize=state.sampling.cfg_normalize,
            img_shapes=state.img_shapes,
            txt_seq_lens=state.txt_seq_lens,
            negative_txt_seq_lens=state.negative_txt_seq_lens,
        )
        noise_pred = pipe.denoise_step(input_batch)
        assert noise_pred is not None, "denoise_step returned None unexpectedly"
        pipe.step_scheduler(state, noise_pred)

    assert state.denoise_completed, f"Expected completion, got step_index={state.step_index}/{state.total_steps}"
    result = pipe.post_decode(state)
    return result.output


class TestFlowEditStepExecution:
    def test_basic_equivalence(self, mock_pipeline):
        """Step execution must produce bit-exact same z_edit as forward()."""
        req, prompt_dict, sampling = _make_request(mock_pipeline)
        fwd = _run_forward(mock_pipeline, req)
        step = _run_step_execution(mock_pipeline, prompt_dict, sampling)
        assert torch.equal(fwd, step), (
            f"Outputs differ! max_diff={( fwd - step).abs().max().item():.2e}"
        )

    def test_equivalence_with_n_max(self, mock_pipeline):
        """n_max trimming must produce the same result in both paths."""
        req, prompt_dict, sampling = _make_request(
            mock_pipeline, num_steps=6, n_max=4,
        )
        fwd = _run_forward(mock_pipeline, req)
        step = _run_step_execution(mock_pipeline, prompt_dict, sampling)
        assert torch.equal(fwd, step), (
            f"n_max: outputs differ! max_diff={(fwd - step).abs().max().item():.2e}"
        )

    def test_equivalence_custom_cfg_scales(self, mock_pipeline):
        """Custom cfg_scale_src != -cfg_scale_tgt must also match."""
        req, prompt_dict, sampling = _make_request(
            mock_pipeline, cfg_scale_tgt=5.0, cfg_scale_src=-3.0,
        )
        fwd = _run_forward(mock_pipeline, req)
        step = _run_step_execution(mock_pipeline, prompt_dict, sampling)
        assert torch.equal(fwd, step), (
            f"Custom CFG: outputs differ! max_diff={(fwd - step).abs().max().item():.2e}"
        )

    def test_denoise_completed_flag(self, mock_pipeline):
        """State must report denoise_completed after all steps."""
        from vllm_omni.diffusion.worker.utils import DiffusionRequestState
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        req, prompt_dict, sampling = _make_request(mock_pipeline, num_steps=3)
        fresh_sampling = OmniDiffusionSamplingParams(
            height=sampling.height,
            width=sampling.width,
            num_inference_steps=3,
            true_cfg_scale=sampling.true_cfg_scale,
            generator=torch.Generator().manual_seed(42),
            guidance_scale=1.0,
        )
        state = DiffusionRequestState(
            request_id="test-flag",
            sampling=fresh_sampling,
            prompts=[prompt_dict],
        )
        state = mock_pipeline.prepare_encode(state)

        assert not state.denoise_completed
        assert state.total_steps == 3

        for i in range(3):
            input_batch = SimpleNamespace(
                latents=state.latents,
                image_latents=state.sampling.image_latent,
                timesteps=state.current_timestep,
                prompt_embeds=state.prompt_embeds,
                prompt_embeds_mask=state.prompt_embeds_mask,
                negative_prompt_embeds=state.negative_prompt_embeds,
                negative_prompt_embeds_mask=state.negative_prompt_embeds_mask,
                guidance=state.guidance,
                do_true_cfg=state.do_true_cfg,
                true_cfg_scale=state.sampling.true_cfg_scale,
                cfg_normalize=state.sampling.cfg_normalize,
                img_shapes=state.img_shapes,
                txt_seq_lens=state.txt_seq_lens,
                negative_txt_seq_lens=state.negative_txt_seq_lens,
            )
            noise_pred = mock_pipeline.denoise_step(input_batch)
            mock_pipeline.step_scheduler(state, noise_pred)

        assert state.denoise_completed
