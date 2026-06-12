#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
FlowEdit Step Execution — Numerical Consistency Test (Pipeline-Level)

Tests numerical equivalence between:
  1. QwenImageFlowEditPipeline.forward() — the monolithic path
  2. prepare_encode → denoise_step loop → step_scheduler → post_decode — the step path

Both paths run on the SAME loaded pipeline (single model load, single GPU),
sharing all weights and state, to eliminate environmental differences.

Usage on Koala:
    uv run python tests/diffusion/models/qwen_image/test_flowedit_step_vs_forward.py

    # With custom parameters
    uv run python tests/diffusion/models/qwen_image/test_flowedit_step_vs_forward.py \
        --model Qwen/Qwen-Image-Edit-2511 \
        --num-inference-steps 10 \
        --seed 42 \
        --cfg-scale-tgt 7.5
"""

import argparse
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image


def init_single_gpu_distributed():
    """Initialize vLLM distributed state for single-GPU testing."""
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")

    from vllm_omni.diffusion.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    init_distributed_environment()
    initialize_model_parallel(
        data_parallel_size=1,
        cfg_parallel_size=1,
        sequence_parallel_size=1,
        ulysses_degree=1,
        ring_degree=1,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
    )


from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image_flowedit import (
    QwenImageFlowEditPipeline,
    get_qwen_image_flowedit_pre_process_func,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.utils import DiffusionRequestState
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def parse_args():
    parser = argparse.ArgumentParser(description="FlowEdit step execution numerical test")
    parser.add_argument("--model", default="Qwen/Qwen-Image-Edit-2511")
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg-scale-tgt", type=float, default=7.5)
    parser.add_argument("--cfg-scale-src", type=float, default=None)
    parser.add_argument("--n-max", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def create_test_image(size):
    rng = np.random.default_rng(12345)
    arr = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def build_request(
    prompts_dict,
    sampling_params,
) -> OmniDiffusionRequest:
    """Build a request suitable for the pipeline's forward()."""
    return OmniDiffusionRequest(
        prompts=[prompts_dict],
        sampling_params=sampling_params,
        request_id="test-fwd-0",
    )


def run_forward(pipeline, prompts_dict, sampling_params):
    """Run the monolithic forward() path."""
    req = build_request(prompts_dict, sampling_params)
    with torch.no_grad():
        result = pipeline.forward(req)
    return result.output


def run_step_execution(pipeline, prompts_dict, sampling_params):
    """Run the step execution path: prepare_encode → loop(denoise_step, step_scheduler) → post_decode."""
    state = DiffusionRequestState(
        request_id="test-step-0",
        sampling=sampling_params,
        prompts=[prompts_dict],
    )

    with torch.no_grad():
        state = pipeline.prepare_encode(state)

        for _ in range(state.total_steps):
            # Build InputBatch-like namespace from state
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
                cfg_normalize=getattr(state.sampling, "cfg_normalize", True),
                img_shapes=state.img_shapes,
                txt_seq_lens=state.txt_seq_lens,
                negative_txt_seq_lens=state.negative_txt_seq_lens,
            )

            noise_pred = pipeline.denoise_step(input_batch)
            if noise_pred is None:
                raise RuntimeError("denoise_step returned None (pipeline interrupted?)")

            pipeline.step_scheduler(state, noise_pred)

        if not state.denoise_completed:
            raise RuntimeError(
                f"Denoise loop ended but state not completed: "
                f"step_index={state.step_index}/{state.total_steps}"
            )

        result = pipeline.post_decode(state)
    return result.output


def main():
    args = parse_args()
    cfg_scale_src = args.cfg_scale_src if args.cfg_scale_src is not None else -args.cfg_scale_tgt

    print(f"{'=' * 60}")
    print("FlowEdit Step Execution — Pipeline-Level Consistency Test")
    print(f"{'=' * 60}")
    print(f"  Model: {args.model}")
    print(f"  Steps: {args.num_inference_steps}, Seed: {args.seed}")
    print(f"  CFG: tgt={args.cfg_scale_tgt}, src={cfg_scale_src}")
    print(f"  n_max: {args.n_max}")
    print(f"  Image: {args.image_size}x{args.image_size}")
    print(f"  Device: {args.device}")
    print(f"{'=' * 60}\n")

    # --- Initialize distributed state ---
    print("Initializing distributed environment...")
    init_single_gpu_distributed()

    # --- Load pipeline ---
    print("Loading pipeline...")
    od_config = OmniDiffusionConfig(
        model=args.model,
        model_class_name="QwenImageFlowEditPipeline",
        parallel_config=DiffusionParallelConfig(),
        step_execution=True,
        max_num_seqs=1,
        dtype=torch.bfloat16,
    )

    from vllm.config import LoadConfig
    from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader

    loader = DiffusersPipelineLoader(LoadConfig(), od_config=od_config)
    pipeline = loader.load_model(load_device=args.device, device=torch.device(args.device))
    # Ensure VAE is same dtype as text encoder (from_pretrained may load in fp32)
    pipeline.vae = pipeline.vae.to(torch.bfloat16)
    print(f"  Pipeline loaded on {args.device}")

    # --- Prepare input ---
    image = create_test_image(args.image_size)
    pre_process = get_qwen_image_flowedit_pre_process_func(
        SimpleNamespace(model=args.model)
    )

    # Common prompt dict
    base_prompt = {
        "prompt": "Make the colors more vibrant and add warm lighting.",
        "negative_prompt": " ",
        "multi_modal_data": {"image": [image, image]},
    }

    # Apply pre-processing
    base_sampling = OmniDiffusionSamplingParams(
        height=None, width=None,
        num_inference_steps=args.num_inference_steps,
        true_cfg_scale=args.cfg_scale_tgt,
        true_cfg_scale_src=cfg_scale_src,
        n_max=args.n_max,
        guidance_scale=1.0,
    )
    dummy_req = SimpleNamespace(prompts=[base_prompt], sampling_params=base_sampling)
    pre_process(dummy_req)
    processed_prompt = dummy_req.prompts[0]

    # --- Run forward path ---
    print("\n[1/2] Running forward() path...")
    sampling_fwd = OmniDiffusionSamplingParams(
        height=dummy_req.sampling_params.height,
        width=dummy_req.sampling_params.width,
        num_inference_steps=args.num_inference_steps,
        true_cfg_scale=args.cfg_scale_tgt,
        true_cfg_scale_src=cfg_scale_src,
        n_max=args.n_max,
        generator=torch.Generator(device=args.device).manual_seed(args.seed),
        guidance_scale=1.0,
    )

    t0 = time.perf_counter()
    output_fwd = run_forward(pipeline, processed_prompt, sampling_fwd)
    t_fwd = time.perf_counter() - t0
    print(f"  Done in {t_fwd:.2f}s, output shape: {output_fwd.shape}")

    # --- Run step execution path ---
    print("[2/2] Running step execution path...")
    sampling_step = OmniDiffusionSamplingParams(
        height=dummy_req.sampling_params.height,
        width=dummy_req.sampling_params.width,
        num_inference_steps=args.num_inference_steps,
        true_cfg_scale=args.cfg_scale_tgt,
        true_cfg_scale_src=cfg_scale_src,
        n_max=args.n_max,
        generator=torch.Generator(device=args.device).manual_seed(args.seed),
        guidance_scale=1.0,
    )

    t0 = time.perf_counter()
    output_step = run_step_execution(pipeline, processed_prompt, sampling_step)
    t_step = time.perf_counter() - t0
    print(f"  Done in {t_step:.2f}s, output shape: {output_step.shape}")

    # --- Compare ---
    print(f"\n{'=' * 60}")
    print("COMPARISON")
    print(f"{'=' * 60}")

    if output_fwd.shape != output_step.shape:
        print(f"✗ FAIL: Shape mismatch! fwd={output_fwd.shape}, step={output_step.shape}")
        sys.exit(1)

    is_exact = torch.equal(output_fwd, output_step)
    max_diff = (output_fwd - output_step).abs().max().item()
    mean_diff = (output_fwd - output_step).abs().mean().item()

    print(f"  Bit-exact match: {is_exact}")
    print(f"  Max abs diff:    {max_diff:.2e}")
    print(f"  Mean abs diff:   {mean_diff:.2e}")
    print(f"  Forward time:    {t_fwd:.2f}s")
    print(f"  Step exec time:  {t_step:.2f}s")
    overhead = ((t_step / t_fwd) - 1) * 100 if t_fwd > 0 else 0
    print(f"  Overhead:        {overhead:.1f}%")

    if is_exact:
        print("\n✓ PASS: Bit-exact match!")
        sys.exit(0)
    elif args.atol > 0 and max_diff <= args.atol:
        print(f"\n✓ PASS: Within tolerance (atol={args.atol})")
        sys.exit(0)
    else:
        print(f"\n✗ FAIL: Outputs differ (max_diff={max_diff:.2e})")
        if max_diff < 1e-4:
            print("  Note: diff is small, likely FP accumulation order difference.")
            print("  Consider rerunning with --atol 1e-4")
        sys.exit(1)


if __name__ == "__main__":
    main()
