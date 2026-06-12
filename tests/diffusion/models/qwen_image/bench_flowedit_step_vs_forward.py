#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
FlowEdit: forward() vs step execution — speed benchmark + output images.

Runs both paths multiple times (warm-up + timed) and reports per-iteration
timing stats. Saves output images for visual comparison.

Usage on Koala:
    /tmp/uv-venv-omni/bin/python tests/diffusion/models/qwen_image/bench_flowedit_step_vs_forward.py

    # Custom config
    /tmp/uv-venv-omni/bin/python tests/diffusion/models/qwen_image/bench_flowedit_step_vs_forward.py \
        --num-inference-steps 28 --warmup 1 --repeats 3 --image-size 512
"""

import argparse
import os
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image


def init_single_gpu_distributed():
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
    p = argparse.ArgumentParser(description="FlowEdit speed benchmark")
    p.add_argument("--model", default="Qwen/Qwen-Image-Edit-2511")
    p.add_argument("--num-inference-steps", type=int, default=28)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cfg-scale-tgt", type=float, default=7.5)
    p.add_argument("--cfg-scale-src", type=float, default=None)
    p.add_argument("--n-max", type=int, default=None)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--output-dir", default="/tmp/flowedit_bench")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def create_test_image(size):
    rng = np.random.default_rng(12345)
    arr = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def make_sampling(args, cfg_scale_src, seed, device):
    return OmniDiffusionSamplingParams(
        height=None, width=None,
        num_inference_steps=args.num_inference_steps,
        true_cfg_scale=args.cfg_scale_tgt,
        true_cfg_scale_src=cfg_scale_src,
        n_max=args.n_max,
        generator=torch.Generator(device=device).manual_seed(seed),
        guidance_scale=1.0,
    )


def run_forward(pipeline, prompt_dict, sampling):
    req = OmniDiffusionRequest(
        prompts=[prompt_dict], sampling_params=sampling, request_id="bench-fwd",
    )
    with torch.no_grad():
        result = pipeline.forward(req)
    return result.output


def run_step_execution(pipeline, prompt_dict, sampling, seed, device):
    fresh_sampling = OmniDiffusionSamplingParams(
        height=sampling.height, width=sampling.width,
        num_inference_steps=sampling.num_inference_steps,
        true_cfg_scale=sampling.true_cfg_scale,
        true_cfg_scale_src=sampling.true_cfg_scale_src,
        n_max=sampling.n_max,
        generator=torch.Generator(device=device).manual_seed(seed),
        guidance_scale=1.0,
    )

    state = DiffusionRequestState(
        request_id="bench-step", sampling=fresh_sampling, prompts=[prompt_dict],
    )

    with torch.no_grad():
        state = pipeline.prepare_encode(state)

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
            noise_pred = pipeline.denoise_step(input_batch)
            pipeline.step_scheduler(state, noise_pred)

        result = pipeline.post_decode(state)
    return result.output


def tensor_to_pil(t):
    """Convert model output tensor to PIL Image."""
    img = (t.squeeze(0) * 0.5 + 0.5).clamp(0, 1).cpu().float()
    if img.shape[0] in (1, 3):
        img = img.permute(1, 2, 0)
    arr = (img.numpy() * 255).clip(0, 255).astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = arr.squeeze(-1)
    return Image.fromarray(arr)


def main():
    args = parse_args()
    cfg_scale_src = args.cfg_scale_src if args.cfg_scale_src is not None else -args.cfg_scale_tgt

    print(f"{'=' * 60}")
    print("FlowEdit Speed Benchmark: forward() vs step execution")
    print(f"{'=' * 60}")
    print(f"  Model:   {args.model}")
    print(f"  Steps:   {args.num_inference_steps}, Seed: {args.seed}")
    print(f"  CFG:     tgt={args.cfg_scale_tgt}, src={cfg_scale_src}")
    print(f"  n_max:   {args.n_max}")
    print(f"  Image:   {args.image_size}x{args.image_size}")
    print(f"  Warmup:  {args.warmup}, Repeats: {args.repeats}")
    print(f"  Output:  {args.output_dir}")
    print(f"{'=' * 60}\n")

    # Init
    print("Initializing distributed environment...")
    init_single_gpu_distributed()

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
    pipeline.vae = pipeline.vae.to(torch.bfloat16)
    print("  Pipeline loaded\n")

    # Prepare input
    image = create_test_image(args.image_size)
    pre_process = get_qwen_image_flowedit_pre_process_func(SimpleNamespace(model=args.model))

    base_prompt = {
        "prompt": "Make the colors more vibrant and add warm lighting.",
        "negative_prompt": " ",
        "multi_modal_data": {"image": [image, image]},
    }

    dummy_sampling = make_sampling(args, cfg_scale_src, args.seed, args.device)
    dummy_req = SimpleNamespace(prompts=[base_prompt], sampling_params=dummy_sampling)
    pre_process(dummy_req)
    prompt_dict = dummy_req.prompts[0]
    # Capture preprocessed height/width
    pp_height = dummy_req.sampling_params.height
    pp_width = dummy_req.sampling_params.width

    def fresh_sampling(seed):
        return OmniDiffusionSamplingParams(
            height=pp_height, width=pp_width,
            num_inference_steps=args.num_inference_steps,
            true_cfg_scale=args.cfg_scale_tgt,
            true_cfg_scale_src=cfg_scale_src,
            n_max=args.n_max,
            generator=torch.Generator(device=args.device).manual_seed(seed),
            guidance_scale=1.0,
        )

    # --- Warmup ---
    print(f"Warming up ({args.warmup} iteration(s) each)...")
    for i in range(args.warmup):
        run_forward(pipeline, prompt_dict, fresh_sampling(args.seed))
        run_step_execution(pipeline, prompt_dict, fresh_sampling(args.seed), args.seed, args.device)
    torch.cuda.synchronize()
    print("  Warmup done\n")

    # --- Benchmark forward() ---
    print(f"Benchmarking forward() ({args.repeats} runs)...")
    fwd_times = []
    fwd_output = None
    for i in range(args.repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = run_forward(pipeline, prompt_dict, fresh_sampling(args.seed))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        fwd_times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.3f}s")
        if fwd_output is None:
            fwd_output = out

    # --- Benchmark step execution ---
    print(f"\nBenchmarking step execution ({args.repeats} runs)...")
    step_times = []
    step_output = None
    for i in range(args.repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = run_step_execution(pipeline, prompt_dict, fresh_sampling(args.seed), args.seed, args.device)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        step_times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.3f}s")
        if step_output is None:
            step_output = out

    # --- Results ---
    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")

    fwd_mean = statistics.mean(fwd_times)
    step_mean = statistics.mean(step_times)
    fwd_std = statistics.stdev(fwd_times) if len(fwd_times) > 1 else 0
    step_std = statistics.stdev(step_times) if len(step_times) > 1 else 0
    overhead = ((step_mean / fwd_mean) - 1) * 100

    print(f"  forward():       {fwd_mean:.3f}s ± {fwd_std:.3f}s  (runs: {fwd_times})")
    print(f"  step execution:  {step_mean:.3f}s ± {step_std:.3f}s  (runs: {step_times})")
    print(f"  overhead:        {overhead:+.1f}%")

    is_exact = torch.equal(fwd_output, step_output)
    max_diff = (fwd_output - step_output).abs().max().item()
    print(f"  bit-exact match: {is_exact}  (max_diff={max_diff:.2e})")

    # --- Save images ---
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        fwd_img = tensor_to_pil(fwd_output)
        step_img = tensor_to_pil(step_output)
        fwd_img.save(out_dir / "forward.png")
        step_img.save(out_dir / "step_exec.png")
        print(f"\n  Saved: {out_dir}/forward.png")
        print(f"  Saved: {out_dir}/step_exec.png")
    except Exception as e:
        print(f"\n  Could not save images: {e}")

    # Save summary
    summary = (
        f"FlowEdit Benchmark\n"
        f"  model: {args.model}\n"
        f"  steps: {args.num_inference_steps}, n_max: {args.n_max}\n"
        f"  cfg: tgt={args.cfg_scale_tgt}, src={cfg_scale_src}\n"
        f"  image: {args.image_size}x{args.image_size}\n"
        f"  warmup: {args.warmup}, repeats: {args.repeats}\n\n"
        f"forward():       {fwd_mean:.3f}s ± {fwd_std:.3f}s\n"
        f"step execution:  {step_mean:.3f}s ± {step_std:.3f}s\n"
        f"overhead:        {overhead:+.1f}%\n"
        f"bit-exact:       {is_exact} (max_diff={max_diff:.2e})\n"
    )
    (out_dir / "summary.txt").write_text(summary)
    print(f"  Saved: {out_dir}/summary.txt")

    sys.exit(0)


if __name__ == "__main__":
    main()
