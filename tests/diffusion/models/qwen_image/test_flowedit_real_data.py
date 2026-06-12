#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
FlowEdit Step Execution — Real Data Consistency Test

Uses pre-rendered TRELLIS2 images (renders + conditions) from S3 to verify
forward() vs step execution produce bit-exact results on real data.

Data layout (S3 or local):
    eval_flowedit/
    ├── renders/{sample}/v{0-7}.png
    ├── conditions/{sample}.png
    └── metadata.json

Usage on Koala:
    # 1. Pull data from S3
    s5cmd sync "s3://arcwm-code-us-west-2/ericzyma/eval_flowedit/" /local-ssd/eval_flowedit/

    # 2. Run test
    /tmp/uv-venv-omni/bin/python tests/diffusion/models/qwen_image/test_flowedit_real_data.py \
        --data-dir /local-ssd/eval_flowedit \
        --num-inference-steps 10 \
        --max-samples 2 --max-views 2

    # Full run (all 5 samples × 8 views, ~40 pairs)
    /tmp/uv-venv-omni/bin/python tests/diffusion/models/qwen_image/test_flowedit_real_data.py \
        --data-dir /local-ssd/eval_flowedit --num-inference-steps 10

    # With optimal eval params
    /tmp/uv-venv-omni/bin/python tests/diffusion/models/qwen_image/test_flowedit_real_data.py \
        --data-dir /local-ssd/eval_flowedit \
        --num-inference-steps 28 --cfg-scale-tgt 7.5 --cfg-scale-src 0 --n-max 21
"""

import argparse
import json
import os
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
    get_qwen_image_flowedit_pre_process_func,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.utils import DiffusionRequestState
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def parse_args():
    p = argparse.ArgumentParser(description="FlowEdit real-data consistency test")
    p.add_argument("--data-dir", required=True, help="Path to eval_flowedit/ with renders/ and conditions/")
    p.add_argument("--model", default="Qwen/Qwen-Image-Edit-2511")
    p.add_argument("--num-inference-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cfg-scale-tgt", type=float, default=7.5)
    p.add_argument("--cfg-scale-src", type=float, default=None)
    p.add_argument("--n-max", type=int, default=None)
    p.add_argument("--max-samples", type=int, default=None, help="Limit number of samples (for quick test)")
    p.add_argument("--max-views", type=int, default=None, help="Limit views per sample (for quick test)")
    p.add_argument("--prompt", default="Rotate the camera. White background.")
    p.add_argument("--save-images", action="store_true", help="Save forward/step output images for visual check")
    p.add_argument("--output-dir", default="/tmp/flowedit_real_test")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def discover_samples(data_dir: Path, max_samples=None, max_views=None):
    """Discover (sample_name, render_path, condition_path) pairs from data dir."""
    renders_dir = data_dir / "renders"
    conditions_dir = data_dir / "conditions"

    if not renders_dir.exists():
        print(f"ERROR: renders dir not found: {renders_dir}")
        sys.exit(1)
    if not conditions_dir.exists():
        print(f"ERROR: conditions dir not found: {conditions_dir}")
        sys.exit(1)

    pairs = []
    sample_dirs = sorted([d for d in renders_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])
    if max_samples:
        sample_dirs = sample_dirs[:max_samples]

    for sample_dir in sample_dirs:
        name = sample_dir.name
        cond_path = conditions_dir / f"{name}.png"
        if not cond_path.exists():
            print(f"  WARN: condition not found for {name}, skipping")
            continue

        view_files = sorted([f for f in sample_dir.glob("v*.png") if not f.name.startswith(".")])
        if max_views:
            view_files = view_files[:max_views]

        for vf in view_files:
            pairs.append((name, vf.stem, vf, cond_path))

    return pairs


def run_forward(pipeline, prompt_dict, sampling_params):
    req = OmniDiffusionRequest(
        prompts=[prompt_dict],
        sampling_params=sampling_params,
        request_id="test-fwd",
    )
    with torch.no_grad():
        result = pipeline.forward(req)
    return result.output


def run_step_execution(pipeline, prompt_dict, sampling_params):
    state = DiffusionRequestState(
        request_id="test-step",
        sampling=sampling_params,
        prompts=[prompt_dict],
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
                cfg_normalize=getattr(state.sampling, "cfg_normalize", True),
                img_shapes=state.img_shapes,
                txt_seq_lens=state.txt_seq_lens,
                negative_txt_seq_lens=state.negative_txt_seq_lens,
            )
            noise_pred = pipeline.denoise_step(input_batch)
            if noise_pred is None:
                raise RuntimeError("denoise_step returned None")
            pipeline.step_scheduler(state, noise_pred)

        result = pipeline.post_decode(state)
    return result.output


def tensor_to_pil(t):
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
    data_dir = Path(args.data_dir)

    print(f"{'=' * 70}")
    print("FlowEdit Step Execution — Real Data Consistency Test")
    print(f"{'=' * 70}")
    print(f"  Model:     {args.model}")
    print(f"  Data:      {data_dir}")
    print(f"  Steps:     {args.num_inference_steps}, Seed: {args.seed}")
    print(f"  CFG:       tgt={args.cfg_scale_tgt}, src={cfg_scale_src}")
    print(f"  n_max:     {args.n_max}")
    print(f"  Prompt:    {args.prompt[:60]}...")
    print(f"  Max:       samples={args.max_samples}, views={args.max_views}")
    print(f"{'=' * 70}\n")

    # Discover test pairs
    pairs = discover_samples(data_dir, args.max_samples, args.max_views)
    if not pairs:
        print("ERROR: no (render, condition) pairs found")
        sys.exit(1)
    print(f"Found {len(pairs)} (render, condition) pairs\n")

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

    pre_process = get_qwen_image_flowedit_pre_process_func(SimpleNamespace(model=args.model))
    print("  Pipeline loaded\n")

    if args.save_images:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    # Run tests
    results = []
    all_pass = True

    for i, (sample_name, view_name, render_path, cond_path) in enumerate(pairs):
        label = f"{sample_name}/{view_name}"
        print(f"[{i+1}/{len(pairs)}] {label}")

        source_img = Image.open(render_path).convert("RGB")
        condition_img = Image.open(cond_path).convert("RGB")

        base_prompt = {
            "prompt": args.prompt,
            "negative_prompt": " ",
            "multi_modal_data": {"image": [source_img, condition_img]},
        }

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
        pp_h = dummy_req.sampling_params.height
        pp_w = dummy_req.sampling_params.width

        def make_sampling(seed):
            return OmniDiffusionSamplingParams(
                height=pp_h, width=pp_w,
                num_inference_steps=args.num_inference_steps,
                true_cfg_scale=args.cfg_scale_tgt,
                true_cfg_scale_src=cfg_scale_src,
                n_max=args.n_max,
                generator=torch.Generator(device=args.device).manual_seed(seed),
                guidance_scale=1.0,
            )

        # Forward
        t0 = time.perf_counter()
        out_fwd = run_forward(pipeline, processed_prompt, make_sampling(args.seed))
        torch.cuda.synchronize()
        t_fwd = time.perf_counter() - t0

        # Step execution
        t0 = time.perf_counter()
        out_step = run_step_execution(pipeline, processed_prompt, make_sampling(args.seed))
        torch.cuda.synchronize()
        t_step = time.perf_counter() - t0

        is_exact = torch.equal(out_fwd, out_step)
        max_diff = (out_fwd - out_step).abs().max().item()
        status = "PASS" if is_exact else "FAIL"
        if not is_exact:
            all_pass = False

        print(f"  {status}  exact={is_exact}  max_diff={max_diff:.2e}  fwd={t_fwd:.1f}s  step={t_step:.1f}s")

        results.append({
            "sample": sample_name, "view": view_name,
            "exact": is_exact, "max_diff": max_diff,
            "fwd_time": round(t_fwd, 2), "step_time": round(t_step, 2),
        })

        if args.save_images:
            sample_dir = out_dir / sample_name
            sample_dir.mkdir(parents=True, exist_ok=True)
            tensor_to_pil(out_fwd).save(sample_dir / f"{view_name}_fwd.png")
            tensor_to_pil(out_step).save(sample_dir / f"{view_name}_step.png")

    # Summary
    n_pass = sum(1 for r in results if r["exact"])
    n_total = len(results)
    avg_fwd = sum(r["fwd_time"] for r in results) / n_total if n_total else 0
    avg_step = sum(r["step_time"] for r in results) / n_total if n_total else 0

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Passed: {n_pass}/{n_total}")
    print(f"  Avg forward time:  {avg_fwd:.2f}s")
    print(f"  Avg step time:     {avg_step:.2f}s")
    print(f"  Max diff (worst):  {max(r['max_diff'] for r in results):.2e}")

    if args.save_images:
        summary_path = out_dir / "results.json"
        with open(summary_path, "w") as f:
            json.dump({"params": vars(args), "results": results}, f, indent=2, default=str)
        print(f"  Results saved: {summary_path}")

    if all_pass:
        print(f"\n  ALL {n_total} PAIRS BIT-EXACT")
    else:
        print(f"\n  {n_total - n_pass} PAIRS FAILED")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
