# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
FlowEdit image editing with Qwen-Image-Edit backbone.

FlowEdit (ICCV 2025) performs inversion-free editing via differential velocity
fields. It requires two images: image[0]=source (being edited), image[1:]=conditions.

Usage (basic):
    python flowedit.py \
        --image source.png condition.png \
        --prompt "Rotate the camera." \
        --output output_flowedit.png

Usage (single-image editing, pass same image twice):
    python flowedit.py \
        --image input.png input.png \
        --prompt "Make it snowy." \
        --output output_flowedit.png

Usage (custom CFG scales):
    python flowedit.py \
        --image source.png condition.png \
        --prompt "Change the lighting." \
        --cfg-scale-tgt 5.5 \
        --cfg-scale-src -5.5 \
        --n-max 20

Usage (with CFG Parallel):
    python flowedit.py \
        --image source.png condition.png \
        --prompt "Edit description" \
        --cfg-parallel-size 2

For more options, run:
    python flowedit.py --help
"""

import argparse
import os
import time
from pathlib import Path

import torch
from PIL import Image

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FlowEdit image editing with Qwen-Image-Edit backbone.")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen-Image-Edit-2511",
        help="Diffusion model name or local path (must support multi-image input).",
    )
    parser.add_argument(
        "--model-class",
        default="QwenImageFlowEditPipeline",
        help="Pipeline class name (default: QwenImageFlowEditPipeline).",
    )
    parser.add_argument(
        "--image",
        type=str,
        nargs="+",
        required=True,
        help="Input images: first is source (edited), rest are conditions. Minimum 2.",
    )
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for editing.")
    parser.add_argument("--negative-prompt", type=str, default=" ", help="Negative prompt for CFG (default: whitespace).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--cfg-scale-tgt",
        type=float,
        default=7.5,
        help="Target branch CFG scale (positive, pushes toward prompt). Default: 7.5.",
    )
    parser.add_argument(
        "--cfg-scale-src",
        type=float,
        default=None,
        help="Source branch CFG scale (negative = push away from prompt). Default: -cfg_scale_tgt.",
    )
    parser.add_argument(
        "--n-max",
        type=int,
        default=None,
        help="Only apply FlowEdit in the last n_max denoising steps. Default: num_inference_steps (all steps).",
    )
    parser.add_argument("--guidance-scale", type=float, default=1.0, help="Guidance embedding scale. Default: 1.0.")
    parser.add_argument("--num-inference-steps", type=int, default=28, help="Number of denoising steps. Default: 28.")
    parser.add_argument("--output", type=str, default="output_flowedit.png", help="Output image path.")
    parser.add_argument("--cfg-parallel-size", type=int, default=1, choices=[1, 2, 3], help="CFG parallel GPU count.")
    parser.add_argument("--enforce-eager", action="store_true", help="Disable torch.compile.")
    parser.add_argument("--enable-cpu-offload", action="store_true", help="Enable CPU offloading.")
    parser.add_argument(
        "--enable-diffusion-pipeline-profiler",
        action="store_true",
        help="Enable pipeline profiler.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if len(args.image) < 2:
        raise ValueError(
            "FlowEdit requires at least 2 images: image[0]=source, image[1:]=conditions. "
            "For single-image editing, pass the same image twice."
        )

    input_images = []
    for image_path in args.image:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image not found: {image_path}")
        input_images.append(Image.open(image_path).convert("RGB"))

    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(args.seed)

    parallel_config = DiffusionParallelConfig(cfg_parallel_size=args.cfg_parallel_size)

    omni = Omni(
        model=args.model,
        model_class=args.model_class,
        parallel_config=parallel_config,
        enforce_eager=args.enforce_eager,
        enable_cpu_offload=args.enable_cpu_offload,
        enable_diffusion_pipeline_profiler=args.enable_diffusion_pipeline_profiler,
    )
    print("Pipeline loaded")

    cfg_scale_src = args.cfg_scale_src if args.cfg_scale_src is not None else -args.cfg_scale_tgt

    print(f"\n{'=' * 60}")
    print("FlowEdit Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Pipeline: {args.model_class}")
    print(f"  Images: {len(input_images)} (1 source + {len(input_images) - 1} condition(s))")
    for idx, img in enumerate(input_images):
        role = "source" if idx == 0 else f"condition-{idx}"
        print(f"    [{role}] {args.image[idx]} ({img.size[0]}x{img.size[1]})")
    print(f"  Prompt: {args.prompt}")
    print(f"  CFG scales: tgt={args.cfg_scale_tgt}, src={cfg_scale_src}")
    print(f"  n_max: {args.n_max}, steps: {args.num_inference_steps}")
    print(f"  CFG parallel: {args.cfg_parallel_size}")
    print(f"{'=' * 60}\n")

    start = time.perf_counter()

    outputs = omni.generate(
        {
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "multi_modal_data": {"image": input_images},
        },
        OmniDiffusionSamplingParams(
            generator=generator,
            true_cfg_scale=args.cfg_scale_tgt,
            true_cfg_scale_src=cfg_scale_src,
            n_max=args.n_max,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
        ),
    )

    elapsed = time.perf_counter() - start
    print(f"Generation time: {elapsed:.2f}s")

    if not outputs:
        raise ValueError("No output generated")

    first_output = outputs[0]
    if not hasattr(first_output, "request_output") or not first_output.request_output:
        raise ValueError("No request_output found")

    req_out = first_output.request_output
    if not isinstance(req_out, OmniRequestOutput) or not hasattr(req_out, "images"):
        raise ValueError("Invalid request_output structure")

    images = req_out.images
    if not images:
        raise ValueError("No images in output")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(output_path)
    print(f"Saved to {os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()
