"""One-time TRELLIS2 inference + multi-view rendering.

Generates 3D from condition images, renders 8 views each, saves to disk and uploads to S3.

Usage:
    /tmp/uv-venv/bin/python scripts/eval/prepare_trellis2_renders.py --n-samples 5
"""
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.environ.get("TRELLIS2_PATH", "/data/work/vllm-omni/third_party/TRELLIS.2"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--data-dir", type=str, default="/local-ssd/alphaimages_v2_formatted/images")
    parser.add_argument("--weights-dir", type=str, default="/local-ssd/pretrained_weights/TRELLIS.2-4B")
    parser.add_argument("--output-dir", type=str, default="/local-ssd/eval_flowedit")
    parser.add_argument("--n-views", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--s3-dest", type=str,
                        default="s3://arcwm-code-us-west-2/ericzyma/eval_flowedit/")
    parser.add_argument("--skip-s3", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def composite_to_white_bg(image: Image.Image) -> Image.Image:
    """RGBA image -> RGB with white background."""
    rgba = image.convert("RGBA")
    bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, rgba).convert("RGB")


def build_envmap(hdri_path: str = None):
    """Build environment map for PBR rendering."""
    import cv2
    from trellis2.renderers.pbr_mesh_renderer import EnvMap

    if hdri_path is None:
        trellis_root = os.environ.get("TRELLIS2_PATH", "/data/work/vllm-omni/third_party/TRELLIS.2")
        hdri_path = os.path.join(trellis_root, "assets/hdri/studio.exr")
    hdr_bgr = cv2.imread(hdri_path, cv2.IMREAD_UNCHANGED)
    if hdr_bgr is None:
        # Fallback: uniform white envmap
        hdr_tensor = torch.ones(64, 128, 3, dtype=torch.float32, device="cuda")
        return EnvMap(hdr_tensor)
    hdr_rgb = cv2.cvtColor(hdr_bgr, cv2.COLOR_BGR2RGB)
    hdr_tensor = torch.tensor(hdr_rgb, dtype=torch.float32, device="cuda")
    return EnvMap(hdr_tensor)


def render_views(sample, n_views: int, resolution: int, envmap=None):
    """Render N views of a MeshWithVoxel at evenly-spaced yaw angles, white bg."""
    from trellis2.utils.render_utils import render_frames, yaw_pitch_r_fov_to_extrinsics_intrinsics

    if envmap is None:
        envmap = build_envmap()

    yaws = torch.linspace(0, 2 * np.pi, n_views + 1)[:-1].tolist()
    pitchs = [0.25] * n_views
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(
        yaws, pitchs, 2.0, 40
    )
    frames = render_frames(
        sample, extrinsics, intrinsics,
        {"resolution": resolution},
        verbose=False,
        envmap=envmap,
    )
    # PbrMeshRenderer returns 'shaded'; MeshRenderer returns 'color'
    color_key = "color" if "color" in frames else "shaded"
    rendered = frames[color_key]
    # Composite alpha onto white background
    if "alpha" in frames:
        result = []
        for i, img in enumerate(rendered):
            alpha = frames["alpha"][i].astype(np.float32) / 255.0
            white = np.ones_like(img, dtype=np.float32) * 255
            composited = (img.astype(np.float32) * alpha + white * (1 - alpha)).astype(np.uint8)
            result.append(composited)
        return result
    return rendered


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    renders_dir = output_dir / "renders"
    cond_dir = output_dir / "conditions"
    renders_dir.mkdir(parents=True, exist_ok=True)
    cond_dir.mkdir(parents=True, exist_ok=True)

    # Collect input images (skip macOS resource forks ._*)
    data_dir = Path(args.data_dir)
    image_paths = sorted(p for p in data_dir.glob("*.png") if not p.name.startswith("."))
    if not image_paths:
        image_paths = sorted(p for p in data_dir.glob("*.jpg") if not p.name.startswith("."))
    image_paths = image_paths[: args.n_samples]
    print(f"Processing {len(image_paths)} images from {data_dir}")

    # Load pipeline
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    print(f"Loading TRELLIS2 from {args.weights_dir}...")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.weights_dir)
    pipeline.cuda()
    envmap = build_envmap()
    print("Pipeline loaded.")

    metadata = {
        "n_samples": len(image_paths),
        "n_views": args.n_views,
        "resolution": args.resolution,
        "seed": args.seed,
        "yaw_angles_deg": [int(i * 360 / args.n_views) for i in range(args.n_views)],
        "samples": [],
    }

    for img_path in tqdm(image_paths, desc="TRELLIS2 inference"):
        name = img_path.stem
        image = Image.open(img_path)

        # Save condition image (white background)
        cond_img = composite_to_white_bg(image)
        cond_img.save(cond_dir / f"{name}.png")

        # Run TRELLIS2 pipeline
        meshes = pipeline.run(image, seed=args.seed)
        sample = meshes[0]

        # Render views
        color_frames = render_views(sample, args.n_views, args.resolution, envmap=envmap)

        # Save renders
        sample_dir = renders_dir / name
        sample_dir.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(color_frames):
            Image.fromarray(frame).save(sample_dir / f"v{i}.png")

        metadata["samples"].append(name)

        # Free GPU memory
        del meshes, sample
        torch.cuda.empty_cache()

    # Save metadata
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nRendering complete: {len(image_paths)} samples × {args.n_views} views")
    print(f"Output: {output_dir}")

    # Upload to S3
    if not args.skip_s3:
        print(f"\nUploading to {args.s3_dest}...")
        cmd = ["s5cmd", "sync", str(output_dir) + "/", args.s3_dest]
        subprocess.run(cmd, check=True)
        print("S3 upload complete.")


if __name__ == "__main__":
    main()
