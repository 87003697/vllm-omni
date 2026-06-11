"""Evaluate FlowEdit as 3D guidance: edit TRELLIS2 renders toward condition image.

Metrics:
- CLIP similarity (before/after vs condition)
- DINO similarity (before/after vs condition)
- Silhouette IoU (render vs edited, shape preservation)

Usage:
    /tmp/uv-venv-omni/bin/python scripts/eval/eval_flowedit_guidance.py \
        --renders /local-ssd/eval_flowedit --server http://localhost:8092
"""
import argparse
import base64
import json
import subprocess
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--renders", type=str, default="/local-ssd/eval_flowedit")
    parser.add_argument("--server", type=str, default="http://localhost:8092")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--s3-source", type=str,
                        default="s3://arcwm-code-us-west-2/ericzyma/eval_flowedit/")
    parser.add_argument("--prompt", type=str, default="Rotate the camera. White background.")
    parser.add_argument("--cfg-tgt", type=float, default=7.5)
    parser.add_argument("--cfg-src", type=float, default=-7.5)
    parser.add_argument("--n-max", type=int, default=28)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-edit", action="store_true", help="Skip editing, only compute metrics on existing results")
    return parser.parse_args()


# ============================================================================
# FlowEdit API
# ============================================================================

def img_to_b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def call_flowedit(server: str, source: Image.Image, condition: Image.Image,
                  prompt: str, cfg_tgt: float, cfg_src: float,
                  n_max: int, steps: int, seed: int) -> Image.Image:
    """Call FlowEdit API and return edited image."""
    payload = {
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_to_b64(source)}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_to_b64(condition)}"}},
            ]
        }],
        "extra_body": {
            "num_inference_steps": steps,
            "guidance_scale": 1,
            "true_cfg_scale": cfg_tgt,
            "true_cfg_scale_src": cfg_src,
            "n_max": n_max,
            "seed": seed,
        }
    }
    resp = requests.post(f"{server}/v1/chat/completions", json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    b64_url = data["choices"][0]["message"]["content"][0]["image_url"]["url"]
    b64_str = b64_url.split(",", 1)[1]
    return Image.open(BytesIO(base64.b64decode(b64_str))).convert("RGB")


def wait_for_server(server: str, timeout: int = 300):
    """Wait for FlowEdit server to be ready."""
    print(f"Waiting for server at {server}...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{server}/health", timeout=5)
            if resp.status_code == 200:
                print("Server ready.")
                return
        except requests.ConnectionError:
            pass
        time.sleep(5)
    raise TimeoutError(f"Server not ready after {timeout}s")


# ============================================================================
# Metrics
# ============================================================================

class CLIPMetric:
    def __init__(self, device="cuda"):
        import os
        from transformers import CLIPModel, CLIPProcessor
        token = os.environ.get("HF_TOKEN")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", token=token).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", token=token)
        self.device = device

    @torch.no_grad()
    def similarity(self, img1: Image.Image, img2: Image.Image) -> float:
        inputs = self.processor(images=[img1, img2], return_tensors="pt").to(self.device)
        vision_outputs = self.model.vision_model(pixel_values=inputs["pixel_values"])
        feats = self.model.visual_projection(vision_outputs.pooler_output)
        feats = F.normalize(feats, dim=-1)
        return feats[0].dot(feats[1]).item()


class DINOMetric:
    def __init__(self, model_name="facebook/dinov2-vitl14", device="cuda"):
        import os
        from transformers import AutoModel, AutoImageProcessor
        token = os.environ.get("HF_TOKEN")
        self.model = AutoModel.from_pretrained(model_name, token=token).to(device).eval()
        self.processor = AutoImageProcessor.from_pretrained(model_name, token=token)
        self.device = device

    @torch.no_grad()
    def similarity(self, img1: Image.Image, img2: Image.Image) -> float:
        inputs = self.processor(images=[img1, img2], return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        feats = outputs.last_hidden_state[:, 0]  # CLS token
        feats = F.normalize(feats, dim=-1)
        return feats[0].dot(feats[1]).item()


def silhouette_iou(img1: Image.Image, img2: Image.Image, threshold: float = 240) -> float:
    """Compute IoU of foreground silhouettes (white-background images)."""
    img1 = img1.convert("RGB")
    img2 = img2.convert("RGB")
    if img1.size != img2.size:
        img2 = img2.resize(img1.size, Image.LANCZOS)
    arr1 = np.array(img1).astype(float)
    arr2 = np.array(img2).astype(float)
    # Foreground = pixels not close to white
    mask1 = np.mean(arr1, axis=-1) < threshold
    mask2 = np.mean(arr2, axis=-1) < threshold
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


# ============================================================================
# Main
# ============================================================================

def make_grid(images: list, ncols: int = 3) -> Image.Image:
    """Create image grid from list of PIL images."""
    if not images:
        return Image.new("RGB", (1, 1))
    w, h = images[0].size
    nrows = (len(images) + ncols - 1) // ncols
    grid = Image.new("RGB", (w * ncols, h * nrows), (255, 255, 255))
    for i, img in enumerate(images):
        grid.paste(img.resize((w, h)), (w * (i % ncols), h * (i // ncols)))
    return grid


def main():
    args = parse_args()
    renders_dir = Path(args.renders)

    # Pull from S3 if local renders don't exist
    if not (renders_dir / "metadata.json").exists():
        print(f"Renders not found locally. Pulling from {args.s3_source}...")
        subprocess.run(
            ["s5cmd", "sync", args.s3_source, str(renders_dir) + "/"],
            check=True
        )

    # Load metadata
    with open(renders_dir / "metadata.json") as f:
        metadata = json.load(f)

    samples = metadata["samples"]
    n_views = metadata["n_views"]
    print(f"Evaluating {len(samples)} samples × {n_views} views = {len(samples) * n_views} edits")

    # Output directory
    output_dir = Path(args.output) if args.output else renders_dir / "results"
    edited_dir = output_dir / "edited"
    edited_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Edit all renders
    if not args.skip_edit:
        wait_for_server(args.server)
        print("\n=== Phase 1: FlowEdit Editing ===")
        for name in tqdm(samples, desc="Samples"):
            cond_path = renders_dir / "conditions" / f"{name}.png"
            condition = Image.open(cond_path).convert("RGB")
            sample_edit_dir = edited_dir / name
            sample_edit_dir.mkdir(parents=True, exist_ok=True)

            for vi in range(n_views):
                out_path = sample_edit_dir / f"v{vi}.png"
                if out_path.exists():
                    continue
                render_path = renders_dir / "renders" / name / f"v{vi}.png"
                source = Image.open(render_path).convert("RGB")
                edited = call_flowedit(
                    args.server, source, condition, args.prompt,
                    args.cfg_tgt, args.cfg_src, args.n_max, args.steps, args.seed,
                )
                edited.save(out_path)

    # Phase 2: Compute metrics
    print("\n=== Phase 2: Computing Metrics ===")
    clip_metric = CLIPMetric()
    try:
        dino_metric = DINOMetric()
    except Exception as e:
        print(f"  WARNING: DINOv2 unavailable ({e}), using CLIP-only")
        dino_metric = None

    all_results = []
    for name in tqdm(samples, desc="Metrics"):
        cond_path = renders_dir / "conditions" / f"{name}.png"
        condition = Image.open(cond_path).convert("RGB")

        for vi in range(n_views):
            render_path = renders_dir / "renders" / name / f"v{vi}.png"
            edited_path = edited_dir / name / f"v{vi}.png"
            if not edited_path.exists():
                continue

            render_img = Image.open(render_path).convert("RGB")
            edited_img = Image.open(edited_path).convert("RGB")

            clip_before = clip_metric.similarity(render_img, condition)
            clip_after = clip_metric.similarity(edited_img, condition)
            if dino_metric:
                dino_before = dino_metric.similarity(render_img, condition)
                dino_after = dino_metric.similarity(edited_img, condition)
            else:
                dino_before = dino_after = 0.0
            sil_iou = silhouette_iou(render_img, edited_img)

            all_results.append({
                "sample": name,
                "view": vi,
                "clip_before": clip_before,
                "clip_after": clip_after,
                "clip_delta": clip_after - clip_before,
                "dino_before": dino_before,
                "dino_after": dino_after,
                "dino_delta": dino_after - dino_before,
                "silhouette_iou": sil_iou,
            })

    # Phase 3: Report
    print("\n=== Results ===")
    if not all_results:
        print("No results to report.")
        return

    clip_deltas = [r["clip_delta"] for r in all_results]
    dino_deltas = [r["dino_delta"] for r in all_results]
    sil_ious = [r["silhouette_iou"] for r in all_results]

    summary = {
        "n_samples": len(samples),
        "n_views": n_views,
        "n_edits": len(all_results),
        "params": {
            "prompt": args.prompt,
            "cfg_tgt": args.cfg_tgt,
            "cfg_src": args.cfg_src,
            "n_max": args.n_max,
            "steps": args.steps,
        },
        "metrics": {
            "clip_before_mean": float(np.mean([r["clip_before"] for r in all_results])),
            "clip_after_mean": float(np.mean([r["clip_after"] for r in all_results])),
            "clip_delta_mean": float(np.mean(clip_deltas)),
            "clip_delta_std": float(np.std(clip_deltas)),
            "dino_before_mean": float(np.mean([r["dino_before"] for r in all_results])),
            "dino_after_mean": float(np.mean([r["dino_after"] for r in all_results])),
            "dino_delta_mean": float(np.mean(dino_deltas)),
            "dino_delta_std": float(np.std(dino_deltas)),
            "silhouette_iou_mean": float(np.mean(sil_ious)),
            "silhouette_iou_std": float(np.std(sil_ious)),
        },
        "success_criteria": {
            "clip_improves": float(np.mean(clip_deltas)) > 0,
            "dino_improves": float(np.mean(dino_deltas)) > 0,
            "shape_preserved": float(np.mean(sil_ious)) > 0.8,
        },
        "per_sample": all_results,
    }

    # Print summary
    m = summary["metrics"]
    print(f"  CLIP:  {m['clip_before_mean']:.4f} → {m['clip_after_mean']:.4f}  (Δ = {m['clip_delta_mean']:+.4f} ± {m['clip_delta_std']:.4f})")
    print(f"  DINO:  {m['dino_before_mean']:.4f} → {m['dino_after_mean']:.4f}  (Δ = {m['dino_delta_mean']:+.4f} ± {m['dino_delta_std']:.4f})")
    print(f"  SilIoU: {m['silhouette_iou_mean']:.4f} ± {m['silhouette_iou_std']:.4f}")
    print()
    sc = summary["success_criteria"]
    print(f"  CLIP improves:    {'✓' if sc['clip_improves'] else '✗'}")
    print(f"  DINO improves:    {'✓' if sc['dino_improves'] else '✗'}")
    print(f"  Shape preserved:  {'✓' if sc['shape_preserved'] else '✗'}")
    verdict = all(sc.values())
    print(f"\n  {'✓ FlowEdit is viable as guidance' if verdict else '✗ FlowEdit NOT effective'}")

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {output_dir / 'results.json'}")

    # Save CSV
    import csv
    with open(output_dir / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)

    # Save visual grid (first sample: render | edited | condition per view)
    if samples:
        name = samples[0]
        grid_images = []
        condition = Image.open(renders_dir / "conditions" / f"{name}.png").convert("RGB")
        for vi in range(min(n_views, 4)):
            render_img = Image.open(renders_dir / "renders" / name / f"v{vi}.png").convert("RGB")
            edited_path = edited_dir / name / f"v{vi}.png"
            if edited_path.exists():
                edited_img = Image.open(edited_path).convert("RGB")
                grid_images.extend([render_img, edited_img, condition])
        if grid_images:
            grid = make_grid(grid_images, ncols=3)
            grid.save(output_dir / "grid.png")
            print(f"Grid saved to {output_dir / 'grid.png'}")


if __name__ == "__main__":
    main()
