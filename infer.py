"""Test a trained DLL-GAN generator on a single image (CPU-friendly).

Two modes:
  * clean image (default): the script degrades it, restores it, and reports
    PSNR/SSIM vs the original -> use this to judge quality.
  * already-degraded image (--already-degraded): just restores it, no metrics.

Examples:
  python infer.py --image owl.png --ckpt checkpoints/generator.pth --task sr --level 4 --size 256
  python infer.py --image my_noisy.png --already-degraded --ckpt checkpoints/generator.pth
"""
import argparse

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from models import Generator
from degradations import apply_degradation
from datasets import _to_tensor


def to_image(t):
    """(3,H,W) in [-1,1] -> HxWx3 in [0,1], RGB."""
    x = (t.detach().cpu().clamp(-1, 1) + 1) / 2
    return x.permute(1, 2, 0).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="path to an input image")
    ap.add_argument("--ckpt", default="checkpoints/generator.pth")
    ap.add_argument("--task", default="sr", choices=["sr", "denoise", "jpeg"])
    ap.add_argument("--level", type=int, default=4,
                    help="degradation value: sr 4/8/16, denoise sigma, jpeg QF")
    ap.add_argument("--size", type=int, default=256, help="must match training (256)")
    ap.add_argument("--already-degraded", action="store_true",
                    help="input is already degraded; skip degradation and metrics")
    ap.add_argument("--out", default="result.png")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # --- load the trained generator ---
    G = Generator().to(device)
    G.load_state_dict(torch.load(args.ckpt, map_location=device))
    G.eval()

    # --- read + prepare the image ---
    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"could not read image: {args.image}")
    img = cv2.resize(img, (args.size, args.size), interpolation=cv2.INTER_CUBIC)

    if args.already_degraded:
        deg_bgr, clean_bgr = img, None
    else:
        deg_bgr, clean_bgr = apply_degradation(img, args.task, args.level), img

    # --- run the generator ---
    with torch.no_grad():
        out = G(_to_tensor(deg_bgr).unsqueeze(0).to(device))[0]

    deg_rgb, out_rgb = to_image(_to_tensor(deg_bgr)), to_image(out)

    # --- save the restored image ---
    out_bgr = cv2.cvtColor((np.clip(out_rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.out, out_bgr)
    print("saved restored image ->", args.out)

    # --- metrics + side-by-side when we have the clean reference ---
    if clean_bgr is not None:
        from skimage.metrics import peak_signal_noise_ratio as psnr
        from skimage.metrics import structural_similarity as ssim
        clean_rgb = to_image(_to_tensor(clean_bgr))
        print(f"PSNR {psnr(clean_rgb, out_rgb, data_range=1.0):.2f} dB | "
              f"SSIM {ssim(clean_rgb, out_rgb, channel_axis=2, data_range=1.0):.4f}")
        panels = [(deg_rgb, "degraded"), (out_rgb, "DLL-GAN"), (clean_rgb, "ground truth")]
    else:
        panels = [(deg_rgb, "input (degraded)"), (out_rgb, "DLL-GAN output")]

    fig, ax = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    for a, (im, ttl) in zip(np.atleast_1d(ax), panels):
        a.imshow(np.clip(im, 0, 1)); a.set_title(ttl); a.axis("off")
    plt.tight_layout()
    plt.savefig("comparison.png", dpi=120)
    print("saved comparison.png")


if __name__ == "__main__":
    main()