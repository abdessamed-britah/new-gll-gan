"""Benchmark a trained DLL-GAN generator on a folder of UNSEEN clean images.

For every image in --data_dir the script:
  1. resizes it to --size x --size (the resolution the generator was trained at),
  2. degrades it with degradations.apply_degradation(task, level),
  3. restores it with the trained Generator in eval() mode,
  4. computes PSNR/SSIM of the restoration AND of the raw degraded input
     (the "no-model baseline") against the clean image.

Printing both means side by side makes the model's actual gain visible: a
restoration is only meaningful if it beats simply doing nothing.

    python evaluate.py --data_dir benchmarks/Set5 --task sr --level 4

Note on citing these numbers: this protocol is self-consistent but it is NOT the
classic Set5/Set14 SR protocol (which keeps the native resolution/aspect ratio,
downsamples by the scale factor, and often scores the Y channel with a border
crop). Report them as "our protocol: 256x256 center-resized, bicubic x4
down/up", not as directly comparable to published Set5/Set14 tables.
"""
import argparse
import os

import cv2
import matplotlib
matplotlib.use("Agg")  # headless: never try to open a window
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Reuse the project's own definitions - the model, the degradation and the
# normalization must be IDENTICAL to training, so nothing is redefined here.
from models import Generator
from degradations import apply_degradation
from datasets import _to_tensor, _list_images


def to_image(t):
    """(3,H,W) float tensor in [-1,1] -> HxWx3 float array in [0,1], RGB.

    Same convention as infer.py. Metrics are computed in this [0,1] RGB space.
    """
    x = (t.detach().cpu().clamp(-1, 1) + 1) / 2
    return x.permute(1, 2, 0).numpy()


def metrics(ref_rgb, test_rgb):
    """PSNR (dB) and SSIM between two HxWx3 arrays in [0,1]."""
    return (
        psnr(ref_rgb, test_rgb, data_range=1.0),
        ssim(ref_rgb, test_rgb, channel_axis=2, data_range=1.0),
    )


def save_comparison(path, deg_rgb, out_rgb, clean_rgb, title):
    """Write a 'degraded | DLL-GAN | ground truth' figure."""
    panels = [(deg_rgb, "degraded"), (out_rgb, "DLL-GAN"), (clean_rgb, "ground truth")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
    for ax, (im, ttl) in zip(axes, panels):
        ax.imshow(np.clip(im, 0, 1))
        ax.set_title(ttl)
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Evaluate DLL-GAN on a benchmark folder.")
    ap.add_argument("--data_dir", required=True, help="folder of CLEAN benchmark images")
    ap.add_argument("--ckpt", default="checkpoints/generator.pth")
    ap.add_argument("--task", default="sr", choices=["sr", "denoise", "jpeg"])
    ap.add_argument("--level", type=int, default=4,
                    help="degradation value: sr 4/8/16, denoise sigma, jpeg QF")
    ap.add_argument("--size", type=int, default=256, help="must match training (256)")
    ap.add_argument("--out_dir", default="eval_out")
    ap.add_argument("--num_save", type=int, default=5,
                    help="how many side-by-side comparison images to write")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for stochastic degradations (denoise), so runs are reproducible")
    args = ap.parse_args()

    if args.size % 4 != 0:
        raise SystemExit("--size must be divisible by 4 (the generator downsamples twice).")

    np.random.seed(args.seed)  # makes the denoise task reproducible
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    # --- load the trained generator (CPU-safe) ---
    G = Generator().to(device)
    G.load_state_dict(torch.load(args.ckpt, map_location=device))
    G.eval()

    files = _list_images(args.data_dir)
    print(f"device: {device} | ckpt: {args.ckpt}")
    print(f"task: {args.task} | level: {args.level} | size: {args.size} | "
          f"{len(files)} image(s) in {args.data_dir}\n")

    header = f"{'image':<28} {'PSNR deg':>9} {'PSNR gen':>9} {'SSIM deg':>9} {'SSIM gen':>9}"
    print(header)
    print("-" * len(header))

    # accumulators: "deg" = degraded input vs clean, "gen" = restored vs clean
    psnr_deg, psnr_gen, ssim_deg, ssim_gen = [], [], [], []
    saved = 0

    for path in files:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"{os.path.basename(path):<28}  !! unreadable, skipped")
            continue

        # Same preprocessing as GANDataset: resize, then degrade.
        img = cv2.resize(img, (args.size, args.size), interpolation=cv2.INTER_CUBIC)
        deg_bgr = apply_degradation(img, args.task, args.level)

        # --- restore ---
        deg_t = _to_tensor(deg_bgr)
        with torch.no_grad():
            out_t = G(deg_t.unsqueeze(0).to(device))[0]

        # Round-trip clean/degraded through _to_tensor too, so all three images
        # go through exactly the same BGR->RGB + scaling path.
        clean_rgb = to_image(_to_tensor(img))
        deg_rgb = to_image(deg_t)
        out_rgb = np.clip(to_image(out_t), 0.0, 1.0)

        p_d, s_d = metrics(clean_rgb, deg_rgb)
        p_g, s_g = metrics(clean_rgb, out_rgb)
        psnr_deg.append(p_d); psnr_gen.append(p_g)
        ssim_deg.append(s_d); ssim_gen.append(s_g)

        name = os.path.basename(path)
        print(f"{name:<28} {p_d:>9.2f} {p_g:>9.2f} {s_d:>9.4f} {s_g:>9.4f}")

        if saved < args.num_save:
            stem = os.path.splitext(name)[0]
            save_comparison(
                os.path.join(args.out_dir, f"{stem}_compare.png"),
                deg_rgb, out_rgb, clean_rgb,
                f"{name} - {args.task} level {args.level} | "
                f"degraded {p_d:.2f} dB -> DLL-GAN {p_g:.2f} dB",
            )
            saved += 1

    if not psnr_gen:
        raise SystemExit("No image could be evaluated.")

    n = len(psnr_gen)
    m_pd, m_pg = float(np.mean(psnr_deg)), float(np.mean(psnr_gen))
    m_sd, m_sg = float(np.mean(ssim_deg)), float(np.mean(ssim_gen))

    print("-" * len(header))
    print(f"\n=== MEAN over {n} image(s) | task={args.task} level={args.level} ===")
    print(f"{'':<22}{'degraded (no model)':>21}{'DLL-GAN':>12}{'gain':>12}")
    print(f"{'MEAN PSNR (dB)':<22}{m_pd:>21.2f}{m_pg:>12.2f}{m_pg - m_pd:>+12.2f}")
    print(f"{'MEAN SSIM':<22}{m_sd:>21.4f}{m_sg:>12.4f}{m_sg - m_sd:>+12.4f}")
    print(f"\nsaved {saved} comparison image(s) -> {args.out_dir}")


if __name__ == "__main__":
    main()
