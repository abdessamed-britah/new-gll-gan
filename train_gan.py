"""Step 2 - train DLL-GAN using the FROZEN regressor as a third loss.

Run train_regressor.py first so that cfg.reg_ckpt exists.

    python train_gan.py

The generator is trained on a single degradation level (cfg.gan_level_value),
matching the paper's independent-per-level training.
"""
import os

import torch
from torch.utils.data import DataLoader

from config import Config
from datasets import GANDataset
from models import Generator, PixelDiscriminator, Regressor
from losses import GANLoss, discriminator_loss, generator_loss


def main():
    cfg = Config()
    device = cfg.device if torch.cuda.is_available() else "cpu"

    ds = GANDataset(cfg.clean_dir, cfg.task, cfg.gan_level_value, cfg.image_size)
    dl = DataLoader(ds, batch_size=cfg.gan_batch_size, shuffle=True, num_workers=4)

    G = Generator().to(device)
    D = PixelDiscriminator().to(device)

    # frozen regressor (loaded from step 1)
    R = Regressor(pretrained=False).to(device)
    R.load_state_dict(torch.load(cfg.reg_ckpt, map_location=device))
    R.eval()
    for p in R.parameters():
        p.requires_grad_(False)

    gan_loss = GANLoss().to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=cfg.gan_lr, betas=cfg.gan_betas)
    opt_d = torch.optim.Adam(D.parameters(), lr=cfg.gan_lr, betas=cfg.gan_betas)

    for epoch in range(cfg.gan_epochs):
        for deg, clean in dl:
            deg, clean = deg.to(device), clean.to(device)
            fake = G(deg)

            # --- update discriminator ---
            opt_d.zero_grad()
            d_loss = discriminator_loss(D, clean, fake, gan_loss)
            d_loss.backward()
            opt_d.step()

            # --- update generator (adversarial + L1 + degradation) ---
            opt_g.zero_grad()
            g_loss, parts = generator_loss(
                D, R, fake, clean, gan_loss,
                cfg.lambda_gan, cfg.lambda_l1, cfg.lambda_r,
            )
            g_loss.backward()
            opt_g.step()

        print(f"[GAN] epoch {epoch + 1}/{cfg.gan_epochs}  "
              f"D={d_loss.item():.3f}  G={g_loss.item():.3f}  "
              f"adv={parts['adv']:.3f} l1={parts['l1']:.3f} deg={parts['deg']:.3f}")

    os.makedirs(os.path.dirname(cfg.gen_ckpt), exist_ok=True)
    torch.save(G.state_dict(), cfg.gen_ckpt)
    print(f"saved generator -> {cfg.gen_ckpt}")


if __name__ == "__main__":
    main()
