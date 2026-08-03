"""Verifies the whole pipeline wires up and trains one step on random data.
No dataset required, CPU-friendly.

    python smoke_test.py
"""
import torch

from models import Generator, PixelDiscriminator, Regressor
from losses import GANLoss, discriminator_loss, generator_loss


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, S = 2, 128  # small size for speed; must be divisible by 4
    deg = torch.rand(B, 3, S, S, device=device) * 2 - 1
    clean = torch.rand(B, 3, S, S, device=device) * 2 - 1
    levels = torch.randint(0, 4, (B,), device=device).float()

    G = Generator().to(device)
    D = PixelDiscriminator().to(device)
    R = Regressor(pretrained=False).to(device)  # skip weight download for the test
    gan_loss = GANLoss().to(device)

    # --- one regressor step ---
    mse = torch.nn.MSELoss()
    r_opt = torch.optim.Adam(R.parameters(), 1e-4)
    r_loss = mse(R(deg), levels)
    r_opt.zero_grad(); r_loss.backward(); r_opt.step()
    print("regressor out shape :", tuple(R(deg).shape), " mse:", round(r_loss.item(), 4))

    # --- freeze R and run one GAN step ---
    R.eval()
    for p in R.parameters():
        p.requires_grad_(False)

    fake = G(deg)
    print("generator out shape :", tuple(fake.shape))
    print("discriminator shape :", tuple(D(deg).shape))

    d_loss = discriminator_loss(D, clean, fake, gan_loss)
    g_loss, parts = generator_loss(D, R, fake, clean, gan_loss, 1.0, 100.0, 1.0)
    d_loss.backward()
    g_loss.backward()

    print("D loss:", round(d_loss.item(), 4), "| G loss:", round(g_loss.item(), 4))
    print("G parts:", {k: round(v, 4) for k, v in parts.items()})
    print("OK: forward + backward run end-to-end.")


if __name__ == "__main__":
    main()
