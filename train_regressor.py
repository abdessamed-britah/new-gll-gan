"""Step 1 - train the degradation-level regressor R.

R sees images corrupted at every level and learns to predict the level index
(0 = clean, higher = more degraded). It is FROZEN afterwards and reused as a
learnable loss for the generator. Run this before train_gan.py.

    python train_regressor.py
"""
import os

import torch
from torch.utils.data import DataLoader

from config import Config
from datasets import RegressorDataset
from models import Regressor


def main():
    cfg = Config()
    device = cfg.device if torch.cuda.is_available() else "cpu"

    ds = RegressorDataset(cfg.clean_dir, cfg.task, cfg.levels, cfg.image_size)
    dl = DataLoader(ds, batch_size=cfg.reg_batch_size, shuffle=True, num_workers=4)

    R = Regressor(pretrained=True).to(device)
    opt = torch.optim.Adam(R.parameters(), lr=cfg.reg_lr)
    mse = torch.nn.MSELoss()

    R.train()
    for epoch in range(cfg.reg_epochs):
        running, seen = 0.0, 0
        for imgs, levels in dl:
            imgs, levels = imgs.to(device), levels.to(device)
            loss = mse(R(imgs), levels)              # paper Eq. 1
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * imgs.size(0)
            seen += imgs.size(0)
        print(f"[R] epoch {epoch + 1}/{cfg.reg_epochs}  mse={running / seen:.4f}")

    os.makedirs(os.path.dirname(cfg.reg_ckpt), exist_ok=True)
    torch.save(R.state_dict(), cfg.reg_ckpt)
    print(f"saved regressor -> {cfg.reg_ckpt}")


if __name__ == "__main__":
    main()
