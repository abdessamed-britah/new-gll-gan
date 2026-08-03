"""Datasets for the two training stages.

RegressorDataset: yields (degraded_image, level_index) across ALL levels, so the
                  regressor learns to estimate any degradation level.
GANDataset:       yields (degraded_image, clean_image) at ONE fixed level, matching
                  the paper's "independent training on each level".

Images are returned as float tensors in [-1, 1], RGB, shape (3, H, W).
"""
import glob
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from degradations import apply_degradation


def _to_tensor(img_bgr):
    """uint8 HxWx3 BGR -> float tensor (3,H,W) in [-1, 1], RGB."""
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = img / 127.5 - 1.0
    return torch.from_numpy(img).permute(2, 0, 1).contiguous()


def _list_images(folder):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    files = []
    for e in exts:
        files += glob.glob(os.path.join(folder, e))
    if not files:
        raise FileNotFoundError(f"No images found in '{folder}'. Put clean images there.")
    return sorted(files)


class RegressorDataset(Dataset):
    """Each clean image is expanded into `num_levels` samples, one per level,
    labelled with its level INDEX (0 = clean)."""

    def __init__(self, folder, task, levels, image_size=256):
        self.files = _list_images(folder)
        self.task = task
        self.levels = levels
        self.size = image_size

    def __len__(self):
        return len(self.files) * len(self.levels)

    def __getitem__(self, idx):
        file_idx = idx // len(self.levels)
        level_idx = idx % len(self.levels)
        img = cv2.imread(self.files[file_idx], cv2.IMREAD_COLOR)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
        deg = apply_degradation(img, self.task, self.levels[level_idx])
        return _to_tensor(deg), torch.tensor(float(level_idx))


class GANDataset(Dataset):
    """Yields (degraded, clean) pairs at a single fixed degradation level."""

    def __init__(self, folder, task, level_value, image_size=256):
        self.files = _list_images(folder)
        self.task = task
        self.level_value = level_value
        self.size = image_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = cv2.imread(self.files[idx], cv2.IMREAD_COLOR)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
        deg = apply_degradation(img, self.task, self.level_value)
        return _to_tensor(deg), _to_tensor(img)
