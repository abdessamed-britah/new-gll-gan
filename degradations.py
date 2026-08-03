"""Degradation operators for the three tasks.

All functions take and return a uint8 HxWx3 BGR image (OpenCV convention).
A level value of 0 always means "clean" (identity).
"""
import cv2
import numpy as np


def degrade_sr(img, factor):
    """Downsample by `factor` then upsample back with bicubic (loses detail)."""
    if factor <= 1:
        return img.copy()
    h, w = img.shape[:2]
    small = cv2.resize(img, (w // factor, h // factor), interpolation=cv2.INTER_CUBIC)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def degrade_noise(img, sigma):
    """Add additive white Gaussian noise with std `sigma` on the 0-255 scale."""
    if sigma <= 0:
        return img.copy()
    noise = np.random.normal(0.0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def degrade_jpeg(img, quality):
    """Re-encode as JPEG at the given quality factor, then decode."""
    if quality <= 0 or quality >= 100:
        return img.copy()
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


_DEGRADE = {"sr": degrade_sr, "denoise": degrade_noise, "jpeg": degrade_jpeg}


def apply_degradation(img, task, level_value):
    """Dispatch to the right operator. `level_value == 0` returns a clean copy."""
    if level_value == 0:
        return img.copy()
    return _DEGRADE[task](img, level_value)
