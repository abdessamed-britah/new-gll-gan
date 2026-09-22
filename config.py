"""Central configuration for the DLL-GAN reimplementation.

Paper: Kas, Chahi, Kajo, Ruichek, "DLL-GAN: Degradation-level-based learnable
adversarial loss for image enhancement", Expert Systems with Applications 237 (2023).
"""
from dataclasses import dataclass

# Which task to run: "sr" (super-resolution), "denoise", or "jpeg".
TASK = "sr"

# Degradation levels per task. Level INDEX 0 is ALWAYS clean.
# The stored value is the parameter handed to the degradation function.
# Note the parameter is not monotonic with the level index for JPEG (lower QF =
# worse), which is exactly why the regressor is trained on the level INDEX, not
# the raw parameter.
LEVELS = {
    "sr":      [0, 4, 8, 16],        # bicubic down/up-sampling factor (0 = clean)
    "denoise": [0, 15, 25, 50, 70],  # additive white Gaussian noise sigma (0-255)
    "jpeg":    [0, 30, 20, 10],      # JPEG quality factor (0 = clean, 10 = worst)
}


@dataclass
class Config:
    task: str = TASK
    image_size: int = 256            # must be divisible by 4
    channels: int = 3

    # The values below are the ones used to produce the results reported in
    # the README. They fit on a 16 GB GPU; if you run out of memory, lower
    # gan_batch_size to 4 (the rest of the settings are unaffected).

    # --- Regressor training (step 1) ---
    reg_epochs: int = 8
    reg_lr: float = 1e-4             # paper: 1e-4, Adam
    reg_batch_size: int = 32

    # --- GAN training (step 2) ---
    gan_epochs: int = 150
    gan_lr: float = 2e-4
    gan_betas: tuple = (0.5, 0.999)
    gan_batch_size: int = 8          # lower to 4 if the GPU runs out of memory
    lambda_gan: float = 1.0          # weight of the adversarial loss
    lambda_l1: float = 100.0         # weight of the L1 loss (pix2pix convention)
    lambda_r: float = 1.0            # weight of the degradation (regressor) loss

    # Which degradation level VALUE to train the generator on. The paper trains
    # one generator per level ("independent training on each level"), so pick a
    # single value here (e.g. 4 for x4 SR, 25 for sigma=25 denoising).
    gan_level_value: int = 4

    # --- paths ---
    clean_dir: str = "data/div2k"    # folder of clean training images
    reg_ckpt: str = "checkpoints/regressor.pth"
    gen_ckpt: str = "checkpoints/generator.pth"

    device: str = "cuda"             # scripts fall back to cpu automatically

    @property
    def levels(self):
        return LEVELS[self.task]

    @property
    def num_levels(self):
        return len(self.levels)
