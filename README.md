# DLL-GAN (independent reimplementation)

A from-scratch, runnable reimplementation of the training pipeline from:

> Kas, Chahi, Kajo, Ruichek, *DLL-GAN: Degradation-level-based learnable
> adversarial loss for image enhancement*, Expert Systems with Applications 237 (2023).

The paper's official repo ships only figures and a README (no code or
checkpoints), so this rebuilds the method from the description. The core idea:
train a CNN **regressor** to estimate how degraded an image is, freeze it, and
use its estimate on the generator's output as a third loss alongside the usual
adversarial + L1 losses.

## Pipeline (two steps)

```
Step 1  train_regressor.py   ->  R learns to predict the degradation level (0 = clean)
Step 2  train_gan.py         ->  G + D train adversarially, with the FROZEN R as a 3rd loss
```

The regressor is trained once, across all degradation levels. The generator is
trained per level (the paper trains one model per level), selected by
`gan_level_value` in `config.py`.

## Files

| file | role | paper reference |
|------|------|-----------------|
| `config.py` | tasks, degradation levels, hyperparameters | Sec. 4.1 |
| `degradations.py` | SR / noise / JPEG corruption operators | Sec. 4.1 |
| `datasets.py` | leveled set for R, paired set for the GAN | Sec. 3.2 |
| `models.py` | ResNet-9 generator, pixel discriminator, ResNet-18 regressor | Figs. 2, 4, 5 |
| `losses.py` | the three losses | Eqs. 1-8 |
| `train_regressor.py` | step 1 | Eq. 1 |
| `train_gan.py` | step 2 | Eqs. 2-8 |
| `smoke_test.py` | runs the whole thing on random tensors (no data) | - |

## Setup

```bash
pip install -r requirements.txt
python smoke_test.py          # verifies everything wires up (no dataset needed)
```

## Data

Put clean training images (e.g. the DIV2K training set) in `data/div2k/`.
Benchmarks for evaluation (Set5, Set14, BSD100/CBSD68, Urban100 for SR/denoise;
Live1, Classic5 for JPEG) are all publicly downloadable. Degradation is applied
on the fly, so you only need the clean images to train.

## Run

```bash
# choose the task in config.py: "sr", "denoise", or "jpeg"
python train_regressor.py     # step 1  -> checkpoints/regressor.pth
python train_gan.py           # step 2  -> checkpoints/generator.pth
```

## How the degradation loss works (the heart of the paper)

In `losses.py`, `generator_loss` computes:

```
adv  = BCE(D(fake), real)            # fool the discriminator      (Eq. 7)
l1   = |fake - clean|                # reconstruction              (Eq. 8)
deg  = R(fake).mean()                # residual degradation -> 0   (Eq. 6)
total = lambda_gan*adv + lambda_l1*l1 + lambda_r*deg               (Eq. 5)
```

`R` is in `eval()` with `requires_grad_(False)`, so its weights stay fixed while
gradients still flow through it into `G`.

## Known caveats / open questions (good things to probe)

- **Gaming a frozen R.** Minimising `R(fake).mean()` can be exploited: the
  generator may learn to make `R` report low degradation without the image truly
  being clean, and nothing bounds the value below 0. A safer variant is
  `R(fake).clamp(min=0).mean()`, or periodically fine-tuning `R`. Left as-is here
  to match the paper.
- **Unseen levels.** `R` is trained on discrete levels; real generator outputs
  are out-of-distribution. Worth checking how it scores intermediate levels.
- **Metrics.** Add LPIPS / NIQE alongside PSNR/SSIM — the paper argues PSNR/SSIM
  correlate poorly with perception yet evaluates mostly on them.
- **Norm.** The paper specifies BatchNorm; `models.py` exposes InstanceNorm too,
  which often trains more stably for generation.

## Status

Verified: forward + backward run end-to-end and the generator loss decreases
over training steps (`smoke_test.py`). Not yet trained to convergence on real
data - that is the next step.
