"""Losses for DLL-GAN.

Discriminator (paper Eqs. 2-4): standard GAN loss, real -> 1, fake -> 0.
Generator     (paper Eqs. 5-8): adversarial + lambda * L1 + degradation loss.

The degradation loss is the whole point of the paper: run the FROZEN regressor
on the generated image and push its estimate toward 0 (clean). Gradients flow
through R into G, but R's own weights are not updated.
"""
import torch
import torch.nn as nn


class GANLoss(nn.Module):
    """Vanilla (BCE) adversarial loss over the pixel-wise logit map."""

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def __call__(self, logits, target_is_real):
        target = torch.ones_like(logits) if target_is_real else torch.zeros_like(logits)
        return self.bce(logits, target)


l1_loss = nn.L1Loss()


def discriminator_loss(D, real, fake, gan_loss):
    """real -> 1, fake (detached so G isn't updated here) -> 0."""
    loss_real = gan_loss(D(real), True)
    loss_fake = gan_loss(D(fake.detach()), False)
    return 0.5 * (loss_real + loss_fake)


def generator_loss(D, R, fake, clean, gan_loss,
                   lambda_gan=1.0, lambda_l1=100.0, lambda_r=1.0):
    """Returns (total_loss, parts_dict)."""
    # 1) adversarial: fool D into calling the generated image real.
    adv = gan_loss(D(fake), True)

    # 2) L1 reconstruction toward the clean target (stabilises early training).
    l1 = l1_loss(fake, clean)

    # 3) degradation loss: estimated residual degradation of the generated image.
    #    NOTE: minimising R(fake).mean() can in principle be "gamed" - the
    #    generator may learn inputs that make R report a low value without the
    #    image truly being clean, and nothing stops the value going below 0.
    #    A safer variant is R(fake).clamp(min=0).mean(); left as-is to match the
    #    paper. This is one of the open questions worth probing.
    deg = R(fake).mean()

    total = lambda_gan * adv + lambda_l1 * l1 + lambda_r * deg
    return total, {"adv": adv.item(), "l1": l1.item(), "deg": deg.item()}
