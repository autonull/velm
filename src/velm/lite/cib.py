import torch

class CIBLoss:
    """Conditional Information Bottleneck proxy: tradeoff between task loss and latent norm.
    Use L2 norm of latents as a compression proxy."""
    def __init__(self, lambda_c=1e-3):
        self.lambda_c = lambda_c

    def __call__(self, task_loss, latents):
        # latents: (B, D)
        compr = latents.norm(p=2, dim=1).mean()
        return task_loss + self.lambda_c * compr
