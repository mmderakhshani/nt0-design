import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding followed by MLP projection.

    Converts scalar t ∈ [0,1] to a d_out-dimensional embedding.
    """

    def __init__(self, d_sinusoidal: int = 128, d_out: int = 64):
        super().__init__()
        self.d_sinusoidal = d_sinusoidal
        self.mlp = nn.Sequential(
            nn.Linear(d_sinusoidal, d_out),
            nn.SiLU(),
            nn.Linear(d_out, d_out),
        )

        # Precompute frequency bands (not a parameter — fixed)
        half = d_sinusoidal // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs)

    def forward(self, t: Tensor) -> Tensor:
        """
        Args:
            t: (B,) float timestep in [0, 1].
        Returns:
            (B, d_out) timestep embedding.
        """
        args = t[:, None] * self.freqs[None, :]  # (B, half)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, d_sinusoidal)
        return self.mlp(emb)


def noise_action(
    action_gt: Tensor,
    t: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Apply flow matching noise to ground truth actions.

    Rectified flow interpolation:
        x_t = (1 - t) * noise + t * action_gt
        v_target = action_gt - noise

    Args:
        action_gt: (B, N, action_dim) ground truth action chunk.
        t: (B,) optional timestep in [0, 1]. Sampled uniformly if None.
    Returns:
        noisy_action: (B, N, action_dim)
        t: (B,) timestep used
        v_target: (B, N, action_dim) velocity target for loss
    """
    B = action_gt.shape[0]
    if t is None:
        t = torch.rand(B, device=action_gt.device, dtype=action_gt.dtype)

    noise = torch.randn_like(action_gt)

    # Reshape t for broadcasting: (B,) → (B, 1, 1)
    t_broadcast = t[:, None, None]
    noisy_action = (1 - t_broadcast) * noise + t_broadcast * action_gt
    v_target = action_gt - noise

    return noisy_action, t, v_target


def flow_matching_loss(predicted_v: Tensor, v_target: Tensor) -> Tensor:
    """MSE loss between predicted and target velocity."""
    return F.mse_loss(predicted_v, v_target)


def pc_mse_loss(predicted_pc: Tensor, target_pc: Tensor) -> Tensor:
    """MSE loss for point cloud prediction (Step 16)."""
    return F.mse_loss(predicted_pc, target_pc)
