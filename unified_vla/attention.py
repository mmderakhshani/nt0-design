import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .core import TokenType


class ExpertQKV(nn.Module):
    """Per-expert Q/K/V projections producing multi-head outputs.

    Supports GQA: num_kv_heads can be less than num_heads (Q heads).
    When num_kv_heads is None, defaults to num_heads (standard MHA).

    Uses separate Q, K, V linear layers (not fused) so that weight
    initialization from VLM layers (Step 12) is a direct copy of
    q_proj, k_proj, v_proj weights.
    """

    def __init__(self, d_expert: int, head_dim: int, num_kv_heads: int | None = None):
        super().__init__()
        assert d_expert % head_dim == 0, f"d_expert={d_expert} not divisible by head_dim={head_dim}"
        self.d_expert = d_expert
        self.head_dim = head_dim
        self.num_heads = d_expert // head_dim
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else self.num_heads
        self.d_kv = self.num_kv_heads * head_dim

        assert self.num_heads % self.num_kv_heads == 0, (
            f"num_heads={self.num_heads} not divisible by num_kv_heads={self.num_kv_heads}"
        )

        self.q_proj = nn.Linear(d_expert, d_expert, bias=True)
        self.k_proj = nn.Linear(d_expert, self.d_kv, bias=True)
        self.v_proj = nn.Linear(d_expert, self.d_kv, bias=True)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            x: (B, N, d_expert)
        Returns:
            q: (B, num_heads, N, head_dim)
            k: (B, num_kv_heads, N, head_dim)
            v: (B, num_kv_heads, N, head_dim)
        """
        B, N, _ = x.shape

        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)

        return q, k, v


def build_attention_mask(token_types: Tensor) -> Tensor:
    """Build left-to-right attention mask from token types.

    Visibility rules (all self-attention is bidirectional within a modality):
        VLM    queries attend to: VLM only
        PC     queries attend to: VLM + PC
        Action queries attend to: VLM + PC + Action

    Args:
        token_types: (B, seq_len) int tensor with TokenType values.
    Returns:
        (B, 1, seq_len, seq_len) bool mask where True = allowed to attend.
        Broadcastable over the heads dimension.
    """
    # (B, seq_len, 1) and (B, 1, seq_len) for pairwise comparison
    q_types = token_types.unsqueeze(2)  # (B, seq_len, 1)
    k_types = token_types.unsqueeze(1)  # (B, 1, seq_len)

    # VLM queries: attend to VLM keys only
    vlm_mask = (q_types == TokenType.VLM) & (k_types == TokenType.VLM)

    # PC queries: attend to VLM + PC keys
    pc_mask = (q_types == TokenType.PC) & (
        (k_types == TokenType.VLM) | (k_types == TokenType.PC)
    )

    # Action queries: attend to VLM + PC + Action keys
    action_mask = (q_types == TokenType.ACTION)  # all keys allowed

    mask = vlm_mask | pc_mask | action_mask  # (B, seq_len, seq_len)
    return mask.unsqueeze(1)  # (B, 1, seq_len, seq_len)


def _match_heads(tensor: Tensor, target_heads: int) -> Tensor:
    """Match K/V head count to the querying expert's head count.

    When experts have different widths (and thus different head counts),
    cross-modal attention requires matching heads. Uses GQA-style grouping:
    - If source has more heads: average groups to reduce
    - If source has fewer heads: repeat-interleave to expand
    """
    current_heads = tensor.shape[1]
    if current_heads == target_heads:
        return tensor
    if current_heads > target_heads:
        assert current_heads % target_heads == 0
        group = current_heads // target_heads
        B, _, N, hd = tensor.shape
        return tensor.reshape(B, target_heads, group, N, hd).mean(dim=2)
    else:
        assert target_heads % current_heads == 0
        return tensor.repeat_interleave(target_heads // current_heads, dim=1)


def detached_cross_modal_attention(
    vlm_q: Tensor,
    vlm_k: Tensor,
    vlm_v: Tensor,
    pc_q: Tensor,
    pc_k: Tensor,
    pc_v: Tensor,
    action_q: Tensor,
    action_k: Tensor,
    action_v: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Three-call attention with VLM K/V detached; PC K/V flow gradients.

    VLM:    self-attention only.
    PC:     self + VLM K/V (detached — VLM is frozen).
    Action: self + VLM K/V (detached) + PC K/V (NOT detached, so action
            flow-matching loss flows back into the PC expert).

    Head counts may differ across experts. Cross-modal K/V is matched
    to the querying expert's head count via _match_heads (GQA-style).

    Returns:
        vlm_out, pc_out, action_out: each (B, num_heads_expert, N_expert, head_dim)
    """
    # VLM: self only
    vlm_out = F.scaled_dot_product_attention(vlm_q, vlm_k, vlm_v)

    # PC: self + VLM(detached)
    h_pc = pc_q.shape[1]
    pc_combined_k = torch.cat([_match_heads(vlm_k.detach(), h_pc), pc_k], dim=2)
    pc_combined_v = torch.cat([_match_heads(vlm_v.detach(), h_pc), pc_v], dim=2)
    pc_out = F.scaled_dot_product_attention(pc_q, pc_combined_k, pc_combined_v)

    # Action: self + VLM(detached) + PC(grad flows)
    h_action = action_q.shape[1]
    action_combined_k = torch.cat(
        [_match_heads(vlm_k.detach(), h_action), _match_heads(pc_k, h_action), action_k],
        dim=2,
    )
    action_combined_v = torch.cat(
        [_match_heads(vlm_v.detach(), h_action), _match_heads(pc_v, h_action), action_v],
        dim=2,
    )
    action_out = F.scaled_dot_product_attention(action_q, action_combined_k, action_combined_v)

    return vlm_out, pc_out, action_out
