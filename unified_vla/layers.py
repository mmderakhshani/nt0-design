import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .core import modulate
from .attention import ExpertQKV, _match_heads


class GatedMLP(nn.Module):
    """Gated MLP matching Qwen2's structure: act(gate(x)) * up(x) → down.

    This allows direct weight initialization from VLM layers.
    """

    def __init__(self, d: int, d_inner: int):
        super().__init__()
        self.gate_proj = nn.Linear(d, d_inner, bias=False)
        self.up_proj = nn.Linear(d, d_inner, bias=False)
        self.down_proj = nn.Linear(d_inner, d, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class ExpertBlock(nn.Module):
    """Single expert transformer block.

    pre-norm → QKV → (optional RoPE) → attention → output proj → residual →
    pre-norm → GatedMLP → residual

    Supports:
    - RMSNorm or LayerNorm (via norm_class parameter)
    - Optional MRoPE on Q/K (via position_embeddings + mrope_section in forward)
    - Optional attention mask (via attn_mask in forward)
    - Optional adaLN modulation for both norms
    - External (detached) K/V from upstream experts for cross-modal attention
    """

    def __init__(
        self,
        d_expert: int,
        head_dim: int,
        num_kv_heads: int | None = None,
        intermediate_size: int | None = None,
        norm_class: type | None = None,
        norm_kwargs: dict | None = None,
    ):
        super().__init__()
        self.d_expert = d_expert
        self.head_dim = head_dim

        # Norm: default LayerNorm(bias=False), can be swapped for RMSNorm
        if norm_class is None:
            self.norm1 = nn.LayerNorm(d_expert, bias=False)
            self.norm2 = nn.LayerNorm(d_expert, bias=False)
        else:
            nkw = norm_kwargs or {}
            self.norm1 = norm_class(d_expert, **nkw)
            self.norm2 = norm_class(d_expert, **nkw)

        self.qkv = ExpertQKV(d_expert, head_dim, num_kv_heads=num_kv_heads)
        self.o_proj = nn.Linear(d_expert, d_expert, bias=False)
        self.mlp = GatedMLP(d_expert, intermediate_size or d_expert * 4)

    def forward(
        self,
        x: Tensor,  # (B, N, d_expert)
        ext_k: Tensor | None = None,  # (B, h_self, N_ext, head_dim) — already head-matched
        ext_v: Tensor | None = None,  # (B, h_self, N_ext, head_dim)
        adaln_mod: tuple[Tensor, ...] | None = None,
        # 6-tuple (s1, sh1, g1, s2, sh2, g2) — DiT adaLN-zero style.
        # s1/sh1 modulate norm1; g1 gates the attention residual.
        # s2/sh2 modulate norm2; g2 gates the MLP residual.
        position_embeddings: tuple[Tensor, Tensor] | None = None,  # (cos, sin) for RoPE
        mrope_section: list[int] | None = None,
        attn_mask: Tensor | None = None,  # (1, 1, N, N_total) bool or float mask
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Returns:
            x: (B, N, d_expert) output hidden states.
            self_k: (B, num_heads, N, head_dim) self K for downstream use.
            self_v: (B, num_heads, N, head_dim) self V for downstream use.
        """
        # --- Attention sub-block ---
        if adaln_mod is not None:
            s1, sh1, g1, s2, sh2, g2 = adaln_mod
            normed = modulate(x, self.norm1, s1, sh1)
        else:
            normed = self.norm1(x)

        q, self_k, self_v = self.qkv(normed)

        # Apply MRoPE to Q and K (before GQA expansion)
        if position_embeddings is not None:
            from qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb
            cos, sin = position_embeddings
            q, self_k = apply_multimodal_rotary_pos_emb(q, self_k, cos, sin, mrope_section)

        # Expand KV heads to match Q heads (GQA)
        if self.qkv.num_kv_heads != self.qkv.num_heads:
            n_rep = self.qkv.num_heads // self.qkv.num_kv_heads
            self_k = self_k.repeat_interleave(n_rep, dim=1)
            self_v = self_v.repeat_interleave(n_rep, dim=1)

        # Concatenate external K/V (from upstream, detached) with self K/V
        if ext_k is not None:
            attn_k = torch.cat([ext_k, self_k], dim=2)
            attn_v = torch.cat([ext_v, self_v], dim=2)
        else:
            attn_k = self_k
            attn_v = self_v

        attn_out = F.scaled_dot_product_attention(q, attn_k, attn_v, attn_mask=attn_mask)

        # Reshape (B, heads, N, hd) → (B, N, d_expert)
        B, _, N, _ = attn_out.shape
        attn_out = attn_out.transpose(1, 2).reshape(B, N, self.d_expert)
        attn_out = self.o_proj(attn_out)
        if adaln_mod is not None:
            attn_out = g1.unsqueeze(1) * attn_out
        x = x + attn_out

        # --- MLP sub-block ---
        if adaln_mod is not None:
            normed2 = modulate(x, self.norm2, s2, sh2)
        else:
            normed2 = self.norm2(x)
        mlp_out = self.mlp(normed2)
        if adaln_mod is not None:
            mlp_out = g2.unsqueeze(1) * mlp_out
        x = x + mlp_out

        return x, self_k, self_v


def init_expert_from_vlm(expert: ExpertBlock, vlm_block: ExpertBlock) -> None:
    """Initialize a downsized expert by copying the first N heads from a VLM block.

    Handles GQA: Q is sliced by expert's num_heads, K/V by expert's num_kv_heads.
    MLP is sliced to expert's intermediate_size.

    Args:
        expert: target ExpertBlock (smaller d_expert).
        vlm_block: source ExpertBlock (larger d_vlm, pretrained weights).
    """
    d = expert.d_expert
    d_kv = expert.qkv.d_kv

    with torch.no_grad():
        # Q projection: slice first d_expert rows (first N Q heads), d_expert columns
        src_q = vlm_block.qkv.q_proj
        dst_q = expert.qkv.q_proj
        dst_q.weight.copy_(src_q.weight[:d, :d])
        if src_q.bias is not None and dst_q.bias is not None:
            dst_q.bias.copy_(src_q.bias[:d])

        # K/V projections: slice first d_kv rows (first N KV heads), d_expert columns
        for proj_name in ("k_proj", "v_proj"):
            src = getattr(vlm_block.qkv, proj_name)
            dst = getattr(expert.qkv, proj_name)
            dst.weight.copy_(src.weight[:d_kv, :d])
            if src.bias is not None and dst.bias is not None:
                dst.bias.copy_(src.bias[:d_kv])

        # Output projection: (d, d) from (d_vlm, d_vlm)
        expert.o_proj.weight.copy_(vlm_block.o_proj.weight[:d, :d])
        if vlm_block.o_proj.bias is not None and expert.o_proj.bias is not None:
            expert.o_proj.bias.copy_(vlm_block.o_proj.bias[:d])

        # LayerNorms
        expert.norm1.weight.copy_(vlm_block.norm1.weight[:d])
        expert.norm2.weight.copy_(vlm_block.norm2.weight[:d])

        # GatedMLP — use expert's actual intermediate size
        d_inner_expert = expert.mlp.gate_proj.weight.shape[0]
        expert.mlp.gate_proj.weight.copy_(vlm_block.mlp.gate_proj.weight[:d_inner_expert, :d])
        expert.mlp.up_proj.weight.copy_(vlm_block.mlp.up_proj.weight[:d_inner_expert, :d])
        expert.mlp.down_proj.weight.copy_(vlm_block.mlp.down_proj.weight[:d, :d_inner_expert])
