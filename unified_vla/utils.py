"""Parameter counting utilities for the unified VLA backbone."""

from collections import defaultdict

import torch.nn as nn


def count_params(module: nn.Module, requires_grad: bool | None = None) -> int:
    """Count parameters in a module.

    Args:
        module: PyTorch module.
        requires_grad: If True, count only trainable. If False, only frozen.
                       If None, count all.
    """
    total = 0
    for p in module.parameters():
        if requires_grad is None or p.requires_grad == requires_grad:
            total += p.numel()
    return total


def format_num(n: int) -> str:
    """Format a number with M/K suffix."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def backbone_param_summary(backbone) -> dict:
    """Detailed parameter breakdown for a UnifiedVLABackbone.

    Returns a dict with counts for each component.
    """
    summary = defaultdict(int)

    for i, layer in enumerate(backbone.layers):
        # VLM layer (frozen) — actual Qwen decoder layer
        vlm = layer.vlm_layer
        summary["vlm_total"] += count_params(vlm)
        summary["vlm_attn"] += count_params(vlm.self_attn)
        summary["vlm_mlp"] += count_params(vlm.mlp)
        summary["vlm_norm"] += count_params(vlm.input_layernorm) + count_params(vlm.post_attention_layernorm)

        # PC block (trainable)
        summary["pc_total"] += count_params(layer.pc_block)
        summary["pc_attn"] += count_params(layer.pc_block.qkv) + count_params(layer.pc_block.o_proj)
        summary["pc_mlp"] += count_params(layer.pc_block.mlp)
        summary["pc_norm"] += count_params(layer.pc_block.norm1) + count_params(layer.pc_block.norm2)

        # Action block (trainable)
        summary["action_total"] += count_params(layer.action_block)
        summary["action_attn"] += count_params(layer.action_block.qkv) + count_params(layer.action_block.o_proj)
        summary["action_mlp"] += count_params(layer.action_block.mlp)
        summary["action_norm"] += count_params(layer.action_block.norm1) + count_params(layer.action_block.norm2)

        # adaLN (trainable)
        summary["adaln_total"] += count_params(layer.action_adaln)

    summary["trainable"] = count_params(backbone, requires_grad=True)
    summary["frozen"] = count_params(backbone, requires_grad=False)
    summary["total"] = count_params(backbone)
    summary["num_layers"] = len(backbone.layers)

    return dict(summary)


def print_param_summary(backbone) -> None:
    """Print a formatted parameter summary table."""
    s = backbone_param_summary(backbone)
    L = s["num_layers"]

    print(f"\n{'='*60}")
    print(f"  Unified VLA Backbone — {L} layers")
    print(f"{'='*60}")
    print(f"  {'Component':<30} {'Total':>12} {'Per Layer':>12}")
    print(f"  {'-'*54}")

    print(f"  {'VLM expert (frozen)':<30} {format_num(s['vlm_total']):>12} {format_num(s['vlm_total']//L):>12}")
    print(f"    {'attention (Q/K/V/O)':<28} {format_num(s['vlm_attn']):>12} {format_num(s['vlm_attn']//L):>12}")
    print(f"    {'MLP (gate/up/down)':<28} {format_num(s['vlm_mlp']):>12} {format_num(s['vlm_mlp']//L):>12}")
    print(f"    {'norms':<28} {format_num(s['vlm_norm']):>12} {format_num(s['vlm_norm']//L):>12}")

    print()
    print(f"  {'PC expert (trainable)':<30} {format_num(s['pc_total']):>12} {format_num(s['pc_total']//L):>12}")
    print(f"    {'attention (Q/K/V/O)':<28} {format_num(s['pc_attn']):>12} {format_num(s['pc_attn']//L):>12}")
    print(f"    {'MLP (gate/up/down)':<28} {format_num(s['pc_mlp']):>12} {format_num(s['pc_mlp']//L):>12}")
    print(f"    {'norms':<28} {format_num(s['pc_norm']):>12} {format_num(s['pc_norm']//L):>12}")

    print()
    print(f"  {'Action expert (trainable)':<30} {format_num(s['action_total']):>12} {format_num(s['action_total']//L):>12}")
    print(f"    {'attention (Q/K/V/O)':<28} {format_num(s['action_attn']):>12} {format_num(s['action_attn']//L):>12}")
    print(f"    {'MLP (gate/up/down)':<28} {format_num(s['action_mlp']):>12} {format_num(s['action_mlp']//L):>12}")
    print(f"    {'norms':<28} {format_num(s['action_norm']):>12} {format_num(s['action_norm']//L):>12}")

    print()
    print(f"  {'adaLN (trainable)':<30} {format_num(s['adaln_total']):>12} {format_num(s['adaln_total']//L):>12}")

    print(f"\n  {'-'*54}")
    print(f"  {'TOTAL':<30} {format_num(s['total']):>12}")
    print(f"  {'Trainable':<30} {format_num(s['trainable']):>12}")
    print(f"  {'Frozen (VLM)':<30} {format_num(s['frozen']):>12}")
    print(f"  {'Trainable %':<30} {100*s['trainable']/s['total']:>11.1f}%")
    print(f"{'='*60}\n")
