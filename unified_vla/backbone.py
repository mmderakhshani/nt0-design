"""Backbone that uses actual Qwen2.5-VL decoder layers for the VLM expert.

Supports two expert modes:
- dense:  PC/Action experts at every layer; cross-attention to VLM only at
          expert_layer_ids, self-attention-only at other layers.
- sparse: PC/Action experts only at expert_layer_ids. Pass-through elsewhere.

Action cross-attends to VLM K/V (current layer) + PC K/V (same layer).
PC runs first within each expert layer, then Action uses that layer's PC K/V.
"""

import torch
import torch.nn as nn
from torch import Tensor

from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

from .layers import ExpertBlock
from .attention import _match_heads


def _build_action_cross_causal_mask(
    n_action: int,
    n_ext: int,
    device: torch.device,
    dtype: torch.dtype = torch.bool,
) -> Tensor:
    """Hybrid mask: full cross-attention + causal self-attention for Action."""
    full_cross = torch.ones(n_action, n_ext, dtype=dtype, device=device)
    causal_self = torch.tril(torch.ones(n_action, n_action, dtype=dtype, device=device))
    mask = torch.cat([full_cross, causal_self], dim=1)
    return mask.unsqueeze(0).unsqueeze(0)


def _build_action_self_causal_mask(
    n_action: int,
    device: torch.device,
    dtype: torch.dtype = torch.bool,
) -> Tensor:
    """Causal self-attention mask for Action (no cross-attention)."""
    mask = torch.tril(torch.ones(n_action, n_action, dtype=dtype, device=device))
    return mask.unsqueeze(0).unsqueeze(0)


class MultiExpertLayer(nn.Module):
    """One backbone layer: actual Qwen decoder layer (VLM) + optional ExpertBlocks (PC, Action).

    At expert layers (is_cross=True): PC cross-attends to VLM, Action cross-attends
    to VLM + same-layer PC K/V.
    At non-expert layers (is_cross=False, dense mode): PC and Action do self-attention only.
    At non-expert layers (sparse mode): pc_block/action_block are None, pass-through.
    """

    def __init__(
        self,
        vlm_layer: nn.Module,
        pc_block: ExpertBlock | None,
        action_block: ExpertBlock | None,
        d_action: int,
        d_cond: int,
        layer_idx: int,
    ):
        super().__init__()
        self.vlm_layer = vlm_layer
        self.pc_block = pc_block
        self.action_block = action_block
        self.layer_idx = layer_idx

        if action_block is not None:
            # adaLN-zero (DiT): 6 mod params per ExpertBlock —
            # (scale1, shift1, gate1, scale2, shift2, gate2).
            self.action_adaln = nn.Sequential(
                nn.Linear(d_cond, d_action),
                nn.SiLU(),
                nn.Linear(d_action, d_action * 6),
            )
            # Zero-init the FINAL Linear → all 6 mod params start at zero →
            # block is identity at step 0; gradient ramps it up smoothly.
            nn.init.zeros_(self.action_adaln[-1].weight)
            nn.init.zeros_(self.action_adaln[-1].bias)

    def forward(
        self,
        vlm: Tensor,
        pc: Tensor,
        action: Tensor,
        cond: Tensor,
        vlm_attention_mask: Tensor | None,
        vlm_position_ids: Tensor | None,
        vlm_position_embeddings: tuple[Tensor, Tensor],
        vlm_cache_position: Tensor,
        kv_cache: DynamicCache,
        is_cross: bool = True,
        pc_position_embeddings: tuple[Tensor, Tensor] | None = None,
        action_position_embeddings: tuple[Tensor, Tensor] | None = None,
        mrope_section: list[int] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Returns:
            vlm, pc, action: updated hidden states.
        """
        # 1. VLM: actual Qwen decoder layer (frozen, always runs)
        vlm_out = self.vlm_layer(
            vlm,
            attention_mask=vlm_attention_mask,
            position_ids=vlm_position_ids,
            past_key_values=kv_cache,
            output_attentions=False,
            use_cache=True,
            cache_position=vlm_cache_position,
            position_embeddings=vlm_position_embeddings,
        )
        vlm = vlm_out[0]

        # Extract VLM K/V
        vlm_k = kv_cache.layers[self.layer_idx].keys.detach()
        vlm_v = kv_cache.layers[self.layer_idx].values.detach()

        # 2. PC expert
        pc_k = None
        pc_v = None
        if self.pc_block is not None:
            if is_cross:
                h_pc = self.pc_block.qkv.num_heads
                pc_ext_k = _match_heads(vlm_k, h_pc)
                pc_ext_v = _match_heads(vlm_v, h_pc)
            else:
                pc_ext_k = None
                pc_ext_v = None

            pc, pc_k, pc_v = self.pc_block(
                pc, ext_k=pc_ext_k, ext_v=pc_ext_v,
                position_embeddings=pc_position_embeddings,
                mrope_section=mrope_section,
                attn_mask=None,  # PC: always bidirectional
            )

        # 3. Action expert
        if self.action_block is not None:
            adaln_out = self.action_adaln(cond)
            s1, sh1, g1, s2, sh2, g2 = adaln_out.chunk(6, dim=-1)

            n_action = action.shape[1]

            if is_cross:
                # Cross-attention to VLM K/V + same-layer PC K/V
                h_action = self.action_block.qkv.num_heads
                ext_parts_k = [_match_heads(vlm_k, h_action)]
                ext_parts_v = [_match_heads(vlm_v, h_action)]
                if pc_k is not None:
                    # PC K/V are NOT detached: action flow-matching loss
                    # flows back into the PC expert.
                    ext_parts_k.append(_match_heads(pc_k, h_action))
                    ext_parts_v.append(_match_heads(pc_v, h_action))

                action_ext_k = torch.cat(ext_parts_k, dim=2)
                action_ext_v = torch.cat(ext_parts_v, dim=2)
                n_ext = action_ext_k.shape[2]
                action_mask = _build_action_cross_causal_mask(
                    n_action, n_ext, device=action.device,
                )
            else:
                # Self-attention only (causal)
                action_ext_k = None
                action_ext_v = None
                action_mask = _build_action_self_causal_mask(
                    n_action, device=action.device,
                )

            action, _, _ = self.action_block(
                action, ext_k=action_ext_k, ext_v=action_ext_v,
                adaln_mod=(s1, sh1, g1, s2, sh2, g2),
                position_embeddings=action_position_embeddings,
                mrope_section=mrope_section,
                attn_mask=action_mask,
            )

        return vlm, pc, action


class Backbone(nn.Module):
    """Full backbone using actual Qwen2.5-VL decoder layers for the VLM expert.

    Expert modes:
    - dense:  Experts at every layer. Cross-attention to VLM at expert_layer_ids,
              self-attention only at other layers. Full depth of processing.
    - sparse: Experts only at expert_layer_ids. Pass-through at other layers.

    Action cross-attends to VLM K/V (current layer) + PC K/V (same layer).
    """

    def __init__(
        self,
        layers: nn.ModuleList,
        vlm_norm: nn.Module,
        vlm_rotary_emb: nn.Module,
        vlm_config,
        expert_layer_ids: list[int],
        has_sliding_layers: bool = False,
    ):
        super().__init__()
        self.layers = layers
        self.vlm_norm = vlm_norm
        self.vlm_rotary_emb = vlm_rotary_emb
        self.vlm_config = vlm_config
        self.has_sliding_layers = has_sliding_layers
        self.num_layers = len(layers)

        self.expert_layer_ids = set(expert_layer_ids)

        rope_scaling = getattr(vlm_config, "rope_scaling", None) or {}
        self.mrope_section = rope_scaling.get("mrope_section", None)

    def _prepare_vlm_inputs(
        self,
        vlm_embeds: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor | None,
    ) -> dict:
        seq_len = vlm_embeds.shape[1]

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            rope_position_ids = position_ids[1:]
        else:
            text_position_ids = None
            rope_position_ids = position_ids

        position_embeddings = self.vlm_rotary_emb(vlm_embeds, rope_position_ids)
        cache_position = torch.arange(seq_len, device=vlm_embeds.device)

        mask_kwargs = {
            "config": self.vlm_config,
            "input_embeds": vlm_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": None,
            "position_ids": text_position_ids,
        }
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
        }
        if self.has_sliding_layers:
            causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(
                **mask_kwargs
            )

        vlm_attention_masks = []
        for layer in self.layers:
            attn_type = getattr(layer.vlm_layer, "attention_type", "full_attention")
            vlm_attention_masks.append(causal_mask_mapping[attn_type])

        return {
            "text_position_ids": text_position_ids,
            "rope_position_ids": rope_position_ids,
            "position_embeddings": position_embeddings,
            "cache_position": cache_position,
            "attention_masks": vlm_attention_masks,
        }

    def _prepare_expert_positions(
        self,
        vlm_rope_position_ids: Tensor,
        n_pc: int,
        n_action: int,
        vlm_embeds: Tensor,
    ) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        device = vlm_embeds.device
        B = vlm_embeds.shape[0]
        max_vlm_pos = vlm_rope_position_ids.max().item()

        pc_start = int(max_vlm_pos) + 1
        pc_pos_3d = (
            torch.arange(pc_start, pc_start + n_pc, device=device)
            .unsqueeze(0).expand(B, -1)
            .unsqueeze(0).expand(3, -1, -1)
        )
        pc_position_embeddings = self.vlm_rotary_emb(vlm_embeds[:, :n_pc], pc_pos_3d)

        action_start = pc_start + n_pc
        action_pos_3d = (
            torch.arange(action_start, action_start + n_action, device=device)
            .unsqueeze(0).expand(B, -1)
            .unsqueeze(0).expand(3, -1, -1)
        )
        action_position_embeddings = self.vlm_rotary_emb(vlm_embeds[:, :n_action], action_pos_3d)

        return pc_position_embeddings, action_position_embeddings

    def forward(
        self,
        vlm_embeds: Tensor,
        pc: Tensor,
        action: Tensor,
        cond: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        vlm_inputs = self._prepare_vlm_inputs(vlm_embeds, position_ids, attention_mask)
        kv_cache = DynamicCache()

        pc_pos_emb, action_pos_emb = self._prepare_expert_positions(
            vlm_inputs["rope_position_ids"],
            n_pc=pc.shape[1],
            n_action=action.shape[1],
            vlm_embeds=vlm_embeds,
        )

        for i, layer in enumerate(self.layers):
            is_cross = (i in self.expert_layer_ids)

            vlm_embeds, pc, action = layer(
                vlm_embeds, pc, action, cond,
                vlm_attention_mask=vlm_inputs["attention_masks"][i],
                vlm_position_ids=vlm_inputs["text_position_ids"],
                vlm_position_embeddings=vlm_inputs["position_embeddings"],
                vlm_cache_position=vlm_inputs["cache_position"],
                kv_cache=kv_cache,
                is_cross=is_cross,
                pc_position_embeddings=pc_pos_emb,
                action_position_embeddings=action_pos_emb,
                mrope_section=self.mrope_section,
            )

        vlm_embeds = self.vlm_norm(vlm_embeds)
        return vlm_embeds, pc, action
