"""Weight surgery: load a pretrained Qwen2.5-VL model and create the backbone."""

import torch
import torch.nn as nn

from .core import SequenceBuilder
from .layers import ExpertBlock, init_expert_from_vlm
from .backbone import MultiExpertLayer, Backbone


def _qwen_layer_to_expert_block(qwen_layer, d_vlm: int, head_dim: int,
                                 num_kv_heads: int, intermediate_size: int) -> ExpertBlock:
    """Convert a Qwen2.5-VL decoder layer into an ExpertBlock for weight init."""
    block = ExpertBlock(
        d_expert=d_vlm, head_dim=head_dim,
        num_kv_heads=num_kv_heads, intermediate_size=intermediate_size,
    )
    with torch.no_grad():
        block.qkv.q_proj.weight.copy_(qwen_layer.self_attn.q_proj.weight)
        block.qkv.q_proj.bias.copy_(qwen_layer.self_attn.q_proj.bias)
        block.qkv.k_proj.weight.copy_(qwen_layer.self_attn.k_proj.weight)
        block.qkv.k_proj.bias.copy_(qwen_layer.self_attn.k_proj.bias)
        block.qkv.v_proj.weight.copy_(qwen_layer.self_attn.v_proj.weight)
        block.qkv.v_proj.bias.copy_(qwen_layer.self_attn.v_proj.bias)
        block.o_proj.weight.copy_(qwen_layer.self_attn.o_proj.weight)
        block.norm1.weight.copy_(qwen_layer.input_layernorm.weight)
        block.norm2.weight.copy_(qwen_layer.post_attention_layernorm.weight)
        block.mlp.gate_proj.weight.copy_(qwen_layer.mlp.gate_proj.weight)
        block.mlp.up_proj.weight.copy_(qwen_layer.mlp.up_proj.weight)
        block.mlp.down_proj.weight.copy_(qwen_layer.mlp.down_proj.weight)
    return block


def create_backbone(
    qwen_model,
    d_pc: int,
    d_action: int,
    d_cond: int,
    num_layers: int | None = None,
    expert_layer_ids: list[int] | None = None,
    expert_mode: str = "dense",
    num_kv_heads_pc: int | None = None,
    num_kv_heads_action: int | None = None,
    intermediate_size_pc: int | None = None,
    intermediate_size_action: int | None = None,
) -> Backbone:
    """Create a Backbone from a pretrained Qwen2.5-VL model.

    VLM: actual Qwen decoder layers (frozen, unmodified) at every layer.
    PC/Action: ExpertBlocks with RMSNorm + MRoPE, initialized from VLM weights.

    Expert modes:
        dense:  PC/Action experts at every layer. Cross-attention to VLM only
                at expert_layer_ids, self-attention only at other layers.
        sparse: PC/Action experts only at expert_layer_ids. Pass-through elsewhere.

    Action cross-attends to VLM K/V (current layer) + PC K/V (same layer).

    Args:
        qwen_model: loaded Qwen2_5_VLForConditionalGeneration.
        d_pc, d_action: expert widths.
        d_cond: conditioning vector dimension.
        num_layers: how many VLM layers to use (None = all).
        expert_layer_ids: layers where cross-attention happens (None = all layers).
        expert_mode: "dense" or "sparse".
    """
    assert expert_mode in ("dense", "sparse")

    text_model = qwen_model.model.language_model
    text_config = qwen_model.config.text_config
    d_vlm = text_config.hidden_size
    head_dim = d_vlm // text_config.num_attention_heads
    num_kv_heads_vlm = text_config.num_key_value_heads
    intermediate_size_vlm = text_config.intermediate_size
    total_layers = text_config.num_hidden_layers

    if num_layers is None:
        num_layers = total_layers
    if expert_layer_ids is None:
        expert_layer_ids = list(range(num_layers))

    # Default expert GQA and MLP sizes
    gqa_ratio = text_config.num_attention_heads // num_kv_heads_vlm
    if num_kv_heads_pc is None:
        num_kv_heads_pc = max(1, (d_pc // head_dim) // gqa_ratio)
    if num_kv_heads_action is None:
        num_kv_heads_action = max(1, (d_action // head_dim) // gqa_ratio)
    mlp_ratio = intermediate_size_vlm / d_vlm
    if intermediate_size_pc is None:
        intermediate_size_pc = int(d_pc * mlp_ratio)
    if intermediate_size_action is None:
        intermediate_size_action = int(d_action * mlp_ratio)

    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
    rms_norm_eps = text_config.rms_norm_eps

    def _make_expert(qwen_layer, d_expert, num_kv_heads_expert, intermediate_size_expert):
        vlm_block = _qwen_layer_to_expert_block(
            qwen_layer, d_vlm, head_dim, num_kv_heads_vlm, intermediate_size_vlm,
        )
        expert = ExpertBlock(
            d_expert, head_dim, num_kv_heads_expert, intermediate_size_expert,
            norm_class=Qwen2RMSNorm, norm_kwargs={"eps": rms_norm_eps},
        )
        init_expert_from_vlm(expert, vlm_block)
        return expert

    expert_set = set(expert_layer_ids)

    # Determine which layers get expert blocks
    if expert_mode == "dense":
        expert_block_layers = set(range(num_layers))  # every layer
    else:
        expert_block_layers = expert_set  # only cross-attention layers

    layers = nn.ModuleList()
    for i in range(num_layers):
        qwen_layer = text_model.layers[i]

        if i in expert_block_layers:
            pc_block = _make_expert(qwen_layer, d_pc, num_kv_heads_pc, intermediate_size_pc)
            action_block = _make_expert(qwen_layer, d_action, num_kv_heads_action, intermediate_size_action)
        else:
            pc_block = None
            action_block = None

        mel = MultiExpertLayer(
            vlm_layer=qwen_layer,
            pc_block=pc_block,
            action_block=action_block,
            d_action=d_action,
            d_cond=d_cond,
            layer_idx=i,
        )

        for p in mel.vlm_layer.parameters():
            p.requires_grad_(False)

        layers.append(mel)

    has_sliding = hasattr(text_model, "has_sliding_layers") and text_model.has_sliding_layers

    backbone = Backbone(
        layers=layers,
        vlm_norm=text_model.norm,
        vlm_rotary_emb=text_model.rotary_emb,
        vlm_config=text_config,
        expert_layer_ids=expert_layer_ids,
        has_sliding_layers=has_sliding,
    )

    for p in backbone.vlm_norm.parameters():
        p.requires_grad_(False)
    for p in backbone.vlm_rotary_emb.parameters():
        p.requires_grad_(False)

    return backbone


def init_special_tokens_from_vlm(
    sequence_builder: SequenceBuilder,
    qwen_model,
    eps: float = 1e-3,
) -> None:
    """Initialize learnable delimiter tokens from VLM embeddings + ε.

    Without this function, `<pc_*_start>`, `<pc_*_end>`, `<proprio_*>`, and
    `<action_*>` are randomly initialized. The frozen VLM has never seen
    those random vectors, so it has no useful prior for them. This function
    bootstraps each new delimiter from the VLM's own vision delimiters:

        base = mean(embed(<|vision_start|>), embed(<|vision_end|>))   # (d_vlm,)

    PC delimiters are sliced to `d_pc`; Action / Proprio delimiters to
    `d_action`. Each parameter gets an independent N(0, eps) sample on top
    of the base so they stay distinct.

    The `<align>` token is left random — it carries a discrete binary
    signal (aligned vs unaligned point clouds), not a delimiter role, so
    the vision-tokens base is not a meaningful prior for it.

    Embodiment is handled separately: `SharedEmbodimentEmbedding` self-
    initializes its base table and zero-initializes its projection heads
    (Recipe 2 — embodiment contributes nothing at step 0). The vision-
    tokens base does not apply across the bottleneck.

    Args:
        sequence_builder: the `SequenceBuilder` whose delimiter Parameters
            get overwritten in place.
        qwen_model: a loaded `Qwen2_5_VLForConditionalGeneration`-like
            model exposing `config.vision_start_token_id`,
            `config.vision_end_token_id`, and
            `model.language_model.embed_tokens`.
        eps: stddev of the Gaussian noise added on top of the base.
    """
    text_model = qwen_model.model.language_model
    embed_table = text_model.embed_tokens.weight  # (V, d_vlm)
    vision_start_id = qwen_model.config.vision_start_token_id
    vision_end_id = qwen_model.config.vision_end_token_id

    base = (embed_table[vision_start_id] + embed_table[vision_end_id]) / 2  # (d_vlm,)
    base = base.detach()

    pc_params = (
        sequence_builder.pc_start_emb,
        sequence_builder.pc_end_emb,
        sequence_builder.pc_wrist_start_emb,
        sequence_builder.pc_wrist_end_emb,
    )
    action_params = (
        sequence_builder.proprio_start_emb,
        sequence_builder.proprio_end_emb,
        sequence_builder.action_start_emb,
        sequence_builder.action_end_emb,
    )

    with torch.no_grad():
        d_pc = sequence_builder.d_pc
        for p in pc_params:
            noise = torch.randn(p.shape, device=p.device, dtype=p.dtype) * eps
            p.copy_(base[:d_pc].to(dtype=p.dtype, device=p.device).view(1, 1, d_pc) + noise)

        d_action = sequence_builder.d_action
        for p in action_params:
            noise = torch.randn(p.shape, device=p.device, dtype=p.dtype) * eps
            p.copy_(base[:d_action].to(dtype=p.dtype, device=p.device).view(1, 1, d_action) + noise)
