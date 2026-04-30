"""
Unified VLA: Diffusion Policy Inside Qwen-VL

A single-model Vision-Language-Action inference pipeline that embeds a diffusion
policy for robot next-state prediction directly into a Qwen-VL transformer backbone.

Architecture:
  - Triple-stream design: VLM (frozen), Point Cloud (trainable), Action (trainable)
  - Causal attention with asymmetric gradient isolation
  - adaLN timestep conditioning on action stream only
  - Flow matching scheduler for action denoising

This is a sketch/draft for use as a starting point.
"""

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor


# =============================================================================
# Output
# =============================================================================

@dataclass
class VLAOutput:
    """Output of the UnifiedVLAPipeline."""
    actions: torch.Tensor  # (batch, action_seq_len, action_dim)


# =============================================================================
# Timestep Embedding
# =============================================================================

class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embeddings + MLP projection."""

    def __init__(self, dim: int, freq_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.freq_dim = freq_dim

    @staticmethod
    def sinusoidal_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / half)
        args = timesteps[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        t_emb = self.sinusoidal_embedding(timestep, self.freq_dim)
        return self.mlp(t_emb.to(self.mlp[0].weight.dtype))


# =============================================================================
# Triple-Stream Attention Processor
# =============================================================================

class TripleStreamAttnProcessor(nn.Module):
    """
    Attention processor for the triple-stream VLA block.

    Three streams with asymmetric gradient isolation:
      - VLM:    frozen Q/K/V/O, K/V detached from everything
      - PC:     trainable Q/K/V/O, K/V live for action queries, action K/V detached for PC queries
      - Action: trainable Q/K/V/O + adaLN, K/V live for action queries
    """

    def __init__(self, dim: int, num_heads: int, head_dim: int, qk_norm: bool = True, eps: float = 1e-6):
        super().__init__()
        inner_dim = num_heads * head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        # VLM projections — will be loaded from pretrained and frozen
        self.vlm_q_proj = nn.Linear(dim, inner_dim, bias=True)
        self.vlm_k_proj = nn.Linear(dim, inner_dim, bias=True)
        self.vlm_v_proj = nn.Linear(dim, inner_dim, bias=True)
        self.vlm_o_proj = nn.Linear(inner_dim, dim, bias=True)

        # Point cloud projections — trainable
        self.pc_q_proj = nn.Linear(dim, inner_dim, bias=True)
        self.pc_k_proj = nn.Linear(dim, inner_dim, bias=True)
        self.pc_v_proj = nn.Linear(dim, inner_dim, bias=True)
        self.pc_o_proj = nn.Linear(inner_dim, dim, bias=True)

        # Action projections — trainable
        self.act_q_proj = nn.Linear(dim, inner_dim, bias=True)
        self.act_k_proj = nn.Linear(dim, inner_dim, bias=True)
        self.act_v_proj = nn.Linear(dim, inner_dim, bias=True)
        self.act_o_proj = nn.Linear(inner_dim, dim, bias=True)

        # QK norm
        self.qk_norm = qk_norm
        if qk_norm:
            self.vlm_q_norm = nn.RMSNorm(head_dim, eps=eps)
            self.vlm_k_norm = nn.RMSNorm(head_dim, eps=eps)
            self.pc_q_norm = nn.RMSNorm(head_dim, eps=eps)
            self.pc_k_norm = nn.RMSNorm(head_dim, eps=eps)
            self.act_q_norm = nn.RMSNorm(head_dim, eps=eps)
            self.act_k_norm = nn.RMSNorm(head_dim, eps=eps)

    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, inner_dim) -> (B, L, num_heads, head_dim)"""
        return x.unflatten(-1, (self.num_heads, self.head_dim))

    def forward(
        self,
        vlm_h: torch.Tensor,       # (B, N_vlm, dim)
        pc_h: torch.Tensor,        # (B, N_pc, dim)
        act_h: torch.Tensor,       # (B, N_act, dim)
        causal_mask: torch.Tensor | None = None,  # (B, 1, L_total, L_total) or broadcastable
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            pc_attn_out:  (B, N_pc, dim)
            act_attn_out: (B, N_act, dim)
        """
        n_vlm = vlm_h.shape[1]
        n_pc = pc_h.shape[1]
        n_act = act_h.shape[1]

        # ── VLM Q/K/V (frozen, K/V detached) ──
        vlm_q = self._reshape(self.vlm_q_proj(vlm_h))
        vlm_k = self._reshape(self.vlm_k_proj(vlm_h)).detach()
        vlm_v = self._reshape(self.vlm_v_proj(vlm_h)).detach()

        # ── PC Q/K/V (trainable) ──
        pc_q = self._reshape(self.pc_q_proj(pc_h))
        pc_k = self._reshape(self.pc_k_proj(pc_h))
        pc_v = self._reshape(self.pc_v_proj(pc_h))

        # ── Action Q/K/V (trainable) ──
        act_q = self._reshape(self.act_q_proj(act_h))
        act_k = self._reshape(self.act_k_proj(act_h))
        act_v = self._reshape(self.act_v_proj(act_h))

        # ── QK norm ──
        if self.qk_norm:
            vlm_q, vlm_k = self.vlm_q_norm(vlm_q), self.vlm_k_norm(vlm_k)
            pc_q, pc_k = self.pc_q_norm(pc_q), self.pc_k_norm(pc_k)
            act_q, act_k = self.act_q_norm(act_q), self.act_k_norm(act_k)

        # ── Separate attention per stream for gradient isolation ──
        # PC attention: VLM(detached) + PC(live) + Action(detached)
        pc_joint_k = torch.cat([vlm_k, pc_k, act_k.detach()], dim=1)
        pc_joint_v = torch.cat([vlm_v, pc_v, act_v.detach()], dim=1)

        # Extract the PC rows from the causal mask
        if causal_mask is not None:
            pc_mask = causal_mask[:, :, n_vlm : n_vlm + n_pc, :]
        else:
            pc_mask = None

        pc_attn_out = F.scaled_dot_product_attention(
            pc_q.transpose(1, 2), pc_joint_k.transpose(1, 2), pc_joint_v.transpose(1, 2),
            attn_mask=pc_mask, dropout_p=0.0, is_causal=False,
        ).transpose(1, 2).flatten(2)

        # Action attention: VLM(detached) + PC(live!) + Action(live)
        act_joint_k = torch.cat([vlm_k, pc_k, act_k], dim=1)
        act_joint_v = torch.cat([vlm_v, pc_v, act_v], dim=1)

        if causal_mask is not None:
            act_mask = causal_mask[:, :, n_vlm + n_pc :, :]
        else:
            act_mask = None

        act_attn_out = F.scaled_dot_product_attention(
            act_q.transpose(1, 2), act_joint_k.transpose(1, 2), act_joint_v.transpose(1, 2),
            attn_mask=act_mask, dropout_p=0.0, is_causal=False,
        ).transpose(1, 2).flatten(2)

        # ── Output projections ──
        pc_attn_out = self.pc_o_proj(pc_attn_out)
        act_attn_out = self.act_o_proj(act_attn_out)

        return pc_attn_out, act_attn_out


# =============================================================================
# Triple-Stream DiT Block
# =============================================================================

class TripleStreamDiTBlock(nn.Module):
    """
    A single transformer block for the triple-stream VLA.

    Streams:
      - VLM:    frozen, no update, no adaLN
      - PC:     trainable, updates with residual, no adaLN
      - Action: trainable, updates with residual, adaLN from timestep
    """

    def __init__(self, dim: int, num_heads: int, head_dim: int, mlp_ratio: float = 4.0, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        mlp_dim = int(dim * mlp_ratio)

        # ── Attention ──
        self.attn = TripleStreamAttnProcessor(dim, num_heads, head_dim, qk_norm=True, eps=eps)

        # ── Point cloud stream (no adaLN) ──
        self.pc_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.pc_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.pc_mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_dim, dim),
        )
        self.pc_gate1 = nn.Parameter(torch.ones(dim))
        self.pc_gate2 = nn.Parameter(torch.ones(dim))

        # ── Action stream (with adaLN) ──
        self.act_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),  # shift1, scale1, gate1, shift2, scale2, gate2
        )
        self.act_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.act_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.act_mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_dim, dim),
        )

    @staticmethod
    def _adaln_modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Apply adaLN: x * (1 + scale) + shift. shift/scale are (B, D), x is (B, L, D)."""
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(
        self,
        vlm_h: torch.Tensor,       # (B, N_vlm, dim) — frozen context
        pc_h: torch.Tensor,        # (B, N_pc, dim) — evolving
        act_h: torch.Tensor,       # (B, N_act, dim) — evolving + denoising
        temb: torch.Tensor,        # (B, dim) — timestep embedding
        causal_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            pc_h:  updated point cloud hidden states
            act_h: updated action hidden states
        """
        # ── adaLN params for action stream ──
        act_mod_params = self.act_mod(temb)  # (B, 6*dim)
        shift1, scale1, gate1, shift2, scale2, gate2 = act_mod_params.chunk(6, dim=-1)

        # ── Norm ──
        vlm_normed = self.pc_norm1(vlm_h)  # VLM uses same norm structure, but doesn't update
        # NOTE: vlm_normed is only used for Q/K/V input to attention, vlm_h itself never changes
        pc_normed = self.pc_norm1(pc_h)
        act_normed = self._adaln_modulate(self.act_norm1(act_h), shift1, scale1)

        # ── Attention (triple-stream with gradient isolation) ──
        pc_attn_out, act_attn_out = self.attn(vlm_normed, pc_normed, act_normed, causal_mask)

        # ── Residual + gating ──
        pc_h = pc_h + self.pc_gate1 * pc_attn_out
        act_h = act_h + gate1.unsqueeze(1) * act_attn_out

        # ── MLP ──
        pc_h = pc_h + self.pc_gate2 * self.pc_mlp(self.pc_norm2(pc_h))
        act_mlp_in = self._adaln_modulate(self.act_norm2(act_h), shift2, scale2)
        act_h = act_h + gate2.unsqueeze(1) * self.act_mlp(act_mlp_in)

        return pc_h, act_h


# =============================================================================
# Unified VLA Transformer
# =============================================================================

class UnifiedVLATransformer(nn.Module):
    """
    The unified VLA transformer. Wraps a frozen Qwen-VL backbone and adds
    triple-stream DiT blocks for the action diffusion policy.

    Architecture:
        Qwen-VL layers [0, vlm_split_layer)    → understanding (run once)
        DiT blocks [0, num_dit_layers)          → denoising (run T times)

    The DiT blocks receive:
        - vlm_h:  cached VLM hidden states from layer vlm_split_layer
        - pc_h:   point cloud encoder output (detached)
        - act_h:  noisy action latents being denoised
    """

    def __init__(
        self,
        vlm: Qwen2_5_VLForConditionalGeneration,
        vlm_split_layer: int = 14,
        dim: int = 3584,
        num_dit_layers: int = 14,
        num_heads: int = 28,
        head_dim: int = 128,
        mlp_ratio: float = 4.0,
        action_dim: int = 64,
        action_seq_len: int = 16,
        pc_input_dim: int = 256,
    ):
        super().__init__()

        self.dim = dim
        self.vlm_split_layer = vlm_split_layer
        self.action_seq_len = action_seq_len

        # ── Frozen VLM backbone (only first vlm_split_layer layers used) ──
        self.vlm = vlm
        for param in self.vlm.parameters():
            param.requires_grad = False

        # ── Input projections ──
        self.action_in = nn.Linear(action_dim, dim)
        self.pc_in = nn.Linear(pc_input_dim, dim)

        # ── Timestep embedding ──
        self.time_embed = TimestepEmbedder(dim)

        # ── DiT blocks ──
        self.dit_blocks = nn.ModuleList([
            TripleStreamDiTBlock(dim, num_heads, head_dim, mlp_ratio)
            for _ in range(num_dit_layers)
        ])

        # ── Output head for action ──
        self.act_out_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.act_out_proj = nn.Linear(dim, action_dim)

        # ── Output head for point cloud (MSE target) ──
        self.pc_out_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.pc_out_proj = nn.Linear(dim, pc_input_dim)

    def _build_causal_mask(
        self, n_vlm: int, n_pc: int, n_act: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Build causal attention mask for the triple-stream sequence.

        Sequence order: [vlm_tokens, pc_tokens, action_tokens]

        Causal: each token attends to itself and all tokens before it.
        This means:
          - VLM tokens see only prior VLM tokens (but VLM doesn't update, so this is a no-op)
          - PC tokens see all VLM tokens + prior PC tokens
          - Action tokens see all VLM + all PC + prior action tokens
        """
        total = n_vlm + n_pc + n_act
        # Standard causal mask: lower triangular
        mask = torch.tril(torch.ones(total, total, device=device, dtype=torch.bool))
        # Expand for batch and head dims: (1, 1, L, L)
        return mask.unsqueeze(0).unsqueeze(0)

    @torch.no_grad()
    def encode_vlm(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Run the frozen VLM up to vlm_split_layer and return cached hidden states.

        This runs ONCE per generation — the output is reused across all denoising steps.

        Returns:
            vlm_hidden_states: (B, N_vlm, dim)
        """
        # Use Qwen2.5-VL's forward with output_hidden_states to get intermediate layer output
        outputs = self.vlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
        )

        # Extract hidden states from the split layer
        # hidden_states is a tuple of (num_layers + 1) tensors (including embedding layer)
        # Index vlm_split_layer + 1 because index 0 is the embedding output
        vlm_h = outputs.hidden_states[self.vlm_split_layer]

        return vlm_h

    def forward(
        self,
        vlm_h: torch.Tensor,           # (B, N_vlm, dim) — cached, frozen
        pc_features: torch.Tensor,      # (B, N_pc, pc_input_dim) — from pretrained encoder
        noisy_actions: torch.Tensor,    # (B, action_seq_len, action_dim) — noisy latent
        timestep: torch.Tensor,         # (B,) — diffusion timestep
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Single denoising step through the DiT blocks.

        Args:
            vlm_h:         Cached VLM hidden states (run encode_vlm once, reuse).
            pc_features:   Point cloud encoder output (detached before calling).
            noisy_actions: Current noisy action latents.
            timestep:      Current diffusion timestep.

        Returns:
            action_pred:   Predicted velocity/noise for the action (B, action_seq_len, action_dim)
            pc_pred:       Point cloud prediction for MSE loss (B, N_pc, pc_input_dim)
        """
        batch_size = vlm_h.shape[0]
        device = vlm_h.device

        # ── Project inputs to model dimension ──
        pc_h = self.pc_in(pc_features.detach())   # detach encoder output
        act_h = self.action_in(noisy_actions)

        # ── Timestep embedding ──
        temb = self.time_embed(timestep)

        # ── Build causal mask ──
        n_vlm = vlm_h.shape[1]
        n_pc = pc_h.shape[1]
        n_act = act_h.shape[1]
        causal_mask = self._build_causal_mask(n_vlm, n_pc, n_act, device, vlm_h.dtype)

        # ── DiT blocks ──
        for block in self.dit_blocks:
            pc_h, act_h = block(vlm_h, pc_h, act_h, temb, causal_mask)

        # ── Output heads ──
        action_pred = self.act_out_proj(self.act_out_norm(act_h))
        pc_pred = self.pc_out_proj(self.pc_out_norm(pc_h))

        return action_pred, pc_pred


# =============================================================================
# Inference Pipeline
# =============================================================================

class UnifiedVLAPipeline:
    """
    Inference pipeline for the Unified VLA.

    Usage:
        pipe = UnifiedVLAPipeline(model, tokenizer, processor, pc_encoder, scheduler)

        actions = pipe(
            prompt="pick up the red cup",
            images=camera_images,           # list of PIL images or tensor
            point_clouds=point_cloud_data,  # per-camera-view point clouds
            num_inference_steps=20,
        )
    """

    def __init__(
        self,
        model: UnifiedVLATransformer,
        tokenizer: Qwen2Tokenizer,
        processor: Qwen2VLProcessor,
        pc_encoder: nn.Module,
        scheduler: Any,  # e.g. FlowMatchEulerDiscreteScheduler
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.pc_encoder = pc_encoder
        self.scheduler = scheduler

        # Freeze point cloud encoder
        for param in self.pc_encoder.parameters():
            param.requires_grad = False

        # Chat template for Qwen2.5-VL style prompt
        self.prompt_template = (
            "<|im_start|>system\n"
            "You are a robot assistant. Observe the scene and follow the instruction.<|im_end|>\n"
            "<|im_start|>user\n"
            "<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        # Number of system tokens to drop from the hidden states (adjust based on your template)
        self.prompt_drop_idx = 20

    @property
    def device(self) -> torch.device:
        return next(self.model.dit_blocks.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.dit_blocks.parameters()).dtype

    # ─────────────────────────────────────────────────────────────
    # Encoding
    # ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def encode_prompt_and_images(
        self,
        prompt: str | list[str],
        images: list | None = None,
    ) -> torch.Tensor:
        """
        Encode text + camera images through the frozen VLM backbone.
        Returns cached hidden states from the VLM split layer.

        Args:
            prompt: Task instruction (e.g. "pick up the red cup")
            images: Camera images (list of PIL images)

        Returns:
            vlm_h: (B, N_vlm, dim)
        """
        if isinstance(prompt, str):
            prompt = [prompt]

        # Format with chat template
        texts = [self.prompt_template.format(p) for p in prompt]

        # Process through Qwen2VL processor (handles text tokenization + image processing)
        model_inputs = self.processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # Run VLM up to split layer
        vlm_h = self.model.encode_vlm(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            pixel_values=getattr(model_inputs, "pixel_values", None),
            image_grid_thw=getattr(model_inputs, "image_grid_thw", None),
        )

        # Drop system prefix tokens
        vlm_h = vlm_h[:, self.prompt_drop_idx:]

        return vlm_h

    @torch.no_grad()
    def encode_point_clouds(
        self,
        point_clouds: list[torch.Tensor] | torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode point clouds from all camera views through the pretrained encoder.

        Args:
            point_clouds: Per-view point clouds. Either:
                - list of (B, N_points, 3+) tensors, one per view
                - single (B, num_views, N_points, 3+) tensor

        Returns:
            pc_features: (B, total_pc_tokens, pc_input_dim) — concatenated across views
        """
        if isinstance(point_clouds, torch.Tensor) and point_clouds.dim() == 4:
            # (B, num_views, N_points, C) -> list of (B, N_points, C)
            point_clouds = [point_clouds[:, i] for i in range(point_clouds.shape[1])]

        view_features = []
        for pc_view in point_clouds:
            pc_view = pc_view.to(device=self.device, dtype=self.dtype)
            features = self.pc_encoder(pc_view)  # (B, N_tokens, pc_input_dim)
            view_features.append(features)

        # Concatenate all views: [view1_tokens, view2_tokens, ...]
        pc_features = torch.cat(view_features, dim=1)  # (B, total_pc_tokens, pc_input_dim)

        return pc_features

    # ─────────────────────────────────────────────────────────────
    # Inference
    # ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def __call__(
        self,
        prompt: str | list[str],
        images: list | None = None,
        point_clouds: list[torch.Tensor] | torch.Tensor = None,
        num_inference_steps: int = 20,
        generator: torch.Generator | None = None,
        sigmas: list[float] | None = None,
        action_seq_len: int | None = None,
        vlm_h: torch.Tensor | None = None,
        pc_features: torch.Tensor | None = None,
        return_dict: bool = True,
    ) -> VLAOutput | torch.Tensor:
        """
        Run the full inference pipeline.

        Args:
            prompt:             Task instruction.
            images:             Camera images (list of PIL images).
            point_clouds:       Per-view point clouds.
            num_inference_steps: Number of denoising steps.
            generator:          Random generator for reproducibility.
            sigmas:             Custom noise schedule (optional).
            action_seq_len:     Override action sequence length (default: model's config).
            vlm_h:              Pre-computed VLM hidden states (skip encoding if provided).
            pc_features:        Pre-computed PC features (skip encoding if provided).
            return_dict:        Whether to return VLAOutput or raw tensor.

        Returns:
            VLAOutput with predicted actions, or raw action tensor.
        """
        action_seq_len = action_seq_len or self.model.action_seq_len
        action_dim = self.model.act_out_proj.out_features

        # ── Step 1: Encode conditioning (run once) ──

        if vlm_h is None:
            vlm_h = self.encode_prompt_and_images(prompt, images)

        if pc_features is None and point_clouds is not None:
            pc_features = self.encode_point_clouds(point_clouds)

        batch_size = vlm_h.shape[0]

        # ── Step 2: Initialize noisy action latents ──
        action_shape = (batch_size, action_seq_len, action_dim)
        actions = torch.randn(action_shape, device=self.device, dtype=self.dtype, generator=generator)

        # ── Step 3: Setup scheduler ──
        if sigmas is None:
            sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
        self.scheduler.set_timesteps(num_inference_steps, device=self.device, sigmas=sigmas)
        timesteps = self.scheduler.timesteps

        # ── Step 4: Denoising loop ──
        for t in timesteps:
            timestep = t.expand(batch_size).to(self.dtype)

            # Forward pass through DiT blocks
            action_pred, _ = self.model(
                vlm_h=vlm_h,
                pc_features=pc_features,
                noisy_actions=actions,
                timestep=timestep / 1000,  # normalize timestep to [0, 1]
            )

            # Scheduler step: x_t -> x_{t-1}
            actions = self.scheduler.step(action_pred, t, actions, return_dict=False)[0]

        # ── Step 5: Return ──
        if not return_dict:
            return actions

        return VLAOutput(actions=actions)


# =============================================================================
# Training Wrapper (sketch)
# =============================================================================

class UnifiedVLATrainingWrapper(nn.Module):
    """
    Training wrapper that handles the asymmetric loss computation.

    Loss:
      - Flow matching on action stream
      - MSE on point cloud stream
      - VLM is frozen (no loss)

    Gradient isolation:
      - Flow matching → action params + PC K/V (PC learns to help action)
      - MSE → PC params only (action K/V detached inside attention)
      - VLM → fully frozen + detached
    """

    def __init__(
        self,
        model: UnifiedVLATransformer,
        pc_loss_weight: float = 1.0,
        action_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.model = model
        self.pc_loss_weight = pc_loss_weight
        self.action_loss_weight = action_loss_weight

    def forward(
        self,
        vlm_h: torch.Tensor,                # (B, N_vlm, dim) — cached, no grad
        pc_features: torch.Tensor,           # (B, N_pc, pc_input_dim) — from encoder
        noisy_actions: torch.Tensor,         # (B, action_seq_len, action_dim)
        timestep: torch.Tensor,              # (B,)
        action_target: torch.Tensor,         # (B, action_seq_len, action_dim) — flow matching target (velocity)
        pc_target: torch.Tensor,             # (B, N_pc, pc_input_dim) — MSE target
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass with loss computation.

        Returns dict with:
            loss:        combined loss
            action_loss: flow matching loss on actions
            pc_loss:     MSE loss on point cloud
        """
        # Forward through model (gradient isolation is handled inside TripleStreamAttnProcessor)
        action_pred, pc_pred = self.model(
            vlm_h=vlm_h,
            pc_features=pc_features,
            noisy_actions=noisy_actions,
            timestep=timestep,
        )

        # ── Flow matching loss on actions ──
        action_loss = F.mse_loss(action_pred, action_target)

        # ── MSE loss on point cloud ──
        pc_loss = F.mse_loss(pc_pred, pc_target)

        # ── Combined ──
        loss = self.action_loss_weight * action_loss + self.pc_loss_weight * pc_loss

        return {
            "loss": loss,
            "action_loss": action_loss,
            "pc_loss": pc_loss,
        }


# =============================================================================
# Example usage
# =============================================================================

if __name__ == "__main__":
    print("Unified VLA Pipeline — Architecture Sketch")
    print("=" * 50)
    print()
    print("Components:")
    print("  1. UnifiedVLATransformer  — model with frozen VLM + trainable DiT blocks")
    print("  2. UnifiedVLAPipeline     — inference pipeline (encode once, denoise T steps)")
    print("  3. UnifiedVLATrainingWrapper — training with asymmetric gradient isolation")
    print()
    print("Triple-stream sequence: [vlm_tokens | pc_tokens | action_tokens]")
    print()
    print("Gradient flow:")
    print("  Flow matching → Action params + PC K/V")
    print("  MSE           → PC params only")
    print("  VLM           → Frozen + detached")
