import enum
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class TokenType(enum.IntEnum):
    VLM = 0
    PC = 1
    ACTION = 2


@dataclass
class TokenSequence:
    """Multi-modal token sequence with per-modality hidden states.

    Hidden states are stored separately per modality because expert widths
    differ (d_vlm != d_pc != d_action). The token_types tensor records the
    global ordering for attention mask construction.
    """

    vlm: Tensor  # (B, N_vlm, d_vlm)
    pc: Tensor  # (B, N_pc_full, d_pc) — includes <align>, per-camera [<pc_start> tokens <pc_end>]
    action: Tensor  # (B, N_action_full, d_action) — includes <action_start>, tokens, <action_end>
    token_types: Tensor  # (B, N_total) int tensor


class SequenceBuilder(nn.Module):
    """Wraps per-modality tokens with learnable boundary embeddings.

    Follows Qwen-VL's delimiter pattern:
    - Images in VLM:  <|vision_start|> <|image_pad|>×N <|vision_end|>  per camera
    - Point clouds:   <|pc_start|> <|pc_pad|>×M <|pc_end|>            per non-wrist camera
                      <|pc_wrist_start|> <|pc_pad|>×M <|pc_wrist_end|> per wrist camera
    - Action:         <|proprio_start|> proprio_tokens <|proprio_end|>
                      <|action_start|> action_tokens <|action_end|>

    PC segment layout for K cameras (mix of base / wrist):
        sink_pc embodiment_pc <align> <pc_*_start> pc_cam1 <pc_*_end> ... per camera

    Action segment layout:
        sink_action embodiment_action
                    <proprio_start> proprio <proprio_end>
                    <action_start> action_tokens <action_end>

    A learnable attention sink sits at position 0 of each segment. Sinks
    have no input data — they are content-free Parameters whose role is
    to absorb the softmax mass that heads can't usefully place elsewhere
    (Xiao et al. 2023). Because they sit at position 0, every later token
    in PC's bidirectional self-attention and every causal Action query
    has them in its key set; Action's cross-attention to PC K/V also
    sees `sink_pc` for the same drain purpose. Two separate Parameters
    (no shared bottleneck) — sinks are not a shared concept across
    streams; what makes a good sink for PC heads is independent of what
    makes a good sink for Action heads.

    Embodiment is supplied as already-projected per-stream tokens (see
    `SharedEmbodimentEmbedding`) at width d_pc and d_action respectively;
    both flow from a single bottleneck embedding so PC and Action share
    one per-robot latent. With sinks present, embodiment now sits at
    position 1 of each segment (sink at 0).

    The <align> token is prepended once (modulated by alignment quality).
    Wrist delimiters apply to PC segment only — VLM-side wrist delimiters
    are deferred until Phase 2 (VLM stays frozen in Phase 1).

    Action segment: proprio tokens (variable count P, e.g. current-only or
    current+history) are wrapped in <proprio_start> / <proprio_end> so the
    model has a learned cue for the boundary even when P varies across
    samples. The whole segment shares a single unified causal mask and
    TokenType.ACTION.
    """

    def __init__(self, d_pc: int, d_action: int):
        super().__init__()
        self.d_pc = d_pc
        self.d_action = d_action

        # Attention sinks — content-free drains at position 0 of each segment.
        # Two separate Parameters (NOT shared); see class docstring.
        self.sink_pc = nn.Parameter(torch.randn(1, 1, d_pc))
        self.sink_action = nn.Parameter(torch.randn(1, 1, d_action))

        # PC boundary tokens — base pair for non-wrist cameras
        self.align_base_emb = nn.Parameter(torch.randn(1, 1, d_pc))
        self.pc_start_emb = nn.Parameter(torch.randn(1, 1, d_pc))
        self.pc_end_emb = nn.Parameter(torch.randn(1, 1, d_pc))

        # PC boundary tokens — wrist pair (separate so the model can learn
        # different priors for wrist-mounted vs scene-mounted cameras)
        self.pc_wrist_start_emb = nn.Parameter(torch.randn(1, 1, d_pc))
        self.pc_wrist_end_emb = nn.Parameter(torch.randn(1, 1, d_pc))

        # Proprio boundary tokens — wrap the variable-length proprio block
        self.proprio_start_emb = nn.Parameter(torch.randn(1, 1, d_action))
        self.proprio_end_emb = nn.Parameter(torch.randn(1, 1, d_action))

        # Action boundary tokens
        self.action_start_emb = nn.Parameter(torch.randn(1, 1, d_action))
        self.action_end_emb = nn.Parameter(torch.randn(1, 1, d_action))

    def forward(
        self,
        vlm_tokens: Tensor,  # (B, N_vlm, d_vlm)
        pc_tokens_per_camera: list[Tensor],  # K tensors, each (B, M_k, d_pc)
        action_tokens: Tensor,  # (B, N_action, d_action)
        proprio_tokens: Tensor,  # (B, P, d_action) — already projected by ProprioEncoder
        embodiment_pc: Tensor,  # (B, 1, d_pc) — from SharedEmbodimentEmbedding
        embodiment_action: Tensor,  # (B, 1, d_action) — from SharedEmbodimentEmbedding
        is_wrist_per_camera: Optional[list[bool]] = None,  # one flag per camera; None = all non-wrist
        align_emb: Optional[Tensor] = None,  # (B, 1, d_pc) from AlignmentToken
    ) -> TokenSequence:
        B = vlm_tokens.shape[0]
        device = vlm_tokens.device

        # Align embedding: use external (modulated) or fall back to base
        if align_emb is None:
            align = self.align_base_emb.expand(B, -1, -1)
        else:
            align = align_emb

        if is_wrist_per_camera is None:
            is_wrist_per_camera = [False] * len(pc_tokens_per_camera)

        pc_start_base = self.pc_start_emb.expand(B, -1, -1)
        pc_end_base = self.pc_end_emb.expand(B, -1, -1)
        pc_start_wrist = self.pc_wrist_start_emb.expand(B, -1, -1)
        pc_end_wrist = self.pc_wrist_end_emb.expand(B, -1, -1)

        sink_pc = self.sink_pc.expand(B, -1, -1)
        sink_action = self.sink_action.expand(B, -1, -1)

        # PC segment: [sink_pc, embodiment_pc, <align>] then per-camera [<pc_*_start> tokens <pc_*_end>]
        pc_parts = [sink_pc, embodiment_pc, align]
        for cam_tokens, is_wrist in zip(pc_tokens_per_camera, is_wrist_per_camera):
            if is_wrist:
                pc_parts.extend([pc_start_wrist, cam_tokens, pc_end_wrist])
            else:
                pc_parts.extend([pc_start_base, cam_tokens, pc_end_base])
        pc_full = torch.cat(pc_parts, dim=1)

        # Action segment:
        #   [sink_action, embodiment_action,
        #    <proprio_start>, proprio, <proprio_end>,
        #    <action_start>, action_tokens, <action_end>]
        proprio_start = self.proprio_start_emb.expand(B, -1, -1)
        proprio_end = self.proprio_end_emb.expand(B, -1, -1)
        action_start = self.action_start_emb.expand(B, -1, -1)
        action_end = self.action_end_emb.expand(B, -1, -1)
        action_full = torch.cat(
            [
                sink_action,
                embodiment_action,
                proprio_start, proprio_tokens, proprio_end,
                action_start, action_tokens, action_end,
            ],
            dim=1,
        )

        # Token types
        N_vlm = vlm_tokens.shape[1]
        N_pc_full = pc_full.shape[1]
        N_action_full = action_full.shape[1]

        token_types = torch.cat(
            [
                torch.full((B, N_vlm), TokenType.VLM, dtype=torch.long, device=device),
                torch.full((B, N_pc_full), TokenType.PC, dtype=torch.long, device=device),
                torch.full((B, N_action_full), TokenType.ACTION, dtype=torch.long, device=device),
            ],
            dim=1,
        )

        return TokenSequence(
            vlm=vlm_tokens,
            pc=pc_full,
            action=action_full,
            token_types=token_types,
        )


class AlignmentToken(nn.Module):
    """Learnable <align> embedding indexed by alignment flag.

    Two discrete states:
        0 = unaligned point clouds
        1 = aligned point clouds
    """

    def __init__(self, d_pc: int):
        super().__init__()
        self.embed = nn.Embedding(2, d_pc)

    def forward(self, a: Tensor) -> Tensor:
        """
        Args:
            a: (B,) int or long tensor with values in {0, 1}.
               0 = unaligned, 1 = aligned.
        Returns:
            (B, 1, d_pc) alignment embedding.
        """
        return self.embed(a.long()).unsqueeze(1)  # (B, 1, d_pc)


class InputProjections(nn.Module):
    """Projects cached encoder outputs to expert widths.

    VLM: identity (already at d_vlm from cached Qwen-VL features).
    PC:  Linear(pc_encoder_dim, d_pc) — trainable.
    Action: Linear(action_dim, d_action) — trainable.

    The dataloader provides pre-computed tensors from frozen encoders,
    so these projections are the first trainable layer for PC and Action.
    """

    def __init__(self, pc_encoder_dim: int, d_pc: int, action_dim: int, d_action: int):
        super().__init__()
        self.pc_proj = nn.Linear(pc_encoder_dim, d_pc)
        self.action_proj = nn.Linear(action_dim, d_action)

    def forward(
        self,
        vlm_tokens: Tensor,  # (B, N_vlm, d_vlm) — passed through unchanged
        pc_tokens: Tensor,  # (B, N_pc, pc_encoder_dim)
        action_tokens: Tensor,  # (B, N_action, action_dim)
    ) -> tuple[Tensor, Tensor, Tensor]:
        return (
            vlm_tokens,
            self.pc_proj(pc_tokens),
            self.action_proj(action_tokens),
        )


class SharedEmbodimentEmbedding(nn.Module):
    """One per-robot bottleneck embedding, projected separately into PC and Action streams.

    A single `nn.Embedding(3, d_emb)` holds the per-robot identity. Two
    `Linear(d_emb, d_*)` heads (no bias) surface the same identity at the
    PC-stream width and at the Action-stream width. Both projections share
    the same base, so the model is forced to represent embodiment in a
    single `d_emb`-dimensional latent that the PC and Action experts
    interpret in their own coordinates.

    Where the spliced tokens sit:
        PC segment:     [embodiment_pc, <align>, <pc_*_start> cam <pc_*_end>, ...]
        Action segment: [embodiment_action, <proprio_start>, proprio, <proprio_end>,
                         <action_start>, action, <action_end>]

    Initialization (Recipe 2 — zero-init projections):
        - `base` uses default `nn.Embedding` init (full-strength categorical).
        - `to_pc` and `to_action` start at all zeros, so embodiment_pc and
          embodiment_action are exactly zero vectors at step 0 — training
          starts identical to a no-embodiment baseline. The projections
          ramp up as gradient flows back from both losses.

    Robot IDs (also kept verbatim in the prompt's `Robot: {description}` line):
        0 = Trossen
        1 = Franka
        2 = SO-101
    """

    ROBOT_IDS = {"trossen": 0, "franka": 1, "so-101": 2}
    NUM_ROBOTS = 3

    def __init__(self, d_pc: int, d_action: int, d_emb: int = 32):
        super().__init__()
        self.d_emb = d_emb
        self.base = nn.Embedding(self.NUM_ROBOTS, d_emb)
        self.to_pc = nn.Linear(d_emb, d_pc, bias=False)
        self.to_action = nn.Linear(d_emb, d_action, bias=False)
        # Recipe-2 init: zero projections so embodiment contributes nothing at step 0.
        nn.init.zeros_(self.to_pc.weight)
        nn.init.zeros_(self.to_action.weight)

    def forward(self, robot_id: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            robot_id: (B,) int tensor with values in {0, 1, 2}.
        Returns:
            embod_pc:     (B, 1, d_pc)
            embod_action: (B, 1, d_action)
        """
        base = self.base(robot_id.long())                # (B, d_emb)
        return (
            self.to_pc(base).unsqueeze(1),               # (B, 1, d_pc)
            self.to_action(base).unsqueeze(1),           # (B, 1, d_action)
        )


class ProprioEncoder(nn.Module):
    """Projects raw proprio readings to action-stream tokens.

    Accepts one or more proprio timesteps and projects each to d_action with a
    single Linear. The output is concatenated with the noisy action chunk
    inside the Action expert under a unified causal mask.

    Embodiment is conveyed via the VLM prompt's "Robot: ..." text and (in a
    later todo) via dedicated prompt tokens — not through this encoder.
    """

    def __init__(self, proprio_dim: int, d_action: int):
        super().__init__()
        self.proj = nn.Linear(proprio_dim, d_action)

    def forward(self, proprio: Tensor) -> Tensor:
        """
        Args:
            proprio: (B, P, proprio_dim) for P proprio timesteps,
                     or (B, proprio_dim) for the current step only.
        Returns:
            (B, P, d_action) — one token per proprio timestep.
        """
        if proprio.dim() == 2:
            proprio = proprio.unsqueeze(1)
        return self.proj(proprio)


class AdaLNConditioner(nn.Module):
    """Produces 6 modulation params per ExpertBlock — DiT adaLN-zero style.

    Maps a timestep embedding through an MLP and chunks the output into:

        (scale1, shift1, gate1,  scale2, shift2, gate2)

    where (scale_i, shift_i) modulate the i-th RMSNorm and gate_i scales the
    i-th sub-block's output before residual addition. Sub-block 1 = attention,
    sub-block 2 = MLP.

    Recipe (DiT, Peebles & Xie 2023): the **final** Linear is zero-initialized
    so all 6 mod params start at zero. With `modulate(x, scale, shift) =
    (1 + scale) * LN(x) + shift`, scale=0 means identity-LN, gate=0 means the
    sub-block contributes nothing to the residual, and the entire block
    behaves as identity at step 0. Training learns to ramp gates up from
    zero, which is empirically much more stable than starting with random
    modulation that immediately distorts whatever the (VLM-initialized)
    expert weights were doing.
    """

    def __init__(self, d_t: int, d_action: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_t, d_action),
            nn.SiLU(),
            nn.Linear(d_action, d_action * 6),
        )
        # adaLN-zero: zero the FINAL Linear so the block starts as identity.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self, t_emb: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Args:
            t_emb: (B, d_t)
        Returns:
            scale1, shift1, gate1, scale2, shift2, gate2: each (B, d_action).
        """
        out = self.mlp(t_emb)
        scale1, shift1, gate1, scale2, shift2, gate2 = out.chunk(6, dim=-1)
        return scale1, shift1, gate1, scale2, shift2, gate2


def modulate(x: Tensor, norm: nn.LayerNorm, scale: Tensor, shift: Tensor) -> Tensor:
    """Apply DiT-style adaLN modulation: (1 + scale) * LayerNorm(x) + shift.

    The `(1 + scale)` form means scale=0 produces identity-LN — combined with
    zero-init of the adaLN MLP's final Linear, this lets the block start as
    the residual stream's identity at step 0 (adaLN-zero). The `+ 1` is the
    only difference vs. the more naive `scale * LN(x) + shift` form.

    Args:
        x: (B, N, d) hidden states.
        norm: LayerNorm to apply before modulation.
        scale: (B, d) scale vector — added to 1 before broadcasting.
        shift: (B, d) shift vector — broadcast over sequence dim.
    Returns:
        (B, N, d) modulated hidden states.
    """
    return (1 + scale).unsqueeze(1) * norm(x) + shift.unsqueeze(1)
