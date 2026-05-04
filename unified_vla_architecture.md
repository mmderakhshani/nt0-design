# Unified VLA: Single-Backbone Multi-Expert Architecture

Qwen2.5-VL transformer backbone with per-layer modality experts for vision-language understanding, point cloud grounding, and action denoising. The VLM expert is the actual unmodified Qwen2.5-VL decoder layer (frozen, zero distribution shift). PC and Action experts are downsized ExpertBlocks initialized from VLM weights, sharing the same RMSNorm and MRoPE. Experts can be placed at all layers or at selected layers only, in either dense or sparse mode.


## Core Idea

The frozen VLM expert is the actual `Qwen2_5_VLDecoderLayer` — not a conversion or approximation. It retains its native 3D MRoPE, RMSNorm, GQA, causal + packed-sequence attention mask, and sliding window layers. VLM runs at every backbone layer. VLM K/V (after RoPE) are extracted from DynamicCache and passed to downstream experts with `.detach()`.

PC and Action experts are `ExpertBlock` modules — downsized transformer blocks that match the VLM's architectural primitives (RMSNorm, GatedMLP, GQA, MRoPE) at reduced width. Initialized from VLM weights by slicing the first N heads and truncating MLP dimensions.

Experts are placed at configurable layers (`expert_layer_ids`). Two modes control what happens at non-expert layers:
- **Dense**: experts exist at every layer; cross-attention to VLM only at `expert_layer_ids`, self-attention only elsewhere.
- **Sparse**: experts only at `expert_layer_ids`; hidden states pass through unchanged elsewhere.


## Conditioning Paths

Each conditioning signal enters the model through one or two paths:

    Signal                  Token sequence (VLM/PC/Action)               adaLN (Action only)
    ------                  ---------------------------------            -------------------
    Camera images (xK)      VLM: <|vision_start|>...<|vision_end|> per cam    --
    Task instruction        VLM: text "<task>...</task>"                 --
    Robot description       VLM: text "Robot: ..."                       --
    Embodiment ID           PC + Action: SharedEmbodimentEmbedding -> embod_pc, embod_action  --
    Attention sinks         PC: sink_pc; Action: sink_action (content-free Parameters)   --
    Point clouds (xK)       PC: <pc_(wrist_)start> tokens <pc_(wrist_)end> per cam   --
    Alignment flag          PC: <align> Embedding(2)[0 or 1]             --
    Proprio (P x 10)        Action: <proprio_start> ProprioEncoder(proprio) <proprio_end>   --
    Noisy action chunk      Action: <action_start> tokens <action_end>   --
    Timestep t              --                                           TimestepEmbedding(t)

Proprio enters as Action-stream tokens: `Linear(proprio_dim, d_action)` projects each proprio reading and the result is wrapped in `<proprio_start>` / `<proprio_end>` learned delimiters before the action chunk inside the Action expert. The single Linear is shared across timesteps, so it works for current-only (P=1) or current+history (P>1); the delimiters give the model a learned cue for the boundary even when P varies sample to sample. Proprio and action tokens share a unified causal mask, letting later action tokens attend to all proprio positions while preserving causality among action tokens.

Embodiment uses a **shared bottleneck** that surfaces in both expert streams. A single `nn.Embedding(NUM_ROBOTS=3, d_emb)` holds the per-robot identity, and two zero-init `Linear(d_emb, d_*, bias=False)` heads project it to `d_pc` and `d_action`. The `embodiment_pc` token is spliced at position 0 of the PC segment (right before `<align>` and the camera blocks); the `embodiment_action` token is spliced at position 0 of the Action segment (right before `<proprio_start>`). Both copies are read directly by the corresponding expert via self-attention, and the Action expert additionally cross-attends to PC K/V (no detach), so action loss reaches `embodiment_pc` too. All three Modules — `base`, `to_pc`, `to_action` — train end-to-end. The textual `Robot: {description}` line in the VLM prompt remains the only path the frozen VLM has to per-robot context.


## Token Sequence

Three segments with separate widths (`d_vlm`, `d_pc`, `d_action`), assembled by `SequenceBuilder`. Each token carries a type label in {vlm, pc, action}.

    ┌───────────────────────────────────────────────────────────────────────┐
    │  VLM segment (rho = vlm, width = d_vlm)                             │
    │                                                                       │
    │  <|im_start|>system ... <|im_end|>                                   │
    │  <|im_start|>user                                                     │
    │    <|vision_start|><|image_pad|>xN<|vision_end|>       cam 1 image   │
    │    <|vision_start|><|image_pad|>xN<|vision_end|>       cam 2 image   │
    │    <|vision_start|><|image_pad|>xN<|vision_end|>       cam 3 image   │
    │    Robot: {description}                                               │
    │    <task>{instruction}</task>                                         │
    │  <|im_end|>                                                           │
    │  <|im_start|>assistant                                                │
    ├───────────────────────────────────────────────────────────────────────┤
    │  PC segment (rho = pc, width = d_pc)                                 │
    │                                                                       │
    │  sink_pc                                       attention sink         │
    │  embodiment_pc                              from SharedEmbodimentEmb │
    │  <align>                                                              │
    │  <pc_start> pc_cam1_tok_1 ... pc_cam1_tok_M <pc_end>          base cam 1 PC  │
    │  <pc_start> pc_cam2_tok_1 ... pc_cam2_tok_M <pc_end>          base cam 2 PC  │
    │  <pc_wrist_start> pc_cam3_tok_1 ... <pc_wrist_end>            wrist cam PC   │
    ├───────────────────────────────────────────────────────────────────────┤
    │  Action segment (rho = action, width = d_action)                     │
    │                                                                       │
    │  sink_action                                   attention sink         │
    │  embodiment_action                          from SharedEmbodimentEmb │
    │  <proprio_start> proprio_tok_1 ... proprio_tok_P <proprio_end>       │
    │  <action_start>  act_tok_1     ...    act_tok_T  <action_end>        │
    └───────────────────────────────────────────────────────────────────────┘

    NOT in token sequence (enter through adaLN only):
      - Timestep embedding

For K cameras, the VLM prompt has K image blocks and the PC segment has K point cloud blocks — one-to-one correspondence. Each camera is bracketed by either `<pc_start>` / `<pc_end>` (non-wrist) or `<pc_wrist_start>` / `<pc_wrist_end>` (wrist), selected by a per-camera `is_wrist` flag passed to `SequenceBuilder`. Within each class the start/end embeddings are shared across cameras (same as Qwen-VL shares `<|vision_start|>` / `<|vision_end|>` across images). Wrist delimiters apply to the PC segment only — VLM-side wrist delimiters are deferred until Phase 2 because the VLM stays frozen.

The Action segment is single-stream: `embodiment_action`, proprio delimiters, proprio tokens, action delimiters, and action tokens all share `TokenType.ACTION`, the same width `d_action`, and a unified causal mask that lets later action tokens attend to all earlier proprio + embodiment positions. P can be 1 (current step only) or larger (current + history), set by what the dataloader feeds into `ProprioEncoder`; the explicit `<proprio_start>` / `<proprio_end>` pair makes the boundary unambiguous when P varies across samples.

The PC segment is bidirectional: `embodiment_pc` participates in PC self-attention with every camera token, and the Action expert reads its processed K/V via cross-attention (no detach), so action loss propagates back through the shared embodiment base.


## Special Tokens

Follows Qwen-VL's delimiter pattern: delimiter tokens wrap each segment, pad tokens are replaced by encoder features.

    Qwen-VL images:     <|vision_start|>    <|image_pad|>xN  <|vision_end|>            per camera
    PC (non-wrist):     <|pc_start|>        <|pc_pad|>xM     <|pc_end|>                per non-wrist camera
    PC (wrist):         <|pc_wrist_start|>  <|pc_pad|>xM     <|pc_wrist_end|>          per wrist camera
    Proprio block:      <|proprio_start|>   proprio_tokens   <|proprio_end|>           wraps P proprio tokens (P variable)
    Our actions:        <|action_start|>    action_tokens    <|action_end|>

    Token                  Expert        Shared across cameras            Purpose
    -----                  ------        -------------------------------  -------
    sink_pc                PC            N/A (one per sample)             Learnable attention sink at position 0 of PC segment; absorbs softmax mass for both PC self-attn and Action's cross-attn to PC K/V
    sink_action            Action        N/A (one per sample)             Learnable attention sink at position 0 of Action segment; absorbs softmax mass for Action self-attn
    embodiment_pc          PC            N/A (one per sample)             Per-robot token from SharedEmbodimentEmbedding.to_pc; position 1 of PC segment
    embodiment_action      Action        N/A (one per sample)             Per-robot token from SharedEmbodimentEmbedding.to_action; position 1 of Action segment
    <|pc_start|>           PC            yes (within non-wrist cameras)   Marks start of a non-wrist camera's point cloud
    <|pc_end|>             PC            yes (within non-wrist cameras)   Marks end of a non-wrist camera's point cloud
    <|pc_wrist_start|>     PC            yes (within wrist cameras)       Marks start of a wrist-camera point cloud
    <|pc_wrist_end|>       PC            yes (within wrist cameras)       Marks end of a wrist-camera point cloud
    <|proprio_start|>      Action        N/A (one per sample)             Marks start of the variable-length proprio block
    <|proprio_end|>        Action        N/A (one per sample)             Marks end of the variable-length proprio block
    <|action_start|>       Action        N/A                              Marks start of action chunk
    <|action_end|>         Action        N/A                              Marks end of action chunk
    <align>                PC            N/A (one per sample)             Alignment flag: Embedding(2, d_pc), 0=unaligned 1=aligned


## Supported Embodiments

    Robot      Description                                        DOF
    -----      -----------                                        ---
    Trossen    "7-DOF Trossen arm with parallel gripper"          7
    Franka     "7-DOF Franka Panda arm with parallel gripper"     7
    SO-101     "6-DOF SO-101 arm with parallel gripper"           6


## VLM Prompt Template

`build_vlm_prompt` returns a single string. There is no embodiment splice on the VLM side — embodiment lives in the PC and Action streams via `SharedEmbodimentEmbedding`. The frozen VLM only sees per-robot context through the textual `Robot: {description}` line.

    <|im_start|>system
    You are a robot manipulation assistant. Given camera observations, robot state, and a
    task instruction, understand the scene by identifying objects, their spatial relationships,
    colors, shapes, sizes, and textures. Pay attention to graspable surfaces, contact points,
    and geometric constraints relevant to manipulation.<|im_end|>
    <|im_start|>user
    <|vision_start|><|image_pad|><|vision_end|>
    <|vision_start|><|image_pad|><|vision_end|>
    <|vision_start|><|image_pad|><|vision_end|>
    Robot: {robot_description}
    <task>{task_instruction}</task><|im_end|>
    <|im_start|>assistant

Notes:
- System prompt is identical for all samples (cacheable across a batch).
- One `<|vision_start|><|image_pad|><|vision_end|>` set per camera (K sets total, matching K point clouds).
- Proprio is not textualized — it enters the Action expert as projected tokens via `ProprioEncoder`.
- Embodiment is not in the VLM segment — `embodiment_pc` and `embodiment_action` are spliced into the PC and Action segments, respectively, via `SharedEmbodimentEmbedding`.
- All tokens from `<|im_start|>system` through `<|im_start|>assistant` carry rho = vlm.


## Input Encoders

All encoders are frozen. Outputs are precomputed and cached by the dataloader. K cameras, each providing one image and one point cloud.

    Input                               Encoder                     Projection
    -----                               -------                     ----------
    Images (xK) + text                  Qwen-VL ViT + tokenizer    native d_vlm (cached)
    Point clouds (xK, one per camera)   Pretrained PC encoder       Linear(encoder_dim, d_pc) per camera
                                                                    + CentroidPosEmbed(centroids -> d_pc) added
    Noisy action chunk                  None (raw, noised at t)     Linear(action_dim, d_action)
    Proprio (P timesteps x 10)          None (raw floats)           ProprioEncoder: Linear(proprio_dim, d_action) -> action stream

`CentroidPosEmbed` mirrors Uni3D's input `pos_embed` (`Linear(3, 128) -> GELU -> Linear(128, d_pc)`) and is applied to the cached FPS centroids; its output is added to the projected Uni3D tokens before they enter `SequenceBuilder`. The final `Linear` is zero-initialized — at step 0 the additive contribution is exactly zero, so the projected PC tokens are bit-for-bit identical to the no-centroid baseline. Same recipe as adaLN-zero: gradient lifts the centroid signal in smoothly. Shared across cameras (wrist and base alike) to mirror Uni3D's pos_embed sharing.

Qwen2.5-VL-3B reference:

    d_vlm = 2048, heads = 16 Q / 2 KV (GQA ratio 8), head_dim = 128
    intermediate_size = 11008, num_layers = 36


## Expert Placement and Modes

Experts are placed via a single `expert_layer_ids` list — both PC and Action experts go at the same layers. If not specified, experts are placed at all layers.

```python
create_backbone(
    qwen_model,
    d_pc=1024, d_action=1024, d_cond=64,
    expert_layer_ids=[0, 4, 8, 12, 16, 20, 24, 28, 32, 35],
    expert_mode="dense",   # or "sparse"
)
```

**Dense mode**: experts at every layer. Cross-attention to VLM at `expert_layer_ids`, self-attention only at other layers. Same trainable param count as all-cross.

**Sparse mode**: experts only at `expert_layer_ids`. Pass-through at other layers. Trainable params proportional to number of expert layers.

Example (8 layers, experts at [0, 4, 7]):

    Mode     Trainable    Sparse/All
    ----     ---------    ----------
    All       342.5M       100%
    Dense     342.5M       100%
    Sparse    128.4M        37.5%


## Architecture Overview

    Cached Encoders (precomputed by dataloader, K cameras)
            |
            v
    Project to expert widths (d_vlm, d_pc per camera, d_action)
            |
            v
    +------------------------------------------------------------------+
    | Backbone (L layers, VLM runs at every layer)                     |
    |                                                                  |
    | Expert layer (is_cross=True):                                    |
    |   VLM:  Qwen decoder layer (frozen) -> extract K/V from cache   |
    |   PC:   self-attn + cross-attn to VLM K/V -> produces pc_k      |
    |   Action: self-attn(causal) + cross-attn to [VLM K/V, pc_k]     |
    |           + adaLN (timestep + embodiment + proprio history)      |
    |                                                                  |
    | Non-expert layer (dense, is_cross=False):                        |
    |   VLM:  Qwen decoder layer (frozen)                             |
    |   PC:   self-attn only (no VLM K/V)                             |
    |   Action: self-attn only (causal, no VLM/PC K/V)                |
    |                                                                  |
    | Non-expert layer (sparse):                                       |
    |   VLM:  Qwen decoder layer (frozen)                             |
    |   PC:   pass-through                                             |
    |   Action: pass-through                                           |
    |                                                                  |
    | Gradient isolation: VLM K/V .detach() (frozen). PC K/V flow grads.|
    +------------------------------------------------------------------+
            |
        +---+-----------+
        |               |
    PC output      Action output
    (MSE loss)     (flow matching loss)


## Cross-Attention Flow at Expert Layers

Both PC and Action experts sit at the same `expert_layer_ids`. At each expert layer, PC runs first, then Action uses the same layer's VLM K/V and PC K/V:

    expert_layer_ids = [0, 4, 8]

    Layer 0 (expert):  PC -> [vlm_k_0]              Action -> [vlm_k_0, pc_k_0]
    Layer 1-3:         (dense: self-attn only / sparse: pass-through)
    Layer 4 (expert):  PC -> [vlm_k_4]              Action -> [vlm_k_4, pc_k_4]
    Layer 5-7:         (dense: self-attn only / sparse: pass-through)
    Layer 8 (expert):  PC -> [vlm_k_8]              Action -> [vlm_k_8, pc_k_8]

At each expert layer: VLM runs -> PC cross-attends to VLM K/V (detached) -> Action cross-attends to VLM K/V (detached) + PC K/V (gradients flow). All three use the same layer's features.


## Per-Layer Expert Details

    Feature      VLM (Qwen layer)              PC (ExpertBlock)                Action (ExpertBlock)
    -------      ----------------              ----------------                --------------------
    Norm         Qwen2RMSNorm                  Qwen2RMSNorm                   Qwen2RMSNorm
    RoPE         3D MRoPE (native)             MRoPE chunk-shared (cont. from VLM)  MRoPE 1D (cont. from PC)
    GQA          native (16Q/2KV for 3B)       proportional to width          proportional to width
    MLP          GatedMLP (native)             GatedMLP (proportional)        GatedMLP (proportional)
    Self-attn    Causal + packed-sequence      Bidirectional                  Causal
    Cross-attn   N/A                           VLM K/V detached (expert layers)  VLM K/V detached + PC K/V w/ grad (expert layers, same layer)
    adaLN        None                          None                           timestep + embodiment + proprio history
    Status       Frozen                        Trainable (init from VLM)      Trainable (init from VLM)


## Position-ID Construction (MRoPE)

A single global counter `p` walks the full token sequence VLM → PC → Action, producing 3-axis MRoPE positions `(t, h, w)` per token. Each segment uses a different rule for advancing the counter:

    Token type                                                      (t, h, w)         Counter advance
    ----------                                                      ---------         ---------------
    VLM text                                                        (p, p, p)         p += 1
    VLM image patch (row r, col c, of HxW grid)                     (p, p+r, p+c)     p += max(H, W) once after the whole image
    All delimiters (<pc_start/end>, <pc_wrist_start/end>, <align>,
        <proprio_start/end>, <action_start/end>, <im_start>,
        <vision_start>, ...)                                        (p, p, p)         p += 1
    Per-segment scalars (sink_pc, embodiment_pc, sink_action,
        embodiment_action)                                          (p, p, p)         p += 1
    PC chunk (M tokens from one Uni3D run, one camera)              all M share (p, p, p)   p += 1 once for the whole chunk
    Proprio token                                                   (p, p, p)         p += 1
    Action token                                                    (p, p, p)         p += 1

**PC chunk rule.** All M tokens in one PC chunk share the same `(p, p, p)`; the counter advances exactly once for the chunk as a whole. Inside a chunk, every pairwise `Δt = Δh = Δw = 0`, so MRoPE contributes no positional signal between chunk tokens — intra-chunk geometry is supplied by two channels: (1) Uni3D's pretrained `pos_embed` + 24 ViT blocks already baked it into the cached patch tokens, and (2) `CentroidPosEmbed` re-injects an explicit, fresh signal directly at PC-expert entry by adding `MLP(centroid_xyz)` to each projected token. Channel (2) is zero-init at step 0 so the prior is preserved; gradient ramps it up if intra-chunk reasoning needs more than what survived the encoder. Across chunks, `Δp ≥ 3` (separated by `<pc_*_end>` and `<pc_*_start>`), so different cameras' PC chunks remain positionally distinguishable.

**Equivalence to 1-D RoPE.** Since `t = h = w` for every non-image-patch token, MRoPE collapses to ordinary 1-D RoPE on the PC and Action segments — the deck's "Scenario A" identity. The only segment where the three axes diverge is the VLM image-patch block, which keeps Qwen2.5-VL's native `(p, p+r, p+c)` rule.

**Counter continuation.** PC positions start at `max_vlm_pos + 1`; action positions start at the PC counter's terminal value. The 1-D position list for the PC segment is built by `build_pc_chunk_position_ids(pc_chunk_sizes, start)` in `core.py`, which returns both the per-token positions (length `n_pc`) and the `next_p` to use for the action segment.

**Layout assumed by the builder** (matches `SequenceBuilder.forward`):

    sink_pc         -> p,    p+=1
    embodiment_pc   -> p,    p+=1
    <align>         -> p,    p+=1
    per camera (M = chunk_size):
        <pc_*_start> -> p,    p+=1
        M tokens     -> all share p, p+=1 once for the whole chunk
        <pc_*_end>   -> p,    p+=1

**API.** `Backbone.forward(..., pc_chunk_sizes=[M_1, ..., M_K])` activates the chunk rule. Passing `pc_chunk_sizes=None` (the default) reproduces the legacy per-token 1-D continuation for backward compatibility.


## adaLN Conditioning — DiT adaLN-zero (6 mod params)

Only on the Action expert, and only timestep enters this path. Embodiment and proprio history were dropped: embodiment lives in the PC + Action streams via `SharedEmbodimentEmbedding`, proprio enters as Action-stream tokens. Per-layer MLP produces **6 mod params per ExpertBlock** — DiT adaLN-zero (Peebles & Xie 2023):

    # Conditioning vector (assembled once per sample)
    cond = TimestepEmbedding(t)        # (B, d_t)

    # Per-layer modulation (inside each MultiExpertLayer)
    s1, sh1, g1, s2, sh2, g2 = action_adaln(cond).chunk(6)   # each (B, d_action)

    # Attention sub-block
    normed = (1 + s1) * RMSNorm(x) + sh1
    attn_out = o_proj(attention(normed, …))
    x = x + g1 * attn_out               # ← gated residual

    # MLP sub-block
    normed2 = (1 + s2) * RMSNorm(x) + sh2
    mlp_out = mlp(normed2)
    x = x + g2 * mlp_out                # ← gated residual

The two gates `g1, g2` scale the modulated path's contribution to the residual stream. The **final Linear of `action_adaln` is zero-initialized**, so all six mod params start at zero. Combined with `(1 + scale)` in `modulate`, this means at step 0 every block computes `x = x + 0*attn + 0*mlp = x` — the entire backbone is the residual stream's identity. Gradient teaches the gates to lift off zero, ramping the modulated path in smoothly from nothing.

Why this matters more for us than for the original DiT: our `ExpertBlock`s are initialized from VLM weights (`init_expert_from_vlm`). Without adaLN-zero, random `s1, sh1` would immediately distort whatever the VLM-init weights were doing. With adaLN-zero, the VLM-init prior is preserved at step 0 and gradient learns when to deviate.


## Attention Masks

    Query \ Key   VLM                    PC                       Action
    -----------   ---                    --                       ------
    VLM           causal + packed-seq    blocked                  blocked
    PC            full (detached K/V)    bidirectional            blocked
    Action        full (detached K/V)    full (PC K/V w/ grad)    causal

Action uses a hybrid mask at expert layers: full cross to VLM+PC columns, causal on self columns. At non-expert layers (dense), Action uses causal self-only mask. Head count mismatches handled by `_match_heads` (GQA-style grouping).


## Gradient Isolation

    Flow matching loss -> Action expert + adaLN + PC expert (via PC K/V)  --X  VLM (detached)
    MSE loss           -> PC expert                                        --X  Action, VLM (detached)

Detach points: VLM K/V from DynamicCache, VLM params frozen.

PC K/V are NOT detached when Action consumes them — the action flow-matching loss propagates back into the PC expert through the cross-attention operands. The PC expert therefore receives gradient from both losses (its own MSE and the action loss). The MSE loss does not flow into the Action expert because PC's forward pass never consumes Action K/V.


## Trainable vs Frozen

**Frozen**: Qwen decoder layers (all L), final RMSNorm, rotary embeddings, ViT and PC encoder (in dataloader).

**Trainable**: PC/Action ExpertBlocks (at expert layers or all layers), Action adaLN per layer (timestep only), input projections (incl. `ProprioEncoder`), special token embeddings, alignment embedding, `sink_pc` and `sink_action` attention sinks, `SharedEmbodimentEmbedding` (`base: Embedding(3, d_emb)` + `to_pc: Linear(d_emb, d_pc)` + `to_action: Linear(d_emb, d_action)`).

**Initialization**: experts copy first N heads of VLM Q/K/V/O + truncated MLP/norm. The new delimiter tokens are bootstrapped from VLM embeddings via `init_special_tokens_from_vlm(sequence_builder, qwen_model, eps=1e-3)`: each `<pc_*>`, `<proprio_*>`, `<action_*>` Parameter is set to `mean(embed(<vision_start>), embed(<vision_end>))[:expert_width] + N(0, ε)`. `SharedEmbodimentEmbedding` self-initializes inside its `__init__` (Recipe 2: default Embedding init for the bottleneck `base`, **zero-init** for both `to_pc` and `to_action` projections). **adaLN-zero** for the per-layer adaLN MLPs (`AdaLNConditioner` and `MultiExpertLayer.action_adaln` both zero-init their final Linear) — all 6 mod params start at zero, so each ExpertBlock is the residual identity at step 0; gates lift off zero through gradient. The `<align>` token and both attention sinks (`sink_pc`, `sink_action`) stay random-initialized — sinks are content-free by design, so no VLM prior applies.

Example (Qwen2.5-VL-3B, 2 layers, d_pc=d_action=1024, all-cross):

    VLM (frozen):    154.15M
    PC (trainable):   38.54M
    Action+adaLN:     47.07M
    Total trainable:  85.61M (35.7%)


## Training

Dataloader provides cached VLM embeddings (text + vision merged), PC features, action ground truth.

1. Project PC/Action to expert widths; noise action at random timestep t
2. Prepare VLM inputs (position IDs, attention mask, RoPE) and expert positions (PC chunk-shared MRoPE via `pc_chunk_sizes`; Action 1-D MRoPE continuation)
3. Forward through backbone (VLM at every layer, experts at configured layers)
4. Losses: flow matching on Action output, MSE on PC output. No loss on VLM.


## Inference

1. Full forward through all layers (VLM + PC computed once, K/V cached)
2. T denoising steps: only action tokens re-enter, attending to cached VLM/PC K/V
3. adaLN injects current timestep at each step


## Design Rationale

**Actual Qwen layers for VLM**: zero distribution shift, bit-for-bit verified identical output.

**RMSNorm + MRoPE on experts**: matches VLM primitives, initialized weights are compatible from step 0.

**PC bidirectional, Action causal**: point clouds have no ordering; action chunks are temporal sequences.

**Dense vs sparse modes**: dense gives full-depth processing with selective cross-attention; sparse minimizes trainable params for faster iteration.

**Cached encoder outputs**: ViT and PC encoder are frozen, run once per sample in preprocessing.

**Proprio as Action-stream tokens**: a single `Linear(proprio_dim, d_action)` projects each proprio reading and the result is prepended to the action chunk under a unified causal mask. Keeps proprio scaling-invariant from the VLM prompt format and lets later action tokens attend back to it; the same Linear handles current-only or current+history without architectural change.

**Embodiment as a shared bottleneck across PC + Action**: `Robot: {description}` text in the VLM prompt gives the frozen VLM semantic context (geometry, DOF, gripper). A separate learned `SharedEmbodimentEmbedding` — one `nn.Embedding(3, d_emb)` base plus zero-init `Linear` heads to `d_pc` and `d_action` — produces two per-stream tokens spliced at position 0 of the PC and Action segments respectively. The shared base is the source of truth: PC MSE loss reaches it through `to_pc`, Action flow-matching loss reaches it through `to_action`, and Action loss also reaches `to_pc` indirectly via the no-detach PC→Action cross-attention. The frozen VLM never sees a learned embodiment token — by design, since gradient cannot reach into a frozen stack with detached cross-attention K/V (the path the previous VLM-side splice implicitly relied on, but never actually trained).

**Alignment flag**: binary embedding (not continuous projection) because the signal is discrete — point clouds are either registered or not.


## Code Structure

    unified_vla/
    ├── core.py         TokenType, TokenSequence, SequenceBuilder, AlignmentToken,
    │                   InputProjections, SharedEmbodimentEmbedding, ProprioEncoder,
    │                   AdaLNConditioner (timestep-only), CentroidPosEmbed, modulate
    ├── attention.py    ExpertQKV (GQA), build_attention_mask, _match_heads,
    │                   detached_cross_modal_attention
    ├── layers.py       GatedMLP, ExpertBlock (RMSNorm + MRoPE + mask + adaLN), init_expert_from_vlm
    ├── backbone.py     MultiExpertLayer, Backbone (dense/sparse, expert_layer_ids)
    ├── surgery.py      create_backbone(qwen_model, ...), init_special_tokens_from_vlm
    ├── losses.py       TimestepEmbedding, noise_action, flow_matching_loss, pc_mse_loss
    ├── prompt.py       build_vlm_prompt (single string; no embodiment splice)
    └── utils.py        count_params, backbone_param_summary, print_param_summary


## Open Questions

- Expert width sweep: d_pc and d_action from 512 to 2048, asymmetric configs.
- Expert layer placement: which layers matter most for cross-attention? Every 2nd, every 4th, last N?
- `flex_attention`: replace Action hybrid bool mask with `torch.nn.attention.flex_attention` for efficiency.
- Phase 2 VLM unfreezing: learning rate ratio, what to do with VLM text output.
- Fused attention: single call with segment-level masking vs three separate calls.


## Implementation Status

243 tests (unit + integration), all passing.

--- Completed ---

- [x] Token type registry, sequence builder, alignment token (`core.py`)
- [x] Input projections, ProprioEncoder, timestep-only AdaLNConditioner, SharedEmbodimentEmbedding (`core.py`)
- [x] `modulate` function (`core.py`)
- [x] ExpertQKV with GQA, attention mask builder, gradient-detached cross-modal attention (`attention.py`)
- [x] ExpertBlock with RMSNorm + MRoPE + GatedMLP + adaLN + mask support (`layers.py`)
- [x] Expert weight initialization from VLM, GQA-aware slicing (`layers.py`)
- [x] MultiExpertLayer: actual Qwen layer + ExpertBlocks, dense/sparse modes (`backbone.py`)
- [x] Backbone with expert_layer_ids, dense/sparse modes (`backbone.py`)
- [x] create_backbone surgery from Qwen2.5-VL (`surgery.py`)
- [x] VLM prompt builder (`prompt.py`) — no proprio text
- [x] Flow matching loss, PC MSE loss, timestep embedding (`losses.py`)
- [x] Param counting utilities (`utils.py`)
- [x] VLM forward exactness: bit-for-bit identical to Qwen2.5-VL-3B at every layer
- [x] Dense/sparse mode tests with param count verification
- [x] **Proprio as Action-stream tokens** — ProprioEncoder prepends projected proprio to the action chunk under a unified causal mask; adaLN shrunk to timestep only; embodiment/proprio-history adaLN paths removed
- [x] **Embodiment as a shared bottleneck across PC + Action** — `SharedEmbodimentEmbedding(d_pc, d_action, d_emb=32)` with one `nn.Embedding(3, d_emb)` base + zero-init `Linear` heads to each stream. `embodiment_pc` lives at position 1 of the PC segment (after `sink_pc`); `embodiment_action` at position 1 of the Action segment. Both losses train the shared base.
- [x] **Per-expert attention sinks** — `sink_pc` and `sink_action` (separate `nn.Parameter` Tensors at width `d_pc` / `d_action`) at position 0 of each segment. Two separate sinks, not shared: PC's bidirectional attention regime and Action's causal regime have independent "good sink" directions. `sink_pc` does double duty as the drain for both PC self-attention and Action's cross-attention to PC K/V. Random-init; not bootstrapped by `init_special_tokens_from_vlm` (sinks are content-free by design, no VLM prior applies).
- [x] **adaLN-zero with explicit gates (DiT)** — `AdaLNConditioner` and `MultiExpertLayer.action_adaln` now produce **6** mod params per ExpertBlock: `(scale1, shift1, gate1, scale2, shift2, gate2)`. `modulate` switched to the `(1 + scale) * LN(x) + shift` form. Final Linear of both adaLN MLPs is zero-initialized → at step 0 every ExpertBlock is the residual identity, preserving the VLM-init prior. Gates lift off zero through gradient; this is the same recipe the original DiT and Xiaomi MiBoT use.
- [x] **Wrist vs non-wrist PC delimiters** — `<pc_wrist_start>` / `<pc_wrist_end>` learnable embeddings added alongside the base pair; `SequenceBuilder.forward` accepts a per-camera `is_wrist_per_camera: list[bool]`. PC segment only — VLM-side wrist delimiters remain deferred to Phase 2.
- [x] **Init new delimiter tokens from VLM embeddings + ε** — `init_special_tokens_from_vlm(sequence_builder, qwen_model, eps=1e-3)` overwrites `<pc_*>`, `<proprio_*>`, `<action_*>` Parameters with `mean(<vision_start>, <vision_end>)[:expert_width] + N(0, ε)`. `<align>` is left random; `SharedEmbodimentEmbedding` self-inits in its `__init__` (Recipe-2: zero-init projections).
- [x] **Remove stop-gradient PC → Action** — `.detach()` dropped from PC K/V in `backbone.py:MultiExpertLayer.forward` and in `attention.py:detached_cross_modal_attention`. Action flow-matching loss now flows back into the PC expert via cross-attention. VLM K/V remain detached (still frozen).
- [x] **Proprio delimiters** — `<proprio_start>` / `<proprio_end>` learnable embeddings (`d_action`) added in `SequenceBuilder`; `init_special_tokens_from_vlm` bootstraps them from the same `mean(<vision_start>, <vision_end>) + N(0, ε)` base used for the action delimiters. Action segment is now `[<proprio_start>, P proprio, <proprio_end>, <action_start>, T action, <action_end>]` so the boundary stays unambiguous when P varies sample to sample.
- [x] **PC-chunk shared MRoPE positions** — `build_pc_chunk_position_ids(pc_chunk_sizes, start)` in `core.py`; `Backbone.forward(..., pc_chunk_sizes=[M_1, ..., M_K])` plumbs the chunk descriptor through. All M tokens in one PC chunk share `(p, p, p)`; sink/embodiment/align/`<pc_*_start/end>` follow the text rule. Replaces the original "3D RoPE for PC expert" backlog item — intra-chunk 3D geometry is already encoded by Uni3D's frozen `pos_embed` + ViT blocks, so MRoPE only signals "which chunk is this" via chunk-level `Δp ≥ 3`. `pc_chunk_sizes=None` (default) preserves the legacy per-token 1-D continuation for backward compatibility.

- [x] **CentroidPosEmbed at PC-expert entry** — `CentroidPosEmbed(d_pc, hidden=128)` in `core.py` mirrors Uni3D's input `pos_embed` (`Linear(3, 128) -> GELU -> Linear(128, d_pc)`) but is applied to the cached FPS centroids on the unified-VLA side, then **added** to the projected Uni3D tokens per camera before `SequenceBuilder`. Final `Linear` is zero-initialized — step-0 contribution is exactly zero, so projected PC tokens are bit-for-bit identical to the no-centroid baseline at init (adaLN-zero recipe). Provides intra-chunk geometric signal that MRoPE can no longer carry under the chunk rule, and a fresh re-injection that doesn't depend on Uni3D's `pos_embed` surviving 24 ViT blocks of mixing. One module shared across all cameras (wrist and base alike).

--- Remaining ---

## Pipeline completion

- [ ] Combined training step (forward + both losses + optimizer)
- [ ] Inference with K/V caching (VLM/PC cached, action iterates)
- [ ] End-to-end smoke test

## Architecture backlog (Phase 1)

- [ ] **Wrist-camera Uni3D encoder** — second pretrained Uni3D model specialized for wrist-cam point clouds (different distance distribution, close-up gripper geometry). Tokens come from the dataloader; the backbone treats wrist-cam chunks identically to base-cam chunks (same `pc_chunk_sizes` rule, separate `<pc_wrist_*>` delimiters already present).

- [ ] **Paired camera shuffle (augmentation)** — DataLoader-level: sample a random camera-order permutation per sample and apply the **same** permutation to the image and PC streams so `(img_i, pc_i)` pairs stay aligned. Invalidates ViT-embedding caches that assume a fixed order — cache keys must include the permutation. The PC permutation must also reorder `is_wrist_per_camera` so each camera keeps its wrist tag.

- [ ] **Paired camera dropping (augmentation)** — with probability `(1 − p)`, drop camera `i` from *both* the image and PC streams simultaneously. Always keep ≥ 1 camera. Precompute `E_zero = ViT(zero_image)` once at startup and cache it; at runtime, replace a dropped camera's cached ViT embedding with `E_zero` before the LLM forward (safer than feeding literal zeros — `E_zero` stays on the frozen VLM's input distribution). Mirror on the PC side with an analogous cached "absent-PC" embedding.

- [ ] **Finetune PC encoder flag** — add `finetune_pc_encoder: bool = False`. When enabled, disable cached PC encoder outputs and train Uni3D end-to-end. **Cost:** step time and memory both grow; caching win is lost. Partial unfreezing (last `k` Uni3D blocks) is a sub-option to explore.

*Suggested execution order (small / isolated → architectural → data-aug → expensive):*
*wrist Uni3D wiring → camera shuffle → camera dropping → PC finetune flag.*
