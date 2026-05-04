# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repo is a **Unified VLA implementation**: a Qwen2.5-VL transformer backbone with per-layer modality experts for vision-language understanding, point-cloud grounding, and action denoising. The frozen VLM expert is the actual `Qwen2_5_VLDecoderLayer` (zero distribution shift); PC and Action are downsized `ExpertBlock` modules initialized from VLM weights.

The folder name and the two reference PDFs (`2503.20314v2.pdf` Wan, `2508.02324v1.pdf` Qwen-Image) plus `conditional_generation_analysis.md` are leftover artifacts from the design-research phase that fed this implementation. They are reference material, not the project deliverable.

## Source of truth

**`unified_vla_architecture.md`** is the authoritative design doc — token sequence layout, conditioning paths, expert placement modes, MRoPE rules, attention masks, gradient isolation, trainable/frozen split, init recipe, and implementation status. Read it before making non-trivial changes; update it when behavior changes.

## Code layout

```
unified_vla/
├── core.py        TokenType, SequenceBuilder, SharedEmbodimentEmbedding,
│                  ProprioEncoder, AdaLNConditioner, build_pc_chunk_position_ids
├── attention.py   ExpertQKV (GQA), build_attention_mask, _match_heads
├── layers.py      ExpertBlock, GatedMLP, init_expert_from_vlm
├── backbone.py    MultiExpertLayer, Backbone (dense/sparse, expert_layer_ids)
├── surgery.py     create_backbone, init_special_tokens_from_vlm
├── losses.py      TimestepEmbedding, noise_action, flow_matching_loss, pc_mse_loss
├── prompt.py      build_vlm_prompt
└── utils.py       count_params, backbone_param_summary

qwen2_5_vl/        Vendored HF Qwen2.5-VL (configuration, modeling, processing)
xiaomi_mibot/      Reference implementation (modeling_mibot.py, processing_mibot.py)
Xiaomi-Robotics-0/ Vendored Xiaomi Robotics reference clone

tests/             231 tests, all passing — unit + integration + VLM-exactness checks
slides/            Slidev deck and exported PDF for the design talk
```

## Working with this repo

- Run tests with `pytest tests/`. The `test_vlm_*_exact.py` suite verifies bit-for-bit identity with vanilla Qwen2.5-VL-3B at every layer — do not break those.
- The frozen VLM contract is load-bearing: VLM K/V are `.detach()`'d when experts cross-attend, but PC K/V are **not** detached so action loss reaches the PC expert.
- New delimiter tokens (`<pc_*>`, `<pc_wrist_*>`, `<proprio_*>`, `<action_*>`) are bootstrapped from `mean(<vision_start>, <vision_end>) + N(0, ε)` via `init_special_tokens_from_vlm`. Keep them as `nn.Parameter`s at the per-stream width, not embeddings in Qwen's vocab.
- adaLN-zero recipe: 6 mod params per block, final Linear zero-init so each ExpertBlock starts as residual identity. Don't change to non-zero init without a reason.
- Web fetch permissions are pre-configured for arxiv.org, semanticscholar.org, github.com (see `.claude/settings.local.json`).

## Reference materials (design phase, not active deliverables)

- `2503.20314v2.pdf` — "Wan" (single-stream DiT, cross-attention text, shared adaLN)
- `2508.02324v1.pdf` — "Qwen-Image" (double-stream MMDiT, joint self-attention, per-block adaLN-zero)
- `conditional_generation_analysis.md` — comparison of conditioning mechanisms across the two papers
- `xiaomi_robotics_0.pdf` — Xiaomi Robotics reference paper

These informed design choices like adaLN-zero (from Qwen-Image / DiT) and the dual-path conditioning structure. Cite them when justifying architecture decisions, but treat the unified VLA implementation as the active project.
