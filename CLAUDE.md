# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research analysis repository focused on **conditional generation mechanisms in Diffusion Transformer (DiT) models**. It contains no runnable code — the primary artifact is a detailed technical analysis document comparing two Alibaba research papers.

## Contents

- **`2503.20314v2.pdf`** — "Wan: Open and Advanced Large-Scale Video Generative Models" (video/image generation using single-stream DiT with cross-attention text conditioning and shared adaLN)
- **`2508.02324v1.pdf`** — "Qwen-Image Technical Report" (image generation using double-stream MMDiT with multimodal LLM conditioning via Qwen2.5-VL)
- **`conditional_generation_analysis.md`** — The core deliverable: a ~500-line structured analysis comparing conditional generation mechanisms across both papers, with section/figure/table citations

## Key Technical Concepts

The analysis covers these conditioning mechanisms in detail:

| Mechanism | Wan | Qwen-Image |
|---|---|---|
| Text conditioning | Cross-attention (umT5, bidirectional) | Joint self-attention (Qwen2.5-VL hidden states) |
| Timestep conditioning | Shared adaLN (one MLP, per-block biases) | Per-block adaLN-zero (independent MLPs) |
| Architecture | Single-stream DiT | Double-stream MMDiT (text + image paths with gating) |
| Position encoding | 3D RoPE (temporal + spatial) | MSRoPE (2D image + diagonal text) |
| Image conditioning (I2V/TI2I) | Dual-path: channel concat + CLIP cross-attention | Channel concat + Qwen2.5-VL features |
| Training | Flow matching with rectified flows | Flow matching with rectified flows + DPO/GRPO post-training |

## Working With This Repository

- When asked to extend or update the analysis, maintain the existing citation style: `**Reference:** Section X.Y, page Z; Figure N, page M`.
- The analysis document is organized as Paper 1 (Wan) then Paper 2 (Qwen-Image), with numbered sections mirroring the paper structure.
- Web fetch permissions are pre-configured for arxiv.org, semanticscholar.org, github.com, and several other research domains (see `.claude/settings.local.json`).
- PDFs can be read directly with the Read tool for verification or additional analysis.
