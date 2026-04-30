# Conditional Generation in DiT Blocks: Detailed Analysis

---

# Paper 1: Wan (arXiv 2503.20314)

**Full Title:** Wan: Open and Advanced Large-Scale Video Generative Models  
**Authors:** Wan Team, Alibaba Group  
**Domain:** Video and Image Generation

---

## 1.1 What is Wan?

Wan is an open-source video foundation model built on the Diffusion Transformer (DiT) paradigm combined with Flow Matching. It supports text-to-video (T2V), text-to-image (T2I), image-to-video (I2V), video editing, video personalization, camera motion control, and real-time video generation. Two model sizes are offered: 1.3B and 14B parameters.

**Reference:** Abstract, page 1; Section 1 "Introduction", page 3.

---

## 1.2 Overall Architecture

Wan consists of three primary components (Figure 9, page 13):

1. **Wan-VAE** (encoder/decoder): Compresses pixel-space video into a compact latent space.
2. **umT5** (text encoder): Encodes text prompts into a sequence of embeddings.
3. **Diffusion Transformer (DiT)**: The denoising backbone that takes noisy latents + text condition + timestep and predicts the velocity field.

The pipeline works as follows:
- A video `V` is encoded by Wan-VAE into latent `x` of shape `[1+T/4, H/8, W/8, 16]`.
- A text prompt is encoded by umT5 into 512 token embeddings `c_txt`.
- The DiT processes the noisy latent `x_t`, conditioned on `c_txt` and timestep `t`, to predict velocity.
- The Wan-Decoder reconstructs the output video from the denoised latent.

**Reference:** Section 4.2, page 13; Figure 9, page 13.

---

## 1.3 The DiT Block in Detail

The DiT is composed of: a **patchify module**, N **transformer blocks**, and an **unpatchify module**.

### Patchify Module
- A 3D convolution with kernel size `(1, 2, 2)` converts the latent `x` into a flat sequence of tokens.
- Token sequence shape: `(B, L, D)` where `L = (1 + T/4) x H/16 x W/16`, `B` is batch size, `D` is the latent dimension.

**Reference:** Section 4.2.1 "Diffusion transformer", page 14.

### Transformer Block Structure (Figure 10, page 14)

Each block has the following components in sequence:

```
V-Tokens (video latent tokens)          Timestep t
         |                                    |
         v                                    v
    [LayerNorm] <--- adaLN modulation    MLP (Linear + SiLU)
         |              (scale, shift)    --> 6 params per block
         v
   Self-Attention (full spatio-temporal)
         |
         v
    [LayerNorm] <--- adaLN modulation
         |
         v                    T-Tokens (from umT5, 512 tokens)
   Cross-Attention  <----------/
   (Q = video tokens,
    K,V = text tokens)
         |
         v
    [LayerNorm] <--- adaLN modulation
         |
         v
       FFN
         |
         v
   Output tokens
```

**Reference:** Section 4.2.1, page 14; Figure 10, page 14.

---

## 1.4 Text Conditioning: Cross-Attention with umT5

### How text enters the model

1. The user's text prompt is encoded by **umT5** (a frozen, multilingual, bidirectional encoder-decoder model, 5.3B params) into a fixed-length sequence of **512 token embeddings**.
2. These text embeddings serve as keys (K) and values (V) in a **cross-attention** layer inside every DiT block.
3. The video latent tokens serve as queries (Q).
4. This means at every layer, each video token can attend to the full text sequence, allowing precise text-video alignment.

### Why umT5 was chosen (over LLMs like Qwen2.5 or GLM-4)

- umT5 uses **bidirectional attention**, unlike decoder-only LLMs which use causal attention. Bidirectional attention is better suited for encoding conditions for diffusion models because the full context is available at every position.
- Strong multilingual support (Chinese + English).
- Faster convergence at the same parameter scale compared to LLM-based encoders.
- An ablation (Table 6, page 25) shows umT5 achieves FID 43.01 vs. Qwen-VL-7B at 42.91 (second-last layer) -- comparable quality but umT5 is much smaller.

**Reference:** Section 4.2.1 "Text encoder", page 14; Section 4.7.2 "Ablation on text encoder", page 25; Table 6, page 25.

---

## 1.5 Timestep Conditioning: Shared adaLN

### How timestep enters the model

1. The diffusion timestep `t` is fed into an **MLP** consisting of a Linear layer + SiLU activation.
2. This MLP produces **6 modulation parameters** per DiT block: a (scale, shift) pair for each of the 3 LayerNorm layers in the block (before self-attention, before cross-attention, before FFN).
3. These parameters modulate the LayerNorm outputs via **adaptive Layer Normalization (adaLN)**:
   - `output = scale * LayerNorm(input) + shift`

### Key design: Shared MLP across all blocks

- The timestep MLP is **shared across all N transformer blocks** -- i.e., there is only one MLP, not one per block.
- Each block learns a **distinct set of biases** that are added on top of the shared MLP output.
- This saves ~25% of the total parameters compared to the non-shared design (where each block has its own independent MLP).

### Ablation results

Four configurations were tested (Section 4.7.2, page 24; Figure 16, page 25):

| Configuration | Parameters | Result |
|---|---|---|
| Full-shared-adaLN-1.3B (chosen design) | 1.3B | Slightly higher loss than 1.5B, but best param efficiency |
| Half-shared-adaLN-1.5B | 1.5B | Worse than full-shared at same param count |
| Full-shared-adaLN-1.5B (extended depth) | 1.5B | **Lowest training loss** |
| Non-shared-AdaLN-1.7B | 1.7B | Does not outperform full-shared-1.5B despite 0.2B more params |

**Conclusion:** Model depth matters more than adaLN parameter volume. Sharing adaLN is the better trade-off.

**Reference:** Section 4.2.1, page 14; Section 4.7.2 "Ablation on adaptive normalization layers", page 24-25.

---

## 1.6 Image Conditioning (I2V Task)

For the image-to-video task, Wan uses a **dual-path** conditioning strategy that integrates image information at two levels (Figure 18, page 27; Section 5.1.1, page 26):

### Path 1: Channel-wise Latent Concatenation (pixel-level)

1. The condition image `I` (shape `C x 1 x H x W`) is concatenated with zero-filled frames along the temporal axis to create guidance frames `I_c` (shape `C x T x H x W`).
2. `I_c` is compressed by Wan-VAE into a condition latent `z_c` (shape `c x 1 x h x w`, where `c=16` channels).
3. A **binary mask** `M` is created: `1` for the preserved (condition) frame, `0` for frames to be generated.
4. The noise latent `z_t`, condition latent `z_c`, and mask `m` are **concatenated along the channel axis**.
5. Since this gives more input channels than the T2V model (2c + s vs. c), a new **projection layer** (zero-initialized) maps it back to the expected dimension.

### Path 2: CLIP Features via Decoupled Cross-Attention (semantic-level)

1. The **CLIP image encoder** extracts global feature representations from the condition image.
2. These features are projected by a **3-layer MLP** to produce a global context vector.
3. This context is injected into the DiT via **decoupled cross-attention** -- a separate cross-attention mechanism from the text cross-attention. This way, image semantic features and text features do not interfere with each other.

### Both paths work simultaneously

- The channel-concatenated latent gives the model **fine-grained spatial information** about the first frame.
- The CLIP cross-attention gives the model **high-level semantic context** about the image content.
- The text prompt (via umT5 cross-attention) provides additional guidance about the desired motion/action.

**Reference:** Section 5.1.1 "Model Design", page 26; Figure 18, page 27.

---

## 1.7 Other Image Conditioning Variants in Wan

### Video Editing (VACE): Context Tokenization + Channel Concatenation

- All input conditions (frames, masks, text) are unified into a **Video Condition Unit (VCU)**: `V = [T; F; M]`.
- A **concept decoupling** strategy separates frames into "reactive" (pixels to modify: `F x M`) and "inactive" (pixels to keep: `F x (1-M)`).
- Both are VAE-encoded and **concatenated channel-wise** with the noisy latent.

**Reference:** Section 5.2.1, page 31; Figure 21, page 31.

### Video Personalization: Latent Prepending + Self-Attention

- Face images are encoded by Wan-VAE (no external feature extractor).
- K face frames are **prepended** to the video in latent space along the temporal dimension.
- Face frames get all-ones masks; video frames get all-zeros masks.
- The model uses its **existing self-attention** to attend from video tokens to face tokens -- no new cross-attention is added.

**Reference:** Section 5.4.1, page 35-37; Figure 26, page 37.

### Camera Motion: Adaptive Normalization Adapter

- Camera parameters (extrinsic `[R,t]` + intrinsic `K_f`) are converted to **Plucker coordinates** per pixel.
- A **Camera Pose Encoder** (ResBlocks) extracts multi-level features.
- A **Camera Adapter** produces per-layer scale (`gamma_i`) and shift (`beta_i`) via zero-initialized convolutions.
- Injected into each DiT block as: `f_i = (gamma_i + 1) * f_{i-1} + beta_i`.

**Reference:** Section 5.5, page 38; Figure 28, page 40.

---

## 1.8 Training Objective

Wan uses **Flow Matching** with **Rectified Flows** (Section 4.2.2, page 14):

```
x_t = t * x_1 + (1 - t) * x_0           -- Eq. 1, page 14
v_t = dx_t/dt = x_1 - x_0               -- Eq. 2, page 14
L = E[||u(x_t, c_txt, t; theta) - v_t||^2]  -- Eq. 3, page 15
```

Where `x_1` is the data latent, `x_0 ~ N(0,I)` is noise, `t ~ logit-normal(0,1)`, `c_txt` is the umT5 embedding, and the model predicts velocity.

---

## 1.9 Summary of All Conditioning Mechanisms in Wan

| Condition | Mechanism | Where it enters | Reference |
|---|---|---|---|
| Text prompt | Cross-attention (Q=video, K/V=umT5 tokens) | Every DiT block | Sec 4.2.1, p14 |
| Timestep | Shared adaLN MLP -> 6 scale/shift params + per-block biases | Every LayerNorm in every block | Sec 4.2.1, p14 |
| Image (I2V, pixel) | VAE-encoded latent + mask, channel-concatenated with noise | DiT input (before first block) | Sec 5.1.1, p26 |
| Image (I2V, semantic) | CLIP encoder -> 3-layer MLP -> decoupled cross-attention | Every DiT block (separate from text) | Sec 5.1.1, p26 |
| Video editing context | Concept-decoupled VAE latents + mask, channel-concatenated | DiT input | Sec 5.2.1, p31 |
| Face identity | VAE-encoded face frames prepended temporally | Self-attention (no new module) | Sec 5.4.1, p35 |
| Camera motion | Plucker coords -> Encoder -> Adapter -> scale/shift | Adaptive norm in every DiT block | Sec 5.5, p38 |
| CFG | Conditional + unconditional forward pass | Inference only | Sec 4.4, p17 |

---
---

# Paper 2: Qwen-Image (arXiv 2508.02324)

**Full Title:** Qwen-Image Technical Report  
**Authors:** Qwen Team (Alibaba)  
**Domain:** Image Generation and Image Editing

---

## 2.1 What is Qwen-Image?

Qwen-Image is an image generation foundation model that excels at complex text rendering (English and Chinese), text-to-image generation (T2I), and image editing (TI2I: text-image-to-image). It uses a Multimodal Diffusion Transformer (MMDiT) architecture conditioned by a frozen multimodal LLM (Qwen2.5-VL). The MMDiT has 20B parameters.

**Reference:** Abstract, page 1; Section 1, page 6.

---

## 2.2 Overall Architecture

Qwen-Image has three core components (Figure 6, page 7; Section 2.1, page 7):

1. **Qwen2.5-VL** (7B, frozen): A multimodal large language model that serves as the condition encoder. It processes both text and images.
2. **VAE** (Wan-2.1-VAE based, shared encoder + image-specific decoder): Compresses images into latent space and decodes them back.
3. **MMDiT** (20B parameters): A double-stream Multimodal Diffusion Transformer that jointly models text and image latents.

The pipeline for **text-to-image (T2I)**:
1. The text prompt is wrapped in a system prompt template (Figure 7, page 8) and fed to Qwen2.5-VL.
2. The **last layer's hidden state** from Qwen2.5-VL becomes the conditioning representation `h`.
3. A random noise `x_1 ~ N(0,I)` is sampled.
4. The MMDiT denoises `x_1` into a clean image latent `x_0`, conditioned on `h` and timestep `t`.
5. The VAE decoder reconstructs the image from `x_0`.

The pipeline for **image editing (TI2I)**:
1. The input image + text instruction are wrapped in a TI2I system prompt (Figure 15, page 18) and fed to Qwen2.5-VL, producing hidden state `h` (semantic understanding).
2. The same input image is **also** encoded by the VAE to get a reconstructive latent (low-level features).
3. Both conditioning signals feed into the MMDiT.

**Reference:** Section 2.1, page 7; Figure 6, page 7.

---

## 2.3 The MMDiT Block in Detail

Unlike Wan's single-stream DiT with cross-attention, Qwen-Image uses a **double-stream MMDiT** where text tokens and image tokens flow through **parallel processing paths** that share attention.

### MMDiT Block Structure (Figure 6, page 7)

```
Text tokens (from Qwen2.5-VL)           Image tokens (from VAE + noise)
         |                                        |
         v                                        v
  Scale & Shift Norm  <---  MLP(t)  --->  Scale & Shift Norm
         |                                        |
         v                                        v
     Linear layer                            Linear layer
    (project to Q,K,V)                     (project to Q,K,V)
         |                                        |
     QK-Norm (RMSNorm)                      QK-Norm (RMSNorm)
         |                                        |
     MSRoPE applied                          MSRoPE applied
         |                                        |
         +----------> JOINT Self-Attention <------+
         |         (all Q,K,V concatenated         |
         |          into one attention op)          |
         |         (results split back)             |
         v                                        v
       Gate                                     Gate
         |                                        |
    + Residual                               + Residual
         |                                        |
  Scale & Shift Norm                      Scale & Shift Norm
         |                                        |
         v                                        v
     Gate MLP                                Gate MLP
         |                                        |
    + Residual                               + Residual
         |                                        |
         v                                        v
  Text tokens (out)                      Image tokens (out)
```

### What makes this different from Wan's cross-attention approach

- In Wan: text tokens are **static** -- they enter only as K,V in cross-attention and never get updated by the visual tokens.
- In Qwen-Image's MMDiT: text tokens and image tokens **attend to each other bidirectionally** through joint self-attention. Both streams are updated at every layer. Text representations evolve as they interact with image tokens, and vice versa.

**Reference:** Section 2.4, page 8; Figure 6, page 7.

---

## 2.4 Text Conditioning: Qwen2.5-VL Hidden States

### How text enters the model

1. The user's text prompt is formatted with a task-specific **system prompt** (Figure 7, page 8 for T2I).
2. This formatted input is processed by the **frozen Qwen2.5-VL** model (7B parameters).
3. The **last layer's hidden state** from the language model backbone is extracted as representation `h`.
4. `h` is a sequence of token embeddings that enters the **text stream** of the MMDiT.
5. Throughout the MMDiT's 60 layers, text tokens participate in joint self-attention with image tokens, allowing deep bidirectional interaction.

### Why Qwen2.5-VL (not a text-only encoder)

Three reasons (Section 2.2, page 7):
1. **Pre-aligned language-vision spaces:** Qwen2.5-VL already understands the relationship between text and visual content, making it more suitable for T2I than a pure language model.
2. **Strong language modeling retained:** Despite being multimodal, it doesn't lose language capability.
3. **Native multimodal input support:** For image editing tasks, the same model can process both the text instruction and the input image together, producing a unified representation.

**Reference:** Section 2.2, page 7; Figure 7, page 8.

---

## 2.5 Timestep Conditioning: Scale & Shift Modulation with Gating

### How timestep enters the model

1. The diffusion timestep `t` is processed by an **MLP** at the base of the architecture.
2. The MLP output produces **scale and shift parameters** that modulate the normalization layers in **both** the text stream and the image stream.
3. Specifically, each stream has two modulated normalizations:
   - **Before self-attention:** Scale & Shift Norm
   - **Before the gated MLP:** Scale & Shift Norm
4. Additionally, **gating** is applied on both the self-attention output and the MLP output before adding to the residual. The gate values are also derived from the timestep conditioning.

### Normalization details

- General normalization: **LayerNorm**
- QK-Norm (applied to Q and K before attention): **RMSNorm**
- The paper states: "The model employs RMSNorm for QK-Norm, while all other normalization layers use LayerNorm."

**Reference:** Figure 6 caption, page 7; Section 2.4, page 8.

---

## 2.6 Image Conditioning for Editing (TI2I Task)

For image editing, Qwen-Image employs a **dual-encoding mechanism** that provides complementary image representations (Section 4.3, page 18; Figure 14, page 18):

### Path 1: Semantic Features via Qwen2.5-VL (enters the text stream)

1. The input image and the text editing instruction are combined using a TI2I system prompt (Figure 15, page 18):
   ```
   <|im_start|>system
   Describe the key features of the input image... then explain how
   the user's text instruction should alter or modify the image...
   <|im_end|>
   <|im_start|>user
   <|vision_start|><|user_image|><|vision_end|><|user_text|><|im_end|>
   <|im_start|>assistant
   ```
2. Qwen2.5-VL processes the image through its internal **ViT encoder**, producing visual patches. These are concatenated with text tokens inside Qwen2.5-VL's transformer.
3. The last layer's hidden state captures **high-level semantic understanding**: what the image contains, what the instruction means, and how the image should change.
4. This hidden state feeds into the **text stream** of the MMDiT.

### Path 2: Reconstructive Features via VAE (enters the image stream)

1. The same input image is **separately** fed through the VAE encoder.
2. The resulting latent captures **low-level visual details** -- textures, colors, edges, fine structure.
3. This VAE latent is **concatenated with the noised target image latent along the sequence dimension** in the image stream of the MMDiT.
4. MSRoPE is extended with a **frame dimension** to let the model distinguish between "input image patches" and "target image patches."

### Why both paths are needed

- The MLLM path provides **semantic understanding** (what to change and why).
- The VAE path provides **pixel-level fidelity** (preserving visual details the instruction doesn't mention).
- The paper states: "This dual-conditioning design enables the model to simultaneously maintain semantic coherence and visual consistency." (Section 1, page 6)

**Reference:** Section 4.3, page 18; Figure 14, page 18; Figure 15, page 18; Section 1, page 6.

---

## 2.7 MSRoPE: Multimodal Scalable RoPE

A novel positional encoding designed for the MMDiT (Section 2.4, page 8; Figure 8, page 9).

### The problem

In MMDiT, text tokens and image tokens are concatenated for joint self-attention. How should their positions be encoded?

- **Naive concatenation** (Figure 8-A): Flatten image into 1D positions (0,1,2,...), append text after. Problem: no 2D spatial structure for images; long position IDs for text.
- **Column-wise 2D encoding** (Figure 8-B, Seedream 3.0): Image uses 2D RoPE centered in grid, text tokens treated as 2D with sequential column positions. Problem: some rows of position encodings become **isomorphic** between text and image tokens (e.g., the 0-th middle row), confusing the model.

### MSRoPE solution (Figure 8-C)

- **Image tokens:** Standard 2D RoPE based on (height, width) grid positions starting from the center, e.g., `(-1,-1), (0,-1), (1,-1), (-1,0), (0,0), (1,0), ...`.
- **Text tokens:** Treated as 2D tensors but with **identical position IDs across both dimensions**. Positions are placed along the **diagonal** of the grid: `(2,2), (3,3), (4,4), ...`.
- This means text positions are always on the diagonal and never overlap with any image position row or column.

### Benefits

- Image tokens retain 2D spatial structure and **resolution scaling** advantages of 2D RoPE.
- Text tokens have functionally equivalent behavior to **1D RoPE** (since both dimensions are identical).
- No ambiguity between text and image positional encodings.

### Extension for multi-image (TI2I)

- For editing tasks with input + target images, MSRoPE adds a **frame dimension** (Figure 14 right, page 18).
- Input image patches and target image patches are distinguished by different frame IDs while sharing the same (height, width) coordinates.

**Reference:** Section 2.4, page 8; Figure 8, page 9; Figure 14 (right), page 18.

---

## 2.8 VAE Design

Qwen-Image uses a custom VAE based on Wan-2.1-VAE with important modifications (Section 2.3, page 8):

- **Single encoder, dual decoder** architecture: shared encoder for images and videos, separate specialized decoders.
- The encoder is frozen from Wan-2.1-VAE; only the **image decoder** is fine-tuned.
- Fine-tuned on text-rich images (PDFs, PowerPoint slides, posters) to enhance small text and fine detail reconstruction.
- Compression: 8x8 spatial, 16 latent channels, patch size 2.
- Parameters: 19M encoder (for images), 25M decoder (Table 2, page 20).

**Reference:** Section 2.3, page 8; Table 1, page 9; Table 2, page 20.

---

## 2.9 Training Objective

Flow Matching with Rectified Flows (Section 4.1, page 15):

```
x_t = t * x_0 + (1 - t) * x_1           -- Eq. 1, page 15
v_t = x_0 - x_1                          -- velocity
L = E[||v_theta(x_t, t, h) - v_t||^2]   -- Eq. 2, page 15
```

Where `x_0` is the data latent (from VAE), `x_1 ~ N(0,I)` is noise, `t ~ logit-normal`, and `h = phi(S)` is the Qwen2.5-VL hidden state.

*Note: Wan and Qwen-Image use swapped notation for x_0/x_1 but the underlying math is identical.*

---

## 2.10 Post-Training

Qwen-Image goes further than Wan's base model with explicit **RL-based alignment** (Section 4.2, pages 17-18):

1. **Supervised Fine-Tuning (SFT):** High-quality human-annotated images with clear, detailed, photorealistic properties.
2. **Direct Preference Optimization (DPO):** Adapted for flow matching. Human annotators select best/worst images from multiple generations. Loss function (Eq. 3, page 17) computes preference differences between chosen and rejected samples at the flow-matching velocity level.
3. **Group Relative Policy Optimization (GRPO):** On-policy RL with a reward model. Groups of images are generated per prompt, and the advantage function (Eq. 4, page 17) normalizes rewards within each group. The sampling process is reformulated as an SDE for exploration (Eq. 6, page 17).

**Reference:** Section 4.2, pages 17-18; Equations 3-8, pages 17-18.

---

## 2.11 Summary of All Conditioning Mechanisms in Qwen-Image

| Condition | Mechanism | Where it enters | Reference |
|---|---|---|---|
| Text prompt (T2I) | Qwen2.5-VL last-layer hidden state -> text stream of MMDiT | Joint self-attention with image tokens, every block | Sec 2.2, p7 |
| Text + Image (TI2I, semantic) | Qwen2.5-VL processes image (via ViT) + text together -> hidden state -> text stream | Joint self-attention, every block | Sec 4.3, p18 |
| Image (TI2I, reconstructive) | VAE-encoded input image latent concatenated along sequence dim in image stream | Image stream, joint self-attention | Sec 4.3, p18 |
| Timestep | MLP -> Scale & Shift params + gating for both streams | Every norm layer in every block | Fig 6, p7 |
| Position (image) | 2D RoPE on (height, width) grid | Q,K in self-attention | Sec 2.4, p8 |
| Position (text) | MSRoPE: diagonal positions (identical IDs on both dims) | Q,K in self-attention | Sec 2.4, p8 |
| Position (multi-image) | MSRoPE + frame dimension | Q,K in self-attention | Fig 14, p18 |
| CFG | Conditional + unconditional forward pass | Inference | Implied in Sec 4.1, p15 |

---
---

# Key Differences at a Glance

| Aspect | Wan | Qwen-Image |
|---|---|---|
| **DiT type** | Single-stream DiT | Double-stream MMDiT |
| **Text-visual interaction** | Cross-attention (text is static K,V) | Joint self-attention (text and image tokens update each other bidirectionally) |
| **Text encoder** | umT5 (5.3B, text-only, bidirectional) | Qwen2.5-VL (7B, multimodal LLM) |
| **Timestep injection** | Shared adaLN MLP + per-block biases (6 params/block) | Per-stream Scale & Shift Norm + gating |
| **Image conditioning (editing)** | Channel-wise concatenation of VAE latents + masks | Dual encoding: MLLM semantic features in text stream + VAE latent in image stream along sequence dim |
| **Positional encoding** | 3D patchify conv (1,2,2) | MSRoPE: 2D for images, diagonal for text |
| **Post-training alignment** | Prompt rewriting (LLM-based) | SFT + DPO + GRPO (full RL pipeline) |
| **Model scale** | 1.3B / 14B (DiT) | 20B (MMDiT) + 7B (Qwen2.5-VL) |
