"""Tests for init_special_tokens_from_vlm.

These use a tiny stub that mimics the parts of a Qwen-VL model the surgery
function reads (no real model download), so they run on CPU in milliseconds.

Embodiment is no longer initialized here — `SharedEmbodimentEmbedding`
self-inits in its `__init__` (Recipe 2: zero-init projections).
"""

from types import SimpleNamespace

import torch
import torch.nn as nn

from unified_vla import (
    SequenceBuilder,
    SharedEmbodimentEmbedding,
    init_special_tokens_from_vlm,
)


D_VLM = 256
D_PC = 64
D_ACTION = 96
VOCAB = 128
VISION_START_ID = 17
VISION_END_ID = 23


def _make_stub_qwen():
    """Mimic qwen_model.model.language_model.embed_tokens + .config.vision_*_token_id."""
    embed = nn.Embedding(VOCAB, D_VLM)
    language_model = SimpleNamespace(embed_tokens=embed)
    inner_model = SimpleNamespace(language_model=language_model)
    config = SimpleNamespace(
        vision_start_token_id=VISION_START_ID,
        vision_end_token_id=VISION_END_ID,
    )
    return SimpleNamespace(model=inner_model, config=config)


def _expected_base(qwen) -> torch.Tensor:
    table = qwen.model.language_model.embed_tokens.weight
    return (table[VISION_START_ID] + table[VISION_END_ID]) / 2


class TestInitSpecialTokensFromVLM:
    def test_pc_delimiters_match_base_when_eps_zero(self):
        sb = SequenceBuilder(d_pc=D_PC, d_action=D_ACTION)
        qwen = _make_stub_qwen()
        base = _expected_base(qwen)

        init_special_tokens_from_vlm(sb, qwen_model=qwen, eps=0.0)

        expected = base[:D_PC].view(1, 1, D_PC)
        for p in (sb.pc_start_emb, sb.pc_end_emb, sb.pc_wrist_start_emb, sb.pc_wrist_end_emb):
            assert torch.allclose(p, expected, atol=0)

    def test_action_delimiters_match_base_when_eps_zero(self):
        sb = SequenceBuilder(d_pc=D_PC, d_action=D_ACTION)
        qwen = _make_stub_qwen()
        base = _expected_base(qwen)

        init_special_tokens_from_vlm(sb, qwen_model=qwen, eps=0.0)

        expected = base[:D_ACTION].view(1, 1, D_ACTION)
        for p in (sb.action_start_emb, sb.action_end_emb):
            assert torch.allclose(p, expected, atol=0)

    def test_proprio_delimiters_match_base_when_eps_zero(self):
        """proprio_start/end use the same d_action-width base as action_start/end."""
        sb = SequenceBuilder(d_pc=D_PC, d_action=D_ACTION)
        qwen = _make_stub_qwen()
        base = _expected_base(qwen)

        init_special_tokens_from_vlm(sb, qwen_model=qwen, eps=0.0)

        expected = base[:D_ACTION].view(1, 1, D_ACTION)
        for p in (sb.proprio_start_emb, sb.proprio_end_emb):
            assert torch.allclose(p, expected, atol=0)

    def test_eps_controls_noise_scale(self):
        torch.manual_seed(0)
        sb = SequenceBuilder(d_pc=D_PC, d_action=D_ACTION)
        qwen = _make_stub_qwen()
        base = _expected_base(qwen)

        init_special_tokens_from_vlm(sb, qwen_model=qwen, eps=1e-3)

        delta = sb.pc_start_emb.flatten() - base[:D_PC]
        assert delta.abs().mean() < 0.1, "noise should be small"
        assert delta.abs().mean() > 0, "noise should be nonzero"

    def test_each_delimiter_gets_independent_noise(self):
        sb = SequenceBuilder(d_pc=D_PC, d_action=D_ACTION)
        qwen = _make_stub_qwen()
        init_special_tokens_from_vlm(sb, qwen_model=qwen, eps=1e-3)

        # No two PC delimiters should be exactly equal after the function runs.
        assert not torch.equal(sb.pc_start_emb, sb.pc_end_emb)
        assert not torch.equal(sb.pc_start_emb, sb.pc_wrist_start_emb)
        assert not torch.equal(sb.pc_wrist_start_emb, sb.pc_wrist_end_emb)
        # Action and proprio delimiters live at d_action width; all four must be distinct.
        assert not torch.equal(sb.action_start_emb, sb.action_end_emb)
        assert not torch.equal(sb.proprio_start_emb, sb.proprio_end_emb)
        assert not torch.equal(sb.proprio_start_emb, sb.action_start_emb)
        assert not torch.equal(sb.proprio_end_emb, sb.action_end_emb)

    def test_alignment_token_left_random(self):
        """The <align> embedding is not in the init list — it should stay as it was."""
        sb = SequenceBuilder(d_pc=D_PC, d_action=D_ACTION)
        qwen = _make_stub_qwen()
        before = sb.align_base_emb.detach().clone()

        init_special_tokens_from_vlm(sb, qwen_model=qwen, eps=1e-3)

        assert torch.equal(sb.align_base_emb, before)

    def test_sink_tokens_left_random(self):
        """Sinks are content-free — vision-tokens base is not a meaningful prior; leave them alone."""
        sb = SequenceBuilder(d_pc=D_PC, d_action=D_ACTION)
        qwen = _make_stub_qwen()
        before_sink_pc = sb.sink_pc.detach().clone()
        before_sink_action = sb.sink_action.detach().clone()

        init_special_tokens_from_vlm(sb, qwen_model=qwen, eps=1e-3)

        assert torch.equal(sb.sink_pc, before_sink_pc)
        assert torch.equal(sb.sink_action, before_sink_action)

    def test_does_not_touch_embodiment(self):
        """SharedEmbodimentEmbedding self-inits; init_special_tokens_from_vlm leaves it alone."""
        sb = SequenceBuilder(d_pc=D_PC, d_action=D_ACTION)
        embod = SharedEmbodimentEmbedding(d_pc=D_PC, d_action=D_ACTION, d_emb=8)
        before_base = embod.base.weight.detach().clone()
        before_to_pc = embod.to_pc.weight.detach().clone()
        before_to_action = embod.to_action.weight.detach().clone()
        qwen = _make_stub_qwen()

        init_special_tokens_from_vlm(sb, qwen_model=qwen, eps=1e-3)

        assert torch.equal(embod.base.weight, before_base)
        assert torch.equal(embod.to_pc.weight, before_to_pc)
        assert torch.equal(embod.to_action.weight, before_to_action)

    def test_init_does_not_break_forward(self):
        """Sanity: running SequenceBuilder.forward after init produces the right shapes."""
        sb = SequenceBuilder(d_pc=D_PC, d_action=D_ACTION)
        embod = SharedEmbodimentEmbedding(d_pc=D_PC, d_action=D_ACTION, d_emb=8)
        qwen = _make_stub_qwen()
        init_special_tokens_from_vlm(sb, qwen_model=qwen, eps=1e-3)

        B, K, M, T, P = 2, 2, 5, 4, 1
        vlm = torch.randn(B, 7, D_VLM)
        pc_per_cam = [torch.randn(B, M, D_PC) for _ in range(K)]
        action = torch.randn(B, T, D_ACTION)
        proprio = torch.randn(B, P, D_ACTION)
        embod_pc, embod_action = embod(torch.zeros(B, dtype=torch.long))

        seq = sb(vlm, pc_per_cam, action, proprio, embod_pc, embod_action)

        # Action segment: sink(1) + embod(1) + <proprio_start>(1) + P + <proprio_end>(1)
        #                  + <action_start>(1) + T + <action_end>(1) = P + T + 6.
        assert seq.action.shape == (B, P + T + 6, D_ACTION)
        # PC segment: sink(1) + embod(1) + <align>(1) + K * (1 + M + 1) = 3 + K * (M + 2).
        assert seq.pc.shape == (B, 3 + K * (M + 2), D_PC)
