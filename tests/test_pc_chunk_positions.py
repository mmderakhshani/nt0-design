"""Tests for the PC-chunk MRoPE position rule.

Each PC chunk (M tokens output by Uni3D for one camera) shares one MRoPE
position; sink/embodiment/align/<pc_*_start>/<pc_*_end> follow the text
rule (counter +1 per token). Within a chunk all M tokens carry
(t, h, w) = (p, p, p) so MRoPE only signals "which chunk is this" via
the chunk-level Δp; intra-chunk geometry is already in the Uni3D patch
tokens.

Layout (matches SequenceBuilder.forward):
    sink_pc          -> p,    p+=1
    embodiment_pc    -> p,    p+=1
    <align>          -> p,    p+=1
    per camera (M = chunk_size):
        <pc_*_start> -> p,    p+=1
        M tokens     -> all share p, p+=1 once for the whole chunk
        <pc_*_end>   -> p,    p+=1
"""

import torch
import pytest

from unified_vla.core import build_pc_chunk_position_ids


class TestBuildPCChunkPositionIds:
    """Unit tests for the position-ID builder."""

    def test_no_cameras_just_pre_chunk(self):
        positions, next_p = build_pc_chunk_position_ids([], start=0)
        # Only sink, embodiment, align — three text-rule tokens.
        assert positions == [0, 1, 2]
        assert next_p == 3

    def test_single_camera(self):
        positions, next_p = build_pc_chunk_position_ids([4], start=0)
        # sink(0), emb(1), align(2), <start>(3), [4]*4, <end>(5)
        assert positions == [0, 1, 2, 3, 4, 4, 4, 4, 5]
        assert next_p == 6

    def test_two_cameras_matches_user_spec(self):
        """Reproduces the user's spec table: each chunk shares one position,
        counter advances +1 per chunk and +1 per delimiter."""
        positions, next_p = build_pc_chunk_position_ids([4, 4], start=0)
        # sink(0), emb(1), align(2),
        # <start>(3), [4]*4, <end>(5),
        # <start>(6), [7]*4, <end>(8)
        assert positions == [0, 1, 2, 3, 4, 4, 4, 4, 5, 6, 7, 7, 7, 7, 8]
        assert next_p == 9

    def test_three_cameras_mixed_chunk_sizes(self):
        positions, next_p = build_pc_chunk_position_ids([2, 3, 1], start=10)
        expected = [
            10, 11, 12,                # sink, embodiment, align
            13, 14, 14, 15,            # cam1: <start>, M=2 chunk, <end>
            16, 17, 17, 17, 18,        # cam2: <start>, M=3 chunk, <end>
            19, 20, 21,                # cam3: <start>, M=1 chunk, <end>
        ]
        assert positions == expected
        assert next_p == 22

    def test_intra_chunk_delta_is_zero(self):
        """All M tokens of one chunk share the same position — Δp = 0 inside."""
        M = 8
        positions, _ = build_pc_chunk_position_ids([M], start=0)
        # Pre-chunk = 3 (sink/embod/align) + 1 (<start>) = 4 tokens.
        chunk_positions = positions[4 : 4 + M]
        assert len(chunk_positions) == M
        assert all(p == chunk_positions[0] for p in chunk_positions)

    def test_cross_chunk_delta_is_three(self):
        """Adjacent chunks separated by <pc_end>, <pc_start> → 3 counter steps
        between their shared chunk positions: chunk_a, end, start, chunk_b ->
        positions p, p+1, p+2, p+3."""
        M = 4
        positions, _ = build_pc_chunk_position_ids([M, M], start=0)
        # Chunk 1 position is at index 4 (after sink/emb/align/<start>).
        # Chunk 2 position is at index 4 + M + 2 (skip chunk1's M tokens + <end> + <start>).
        chunk1_p = positions[4]
        chunk2_p = positions[4 + M + 2]
        assert chunk2_p - chunk1_p == 3

    def test_total_length_formula(self):
        """Total positions = 3 (pre-chunk) + sum(M_k + 2) for K cameras."""
        for chunk_sizes in [[], [1], [4], [4, 4], [2, 3, 5, 7]]:
            positions, _ = build_pc_chunk_position_ids(chunk_sizes, start=0)
            expected = 3 + sum(M + 2 for M in chunk_sizes)
            assert len(positions) == expected

    def test_next_p_formula(self):
        """next_p - start = 3 + 3 * K (always 3 per camera: <start>, chunk, <end>)."""
        for chunk_sizes in [[], [4], [4, 4], [2, 3, 5]]:
            _, next_p = build_pc_chunk_position_ids(chunk_sizes, start=100)
            expected = 100 + 3 + 3 * len(chunk_sizes)
            assert next_p == expected

    def test_start_offset(self):
        """Counter starts wherever caller specifies (typically max_vlm_pos + 1)."""
        positions_a, next_p_a = build_pc_chunk_position_ids([4], start=0)
        positions_b, next_p_b = build_pc_chunk_position_ids([4], start=50)
        assert positions_b == [p + 50 for p in positions_a]
        assert next_p_b == next_p_a + 50

    def test_no_sink(self):
        positions, next_p = build_pc_chunk_position_ids([4], start=0, has_sink=False)
        assert positions == [0, 1, 2, 3, 3, 3, 3, 4]  # only emb, align, then chunk
        assert next_p == 5

    def test_no_align(self):
        positions, next_p = build_pc_chunk_position_ids([4], start=0, has_align=False)
        assert positions == [0, 1, 2, 3, 3, 3, 3, 4]
        assert next_p == 5

    def test_minimal_no_extras(self):
        """Pure chunk, no prefix tokens — should still satisfy the chunk rule."""
        positions, next_p = build_pc_chunk_position_ids(
            [3], start=0, has_sink=False, has_embodiment=False, has_align=False,
        )
        assert positions == [0, 1, 1, 1, 2]  # <start>, M=3 share, <end>
        assert next_p == 3


class TestPositionDeltasMatchExpectedSemantics:
    """High-level checks: verify the resulting position deltas match the
    intended attention semantics (Δp = 0 inside chunk, non-zero across)."""

    def test_intra_chunk_attention_sees_zero_delta(self):
        # Layout for [4, 4] start=0:
        # idx 0..2: sink/emb/align, idx 3: <start>cam1, idx 4..7: chunk 1,
        # idx 8: <end>cam1, idx 9: <start>cam2, idx 10..13: chunk 2, idx 14: <end>cam2.
        positions, _ = build_pc_chunk_position_ids([4, 4], start=0)
        chunk1 = positions[4:8]
        chunk2 = positions[10:14]
        for i in range(len(chunk1)):
            for j in range(len(chunk1)):
                assert chunk1[i] - chunk1[j] == 0
        for i in range(len(chunk2)):
            for j in range(len(chunk2)):
                assert chunk2[i] - chunk2[j] == 0

    def test_cross_chunk_attention_sees_nonzero_delta(self):
        positions, _ = build_pc_chunk_position_ids([4, 4], start=0)
        chunk1 = positions[4:8]
        chunk2 = positions[10:14]
        for c1 in chunk1:
            for c2 in chunk2:
                assert c2 - c1 != 0

    def test_chunk_to_text_text_rule_tokens_have_distinct_positions(self):
        """Sink, embodiment, align, and the delimiters all sit at unique positions."""
        positions, _ = build_pc_chunk_position_ids([4], start=0)
        text_rule_indices = [0, 1, 2, 3, 8]  # sink, emb, align, <start>, <end>
        text_rule_positions = [positions[i] for i in text_rule_indices]
        assert len(set(text_rule_positions)) == len(text_rule_positions)

    def test_action_starts_immediately_after_pc(self):
        """Action segment uses next_p as its starting counter — gap-free."""
        _, pc_next = build_pc_chunk_position_ids([4, 4], start=10)
        # If action then takes N_action positions starting at pc_next, those
        # positions should not overlap any PC position.
        pc_positions, _ = build_pc_chunk_position_ids([4, 4], start=10)
        action_positions = list(range(pc_next, pc_next + 5))
        assert set(pc_positions).isdisjoint(action_positions)


class TestBackboneIntegration:
    """End-to-end check: chunk rule plumbs through Backbone.forward."""

    MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
    D_PC = 1024
    D_ACTION = 1024
    D_COND = 64
    NUM_LAYERS = 2
    B = 1
    D_VLM = 2048

    @pytest.fixture(scope="class")
    def transformers_available(self):
        try:
            import transformers  # noqa: F401
            return True
        except ImportError:
            return False

    @pytest.fixture(scope="class")
    def backbone(self, transformers_available):
        if not transformers_available:
            pytest.skip("transformers not installed")
        from transformers import Qwen2_5_VLForConditionalGeneration
        from unified_vla.surgery import create_backbone

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.MODEL_ID, dtype=torch.float32,
        )
        return create_backbone(
            model, d_pc=self.D_PC, d_action=self.D_ACTION,
            d_cond=self.D_COND, num_layers=self.NUM_LAYERS,
        )

    def test_chunk_sizes_route_through_forward(self, backbone):
        """Backbone.forward(pc_chunk_sizes=...) doesn't crash and produces correct shapes."""
        K, M = 2, 4
        n_pc = 3 + K * (M + 2)  # 15
        n_vlm = 5
        n_action = 4

        vlm = torch.randn(self.B, n_vlm, self.D_VLM)
        pc = torch.randn(self.B, n_pc, self.D_PC)
        action = torch.randn(self.B, n_action, self.D_ACTION)
        cond = torch.randn(self.B, self.D_COND)
        pos_ids = torch.arange(n_vlm).unsqueeze(0).expand(3, self.B, -1)

        with torch.no_grad():
            v, p, a = backbone(
                vlm, pc, action, cond, pos_ids,
                pc_chunk_sizes=[M, M],
            )
        assert v.shape == (self.B, n_vlm, self.D_VLM)
        assert p.shape == (self.B, n_pc, self.D_PC)
        assert a.shape == (self.B, n_action, self.D_ACTION)
        assert not torch.isnan(v).any()
        assert not torch.isnan(p).any()
        assert not torch.isnan(a).any()

    def test_chunk_rule_changes_pc_output_vs_per_token_rule(self, backbone):
        """Same PC tokens, different position rules → different PC outputs.
        Sanity that the pc_chunk_sizes path actually does something."""
        K, M = 2, 4
        n_pc = 3 + K * (M + 2)
        n_vlm = 5
        n_action = 4

        vlm = torch.randn(self.B, n_vlm, self.D_VLM)
        pc = torch.randn(self.B, n_pc, self.D_PC)
        action = torch.randn(self.B, n_action, self.D_ACTION)
        cond = torch.randn(self.B, self.D_COND)
        pos_ids = torch.arange(n_vlm).unsqueeze(0).expand(3, self.B, -1)

        with torch.no_grad():
            _, pc_chunked, _ = backbone(
                vlm, pc, action, cond, pos_ids,
                pc_chunk_sizes=[M, M],
            )
            _, pc_per_token, _ = backbone(
                vlm, pc, action, cond, pos_ids,
                pc_chunk_sizes=None,
            )

        # The two PC outputs must differ — they used different RoPE rotations
        # at the chunk positions.
        assert not torch.allclose(pc_chunked, pc_per_token)

    def test_layout_mismatch_raises(self, backbone):
        """If pc_chunk_sizes claims a layout that doesn't match n_pc, raise."""
        n_pc = 10  # doesn't match 3 + 1 * (4 + 2) = 9 nor 3 + 2 * (4 + 2) = 15
        n_vlm = 5
        n_action = 4

        vlm = torch.randn(self.B, n_vlm, self.D_VLM)
        pc = torch.randn(self.B, n_pc, self.D_PC)
        action = torch.randn(self.B, n_action, self.D_ACTION)
        cond = torch.randn(self.B, self.D_COND)
        pos_ids = torch.arange(n_vlm).unsqueeze(0).expand(3, self.B, -1)

        with pytest.raises(ValueError, match="PC layout mismatch"):
            backbone(vlm, pc, action, cond, pos_ids, pc_chunk_sizes=[4])

    def test_none_falls_back_to_legacy_per_token_rule(self, backbone):
        """pc_chunk_sizes=None must reproduce the original per-token arange behavior."""
        n_pc = 3
        n_vlm = 5
        n_action = 4

        vlm = torch.randn(self.B, n_vlm, self.D_VLM)
        pc = torch.randn(self.B, n_pc, self.D_PC)
        action = torch.randn(self.B, n_action, self.D_ACTION)
        cond = torch.randn(self.B, self.D_COND)
        pos_ids = torch.arange(n_vlm).unsqueeze(0).expand(3, self.B, -1)

        with torch.no_grad():
            v_a, p_a, a_a = backbone(vlm, pc, action, cond, pos_ids)
            v_b, p_b, a_b = backbone(vlm, pc, action, cond, pos_ids, pc_chunk_sizes=None)

        assert torch.equal(v_a, v_b)
        assert torch.equal(p_a, p_b)
        assert torch.equal(a_a, a_b)
