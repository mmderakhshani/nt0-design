import torch
import pytest

from unified_vla.core import (
    TokenType, SequenceBuilder, TokenSequence, AlignmentToken,
    InputProjections, SharedEmbodimentEmbedding, ProprioEncoder, AdaLNConditioner,
    modulate,
)


def _embodiment(B: int, d_pc: int, d_action: int):
    """Random per-stream embodiment tokens for layout tests (values irrelevant)."""
    return torch.randn(B, 1, d_pc), torch.randn(B, 1, d_action)

# Test dimensions (small, CPU-friendly)
D_VLM = 256
D_PC = 128
D_ACTION = 128
PC_ENCODER_DIM = 512
ACTION_DIM = 10
PROPRIO_DIM = 10
N_PROPRIO = 1  # tokens per sample (1 = current step only)
B = 2
N_VLM = 10
N_PC = 5  # tokens per camera
N_ACTION = 8
K_CAMERAS = 3


class TestTokenType:
    def test_values(self):
        assert TokenType.VLM == 0
        assert TokenType.PC == 1
        assert TokenType.ACTION == 2

    def test_members(self):
        assert len(TokenType) == 3


class TestSequenceBuilder:
    """Layout indices after the sink-tokens addition.

    PC segment:     [sink_pc, embodiment_pc, <align>, <pc_*_start> cam <pc_*_end> ...]
    Action segment: [sink_action, embodiment_action,
                     <proprio_start>, proprio, <proprio_end>,
                     <action_start>, action, <action_end>]

    PC length:     3 + K * (M + 2)
    Action length: P + T + 6
    """

    @pytest.fixture
    def builder(self):
        return SequenceBuilder(d_pc=D_PC, d_action=D_ACTION)

    @pytest.fixture
    def sample_tokens(self):
        vlm = torch.randn(B, N_VLM, D_VLM)
        pc_per_cam = [torch.randn(B, N_PC, D_PC) for _ in range(K_CAMERAS)]
        action = torch.randn(B, N_ACTION, D_ACTION)
        proprio = torch.randn(B, N_PROPRIO, D_ACTION)
        embod_pc, embod_action = _embodiment(B, D_PC, D_ACTION)
        return vlm, pc_per_cam, action, proprio, embod_pc, embod_action

    def test_output_sequence_length(self, builder, sample_tokens):
        vlm, pc_per_cam, action, proprio, embod_pc, embod_action = sample_tokens
        seq = builder(vlm, pc_per_cam, action, proprio, embod_pc, embod_action)

        # PC: sink_pc + embodiment_pc + <align> + K * (<pc_start> + N_PC + <pc_end>)
        pc_total = 3 + K_CAMERAS * (N_PC + 2)
        # Action: sink_action + embodiment_action + <proprio_start> + P + <proprio_end>
        #         + <action_start> + N_ACTION + <action_end>
        action_total = N_PROPRIO + N_ACTION + 6
        expected_total = N_VLM + pc_total + action_total

        assert seq.token_types.shape == (B, expected_total)
        assert seq.vlm.shape[1] == N_VLM
        assert seq.pc.shape[1] == pc_total
        assert seq.action.shape[1] == action_total

    def test_pc_segment_structure(self, builder, sample_tokens):
        """PC: [sink_pc, embod_pc, <align>, <pc_start> cam_i_tokens <pc_end> ...]."""
        vlm, pc_per_cam, action, proprio, embod_pc, embod_action = sample_tokens
        seq = builder(vlm, pc_per_cam, action, proprio, embod_pc, embod_action)

        # Position 0: sink_pc; 1: embodiment_pc; 2: <align>; per cam at offset 3 + i*(N_PC+2)
        assert torch.equal(seq.pc[:, 1:2, :], embod_pc)
        offset = 3  # past sink + embodiment + align
        for i, cam_tokens in enumerate(pc_per_cam):
            start = offset + 1  # skip <pc_start>
            end = start + N_PC
            assert torch.equal(seq.pc[:, start:end, :], cam_tokens), f"Camera {i} tokens mismatch"
            offset += N_PC + 2

    def test_action_segment_structure(self, builder, sample_tokens):
        """Action: [sink_action, embod_action, <proprio_start>, proprio, <proprio_end>, <action_start>, action, <action_end>]."""
        vlm, pc_per_cam, action, proprio, embod_pc, embod_action = sample_tokens
        seq = builder(vlm, pc_per_cam, action, proprio, embod_pc, embod_action)

        # Position 0: sink_action; 1: embodiment_action; 2: <proprio_start>; proprio at [3, 3+P).
        assert torch.equal(seq.action[:, 1:2, :], embod_action)
        assert torch.equal(seq.action[:, 3 : 3 + N_PROPRIO, :], proprio)
        # action_start sits at N_PROPRIO + 4 (after sink, embod, proprio_start, P, proprio_end).
        action_start_idx = N_PROPRIO + 4
        action_chunk_idx = action_start_idx + 1
        action_chunk_end = action_chunk_idx + N_ACTION
        assert torch.equal(seq.action[:, action_chunk_idx:action_chunk_end, :], action)

    def test_token_types_correctness(self, builder, sample_tokens):
        vlm, pc_per_cam, action, proprio, embod_pc, embod_action = sample_tokens
        seq = builder(vlm, pc_per_cam, action, proprio, embod_pc, embod_action)

        types = seq.token_types[0]
        pc_total = 3 + K_CAMERAS * (N_PC + 2)
        action_total = N_PROPRIO + N_ACTION + 6

        assert (types[:N_VLM] == TokenType.VLM).all()
        assert (types[N_VLM : N_VLM + pc_total] == TokenType.PC).all()
        assert (types[N_VLM + pc_total : N_VLM + pc_total + action_total] == TokenType.ACTION).all()

    def test_proprio_tokens_share_action_type(self, builder, sample_tokens):
        """Sink + embodiment + proprio block all carry TokenType.ACTION."""
        vlm, pc_per_cam, action, proprio, embod_pc, embod_action = sample_tokens
        seq = builder(vlm, pc_per_cam, action, proprio, embod_pc, embod_action)
        pc_total = 3 + K_CAMERAS * (N_PC + 2)
        action_segment_start = N_VLM + pc_total
        # sink(1) + embodiment(1) + proprio_start(1) + P + proprio_end(1) = P + 4 tokens at front
        block_types = seq.token_types[
            0, action_segment_start : action_segment_start + N_PROPRIO + 4
        ]
        assert (block_types == TokenType.ACTION).all()

    def test_embodiment_pc_carries_pc_token_type(self, builder, sample_tokens):
        vlm, pc_per_cam, action, proprio, embod_pc, embod_action = sample_tokens
        seq = builder(vlm, pc_per_cam, action, proprio, embod_pc, embod_action)
        # PC segment starts immediately after VLM; both position 0 (sink_pc)
        # and position 1 (embodiment_pc) carry TokenType.PC.
        pc_segment_start = N_VLM
        assert seq.token_types[0, pc_segment_start] == TokenType.PC
        assert seq.token_types[0, pc_segment_start + 1] == TokenType.PC

    def test_token_types_consistent_across_batch(self, builder, sample_tokens):
        vlm, pc_per_cam, action, proprio, embod_pc, embod_action = sample_tokens
        seq = builder(vlm, pc_per_cam, action, proprio, embod_pc, embod_action)
        assert (seq.token_types[0] == seq.token_types[1]).all()

    def test_boundary_tokens_are_learnable(self, builder):
        param_names = {name for name, _ in builder.named_parameters()}
        expected = {
            "sink_pc", "sink_action",
            "align_base_emb",
            "pc_start_emb", "pc_end_emb",
            "pc_wrist_start_emb", "pc_wrist_end_emb",
            "proprio_start_emb", "proprio_end_emb",
            "action_start_emb", "action_end_emb",
        }
        assert param_names == expected
        for _, param in builder.named_parameters():
            assert param.requires_grad

    def test_vlm_tokens_passthrough(self, builder, sample_tokens):
        vlm, pc_per_cam, action, proprio, embod_pc, embod_action = sample_tokens
        seq = builder(vlm, pc_per_cam, action, proprio, embod_pc, embod_action)
        assert torch.equal(seq.vlm, vlm)

    def test_single_camera(self, builder):
        """Works with K=1 camera."""
        vlm = torch.randn(B, N_VLM, D_VLM)
        pc_1cam = [torch.randn(B, N_PC, D_PC)]
        action = torch.randn(B, N_ACTION, D_ACTION)
        proprio = torch.randn(B, N_PROPRIO, D_ACTION)
        embod_pc, embod_action = _embodiment(B, D_PC, D_ACTION)
        seq = builder(vlm, pc_1cam, action, proprio, embod_pc, embod_action)
        # sink + embodiment + <align> + 1 * (1 + N_PC + 1) = 3 + N_PC + 2
        assert seq.pc.shape[1] == 3 + N_PC + 2

    def test_variable_cameras(self, builder):
        """Works with different number of cameras."""
        vlm = torch.randn(B, N_VLM, D_VLM)
        proprio = torch.randn(B, N_PROPRIO, D_ACTION)
        embod_pc, embod_action = _embodiment(B, D_PC, D_ACTION)
        for k in [1, 2, 3, 5]:
            pc_k = [torch.randn(B, N_PC, D_PC) for _ in range(k)]
            seq = builder(
                vlm, pc_k, torch.randn(B, N_ACTION, D_ACTION), proprio,
                embod_pc, embod_action,
            )
            expected_pc_len = 3 + k * (N_PC + 2)
            assert seq.pc.shape[1] == expected_pc_len, f"Failed for K={k}"

    def test_variable_proprio_steps(self, builder):
        """Works with different proprio token counts (current vs. history).

        Length = sink_action(1) + embodiment_action(1) + <proprio_start>(1) + P
                 + <proprio_end>(1) + <action_start>(1) + N_ACTION + <action_end>(1)
                 = P + N_ACTION + 6.
        """
        vlm = torch.randn(B, N_VLM, D_VLM)
        pc_per_cam = [torch.randn(B, N_PC, D_PC) for _ in range(K_CAMERAS)]
        action = torch.randn(B, N_ACTION, D_ACTION)
        embod_pc, embod_action = _embodiment(B, D_PC, D_ACTION)
        for p in [1, 4, 8]:
            proprio = torch.randn(B, p, D_ACTION)
            seq = builder(vlm, pc_per_cam, action, proprio, embod_pc, embod_action)
            assert seq.action.shape[1] == p + N_ACTION + 6

    def test_pc_start_end_shared_across_cameras(self, builder, sample_tokens):
        """All cameras use the same <pc_start> and <pc_end> embeddings."""
        vlm, pc_per_cam, action, proprio, embod_pc, embod_action = sample_tokens
        seq = builder(vlm, pc_per_cam, action, proprio, embod_pc, embod_action)

        # <pc_start> positions: 3, 3+(N_PC+2), 3+2*(N_PC+2), ...
        for i in range(K_CAMERAS):
            start_pos = 3 + i * (N_PC + 2)
            if i > 0:
                first_start_pos = 3
                assert torch.equal(
                    seq.pc[:, start_pos:start_pos + 1, :],
                    seq.pc[:, first_start_pos:first_start_pos + 1, :],
                )

    def test_external_align_emb(self, builder, sample_tokens):
        vlm, pc_per_cam, action, proprio, embod_pc, embod_action = sample_tokens
        custom_align = torch.ones(B, 1, D_PC) * 42.0
        seq = builder(
            vlm, pc_per_cam, action, proprio, embod_pc, embod_action,
            align_emb=custom_align,
        )
        # <align> sits at index 2 (sink_pc=0, embodiment_pc=1, align=2).
        assert torch.equal(seq.pc[:, 2:3, :], custom_align)

    def test_default_align_emb(self, builder, sample_tokens):
        vlm, pc_per_cam, action, proprio, embod_pc, embod_action = sample_tokens
        seq = builder(vlm, pc_per_cam, action, proprio, embod_pc, embod_action)
        expected = builder.align_base_emb.expand(B, -1, -1)
        assert torch.equal(seq.pc[:, 2:3, :], expected)

    def test_embodiment_pc_at_position_one(self, builder, sample_tokens):
        """sink_pc is at position 0; embodiment_pc moves to position 1."""
        vlm, pc_per_cam, action, proprio, embod_pc, embod_action = sample_tokens
        seq = builder(vlm, pc_per_cam, action, proprio, embod_pc, embod_action)
        assert torch.equal(seq.pc[:, 1:2, :], embod_pc)

    def test_embodiment_action_at_position_one(self, builder, sample_tokens):
        """sink_action is at position 0; embodiment_action moves to position 1."""
        vlm, pc_per_cam, action, proprio, embod_pc, embod_action = sample_tokens
        seq = builder(vlm, pc_per_cam, action, proprio, embod_pc, embod_action)
        assert torch.equal(seq.action[:, 1:2, :], embod_action)


class TestWristDelimiters:
    """Wrist vs non-wrist PC delimiters (PC segment only; VLM stays unchanged)."""

    @pytest.fixture
    def builder(self):
        return SequenceBuilder(d_pc=D_PC, d_action=D_ACTION)

    @pytest.fixture
    def sample_tokens(self):
        vlm = torch.randn(B, N_VLM, D_VLM)
        pc_per_cam = [torch.randn(B, N_PC, D_PC) for _ in range(K_CAMERAS)]
        action = torch.randn(B, N_ACTION, D_ACTION)
        proprio = torch.randn(B, N_PROPRIO, D_ACTION)
        embod_pc, embod_action = _embodiment(B, D_PC, D_ACTION)
        return vlm, pc_per_cam, action, proprio, embod_pc, embod_action

    def _start_pos(self, cam_idx: int) -> int:
        # PC layout is [sink_pc, embodiment_pc, <align>, ...], so cam_idx 0's
        # <pc_*_start> sits at index 3.
        return 3 + cam_idx * (N_PC + 2)

    def test_default_is_all_non_wrist(self, builder, sample_tokens):
        """Omitting is_wrist_per_camera matches passing all-False."""
        vlm, pc_per_cam, action, proprio, embod_pc, embod_action = sample_tokens
        seq_default = builder(vlm, pc_per_cam, action, proprio, embod_pc, embod_action)
        seq_explicit = builder(
            vlm, pc_per_cam, action, proprio, embod_pc, embod_action,
            is_wrist_per_camera=[False] * K_CAMERAS,
        )
        assert torch.equal(seq_default.pc, seq_explicit.pc)

    def test_wrist_camera_uses_wrist_delimiters(self, builder, sample_tokens):
        vlm, pc_per_cam, action, proprio, embod_pc, embod_action = sample_tokens
        is_wrist = [False, True, False]  # only camera 1 is wrist
        seq = builder(
            vlm, pc_per_cam, action, proprio, embod_pc, embod_action,
            is_wrist_per_camera=is_wrist,
        )

        wrist_start_pos = self._start_pos(1)
        wrist_end_pos = wrist_start_pos + N_PC + 1
        expected_start = builder.pc_wrist_start_emb.expand(B, -1, -1)
        expected_end = builder.pc_wrist_end_emb.expand(B, -1, -1)
        assert torch.equal(seq.pc[:, wrist_start_pos : wrist_start_pos + 1, :], expected_start)
        assert torch.equal(seq.pc[:, wrist_end_pos : wrist_end_pos + 1, :], expected_end)

    def test_non_wrist_camera_keeps_base_delimiters(self, builder, sample_tokens):
        vlm, pc_per_cam, action, proprio, embod_pc, embod_action = sample_tokens
        is_wrist = [False, True, False]
        seq = builder(
            vlm, pc_per_cam, action, proprio, embod_pc, embod_action,
            is_wrist_per_camera=is_wrist,
        )

        for cam_idx in (0, 2):
            start_pos = self._start_pos(cam_idx)
            end_pos = start_pos + N_PC + 1
            expected_start = builder.pc_start_emb.expand(B, -1, -1)
            expected_end = builder.pc_end_emb.expand(B, -1, -1)
            assert torch.equal(seq.pc[:, start_pos : start_pos + 1, :], expected_start)
            assert torch.equal(seq.pc[:, end_pos : end_pos + 1, :], expected_end)

    def test_pc_token_payload_unchanged_by_wrist_flag(self, builder, sample_tokens):
        """Only the bracketing tokens differ — the camera's PC tokens themselves are identical."""
        vlm, pc_per_cam, action, proprio, embod_pc, embod_action = sample_tokens
        seq_base = builder(vlm, pc_per_cam, action, proprio, embod_pc, embod_action)
        seq_wrist = builder(
            vlm, pc_per_cam, action, proprio, embod_pc, embod_action,
            is_wrist_per_camera=[True] * K_CAMERAS,
        )
        for cam_idx in range(K_CAMERAS):
            start_pos = self._start_pos(cam_idx) + 1  # past the start delimiter
            end_pos = start_pos + N_PC
            assert torch.equal(seq_base.pc[:, start_pos:end_pos, :], pc_per_cam[cam_idx])
            assert torch.equal(seq_wrist.pc[:, start_pos:end_pos, :], pc_per_cam[cam_idx])

    def test_wrist_and_base_delimiters_are_distinct(self, builder):
        """Random init should make the two pairs different (otherwise the model can't tell)."""
        assert not torch.equal(builder.pc_start_emb, builder.pc_wrist_start_emb)
        assert not torch.equal(builder.pc_end_emb, builder.pc_wrist_end_emb)

    def test_wrist_delimiters_shared_across_wrist_cameras(self, builder, sample_tokens):
        """Two wrist cameras use the same pc_wrist_start / pc_wrist_end embeddings."""
        vlm, pc_per_cam, action, proprio, embod_pc, embod_action = sample_tokens
        is_wrist = [True, False, True]
        seq = builder(
            vlm, pc_per_cam, action, proprio, embod_pc, embod_action,
            is_wrist_per_camera=is_wrist,
        )

        cam0_start = seq.pc[:, self._start_pos(0) : self._start_pos(0) + 1, :]
        cam2_start = seq.pc[:, self._start_pos(2) : self._start_pos(2) + 1, :]
        assert torch.equal(cam0_start, cam2_start)

    def test_wrist_emb_trainable(self, builder):
        assert builder.pc_wrist_start_emb.requires_grad
        assert builder.pc_wrist_end_emb.requires_grad


class TestProprioDelimiters:
    """<proprio_start> / <proprio_end> wrap the (variable-length) proprio block."""

    @pytest.fixture
    def builder(self):
        return SequenceBuilder(d_pc=D_PC, d_action=D_ACTION)

    @pytest.fixture
    def sample_tokens(self):
        vlm = torch.randn(B, N_VLM, D_VLM)
        pc_per_cam = [torch.randn(B, N_PC, D_PC) for _ in range(K_CAMERAS)]
        action = torch.randn(B, N_ACTION, D_ACTION)
        proprio = torch.randn(B, N_PROPRIO, D_ACTION)
        embod_pc, embod_action = _embodiment(B, D_PC, D_ACTION)
        return vlm, pc_per_cam, action, proprio, embod_pc, embod_action

    def test_proprio_start_after_embodiment_action(self, builder, sample_tokens):
        """sink_action(0), embodiment_action(1), proprio_start(2)."""
        vlm, pc_per_cam, action, proprio, embod_pc, embod_action = sample_tokens
        seq = builder(vlm, pc_per_cam, action, proprio, embod_pc, embod_action)
        expected = builder.proprio_start_emb.expand(B, -1, -1)
        assert torch.equal(seq.action[:, 2:3, :], expected)

    def test_proprio_end_after_proprio_block(self, builder, sample_tokens):
        vlm, pc_per_cam, action, proprio, embod_pc, embod_action = sample_tokens
        seq = builder(vlm, pc_per_cam, action, proprio, embod_pc, embod_action)
        # sink(1) + embodiment(1) + proprio_start(1) + P proprio = 3 + P, so proprio_end at 3 + P.
        end_idx = 3 + N_PROPRIO
        expected = builder.proprio_end_emb.expand(B, -1, -1)
        assert torch.equal(seq.action[:, end_idx : end_idx + 1, :], expected)

    def test_action_start_immediately_after_proprio_end(self, builder, sample_tokens):
        vlm, pc_per_cam, action, proprio, embod_pc, embod_action = sample_tokens
        seq = builder(vlm, pc_per_cam, action, proprio, embod_pc, embod_action)
        action_start_idx = N_PROPRIO + 4
        expected = builder.action_start_emb.expand(B, -1, -1)
        assert torch.equal(seq.action[:, action_start_idx : action_start_idx + 1, :], expected)

    def test_delimiters_distinct_from_action_pair_after_init(self, builder):
        """Random-init parameters should not collide with the action pair."""
        assert not torch.equal(builder.proprio_start_emb, builder.action_start_emb)
        assert not torch.equal(builder.proprio_end_emb, builder.action_end_emb)
        assert not torch.equal(builder.proprio_start_emb, builder.proprio_end_emb)

    def test_works_with_variable_p(self, builder):
        """For each P, <proprio_start> / <proprio_end> wrap exactly P proprio tokens."""
        vlm = torch.randn(B, N_VLM, D_VLM)
        pc_per_cam = [torch.randn(B, N_PC, D_PC) for _ in range(K_CAMERAS)]
        action = torch.randn(B, N_ACTION, D_ACTION)
        embod_pc, embod_action = _embodiment(B, D_PC, D_ACTION)
        for p in (1, 4, 8):
            proprio = torch.randn(B, p, D_ACTION)
            seq = builder(vlm, pc_per_cam, action, proprio, embod_pc, embod_action)
            # sink at 0, embod at 1, proprio_start at 2, proprio at [3, 3+p), proprio_end at 3+p.
            ps = builder.proprio_start_emb.expand(B, -1, -1)
            pe = builder.proprio_end_emb.expand(B, -1, -1)
            assert torch.equal(seq.action[:, 2:3, :], ps), f"P={p}: proprio_start misplaced"
            assert torch.equal(seq.action[:, 3 : 3 + p, :], proprio), f"P={p}: proprio body misplaced"
            assert torch.equal(seq.action[:, 3 + p : 4 + p, :], pe), f"P={p}: proprio_end misplaced"

    def test_proprio_delimiter_widths_match_d_action(self, builder):
        assert builder.proprio_start_emb.shape == (1, 1, D_ACTION)
        assert builder.proprio_end_emb.shape == (1, 1, D_ACTION)

    def test_proprio_delimiters_trainable(self, builder):
        assert builder.proprio_start_emb.requires_grad
        assert builder.proprio_end_emb.requires_grad

    def test_proprio_delimiters_carry_action_token_type(self, builder, sample_tokens):
        vlm, pc_per_cam, action, proprio, embod_pc, embod_action = sample_tokens
        seq = builder(vlm, pc_per_cam, action, proprio, embod_pc, embod_action)
        pc_total = 3 + K_CAMERAS * (N_PC + 2)
        action_segment_start = N_VLM + pc_total
        # All six delimiter / context positions should carry TokenType.ACTION:
        # sink(0), embodiment(1), proprio_start(2), proprio_end(3+P),
        # action_start(4+P), action_end(5+P+T).
        for offset in (0, 1, 2, 3 + N_PROPRIO, 4 + N_PROPRIO, 5 + N_PROPRIO + N_ACTION):
            assert seq.token_types[0, action_segment_start + offset] == TokenType.ACTION


class TestSinkTokens:
    """Per-expert attention sinks at position 0 of PC and Action segments.

    Sinks are content-free Parameters whose role is to absorb softmax mass
    that heads can't usefully place on real tokens (Xiao et al. 2023). PC
    and Action have separate sinks because their attention regimes differ
    (PC bidirectional, Action causal) and a "good sink direction" for one
    is not the same as for the other.
    """

    @pytest.fixture
    def builder(self):
        return SequenceBuilder(d_pc=D_PC, d_action=D_ACTION)

    @pytest.fixture
    def sample_tokens(self):
        vlm = torch.randn(B, N_VLM, D_VLM)
        pc_per_cam = [torch.randn(B, N_PC, D_PC) for _ in range(K_CAMERAS)]
        action = torch.randn(B, N_ACTION, D_ACTION)
        proprio = torch.randn(B, N_PROPRIO, D_ACTION)
        embod_pc, embod_action = _embodiment(B, D_PC, D_ACTION)
        return vlm, pc_per_cam, action, proprio, embod_pc, embod_action

    def test_sink_pc_at_pc_position_zero(self, builder, sample_tokens):
        vlm, pc_per_cam, action, proprio, embod_pc, embod_action = sample_tokens
        seq = builder(vlm, pc_per_cam, action, proprio, embod_pc, embod_action)
        expected = builder.sink_pc.expand(B, -1, -1)
        assert torch.equal(seq.pc[:, 0:1, :], expected)

    def test_sink_action_at_action_position_zero(self, builder, sample_tokens):
        vlm, pc_per_cam, action, proprio, embod_pc, embod_action = sample_tokens
        seq = builder(vlm, pc_per_cam, action, proprio, embod_pc, embod_action)
        expected = builder.sink_action.expand(B, -1, -1)
        assert torch.equal(seq.action[:, 0:1, :], expected)

    def test_sinks_are_separate_parameters(self, builder):
        """sink_pc and sink_action live at different widths and are NOT the same Parameter."""
        assert builder.sink_pc is not builder.sink_action
        assert builder.sink_pc.shape == (1, 1, D_PC)
        assert builder.sink_action.shape == (1, 1, D_ACTION)
        # Two distinct entries in the parameter table.
        ids = {id(builder.sink_pc), id(builder.sink_action)}
        assert len(ids) == 2

    def test_sinks_are_trainable(self, builder):
        assert builder.sink_pc.requires_grad
        assert builder.sink_action.requires_grad

    def test_sinks_distinct_from_embodiment_outputs(self, builder, sample_tokens):
        """Sinks are independent of the embodiment tokens (different positions, different content)."""
        vlm, pc_per_cam, action, proprio, embod_pc, embod_action = sample_tokens
        seq = builder(vlm, pc_per_cam, action, proprio, embod_pc, embod_action)
        # Position 0 must be the sink, position 1 must be embodiment, and they
        # must differ (since embodiment is random and sink is random — collisions
        # would be astronomically unlikely with default Gaussian init).
        assert not torch.equal(seq.pc[:, 0:1, :], seq.pc[:, 1:2, :])
        assert not torch.equal(seq.action[:, 0:1, :], seq.action[:, 1:2, :])

    def test_sinks_carry_correct_token_type(self, builder, sample_tokens):
        vlm, pc_per_cam, action, proprio, embod_pc, embod_action = sample_tokens
        seq = builder(vlm, pc_per_cam, action, proprio, embod_pc, embod_action)
        pc_segment_start = N_VLM
        pc_total = 3 + K_CAMERAS * (N_PC + 2)
        action_segment_start = N_VLM + pc_total
        assert seq.token_types[0, pc_segment_start] == TokenType.PC
        assert seq.token_types[0, action_segment_start] == TokenType.ACTION

    def test_sink_value_actually_threads_through_segment(self, builder):
        """If we change sink_pc, the position-0 PC token in the output changes correspondingly."""
        vlm = torch.randn(B, N_VLM, D_VLM)
        pc_per_cam = [torch.randn(B, N_PC, D_PC) for _ in range(K_CAMERAS)]
        action = torch.randn(B, N_ACTION, D_ACTION)
        proprio = torch.randn(B, N_PROPRIO, D_ACTION)
        embod_pc, embod_action = _embodiment(B, D_PC, D_ACTION)

        seq_a = builder(vlm, pc_per_cam, action, proprio, embod_pc, embod_action)
        with torch.no_grad():
            builder.sink_pc.data.add_(1.0)  # bump every dim by 1
        seq_b = builder(vlm, pc_per_cam, action, proprio, embod_pc, embod_action)

        delta = seq_b.pc[:, 0:1, :] - seq_a.pc[:, 0:1, :]
        assert torch.allclose(delta, torch.ones_like(delta), atol=1e-6)
        # And only position 0 changes — everything else is identical.
        assert torch.equal(seq_a.pc[:, 1:, :], seq_b.pc[:, 1:, :])

    def test_sink_pc_gets_grad_from_pc_loss_path(self, builder):
        """Sanity: a loss touching the PC segment propagates gradient into sink_pc."""
        vlm = torch.randn(B, N_VLM, D_VLM)
        pc_per_cam = [torch.randn(B, N_PC, D_PC) for _ in range(K_CAMERAS)]
        action = torch.randn(B, N_ACTION, D_ACTION)
        proprio = torch.randn(B, N_PROPRIO, D_ACTION)
        embod_pc, embod_action = _embodiment(B, D_PC, D_ACTION)

        seq = builder(vlm, pc_per_cam, action, proprio, embod_pc, embod_action)
        seq.pc.sum().backward()

        assert builder.sink_pc.grad is not None
        assert builder.sink_pc.grad.abs().sum() > 0
        # sink_action shouldn't have received grad from a PC-only loss.
        assert builder.sink_action.grad is None or builder.sink_action.grad.abs().sum() == 0


class TestAlignmentToken:
    @pytest.fixture
    def align_mod(self):
        return AlignmentToken(d_pc=D_PC)

    def test_output_shape(self, align_mod):
        a = torch.tensor([0, 1])  # (B,)
        out = align_mod(a)
        assert out.shape == (B, 1, D_PC)

    def test_accepts_int_and_long(self, align_mod):
        a_int = torch.tensor([0, 1], dtype=torch.int32)
        a_long = torch.tensor([0, 1], dtype=torch.int64)
        assert torch.equal(align_mod(a_int), align_mod(a_long))

    def test_different_a_different_output(self, align_mod):
        a0 = torch.tensor([0, 0])
        a1 = torch.tensor([1, 1])
        out0 = align_mod(a0)
        out1 = align_mod(a1)
        assert not torch.equal(out0, out1)

    def test_deterministic(self, align_mod):
        a = torch.tensor([1, 1])
        out1 = align_mod(a)
        out2 = align_mod(a)
        assert torch.equal(out1, out2)

    def test_parameters_are_learnable(self, align_mod):
        param_names = {name for name, _ in align_mod.named_parameters()}
        assert "embed.weight" in param_names
        assert align_mod.embed.weight.requires_grad

    def test_two_embeddings(self, align_mod):
        assert align_mod.embed.num_embeddings == 2

    def test_integrates_with_sequence_builder(self, align_mod):
        """AlignmentToken output plugs into SequenceBuilder's align_emb arg."""
        builder = SequenceBuilder(d_pc=D_PC, d_action=D_ACTION)
        vlm = torch.randn(B, N_VLM, D_VLM)
        pc_per_cam = [torch.randn(B, N_PC, D_PC) for _ in range(K_CAMERAS)]
        action = torch.randn(B, N_ACTION, D_ACTION)
        proprio = torch.randn(B, N_PROPRIO, D_ACTION)
        embod_pc, embod_action = _embodiment(B, D_PC, D_ACTION)

        a = torch.tensor([1, 0])
        align_emb = align_mod(a)
        seq = builder(
            vlm, pc_per_cam, action, proprio, embod_pc, embod_action,
            align_emb=align_emb,
        )

        # <align> is at index 2 (sink_pc=0, embodiment_pc=1, align=2).
        assert torch.equal(seq.pc[:, 2:3, :], align_emb)


class TestInputProjections:
    @pytest.fixture
    def proj(self):
        return InputProjections(
            pc_encoder_dim=PC_ENCODER_DIM,
            d_pc=D_PC,
            action_dim=ACTION_DIM,
            d_action=D_ACTION,
        )

    def test_output_shapes(self, proj):
        vlm = torch.randn(B, N_VLM, D_VLM)
        pc = torch.randn(B, N_PC, PC_ENCODER_DIM)
        action = torch.randn(B, N_ACTION, ACTION_DIM)

        vlm_out, pc_out, action_out = proj(vlm, pc, action)

        assert vlm_out.shape == (B, N_VLM, D_VLM)
        assert pc_out.shape == (B, N_PC, D_PC)
        assert action_out.shape == (B, N_ACTION, D_ACTION)

    def test_vlm_passthrough(self, proj):
        """VLM tokens should be returned unchanged — no projection."""
        vlm = torch.randn(B, N_VLM, D_VLM)
        pc = torch.randn(B, N_PC, PC_ENCODER_DIM)
        action = torch.randn(B, N_ACTION, ACTION_DIM)

        vlm_out, _, _ = proj(vlm, pc, action)
        assert torch.equal(vlm_out, vlm)

    def test_vlm_no_grad_contribution(self, proj):
        """VLM path should not contribute to proj's gradients."""
        vlm = torch.randn(B, N_VLM, D_VLM, requires_grad=True)
        pc = torch.randn(B, N_PC, PC_ENCODER_DIM)
        action = torch.randn(B, N_ACTION, ACTION_DIM)

        vlm_out, _, _ = proj(vlm, pc, action)
        loss = vlm_out.sum()
        loss.backward()

        # No proj parameters should have received gradients from vlm path
        for name, param in proj.named_parameters():
            assert param.grad is None or param.grad.abs().sum() == 0, (
                f"{name} should have no grad from VLM path"
            )

    def test_pc_projection_trainable(self, proj):
        pc = torch.randn(B, N_PC, PC_ENCODER_DIM)
        vlm = torch.randn(B, N_VLM, D_VLM)
        action = torch.randn(B, N_ACTION, ACTION_DIM)

        _, pc_out, _ = proj(vlm, pc, action)
        loss = pc_out.sum()
        loss.backward()

        assert proj.pc_proj.weight.grad is not None
        assert proj.pc_proj.weight.grad.abs().sum() > 0

    def test_action_projection_trainable(self, proj):
        vlm = torch.randn(B, N_VLM, D_VLM)
        pc = torch.randn(B, N_PC, PC_ENCODER_DIM)
        action = torch.randn(B, N_ACTION, ACTION_DIM)

        _, _, action_out = proj(vlm, pc, action)
        loss = action_out.sum()
        loss.backward()

        assert proj.action_proj.weight.grad is not None
        assert proj.action_proj.weight.grad.abs().sum() > 0

    def test_pc_and_action_independent(self, proj):
        """PC and Action projections should have independent parameters."""
        pc_params = {id(p) for p in proj.pc_proj.parameters()}
        action_params = {id(p) for p in proj.action_proj.parameters()}
        assert pc_params.isdisjoint(action_params)


D_EMB = 32


class TestSharedEmbodimentEmbedding:
    """One nn.Embedding(3, d_emb) base + per-stream zero-init projections."""

    @pytest.fixture
    def embod(self):
        return SharedEmbodimentEmbedding(d_pc=D_PC, d_action=D_ACTION, d_emb=D_EMB)

    def test_num_entries(self, embod):
        assert embod.base.num_embeddings == 3

    def test_base_width(self, embod):
        assert embod.base.embedding_dim == D_EMB

    def test_output_shapes(self, embod):
        robot_id = torch.tensor([0, 1])
        embod_pc, embod_action = embod(robot_id)
        assert embod_pc.shape == (B, 1, D_PC)
        assert embod_action.shape == (B, 1, D_ACTION)

    def test_zero_init_projections_yield_zero_output(self, embod):
        """Recipe-2 init: at step 0 both outputs are exact zeros, so embodiment
        contributes nothing to the model on the first forward pass."""
        robot_id = torch.tensor([0, 1, 2])
        embod_pc, embod_action = embod(robot_id)
        assert torch.equal(embod_pc, torch.zeros_like(embod_pc))
        assert torch.equal(embod_action, torch.zeros_like(embod_action))

    def test_base_rows_distinct(self, embod):
        """Base table is randomly initialized — rows should differ."""
        rows = embod.base.weight
        assert not torch.equal(rows[0], rows[1])
        assert not torch.equal(rows[1], rows[2])
        assert not torch.equal(rows[0], rows[2])

    def test_accepts_int_and_long(self, embod):
        a_int = torch.tensor([0, 1], dtype=torch.int32)
        a_long = torch.tensor([0, 1], dtype=torch.int64)
        pc_a, act_a = embod(a_int)
        pc_b, act_b = embod(a_long)
        assert torch.equal(pc_a, pc_b)
        assert torch.equal(act_a, act_b)

    def test_all_three_modules_trainable(self, embod):
        for name, p in embod.named_parameters():
            assert p.requires_grad, name

    def test_gradient_flows_back_to_base_and_projections(self, embod):
        """Loss on either output should accumulate grad on base + the corresponding projection."""
        # After projection weights have been bumped off zero, grads flow normally.
        torch.nn.init.normal_(embod.to_pc.weight, std=1e-2)
        torch.nn.init.normal_(embod.to_action.weight, std=1e-2)
        embod_pc, embod_action = embod(torch.tensor([0, 1]))
        (embod_pc.sum() + embod_action.sum()).backward()
        assert embod.base.weight.grad is not None
        assert embod.base.weight.grad.abs().sum() > 0
        assert embod.to_pc.weight.grad is not None
        assert embod.to_pc.weight.grad.abs().sum() > 0
        assert embod.to_action.weight.grad is not None
        assert embod.to_action.weight.grad.abs().sum() > 0

    def test_pc_loss_does_not_train_action_projection(self, embod):
        """A loss touching only the PC stream output must not give grad to to_action."""
        torch.nn.init.normal_(embod.to_pc.weight, std=1e-2)
        torch.nn.init.normal_(embod.to_action.weight, std=1e-2)
        embod_pc, _ = embod(torch.tensor([0, 1]))
        embod_pc.sum().backward()
        assert embod.to_action.weight.grad is None or embod.to_action.weight.grad.abs().sum() == 0

    def test_robot_ids_dict(self):
        assert SharedEmbodimentEmbedding.ROBOT_IDS == {"trossen": 0, "franka": 1, "so-101": 2}
        assert SharedEmbodimentEmbedding.NUM_ROBOTS == 3

    def test_no_bias_on_projections(self, embod):
        """Projections use bias=False so the base row is the sole signal source."""
        assert embod.to_pc.bias is None
        assert embod.to_action.bias is None

    def test_d_emb_default(self):
        """Default d_emb is 32 — small enough to bottleneck, large enough to avoid degeneracy."""
        e = SharedEmbodimentEmbedding(d_pc=D_PC, d_action=D_ACTION)
        assert e.base.embedding_dim == 32

    def test_integrates_with_sequence_builder(self, embod):
        """SharedEmbodimentEmbedding output plugs straight into SequenceBuilder."""
        # Bump projections off zero so the spliced positions are visibly the embodiment outputs.
        torch.nn.init.normal_(embod.to_pc.weight, std=1.0)
        torch.nn.init.normal_(embod.to_action.weight, std=1.0)
        embod_pc, embod_action = embod(torch.tensor([0, 1]))

        builder = SequenceBuilder(d_pc=D_PC, d_action=D_ACTION)
        vlm = torch.randn(B, N_VLM, D_VLM)
        pc_per_cam = [torch.randn(B, N_PC, D_PC) for _ in range(K_CAMERAS)]
        action = torch.randn(B, N_ACTION, D_ACTION)
        proprio = torch.randn(B, N_PROPRIO, D_ACTION)

        seq = builder(vlm, pc_per_cam, action, proprio, embod_pc, embod_action)
        # With sinks at position 0, embodiment moves to position 1.
        assert torch.equal(seq.pc[:, 1:2, :], embod_pc)
        assert torch.equal(seq.action[:, 1:2, :], embod_action)


D_T = 32


class TestProprioEncoder:
    @pytest.fixture
    def encoder(self):
        return ProprioEncoder(proprio_dim=PROPRIO_DIM, d_action=D_ACTION)

    def test_output_shape_3d(self, encoder):
        proprio = torch.randn(B, 4, PROPRIO_DIM)
        out = encoder(proprio)
        assert out.shape == (B, 4, D_ACTION)

    def test_output_shape_2d_promotes_to_single_token(self, encoder):
        proprio = torch.randn(B, PROPRIO_DIM)
        out = encoder(proprio)
        assert out.shape == (B, 1, D_ACTION)

    def test_2d_matches_3d_with_unsqueeze(self, encoder):
        proprio_2d = torch.randn(B, PROPRIO_DIM)
        proprio_3d = proprio_2d.unsqueeze(1)
        assert torch.equal(encoder(proprio_2d), encoder(proprio_3d))

    def test_trainable(self, encoder):
        proprio = torch.randn(B, 1, PROPRIO_DIM)
        out = encoder(proprio)
        out.sum().backward()
        assert encoder.proj.weight.grad is not None
        assert encoder.proj.weight.grad.abs().sum() > 0

    def test_per_step_independent_projection(self, encoder):
        """Each proprio step is projected independently — Linear is shared, no mixing across time."""
        a = torch.randn(B, 3, PROPRIO_DIM)
        out_full = encoder(a)
        # Shuffling step order should permute the output the same way.
        perm = torch.tensor([2, 0, 1])
        out_perm = encoder(a[:, perm, :])
        assert torch.equal(out_full[:, perm, :], out_perm)


class TestAdaLNConditioner:
    @pytest.fixture
    def adaln(self):
        return AdaLNConditioner(d_t=D_T, d_action=D_ACTION)

    def test_output_shapes(self, adaln):
        """6-tuple: (s1, sh1, g1, s2, sh2, g2) — DiT adaLN-zero with explicit gates."""
        t_emb = torch.randn(B, D_T)
        outs = adaln(t_emb)
        assert len(outs) == 6
        for tensor in outs:
            assert tensor.shape == (B, D_ACTION)

    def test_zero_init_yields_zero_outputs(self, adaln):
        """Recipe (DiT adaLN-zero): final Linear is zero-init, so all six
        modulation parameters start at exactly zero — block is identity at step 0."""
        for t_emb in (torch.zeros(B, D_T), torch.ones(B, D_T), torch.randn(B, D_T)):
            outs = adaln(t_emb)
            for tensor in outs:
                assert torch.equal(tensor, torch.zeros_like(tensor)), \
                    "outputs should be exactly zero before any training step"

    def test_different_t_after_perturbing_weights(self, adaln):
        """After bumping weights off zero, different t inputs produce different outputs."""
        # Bump the final Linear off zero so the conditioner becomes a real function.
        with torch.no_grad():
            adaln.mlp[-1].weight.add_(torch.randn_like(adaln.mlp[-1].weight) * 0.1)
            adaln.mlp[-1].bias.add_(torch.randn_like(adaln.mlp[-1].bias) * 0.1)

        t0 = torch.zeros(B, D_T)
        t1 = torch.ones(B, D_T)
        out0 = adaln(t0)
        out1 = adaln(t1)
        # At least one of the six should differ between t0 and t1.
        assert any(not torch.equal(a, b) for a, b in zip(out0, out1))

    def test_trainable(self, adaln):
        t_emb = torch.randn(B, D_T)
        outs = adaln(t_emb)
        sum(o.sum() for o in outs).backward()

        # The final Linear should pick up gradient even with zero-init —
        # its weight gets gradient = silu_output, its bias gets gradient = 1.
        assert adaln.mlp[-1].weight.grad is not None
        assert adaln.mlp[-1].weight.grad.abs().sum() > 0
        assert adaln.mlp[-1].bias.grad is not None
        assert adaln.mlp[-1].bias.grad.abs().sum() > 0

    def test_dtype_device_match(self, adaln):
        t_emb = torch.randn(B, D_T)
        outs = adaln(t_emb)
        for tensor in outs:
            assert tensor.dtype == t_emb.dtype
            assert tensor.device == t_emb.device

    def test_signature_takes_only_timestep(self, adaln):
        """Regression: AdaLNConditioner forward must accept only t_emb."""
        import inspect
        sig = inspect.signature(adaln.forward)
        params = [p for p in sig.parameters if p != "self"]
        assert params == ["t_emb"], f"AdaLNConditioner.forward signature is {params}"


class TestModulate:
    def test_identity_with_zeros_zeros(self):
        """DiT form: (1+scale)*LN(x) + shift. scale=0, shift=0 gives plain LayerNorm."""
        d = D_ACTION
        norm = torch.nn.LayerNorm(d)
        x = torch.randn(B, N_ACTION, d)
        scale = torch.zeros(B, d)
        shift = torch.zeros(B, d)

        out = modulate(x, norm, scale, shift)
        expected = norm(x)
        assert torch.allclose(out, expected)

    def test_known_values(self):
        """scale=2, shift=3 should give (1+2)*LN(x) + 3 = 3*LN(x) + 3."""
        d = D_ACTION
        norm = torch.nn.LayerNorm(d)
        x = torch.randn(B, N_ACTION, d)
        scale = torch.full((B, d), 2.0)
        shift = torch.full((B, d), 3.0)

        out = modulate(x, norm, scale, shift)
        expected = 3.0 * norm(x) + 3.0
        assert torch.allclose(out, expected)

    def test_output_shape(self):
        d = D_ACTION
        norm = torch.nn.LayerNorm(d)
        x = torch.randn(B, N_ACTION, d)
        scale = torch.randn(B, d)
        shift = torch.randn(B, d)

        out = modulate(x, norm, scale, shift)
        assert out.shape == (B, N_ACTION, d)

    def test_grad_flows_through_x(self):
        d = D_ACTION
        norm = torch.nn.LayerNorm(d)
        x = torch.randn(B, N_ACTION, d, requires_grad=True)
        scale = torch.ones(B, d)
        shift = torch.zeros(B, d)

        out = modulate(x, norm, scale, shift)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_grad_flows_through_scale(self):
        d = D_ACTION
        norm = torch.nn.LayerNorm(d)
        x = torch.randn(B, N_ACTION, d)
        scale = torch.randn(B, d, requires_grad=True)
        shift = torch.zeros(B, d)

        out = modulate(x, norm, scale, shift)
        out.sum().backward()
        assert scale.grad is not None
        assert scale.grad.abs().sum() > 0

    def test_grad_flows_through_shift(self):
        d = D_ACTION
        norm = torch.nn.LayerNorm(d)
        x = torch.randn(B, N_ACTION, d)
        scale = torch.ones(B, d)
        shift = torch.randn(B, d, requires_grad=True)

        out = modulate(x, norm, scale, shift)
        out.sum().backward()
        assert shift.grad is not None
        assert shift.grad.abs().sum() > 0
