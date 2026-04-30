import torch
import pytest

from unified_vla.attention import ExpertQKV, build_attention_mask, detached_cross_modal_attention, _match_heads
from unified_vla.core import TokenType

# Test dimensions (small, CPU-friendly)
D_VLM = 256
D_PC = 128
D_ACTION = 128
HEAD_DIM = 32
B = 2
N = 10


class TestExpertQKV:
    @pytest.fixture
    def vlm_qkv(self):
        return ExpertQKV(d_expert=D_VLM, head_dim=HEAD_DIM)

    @pytest.fixture
    def pc_qkv(self):
        return ExpertQKV(d_expert=D_PC, head_dim=HEAD_DIM)

    @pytest.fixture
    def action_qkv(self):
        return ExpertQKV(d_expert=D_ACTION, head_dim=HEAD_DIM)

    def test_vlm_output_shapes(self, vlm_qkv):
        x = torch.randn(B, N, D_VLM)
        q, k, v = vlm_qkv(x)
        num_heads = D_VLM // HEAD_DIM  # 8
        assert q.shape == (B, num_heads, N, HEAD_DIM)
        assert k.shape == (B, num_heads, N, HEAD_DIM)
        assert v.shape == (B, num_heads, N, HEAD_DIM)

    def test_pc_output_shapes(self, pc_qkv):
        x = torch.randn(B, N, D_PC)
        q, k, v = pc_qkv(x)
        num_heads = D_PC // HEAD_DIM  # 4
        assert q.shape == (B, num_heads, N, HEAD_DIM)
        assert k.shape == (B, num_heads, N, HEAD_DIM)
        assert v.shape == (B, num_heads, N, HEAD_DIM)

    def test_action_output_shapes(self, action_qkv):
        x = torch.randn(B, N, D_ACTION)
        q, k, v = action_qkv(x)
        num_heads = D_ACTION // HEAD_DIM  # 4
        assert q.shape == (B, num_heads, N, HEAD_DIM)
        assert k.shape == (B, num_heads, N, HEAD_DIM)
        assert v.shape == (B, num_heads, N, HEAD_DIM)

    def test_head_counts(self):
        assert ExpertQKV(D_VLM, HEAD_DIM).num_heads == D_VLM // HEAD_DIM
        assert ExpertQKV(D_PC, HEAD_DIM).num_heads == D_PC // HEAD_DIM
        assert ExpertQKV(D_ACTION, HEAD_DIM).num_heads == D_ACTION // HEAD_DIM

    def test_independent_parameters(self):
        """Three expert QKV modules should have no shared parameters."""
        vlm = ExpertQKV(D_VLM, HEAD_DIM)
        pc = ExpertQKV(D_PC, HEAD_DIM)
        action = ExpertQKV(D_ACTION, HEAD_DIM)

        vlm_ids = {id(p) for p in vlm.parameters()}
        pc_ids = {id(p) for p in pc.parameters()}
        action_ids = {id(p) for p in action.parameters()}

        assert vlm_ids.isdisjoint(pc_ids)
        assert vlm_ids.isdisjoint(action_ids)
        assert pc_ids.isdisjoint(action_ids)

    def test_separate_q_k_v_projections(self, vlm_qkv):
        """Q, K, V should come from independent linear layers."""
        q_ids = {id(p) for p in vlm_qkv.q_proj.parameters()}
        k_ids = {id(p) for p in vlm_qkv.k_proj.parameters()}
        v_ids = {id(p) for p in vlm_qkv.v_proj.parameters()}

        assert q_ids.isdisjoint(k_ids)
        assert q_ids.isdisjoint(v_ids)
        assert k_ids.isdisjoint(v_ids)

    def test_invalid_dim_raises(self):
        with pytest.raises(AssertionError):
            ExpertQKV(d_expert=100, head_dim=32)  # 100 not divisible by 32


class TestBuildAttentionMask:
    @pytest.fixture
    def token_types(self):
        """[vlm, vlm, pc, pc, action, action] — batch of 1."""
        types = torch.tensor([[
            TokenType.VLM, TokenType.VLM,
            TokenType.PC, TokenType.PC,
            TokenType.ACTION, TokenType.ACTION,
        ]])
        return types

    def test_output_shape(self, token_types):
        mask = build_attention_mask(token_types)
        B, seq_len = token_types.shape
        assert mask.shape == (B, 1, seq_len, seq_len)

    def test_vlm_rows_block_pc_and_action(self, token_types):
        mask = build_attention_mask(token_types)
        m = mask[0, 0]  # (6, 6)

        # VLM rows (0, 1): should allow cols 0,1 (VLM), block 2,3 (PC) and 4,5 (Action)
        assert m[0, 0] and m[0, 1]
        assert not m[0, 2] and not m[0, 3]
        assert not m[0, 4] and not m[0, 5]
        assert m[1, 0] and m[1, 1]
        assert not m[1, 2] and not m[1, 3]
        assert not m[1, 4] and not m[1, 5]

    def test_pc_rows_allow_vlm_and_pc_block_action(self, token_types):
        mask = build_attention_mask(token_types)
        m = mask[0, 0]

        # PC rows (2, 3): allow cols 0,1 (VLM) + 2,3 (PC), block 4,5 (Action)
        for row in [2, 3]:
            assert m[row, 0] and m[row, 1]    # VLM
            assert m[row, 2] and m[row, 3]    # PC
            assert not m[row, 4] and not m[row, 5]  # Action

    def test_action_rows_allow_all(self, token_types):
        mask = build_attention_mask(token_types)
        m = mask[0, 0]

        # Action rows (4, 5): allow all columns
        for row in [4, 5]:
            for col in range(6):
                assert m[row, col], f"Action row {row} should attend to col {col}"

    def test_broadcastable_over_heads(self, token_types):
        mask = build_attention_mask(token_types)
        # dim 1 is 1, so it broadcasts over any number of heads
        assert mask.shape[1] == 1

    def test_batched(self):
        """Mask should work with batch size > 1."""
        types = torch.tensor([
            [TokenType.VLM, TokenType.PC, TokenType.ACTION],
            [TokenType.VLM, TokenType.PC, TokenType.ACTION],
        ])
        mask = build_attention_mask(types)
        assert mask.shape == (2, 1, 3, 3)
        # Both batch elements should be identical
        assert torch.equal(mask[0], mask[1])

    def test_vlm_self_bidirectional(self, token_types):
        mask = build_attention_mask(token_types)
        m = mask[0, 0]
        # VLM tokens attend to each other in both directions
        assert m[0, 1] and m[1, 0]

    def test_pc_self_bidirectional(self, token_types):
        mask = build_attention_mask(token_types)
        m = mask[0, 0]
        assert m[2, 3] and m[3, 2]

    def test_action_self_bidirectional(self, token_types):
        mask = build_attention_mask(token_types)
        m = mask[0, 0]
        assert m[4, 5] and m[5, 4]


N_VLM = 10
N_PC = 5
N_ACTION = 8


class TestMatchHeads:
    def test_same_heads_noop(self):
        x = torch.randn(B, 4, N, HEAD_DIM)
        out = _match_heads(x, 4)
        assert torch.equal(out, x)

    def test_reduce_heads(self):
        x = torch.randn(B, 8, N, HEAD_DIM)
        out = _match_heads(x, 4)
        assert out.shape == (B, 4, N, HEAD_DIM)

    def test_expand_heads(self):
        x = torch.randn(B, 4, N, HEAD_DIM)
        out = _match_heads(x, 8)
        assert out.shape == (B, 8, N, HEAD_DIM)

    def test_reduce_is_group_average(self):
        x = torch.randn(B, 8, N, HEAD_DIM)
        out = _match_heads(x, 4)
        # Group 0 should be average of heads 0 and 1
        expected = (x[:, 0] + x[:, 1]) / 2
        assert torch.allclose(out[:, 0], expected)


class TestDetachedCrossModalAttention:
    @pytest.fixture
    def qkv_modules(self):
        vlm_qkv = ExpertQKV(D_VLM, HEAD_DIM)
        pc_qkv = ExpertQKV(D_PC, HEAD_DIM)
        action_qkv = ExpertQKV(D_ACTION, HEAD_DIM)
        return vlm_qkv, pc_qkv, action_qkv

    @pytest.fixture
    def inputs(self):
        vlm_x = torch.randn(B, N_VLM, D_VLM)
        pc_x = torch.randn(B, N_PC, D_PC)
        action_x = torch.randn(B, N_ACTION, D_ACTION)
        return vlm_x, pc_x, action_x

    def test_output_shapes(self, qkv_modules, inputs):
        vlm_qkv, pc_qkv, action_qkv = qkv_modules
        vlm_x, pc_x, action_x = inputs

        vlm_q, vlm_k, vlm_v = vlm_qkv(vlm_x)
        pc_q, pc_k, pc_v = pc_qkv(pc_x)
        action_q, action_k, action_v = action_qkv(action_x)

        vlm_out, pc_out, action_out = detached_cross_modal_attention(
            vlm_q, vlm_k, vlm_v,
            pc_q, pc_k, pc_v,
            action_q, action_k, action_v,
        )

        h_vlm = D_VLM // HEAD_DIM
        h_pc = D_PC // HEAD_DIM
        h_action = D_ACTION // HEAD_DIM
        assert vlm_out.shape == (B, h_vlm, N_VLM, HEAD_DIM)
        assert pc_out.shape == (B, h_pc, N_PC, HEAD_DIM)
        assert action_out.shape == (B, h_action, N_ACTION, HEAD_DIM)

    def test_vlm_gets_no_grad_from_pc_loss(self, qkv_modules, inputs):
        vlm_qkv, pc_qkv, action_qkv = qkv_modules
        vlm_x, pc_x, action_x = inputs

        vlm_q, vlm_k, vlm_v = vlm_qkv(vlm_x)
        pc_q, pc_k, pc_v = pc_qkv(pc_x)
        action_q, action_k, action_v = action_qkv(action_x)

        _, pc_out, _ = detached_cross_modal_attention(
            vlm_q, vlm_k, vlm_v,
            pc_q, pc_k, pc_v,
            action_q, action_k, action_v,
        )

        pc_loss = pc_out.sum()
        pc_loss.backward()

        for name, p in vlm_qkv.named_parameters():
            assert p.grad is None or p.grad.abs().sum() == 0, (
                f"VLM {name} should get no grad from PC loss"
            )

    def test_vlm_gets_no_grad_from_action_loss(self, qkv_modules, inputs):
        vlm_qkv, pc_qkv, action_qkv = qkv_modules
        vlm_x, pc_x, action_x = inputs

        vlm_q, vlm_k, vlm_v = vlm_qkv(vlm_x)
        pc_q, pc_k, pc_v = pc_qkv(pc_x)
        action_q, action_k, action_v = action_qkv(action_x)

        _, _, action_out = detached_cross_modal_attention(
            vlm_q, vlm_k, vlm_v,
            pc_q, pc_k, pc_v,
            action_q, action_k, action_v,
        )

        action_loss = action_out.sum()
        action_loss.backward()

        for name, p in vlm_qkv.named_parameters():
            assert p.grad is None or p.grad.abs().sum() == 0, (
                f"VLM {name} should get no grad from Action loss"
            )

    def test_pc_gets_grad_from_action_loss(self, qkv_modules, inputs):
        """Stop-gradient PC -> Action was removed: action loss must flow into PC."""
        vlm_qkv, pc_qkv, action_qkv = qkv_modules
        vlm_x, pc_x, action_x = inputs

        vlm_q, vlm_k, vlm_v = vlm_qkv(vlm_x)
        pc_q, pc_k, pc_v = pc_qkv(pc_x)
        action_q, action_k, action_v = action_qkv(action_x)

        _, _, action_out = detached_cross_modal_attention(
            vlm_q, vlm_k, vlm_v,
            pc_q, pc_k, pc_v,
            action_q, action_k, action_v,
        )

        action_loss = action_out.sum()
        action_loss.backward()

        # Specifically the K and V projections (consumed by Action's cross-attn)
        # must receive gradient.
        assert pc_qkv.k_proj.weight.grad is not None
        assert pc_qkv.k_proj.weight.grad.abs().sum() > 0
        assert pc_qkv.v_proj.weight.grad is not None
        assert pc_qkv.v_proj.weight.grad.abs().sum() > 0

    def test_action_gets_grad_from_action_loss(self, qkv_modules, inputs):
        vlm_qkv, pc_qkv, action_qkv = qkv_modules
        vlm_x, pc_x, action_x = inputs

        vlm_q, vlm_k, vlm_v = vlm_qkv(vlm_x)
        pc_q, pc_k, pc_v = pc_qkv(pc_x)
        action_q, action_k, action_v = action_qkv(action_x)

        _, _, action_out = detached_cross_modal_attention(
            vlm_q, vlm_k, vlm_v,
            pc_q, pc_k, pc_v,
            action_q, action_k, action_v,
        )

        action_loss = action_out.sum()
        action_loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in action_qkv.parameters()
        )
        assert has_grad, "Action QKV should get grad from Action loss"

    def test_pc_gets_grad_from_pc_loss(self, qkv_modules, inputs):
        vlm_qkv, pc_qkv, action_qkv = qkv_modules
        vlm_x, pc_x, action_x = inputs

        vlm_q, vlm_k, vlm_v = vlm_qkv(vlm_x)
        pc_q, pc_k, pc_v = pc_qkv(pc_x)
        action_q, action_k, action_v = action_qkv(action_x)

        _, pc_out, _ = detached_cross_modal_attention(
            vlm_q, vlm_k, vlm_v,
            pc_q, pc_k, pc_v,
            action_q, action_k, action_v,
        )

        pc_loss = pc_out.sum()
        pc_loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in pc_qkv.parameters()
        )
        assert has_grad, "PC QKV should get grad from PC loss"
