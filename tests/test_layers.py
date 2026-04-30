import torch
import pytest

from unified_vla.layers import ExpertBlock, init_expert_from_vlm

# Test dimensions (small, CPU-friendly)
D_VLM = 256
D_PC = 128
D_ACTION = 128
D_EXPERT = 128
D_COND = 64
HEAD_DIM = 32
B = 2
N = 8
N_VLM = 10
N_PC = 5
N_ACTION = 8


class TestExpertBlock:
    @pytest.fixture
    def block(self):
        return ExpertBlock(d_expert=D_EXPERT, head_dim=HEAD_DIM)

    def test_output_shape(self, block):
        x = torch.randn(B, N, D_EXPERT)
        out, self_k, self_v = block(x)
        assert out.shape == (B, N, D_EXPERT)

    def test_kv_output_shape(self, block):
        x = torch.randn(B, N, D_EXPERT)
        _, self_k, self_v = block(x)
        num_heads = D_EXPERT // HEAD_DIM
        assert self_k.shape == (B, num_heads, N, HEAD_DIM)
        assert self_v.shape == (B, num_heads, N, HEAD_DIM)

    def test_residual_changes_output(self, block):
        """Output should differ from just the MLP output (residual connection active)."""
        x = torch.randn(B, N, D_EXPERT)
        out, _, _ = block(x)
        # If residual were absent, output would be purely from MLP.
        # With residual, output includes the input x.
        assert not torch.equal(out, x)

    def test_plain_layernorm_without_adaln(self, block):
        """Without adaln_mod, block should use plain LayerNorm."""
        x = torch.randn(B, N, D_EXPERT)
        out1, _, _ = block(x)
        out2, _, _ = block(x)
        # Deterministic — same input, same output
        assert torch.equal(out1, out2)

    def test_adaln_changes_output(self, block):
        """With adaln_mod provided, output should differ from without."""
        x = torch.randn(B, N, D_EXPERT)

        out_plain, _, _ = block(x, adaln_mod=None)

        # 6-tuple: (s1, sh1, g1, s2, sh2, g2). Non-zero gates ensure the
        # modulated path actually contributes (zero gates would make the
        # block identity, matching `out_plain` at zero modulation).
        s1 = torch.ones(B, D_EXPERT) * 2.0
        sh1 = torch.ones(B, D_EXPERT) * 0.5
        g1 = torch.ones(B, D_EXPERT)
        s2 = torch.ones(B, D_EXPERT) * 2.0
        sh2 = torch.ones(B, D_EXPERT) * 0.5
        g2 = torch.ones(B, D_EXPERT)
        out_adaln, _, _ = block(x, adaln_mod=(s1, sh1, g1, s2, sh2, g2))

        assert not torch.equal(out_plain, out_adaln)

    def test_different_adaln_different_output(self, block):
        """Different adaLN modulations should produce different outputs."""
        x = torch.randn(B, N, D_EXPERT)

        # 6-tuple modulations with non-zero gates so the modulated path
        # is actually live.
        mod_a = (
            torch.ones(B, D_EXPERT),       # s1
            torch.zeros(B, D_EXPERT),      # sh1
            torch.ones(B, D_EXPERT),       # g1
            torch.ones(B, D_EXPERT),       # s2
            torch.zeros(B, D_EXPERT),      # sh2
            torch.ones(B, D_EXPERT),       # g2
        )
        mod_b = (
            torch.ones(B, D_EXPERT) * 3.0,
            torch.ones(B, D_EXPERT),
            torch.ones(B, D_EXPERT),
            torch.ones(B, D_EXPERT) * 3.0,
            torch.ones(B, D_EXPERT),
            torch.ones(B, D_EXPERT),
        )

        out_a, _, _ = block(x, adaln_mod=mod_a)
        out_b, _, _ = block(x, adaln_mod=mod_b)
        assert not torch.equal(out_a, out_b)

    def test_zero_mod_is_identity_path(self, block):
        """With all-zero adaLN params (DiT zero-init recipe), the modulated
        block adds the LN-attn-MLP delta only; gates=0 means it should match
        the no-residual reference: just `x` (residual stream unchanged)."""
        x = torch.randn(B, N, D_EXPERT)

        zeros = torch.zeros(B, D_EXPERT)
        # All six params zero: scales=0 → identity-LN, gates=0 → no residual delta.
        out_zero_mod, _, _ = block(
            x, adaln_mod=(zeros, zeros, zeros, zeros, zeros, zeros)
        )

        assert torch.allclose(out_zero_mod, x, atol=1e-5)

    def test_external_kv_attention(self, block):
        """Block should attend to external K/V when provided."""
        x = torch.randn(B, N, D_EXPERT)
        num_heads = D_EXPERT // HEAD_DIM
        N_ext = 5
        ext_k = torch.randn(B, num_heads, N_ext, HEAD_DIM)
        ext_v = torch.randn(B, num_heads, N_ext, HEAD_DIM)

        out_self, _, _ = block(x)
        out_cross, _, _ = block(x, ext_k=ext_k, ext_v=ext_v)

        # Attending to extra K/V should change the output
        assert not torch.equal(out_self, out_cross)

    def test_parameter_count(self, block):
        """Verify expected components exist in parameters."""
        param_names = {name for name, _ in block.named_parameters()}

        # QKV
        assert "qkv.q_proj.weight" in param_names
        assert "qkv.k_proj.weight" in param_names
        assert "qkv.v_proj.weight" in param_names

        # Output projection
        assert "o_proj.weight" in param_names

        # Two LayerNorms
        assert "norm1.weight" in param_names
        assert "norm2.weight" in param_names

        # GatedMLP (gate_proj, up_proj, down_proj)
        assert "mlp.gate_proj.weight" in param_names
        assert "mlp.up_proj.weight" in param_names
        assert "mlp.down_proj.weight" in param_names

    def test_grad_flows(self, block):
        x = torch.randn(B, N, D_EXPERT)
        out, _, _ = block(x)
        out.sum().backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in block.parameters()
        )
        assert has_grad

    def test_self_kv_is_pre_concatenation(self, block):
        """Returned self_k/self_v should be from self QKV only, not including ext."""
        x = torch.randn(B, N, D_EXPERT)
        num_heads = D_EXPERT // HEAD_DIM
        ext_k = torch.randn(B, num_heads, 5, HEAD_DIM)
        ext_v = torch.randn(B, num_heads, 5, HEAD_DIM)

        _, self_k, self_v = block(x, ext_k=ext_k, ext_v=ext_v)

        # self_k should have N positions (self only), not N + 5
        assert self_k.shape[2] == N
        assert self_v.shape[2] == N


class TestInitExpertFromVLM:
    """Step 12: Expert weight initialization from VLM."""

    @pytest.fixture
    def vlm_block(self):
        """VLM block with d_vlm=256, 8 heads."""
        torch.manual_seed(42)
        return ExpertBlock(d_expert=D_VLM, head_dim=HEAD_DIM)

    @pytest.fixture
    def pc_expert(self):
        """PC expert with d_pc=128, 4 heads — to be initialized from VLM."""
        return ExpertBlock(d_expert=D_PC, head_dim=HEAD_DIM)

    def test_qkv_weight_slices_match(self, vlm_block, pc_expert):
        init_expert_from_vlm(pc_expert, vlm_block)

        d = D_PC
        for proj_name in ("q_proj", "k_proj", "v_proj"):
            src = getattr(vlm_block.qkv, proj_name).weight
            dst = getattr(pc_expert.qkv, proj_name).weight
            assert torch.equal(dst, src[:d, :d]), f"{proj_name} weights should match VLM slice"

    def test_o_proj_weight_slice(self, vlm_block, pc_expert):
        init_expert_from_vlm(pc_expert, vlm_block)
        d = D_PC
        assert torch.equal(pc_expert.o_proj.weight, vlm_block.o_proj.weight[:d, :d])

    def test_layernorm_weights_copied(self, vlm_block, pc_expert):
        init_expert_from_vlm(pc_expert, vlm_block)
        d = D_PC
        assert torch.equal(pc_expert.norm1.weight, vlm_block.norm1.weight[:d])
        assert torch.equal(pc_expert.norm2.weight, vlm_block.norm2.weight[:d])

    def test_mlp_dims(self, vlm_block, pc_expert):
        init_expert_from_vlm(pc_expert, vlm_block)
        d = D_PC
        d_inner = d * 4

        # gate_proj and up_proj: (d_inner, d)
        assert pc_expert.mlp.gate_proj.weight.shape == (d_inner, d)
        assert pc_expert.mlp.up_proj.weight.shape == (d_inner, d)
        # down_proj: (d, d_inner)
        assert pc_expert.mlp.down_proj.weight.shape == (d, d_inner)

    def test_mlp_weight_slices_match(self, vlm_block, pc_expert):
        init_expert_from_vlm(pc_expert, vlm_block)
        d = D_PC
        d_inner = d * 4

        assert torch.equal(
            pc_expert.mlp.gate_proj.weight,
            vlm_block.mlp.gate_proj.weight[:d_inner, :d],
        )
        assert torch.equal(
            pc_expert.mlp.up_proj.weight,
            vlm_block.mlp.up_proj.weight[:d_inner, :d],
        )
        assert torch.equal(
            pc_expert.mlp.down_proj.weight,
            vlm_block.mlp.down_proj.weight[:d, :d_inner],
        )

    def test_full_width_is_identity(self):
        """When d_expert == d_vlm, init should be an exact copy."""
        torch.manual_seed(42)
        vlm = ExpertBlock(d_expert=D_VLM, head_dim=HEAD_DIM)
        expert = ExpertBlock(d_expert=D_VLM, head_dim=HEAD_DIM)
        init_expert_from_vlm(expert, vlm)

        for (n1, p1), (n2, p2) in zip(vlm.named_parameters(), expert.named_parameters()):
            assert torch.equal(p1, p2), f"{n1} should be identical at full width"

    def test_forward_works_after_init(self, vlm_block, pc_expert):
        """Expert should produce valid output after initialization."""
        init_expert_from_vlm(pc_expert, vlm_block)
        x = torch.randn(B, N, D_PC)
        out, k, v = pc_expert(x)
        assert out.shape == (B, N, D_PC)
        assert not torch.isnan(out).any()
