"""Tests for CentroidPosEmbed.

CentroidPosEmbed mirrors Uni3D's pos_embed recipe (Linear(3, 128) -> GELU ->
Linear(128, d_pc)) but applied at PC-expert entry. Final Linear is zero-init
so the step-0 output is exactly zero — additive contribution to projected PC
tokens vanishes at initialization, preserving the VLM-init prior on PC
ExpertBlocks. Gradient must flow back to centroids and to all parameters
(both fc layers).
"""

import torch
import torch.nn as nn

from unified_vla.core import CentroidPosEmbed


class TestCentroidPosEmbedShape:
    def test_default_hidden(self):
        m = CentroidPosEmbed(d_pc=1024)
        assert isinstance(m.fc1, nn.Linear)
        assert m.fc1.in_features == 3
        assert m.fc1.out_features == 128
        assert m.fc2.in_features == 128
        assert m.fc2.out_features == 1024

    def test_custom_hidden(self):
        m = CentroidPosEmbed(d_pc=512, hidden=64)
        assert m.fc1.out_features == 64
        assert m.fc2.in_features == 64
        assert m.fc2.out_features == 512

    def test_forward_shape_single_camera(self):
        m = CentroidPosEmbed(d_pc=1024)
        centroids = torch.randn(2, 32, 3)
        out = m(centroids)
        assert out.shape == (2, 32, 1024)

    def test_forward_shape_varying_M(self):
        """Per-camera chunks may have different M; module is M-agnostic."""
        m = CentroidPosEmbed(d_pc=256)
        for M in (1, 8, 64, 256):
            centroids = torch.randn(3, M, 3)
            out = m(centroids)
            assert out.shape == (3, M, 256)


class TestCentroidPosEmbedZeroInit:
    """At step 0 the module is the constant-zero map.

    Both fc2.weight and fc2.bias are zero → output is exactly the zero
    tensor for any input centroids. This is the adaLN-zero recipe applied
    to additive positional injection: the projected PC tokens entering
    SequenceBuilder are bit-for-bit identical to the no-centroid baseline
    until gradient lifts fc2 off zero.
    """

    def test_fc2_initialised_to_zero(self):
        m = CentroidPosEmbed(d_pc=1024)
        assert torch.equal(m.fc2.weight, torch.zeros_like(m.fc2.weight))
        assert torch.equal(m.fc2.bias, torch.zeros_like(m.fc2.bias))

    def test_output_is_zero_at_init(self):
        m = CentroidPosEmbed(d_pc=1024)
        centroids = torch.randn(4, 16, 3) * 10  # arbitrary scale
        out = m(centroids)
        assert torch.equal(out, torch.zeros_like(out))

    def test_output_zero_for_any_hidden_width(self):
        for hidden in (16, 64, 128, 256):
            m = CentroidPosEmbed(d_pc=64, hidden=hidden)
            centroids = torch.randn(2, 8, 3)
            assert torch.equal(m(centroids), torch.zeros(2, 8, 64))


class TestCentroidPosEmbedGradient:
    def test_gradient_flows_to_centroids(self):
        m = CentroidPosEmbed(d_pc=64)
        # Lift fc2 off zero so a gradient signal can reach the input.
        with torch.no_grad():
            m.fc2.weight.normal_(0, 0.02)
            m.fc2.bias.normal_(0, 0.02)
        centroids = torch.randn(2, 8, 3, requires_grad=True)
        out = m(centroids)
        out.sum().backward()
        assert centroids.grad is not None
        assert centroids.grad.shape == centroids.shape
        assert centroids.grad.abs().sum() > 0

    def test_gradient_flows_to_all_parameters(self):
        """At step 0 the loss surface is flat through fc2, but fc1 still
        receives gradient via the GELU activation only when fc2 is non-zero.
        Verify with non-zero fc2 (i.e., simulating a step 1+ state)."""
        m = CentroidPosEmbed(d_pc=64)
        with torch.no_grad():
            m.fc2.weight.normal_(0, 0.02)
            m.fc2.bias.normal_(0, 0.02)
        centroids = torch.randn(2, 8, 3)
        out = m(centroids)
        out.sum().backward()
        for name, p in m.named_parameters():
            assert p.grad is not None, f"{name} has no grad"
            assert p.grad.abs().sum() > 0, f"{name} grad is all zero"

    def test_no_grad_to_fc1_at_step_zero(self):
        """At init, fc2.weight = 0, so the gradient flowing back through
        fc2 to GELU(fc1(x)) is zero — fc1 receives zero gradient on step 0.
        This is expected and harmless: fc2 lifts off zero first, then fc1
        starts learning. Same dynamics as adaLN-zero gates."""
        m = CentroidPosEmbed(d_pc=64)
        centroids = torch.randn(2, 8, 3)
        out = m(centroids)
        out.sum().backward()
        # fc2.bias gets grad from sum-reduction — every output dim gets +1
        assert m.fc2.bias.grad.abs().sum() > 0
        # fc1 gets zero grad because fc2.weight = 0
        assert torch.equal(m.fc1.weight.grad, torch.zeros_like(m.fc1.weight))
        assert torch.equal(m.fc1.bias.grad, torch.zeros_like(m.fc1.bias))


class TestCentroidPosEmbedAdditive:
    """Verify the intended use pattern: add to projected PC tokens."""

    def test_addition_preserves_pc_tokens_at_init(self):
        d_pc = 128
        pos = CentroidPosEmbed(d_pc=d_pc)
        pc_proj = nn.Linear(1024, d_pc)
        encoder_features = torch.randn(2, 32, 1024)
        centroids = torch.randn(2, 32, 3)
        baseline = pc_proj(encoder_features)
        with_pos = pc_proj(encoder_features) + pos(centroids)
        # At step 0, pos(centroids) = 0, so the sum equals the baseline exactly.
        assert torch.equal(with_pos, baseline)

    def test_shared_across_cameras(self):
        """One module applied to per-camera centroids in a loop."""
        pos = CentroidPosEmbed(d_pc=64)
        # Simulate 3 cameras with different M_k.
        centroids_per_cam = [torch.randn(2, M, 3) for M in (16, 24, 8)]
        outputs = [pos(c) for c in centroids_per_cam]
        for out, c in zip(outputs, centroids_per_cam):
            assert out.shape == (c.shape[0], c.shape[1], 64)
        # All come from the same parameters.
        params_a = list(pos.parameters())
        params_b = list(pos.parameters())
        for pa, pb in zip(params_a, params_b):
            assert pa is pb
