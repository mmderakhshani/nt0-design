import torch
import pytest

from unified_vla.losses import (
    TimestepEmbedding, noise_action, flow_matching_loss, pc_mse_loss,
)

# Test dimensions
D_VLM = 256
D_PC = 128
D_ACTION = 128
HEAD_DIM = 32
D_COND = 64
D_T = 64
NUM_LAYERS = 2
B = 2
N_VLM = 10
N_PC = 5
N_ACTION = 8
ACTION_DIM = 10


class TestTimestepEmbedding:
    @pytest.fixture
    def t_emb(self):
        return TimestepEmbedding(d_sinusoidal=128, d_out=D_T)

    def test_output_shape(self, t_emb):
        t = torch.rand(B)
        out = t_emb(t)
        assert out.shape == (B, D_T)

    def test_different_t_different_output(self, t_emb):
        t0 = torch.tensor([0.0, 0.0])
        t1 = torch.tensor([1.0, 1.0])
        assert not torch.equal(t_emb(t0), t_emb(t1))

    def test_deterministic(self, t_emb):
        t = torch.tensor([0.5, 0.3])
        assert torch.equal(t_emb(t), t_emb(t))

    def test_trainable(self, t_emb):
        t = torch.rand(B)
        out = t_emb(t)
        out.sum().backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in t_emb.parameters())
        assert has_grad


class TestNoiseAction:
    def test_output_shapes(self):
        action_gt = torch.randn(B, N_ACTION, ACTION_DIM)
        noisy, t, v_target = noise_action(action_gt)

        assert noisy.shape == action_gt.shape
        assert t.shape == (B,)
        assert v_target.shape == action_gt.shape

    def test_t_in_range(self):
        action_gt = torch.randn(B, N_ACTION, ACTION_DIM)
        _, t, _ = noise_action(action_gt)
        assert (t >= 0).all() and (t <= 1).all()

    def test_custom_t(self):
        action_gt = torch.randn(B, N_ACTION, ACTION_DIM)
        t_custom = torch.tensor([0.5, 0.5])
        _, t_out, _ = noise_action(action_gt, t=t_custom)
        assert torch.equal(t_out, t_custom)

    def test_deterministic_with_seed(self):
        action_gt = torch.randn(B, N_ACTION, ACTION_DIM)
        t = torch.tensor([0.5, 0.5])

        torch.manual_seed(0)
        noisy1, _, v1 = noise_action(action_gt, t=t)
        torch.manual_seed(0)
        noisy2, _, v2 = noise_action(action_gt, t=t)

        assert torch.equal(noisy1, noisy2)
        assert torch.equal(v1, v2)

    def test_t0_is_pure_noise(self):
        """At t=0, x_t should equal the noise (no data component)."""
        action_gt = torch.randn(B, N_ACTION, ACTION_DIM)
        t = torch.zeros(B)
        torch.manual_seed(42)
        noisy, _, v_target = noise_action(action_gt, t=t)

        # x_t = (1-0)*noise + 0*action_gt = noise
        # v_target = action_gt - noise → noise = action_gt - v_target
        reconstructed_noise = action_gt - v_target
        assert torch.allclose(noisy, reconstructed_noise, atol=1e-6)

    def test_t1_is_pure_data(self):
        """At t=1, x_t should equal the ground truth action."""
        action_gt = torch.randn(B, N_ACTION, ACTION_DIM)
        t = torch.ones(B)
        noisy, _, _ = noise_action(action_gt, t=t)
        assert torch.allclose(noisy, action_gt)


class TestFlowMatchingLoss:
    def test_positive_scalar(self):
        pred = torch.randn(B, N_ACTION, ACTION_DIM)
        target = torch.randn(B, N_ACTION, ACTION_DIM)
        loss = flow_matching_loss(pred, target)

        assert loss.dim() == 0  # scalar
        assert loss.item() > 0

    def test_zero_when_perfect(self):
        pred = torch.randn(B, N_ACTION, ACTION_DIM)
        loss = flow_matching_loss(pred, pred)
        assert loss.item() == 0.0

    def test_loss_decreases_with_optimizer(self):
        """On a repeated trivial batch, loss should decrease after one step."""
        torch.manual_seed(42)

        # Simple linear model as stand-in
        model = torch.nn.Linear(ACTION_DIM, ACTION_DIM)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        action_gt = torch.randn(1, N_ACTION, ACTION_DIM)
        t = torch.tensor([0.5])

        torch.manual_seed(0)
        noisy, _, v_target = noise_action(action_gt, t=t)

        # Step 1
        pred = model(noisy)
        loss1 = flow_matching_loss(pred, v_target)
        loss1.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Step 2 (same data)
        torch.manual_seed(0)
        noisy, _, v_target = noise_action(action_gt, t=t)
        pred = model(noisy)
        loss2 = flow_matching_loss(pred, v_target)

        assert loss2.item() < loss1.item()


class TestPCMSELoss:
    def test_positive_scalar(self):
        pred = torch.randn(B, N_PC, D_PC)
        target = torch.randn(B, N_PC, D_PC)
        loss = pc_mse_loss(pred, target)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_zero_when_perfect(self):
        pred = torch.randn(B, N_PC, D_PC)
        loss = pc_mse_loss(pred, pred)
        assert loss.item() == 0.0

    def test_loss_decreases_with_optimizer(self):
        torch.manual_seed(42)
        model = torch.nn.Linear(D_PC, D_PC)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        pc_input = torch.randn(1, N_PC, D_PC)
        pc_target = torch.randn(1, N_PC, D_PC)

        pred1 = model(pc_input)
        loss1 = pc_mse_loss(pred1, pc_target)
        loss1.backward()
        optimizer.step()
        optimizer.zero_grad()

        pred2 = model(pc_input)
        loss2 = pc_mse_loss(pred2, pc_target)
        assert loss2.item() < loss1.item()


