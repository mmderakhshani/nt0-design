"""Tests for the Backbone with real Qwen2.5-VL weights.

These tests require the Qwen model to be available. For fast unit tests
of individual components (ExpertBlock, attention, etc.), see the other test files.
"""
import torch
import pytest

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from unified_vla.surgery import create_backbone


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
class TestBackboneWithQwen:
    MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
    D_PC = 1024
    D_ACTION = 1024
    D_COND = 64
    NUM_LAYERS = 2
    B = 1
    N_VLM = 5
    N_PC = 3
    N_ACTION = 4
    D_VLM = 2048

    @pytest.fixture(scope="class")
    def backbone(self):
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.MODEL_ID, dtype=torch.float32,
        )
        return create_backbone(
            model, d_pc=self.D_PC, d_action=self.D_ACTION,
            d_cond=self.D_COND, num_layers=self.NUM_LAYERS,
        )

    def test_num_layers(self, backbone):
        assert len(backbone.layers) == self.NUM_LAYERS

    def test_vlm_frozen(self, backbone):
        for layer in backbone.layers:
            for name, p in layer.vlm_layer.named_parameters():
                assert not p.requires_grad, f"VLM {name} should be frozen"

    def test_experts_trainable(self, backbone):
        for layer in backbone.layers:
            assert any(p.requires_grad for p in layer.pc_block.parameters())
            assert any(p.requires_grad for p in layer.action_block.parameters())

    def test_forward_pass(self, backbone):
        vlm = torch.randn(self.B, self.N_VLM, self.D_VLM)
        pc = torch.randn(self.B, self.N_PC, self.D_PC)
        action = torch.randn(self.B, self.N_ACTION, self.D_ACTION)
        cond = torch.randn(self.B, self.D_COND)
        pos_ids = torch.arange(self.N_VLM).unsqueeze(0).expand(3, self.B, -1)

        with torch.no_grad():
            v, p, a = backbone(vlm, pc, action, cond, pos_ids)

        assert v.shape == (self.B, self.N_VLM, self.D_VLM)
        assert p.shape == (self.B, self.N_PC, self.D_PC)
        assert a.shape == (self.B, self.N_ACTION, self.D_ACTION)
        assert not torch.isnan(v).any()
        assert not torch.isnan(p).any()
        assert not torch.isnan(a).any()

    def test_backward_pass(self, backbone):
        vlm = torch.randn(self.B, self.N_VLM, self.D_VLM)
        pc = torch.randn(self.B, self.N_PC, self.D_PC)
        action = torch.randn(self.B, self.N_ACTION, self.D_ACTION)
        cond = torch.randn(self.B, self.D_COND)
        pos_ids = torch.arange(self.N_VLM).unsqueeze(0).expand(3, self.B, -1)

        _, pc_out, action_out = backbone(vlm, pc, action, cond, pos_ids)
        loss = pc_out.sum() + action_out.sum()
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in backbone.parameters() if p.requires_grad
        )
        assert has_grad
