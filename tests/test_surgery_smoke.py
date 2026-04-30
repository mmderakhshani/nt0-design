"""Smoke test: load Qwen2.5-VL-3B, perform weight surgery, run forward pass."""

import torch
import pytest

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from unified_vla.surgery import create_backbone


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
class TestQwenSurgerySmoke:
    MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

    # Qwen2.5-VL-3B specs
    D_VLM = 2048
    HEAD_DIM = 128
    NUM_KV_HEADS_VLM = 2
    TOTAL_LAYERS = 36

    # Expert config
    D_PC = 1024       # 8 Q heads
    D_ACTION = 1024   # 8 Q heads
    D_COND = 64
    NUM_LAYERS = 2    # only use 2 layers for the smoke test

    B = 1
    N_VLM = 5
    N_PC = 3
    N_ACTION = 4

    @pytest.fixture(scope="class")
    def qwen_model(self):
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            dtype=torch.float32,
        )
        model.eval()
        return model

    @pytest.fixture(scope="class")
    def backbone(self, qwen_model):
        return create_backbone(
            qwen_model,
            d_pc=self.D_PC,
            d_action=self.D_ACTION,
            d_cond=self.D_COND,
            num_layers=self.NUM_LAYERS,
        )

    def test_backbone_created(self, backbone):
        assert len(backbone.layers) == self.NUM_LAYERS

    def test_vlm_weights_match_qwen(self, backbone, qwen_model):
        """VLM block weights should exactly match the original Qwen layer."""
        qwen_layer_0 = qwen_model.model.language_model.layers[0]
        vlm_layer_0 = backbone.layers[0].vlm_layer

        # VLM layer IS the actual Qwen layer — same object
        assert vlm_layer_0 is qwen_layer_0

    def test_vlm_frozen(self, backbone):
        for layer in backbone.layers:
            for name, p in layer.vlm_layer.named_parameters():
                assert not p.requires_grad, f"VLM param {name} should be frozen"

    def test_experts_trainable(self, backbone):
        for layer in backbone.layers:
            pc_trainable = any(p.requires_grad for p in layer.pc_block.parameters())
            action_trainable = any(p.requires_grad for p in layer.action_block.parameters())
            assert pc_trainable
            assert action_trainable

    def test_expert_initialized_from_vlm(self, backbone):
        """PC expert Q weights should be a slice of VLM Q weights."""
        vlm_q = backbone.layers[0].vlm_layer.self_attn.q_proj.weight
        pc_q = backbone.layers[0].pc_block.qkv.q_proj.weight
        d_pc = self.D_PC
        assert torch.equal(pc_q, vlm_q[:d_pc, :d_pc])

    def _make_inputs(self):
        vlm = torch.randn(self.B, self.N_VLM, self.D_VLM)
        pc = torch.randn(self.B, self.N_PC, self.D_PC)
        action = torch.randn(self.B, self.N_ACTION, self.D_ACTION)
        cond = torch.randn(self.B, self.D_COND)
        pos_ids = torch.arange(self.N_VLM).unsqueeze(0).expand(3, self.B, -1)
        return vlm, pc, action, cond, pos_ids

    def test_forward_pass(self, backbone):
        """Full forward pass should produce valid outputs."""
        vlm, pc, action, cond, pos_ids = self._make_inputs()

        with torch.no_grad():
            vlm_out, pc_out, action_out = backbone(vlm, pc, action, cond, pos_ids)

        assert vlm_out.shape == (self.B, self.N_VLM, self.D_VLM)
        assert pc_out.shape == (self.B, self.N_PC, self.D_PC)
        assert action_out.shape == (self.B, self.N_ACTION, self.D_ACTION)

        assert not torch.isnan(vlm_out).any()
        assert not torch.isnan(pc_out).any()
        assert not torch.isnan(action_out).any()

    def test_backward_pass(self, backbone):
        """Backward through Action + PC losses should work and respect isolation.

        Note: adaLN-zero means the action_block's inner attention/MLP weights
        get **zero gradient on step 1** (gates are zero, modulated path is
        dormant). What does pick up gradient on step 1 is the action_adaln
        MLP's final Linear — that lifts the gates off zero so subsequent
        steps train the rest of the block. We assert that here.
        """
        vlm, pc, action, cond, pos_ids = self._make_inputs()

        _, pc_out, action_out = backbone(vlm, pc, action, cond, pos_ids)
        loss = pc_out.sum() + action_out.sum()
        loss.backward()

        # adaLN MLP should have grad (the path through adaLN params is non-zero)
        has_adaln_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for layer in backbone.layers
            if hasattr(layer, "action_adaln")
            for p in layer.action_adaln.parameters()
        )
        assert has_adaln_grad, "action_adaln should pick up grad on step 1"

        # PC expert should have grads (no zero-init gating on PC)
        has_pc_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for layer in backbone.layers
            if layer.pc_block is not None
            for p in layer.pc_block.parameters()
        )
        assert has_pc_grad

        # VLM should have no grads (frozen)
        for layer in backbone.layers:
            for name, p in layer.vlm_layer.named_parameters():
                assert p.grad is None, f"VLM {name} should have no grad"

    def test_action_block_grads_after_adaln_warmup(self, backbone):
        """After bumping action_adaln off zero, the action_block's inner weights
        DO get gradient — this confirms the gates correctly mediate signal flow."""
        vlm, pc, action, cond, pos_ids = self._make_inputs()

        # Warm up: bump action_adaln final Linear off zero so gates aren't zero anymore.
        with torch.no_grad():
            for layer in backbone.layers:
                if hasattr(layer, "action_adaln"):
                    layer.action_adaln[-1].weight.add_(
                        torch.randn_like(layer.action_adaln[-1].weight) * 0.05
                    )

        _, _, action_out = backbone(vlm, pc, action, cond, pos_ids)
        action_out.sum().backward()

        has_action_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for layer in backbone.layers
            for p in layer.action_block.parameters()
        )
        assert has_action_grad, (
            "after the gates are non-zero, action_block weights must receive grad"
        )

    def test_param_summary(self, backbone):
        """Print parameter counts for inspection."""
        total = sum(p.numel() for p in backbone.parameters())
        trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        frozen = total - trainable
        print(f"\n  Total params:     {total:,}")
        print(f"  Trainable params: {trainable:,}")
        print(f"  Frozen params:    {frozen:,}")
        print(f"  Layers used:      {len(backbone.layers)} / {self.TOTAL_LAYERS}")
        assert trainable > 0
        assert frozen > 0
