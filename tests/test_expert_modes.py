"""Test sparse and dense expert modes with real Qwen2.5-VL weights."""

import torch
import pytest

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from unified_vla.surgery import create_backbone
from unified_vla.utils import count_params


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
class TestExpertModes:
    MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
    D_PC = 1024
    D_ACTION = 1024
    D_COND = 64
    NUM_LAYERS = 8
    D_VLM = 2048
    B = 1
    N_VLM = 5
    N_PC = 3
    N_ACTION = 4

    EXPERT_LAYERS = [0, 4, 7]

    @pytest.fixture(scope="class")
    def qwen_model(self):
        return Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.MODEL_ID, dtype=torch.float32,
        )

    @pytest.fixture(scope="class")
    def dense_backbone(self, qwen_model):
        return create_backbone(
            qwen_model, d_pc=self.D_PC, d_action=self.D_ACTION, d_cond=self.D_COND,
            num_layers=self.NUM_LAYERS,
            expert_layer_ids=self.EXPERT_LAYERS,
            expert_mode="dense",
        )

    @pytest.fixture(scope="class")
    def sparse_backbone(self, qwen_model):
        return create_backbone(
            qwen_model, d_pc=self.D_PC, d_action=self.D_ACTION, d_cond=self.D_COND,
            num_layers=self.NUM_LAYERS,
            expert_layer_ids=self.EXPERT_LAYERS,
            expert_mode="sparse",
        )

    @pytest.fixture
    def inputs(self):
        vlm = torch.randn(self.B, self.N_VLM, self.D_VLM)
        pc = torch.randn(self.B, self.N_PC, self.D_PC)
        action = torch.randn(self.B, self.N_ACTION, self.D_ACTION)
        cond = torch.randn(self.B, self.D_COND)
        pos_ids = torch.arange(self.N_VLM).unsqueeze(0).expand(3, self.B, -1)
        return vlm, pc, action, cond, pos_ids

    # --- Dense mode ---

    def test_dense_forward(self, dense_backbone, inputs):
        vlm, pc, action, cond, pos_ids = inputs
        with torch.no_grad():
            v, p, a = dense_backbone(vlm, pc, action, cond, pos_ids)
        assert v.shape == (self.B, self.N_VLM, self.D_VLM)
        assert p.shape == (self.B, self.N_PC, self.D_PC)
        assert a.shape == (self.B, self.N_ACTION, self.D_ACTION)
        assert not torch.isnan(p).any() and not torch.isnan(a).any()

    def test_dense_has_experts_at_every_layer(self, dense_backbone):
        for i, layer in enumerate(dense_backbone.layers):
            assert layer.pc_block is not None, f"Dense layer {i} should have PC expert"
            assert layer.action_block is not None, f"Dense layer {i} should have Action expert"

    def test_dense_backward(self, dense_backbone, inputs):
        vlm, pc, action, cond, pos_ids = inputs
        _, p, a = dense_backbone(vlm, pc, action, cond, pos_ids)
        (p.sum() + a.sum()).backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in dense_backbone.parameters() if p.requires_grad
        )
        assert has_grad

    # --- Sparse mode ---

    def test_sparse_forward(self, sparse_backbone, inputs):
        vlm, pc, action, cond, pos_ids = inputs
        with torch.no_grad():
            v, p, a = sparse_backbone(vlm, pc, action, cond, pos_ids)
        assert v.shape == (self.B, self.N_VLM, self.D_VLM)
        assert p.shape == (self.B, self.N_PC, self.D_PC)
        assert a.shape == (self.B, self.N_ACTION, self.D_ACTION)
        assert not torch.isnan(p).any() and not torch.isnan(a).any()

    def test_sparse_has_experts_only_at_expert_layers(self, sparse_backbone):
        expert_set = set(self.EXPERT_LAYERS)
        for i, layer in enumerate(sparse_backbone.layers):
            if i in expert_set:
                assert layer.pc_block is not None, f"Sparse layer {i} should have experts"
                assert layer.action_block is not None
            else:
                assert layer.pc_block is None, f"Sparse layer {i} should NOT have experts"
                assert layer.action_block is None

    def test_sparse_backward(self, sparse_backbone, inputs):
        vlm, pc, action, cond, pos_ids = inputs
        _, p, a = sparse_backbone(vlm, pc, action, cond, pos_ids)
        (p.sum() + a.sum()).backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in sparse_backbone.parameters() if p.requires_grad
        )
        assert has_grad

    # --- Comparisons ---

    def test_sparse_fewer_params_than_dense(self, dense_backbone, sparse_backbone):
        dense_t = count_params(dense_backbone, requires_grad=True)
        sparse_t = count_params(sparse_backbone, requires_grad=True)
        assert sparse_t < dense_t

    def test_vlm_output_same_across_modes(self, dense_backbone, sparse_backbone, inputs):
        vlm, pc, action, cond, pos_ids = inputs
        with torch.no_grad():
            vlm_dense, _, _ = dense_backbone(vlm, pc, action, cond, pos_ids)
            vlm_sparse, _, _ = sparse_backbone(vlm, pc, action, cond, pos_ids)
        assert torch.equal(vlm_dense, vlm_sparse)

    def test_param_counts(self, dense_backbone, sparse_backbone, qwen_model):
        all_cross = create_backbone(
            qwen_model, d_pc=self.D_PC, d_action=self.D_ACTION, d_cond=self.D_COND,
            num_layers=self.NUM_LAYERS,
        )
        all_t = count_params(all_cross, requires_grad=True)
        dense_t = count_params(dense_backbone, requires_grad=True)
        sparse_t = count_params(sparse_backbone, requires_grad=True)

        print(f"\n  {self.NUM_LAYERS} layers, experts at {self.EXPERT_LAYERS}:")
        print(f"  All-cross trainable:  {all_t:>12,}")
        print(f"  Dense trainable:      {dense_t:>12,}")
        print(f"  Sparse trainable:     {sparse_t:>12,}")
        print(f"  Sparse/All ratio:     {sparse_t/all_t:>11.1%}")

    # --- Action attends to previous layer's PC ---

    def test_action_first_layer_no_prev_pc(self, sparse_backbone, inputs):
        """At the first expert layer, Action has no previous PC K/V — should still work."""
        vlm, pc, action, cond, pos_ids = inputs
        with torch.no_grad():
            _, _, a = sparse_backbone(vlm, pc, action, cond, pos_ids)
        assert not torch.isnan(a).any()

    # --- Default: no expert_layer_ids = all layers ---

    def test_default_all_layers(self, qwen_model, inputs):
        bb = create_backbone(
            qwen_model, d_pc=self.D_PC, d_action=self.D_ACTION, d_cond=self.D_COND,
            num_layers=4,
            # no expert_layer_ids → all layers
        )
        assert bb.expert_layer_ids == {0, 1, 2, 3}

        vlm, pc, action, cond, pos_ids = inputs
        with torch.no_grad():
            v, p, a = bb(vlm, pc, action, cond, pos_ids)
        assert not torch.isnan(p).any() and not torch.isnan(a).any()
