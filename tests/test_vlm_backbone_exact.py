"""Prove that our QwenVLABackbone's VLM path produces bit-for-bit identical
output to the original Qwen2.5-VL model.

Strategy:
1. Process 3 fake images + text through the full Qwen2.5-VL model
2. Capture the inputs_embeds and position_ids that enter the text decoder
3. Build our QwenVLABackbone from the same model
4. Run the same inputs_embeds through our backbone
5. Compare VLM hidden states at every layer — must be exact match
"""

import torch
import pytest
import numpy as np

try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from PIL import Image

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

from unified_vla.surgery import create_backbone


class CaptureHook:
    def __init__(self):
        self.kwargs = None

    def __call__(self, module, args, kwargs, output):
        self.kwargs = {k: v for k, v in kwargs.items()}
        return output


@pytest.mark.skipif(not HAS_DEPS, reason="transformers or PIL not installed")
class TestQwenBackboneVLMExact:
    MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
    D_PC = 1024
    D_ACTION = 1024
    D_COND = 64
    NUM_LAYERS = 4  # test with first 4 layers for speed

    @pytest.fixture(scope="class")
    def model(self):
        m = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.MODEL_ID, dtype=torch.float32,
        )
        m.eval()
        return m

    @pytest.fixture(scope="class")
    def processor(self):
        return AutoProcessor.from_pretrained(self.MODEL_ID)

    @pytest.fixture(scope="class")
    def fake_inputs(self, processor):
        np.random.seed(42)
        imgs = [
            Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            for _ in range(3)
        ]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": (
                            "Robot: 7-DOF Trossen arm with parallel gripper\n"
                            "State: tx=0.32 ty=-0.15 tz=1.07 r1=0.88 r2=-0.42 "
                            "r3=0.61 r4=0.03 r5=0.12 r6=-0.08 grip=0.80\n"
                            "<task>Pick up the red cup and place it on the white plate.</task>"
                        ),
                    },
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=imgs, return_tensors="pt")
        return inputs

    @pytest.fixture(scope="class")
    def reference_data(self, model, fake_inputs):
        """Run full model, capture text decoder inputs and per-layer hidden states."""
        hook = CaptureHook()
        handle = model.language_model.register_forward_hook(hook, with_kwargs=True)

        with torch.no_grad():
            ref_output = model(**fake_inputs, output_hidden_states=True)

        handle.remove()

        # Also get reference hidden states directly from text model
        kwargs = {k: v for k, v in hook.kwargs.items()
                  if k not in ("output_hidden_states", "return_dict")}
        with torch.no_grad():
            text_output = model.language_model(
                **kwargs, output_hidden_states=True, return_dict=True,
            )

        return {
            "inputs_embeds": hook.kwargs["inputs_embeds"],
            "position_ids": hook.kwargs["position_ids"],
            "attention_mask": hook.kwargs.get("attention_mask"),
            "ref_hidden_states": text_output.hidden_states,
        }

    @pytest.fixture(scope="class")
    def backbone(self, model):
        return create_backbone(
            model,
            d_pc=self.D_PC,
            d_action=self.D_ACTION,
            d_cond=self.D_COND,
            num_layers=self.NUM_LAYERS,
        )

    def test_vlm_output_matches_at_every_layer(self, backbone, reference_data):
        """VLM hidden states from our backbone must match the original model exactly."""
        vlm_embeds = reference_data["inputs_embeds"]
        position_ids = reference_data["position_ids"]
        attention_mask = reference_data["attention_mask"]
        ref_hidden_states = reference_data["ref_hidden_states"]

        B = vlm_embeds.shape[0]
        pc = torch.randn(B, 3, self.D_PC)
        action = torch.randn(B, 4, self.D_ACTION)
        cond = torch.randn(B, self.D_COND)

        # Run our backbone layer by layer, comparing VLM at each step
        vlm_inputs = backbone._prepare_vlm_inputs(vlm_embeds, position_ids, attention_mask)
        from transformers.cache_utils import DynamicCache
        kv_cache = DynamicCache()

        vlm = vlm_embeds
        with torch.no_grad():
            # Check input embeddings match
            assert torch.equal(vlm, ref_hidden_states[0]), "Input embeddings don't match"

            for i, layer in enumerate(backbone.layers):
                vlm, pc, action = layer(
                    vlm, pc, action, cond,
                    vlm_attention_mask=vlm_inputs["attention_masks"][i],
                    vlm_position_ids=vlm_inputs["text_position_ids"],
                    vlm_position_embeddings=vlm_inputs["position_embeddings"],
                    vlm_cache_position=vlm_inputs["cache_position"],
                    kv_cache=kv_cache,
                )

                assert torch.equal(vlm, ref_hidden_states[i + 1]), (
                    f"Layer {i} VLM output differs. "
                    f"Max diff: {(vlm - ref_hidden_states[i + 1]).abs().max().item()}"
                )

        print(f"\n  VLM output matches exactly at all {self.NUM_LAYERS} layers.")

    def test_vlm_final_norm_matches(self, backbone, reference_data):
        """VLM output after final norm should match reference."""
        vlm_embeds = reference_data["inputs_embeds"]
        position_ids = reference_data["position_ids"]
        attention_mask = reference_data["attention_mask"]
        ref_hidden_states = reference_data["ref_hidden_states"]

        B = vlm_embeds.shape[0]
        pc = torch.randn(B, 3, self.D_PC)
        action = torch.randn(B, 4, self.D_ACTION)
        cond = torch.randn(B, self.D_COND)

        with torch.no_grad():
            vlm_out, _, _ = backbone(
                vlm_embeds, pc, action, cond, position_ids, attention_mask,
            )

        # Our backbone applies vlm_norm at the end.
        # Reference: hidden_states[-1] for NUM_LAYERS < total_layers is the
        # output of layer NUM_LAYERS-1 (pre-norm). We need to apply norm ourselves
        # for comparison since we only ran NUM_LAYERS layers.
        ref_layer_out = ref_hidden_states[self.NUM_LAYERS]
        ref_normed = backbone.vlm_norm(ref_layer_out)

        assert torch.equal(vlm_out, ref_normed), (
            f"Final normed VLM output differs. "
            f"Max diff: {(vlm_out - ref_normed).abs().max().item()}"
        )

        print(f"\n  Final normed VLM output matches exactly.")

    def test_pc_action_dont_affect_vlm(self, backbone, reference_data):
        """VLM output should be identical regardless of PC/Action input values."""
        vlm_embeds = reference_data["inputs_embeds"]
        position_ids = reference_data["position_ids"]
        attention_mask = reference_data["attention_mask"]

        B = vlm_embeds.shape[0]

        with torch.no_grad():
            vlm_a, _, _ = backbone(
                vlm_embeds,
                torch.randn(B, 3, self.D_PC),
                torch.randn(B, 4, self.D_ACTION),
                torch.randn(B, self.D_COND),
                position_ids, attention_mask,
            )
            vlm_b, _, _ = backbone(
                vlm_embeds,
                torch.randn(B, 3, self.D_PC) * 100,  # very different PC
                torch.randn(B, 4, self.D_ACTION) * 100,  # very different Action
                torch.randn(B, self.D_COND) * 100,
                position_ids, attention_mask,
            )

        assert torch.equal(vlm_a, vlm_b), "VLM output should not depend on PC/Action inputs"
        print("\n  VLM output is independent of PC/Action inputs.")

    def test_summary(self, backbone, reference_data):
        """Print shapes for inspection."""
        vlm_embeds = reference_data["inputs_embeds"]
        position_ids = reference_data["position_ids"]
        print(f"\n  VLM sequence length: {vlm_embeds.shape[1]} tokens")
        print(f"  Position IDs shape: {position_ids.shape}")
        print(f"  Backbone layers: {backbone.num_layers}")
        print(f"  VLM hidden dim: {vlm_embeds.shape[2]}")

        from unified_vla.utils import count_params
        trainable = count_params(backbone, requires_grad=True)
        frozen = count_params(backbone, requires_grad=False)
        print(f"  Trainable: {trainable:,}")
        print(f"  Frozen: {frozen:,}")
