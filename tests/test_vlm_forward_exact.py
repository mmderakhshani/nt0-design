"""Verify that we can replicate Qwen2.5-VL's text decoder forward pass exactly.

Strategy:
1. Run the full Qwen2.5-VL model with 3 fake images + text
2. Hook into model.language_model to capture the exact (inputs_embeds, position_ids,
   attention_mask) that the model passes to its text decoder
3. Call model.language_model directly with those captured inputs
4. Verify the outputs are bit-for-bit identical

If this passes, we know exactly how to call the VLM decoder to get zero distribution shift.
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


class CaptureHook:
    """Forward hook that captures args and kwargs passed to a module."""

    def __init__(self):
        self.args = None
        self.kwargs = None
        self.output = None

    def __call__(self, module, args, kwargs, output):
        self.args = args
        self.kwargs = {k: v for k, v in kwargs.items()}
        self.output = output
        return output


@pytest.mark.skipif(not HAS_DEPS, reason="transformers or PIL not installed")
class TestQwenVLMForwardExact:
    MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

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
        """3 fake images + robot prompt, processed through Qwen's processor."""
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
    def reference_and_captured(self, model, fake_inputs):
        """Run full model, capture language_model inputs, return both."""
        hook = CaptureHook()
        handle = model.language_model.register_forward_hook(hook, with_kwargs=True)

        with torch.no_grad():
            ref_output = model(**fake_inputs, output_hidden_states=True)

        handle.remove()
        return ref_output, hook

    def test_hook_captured_inputs(self, reference_and_captured):
        """Verify the hook captured meaningful inputs."""
        _, hook = reference_and_captured
        assert hook.kwargs is not None
        assert "inputs_embeds" in hook.kwargs
        assert "position_ids" in hook.kwargs
        assert hook.kwargs["inputs_embeds"] is not None

    def test_captured_shapes(self, reference_and_captured, fake_inputs):
        """Captured inputs_embeds should match the input sequence length."""
        _, hook = reference_and_captured
        embeds = hook.kwargs["inputs_embeds"]
        print(f"\n  inputs_embeds shape: {embeds.shape}")
        print(f"  position_ids shape:  {hook.kwargs['position_ids'].shape}")
        # Should be (1, seq_len, 2048)
        assert embeds.ndim == 3
        assert embeds.shape[0] == 1
        assert embeds.shape[2] == 2048

    def test_direct_call_matches_reference(self, model, reference_and_captured):
        """Calling language_model directly with captured inputs produces identical output."""
        ref_output, hook = reference_and_captured

        # Remove keys that we'll pass explicitly to avoid duplicates
        kwargs = {k: v for k, v in hook.kwargs.items()
                  if k not in ("output_hidden_states", "return_dict")}

        with torch.no_grad():
            direct_output = model.language_model(
                **kwargs,
                output_hidden_states=True,
                return_dict=True,
            )

        # Compare last hidden state (after final norm)
        ref_hidden = ref_output.hidden_states[-1]
        direct_hidden = direct_output.hidden_states[-1]
        assert torch.equal(ref_hidden, direct_hidden), (
            f"Max diff: {(ref_hidden - direct_hidden).abs().max().item()}"
        )

    def test_all_hidden_states_match(self, model, reference_and_captured):
        """Every intermediate hidden state should match exactly."""
        ref_output, hook = reference_and_captured

        kwargs = {k: v for k, v in hook.kwargs.items()
                  if k not in ("output_hidden_states", "return_dict")}

        with torch.no_grad():
            direct_output = model.language_model(
                **kwargs,
                output_hidden_states=True,
                return_dict=True,
            )

        assert len(ref_output.hidden_states) == len(direct_output.hidden_states)

        for i, (ref_hs, direct_hs) in enumerate(
            zip(ref_output.hidden_states, direct_output.hidden_states)
        ):
            assert torch.equal(ref_hs, direct_hs), (
                f"Layer {i} hidden states differ. Max diff: "
                f"{(ref_hs - direct_hs).abs().max().item()}"
            )

        print(f"\n  All {len(ref_output.hidden_states)} hidden states match exactly.")

    def test_layer_by_layer_matches(self, model, reference_and_captured):
        """Run decoder layers one at a time and verify each matches.

        Reference hidden_states from Qwen2_5_VLTextModel with output_hidden_states=True:
            hidden_states[0]  = input embeddings
            hidden_states[i+1] = output of decoder layer i (before final norm)
            (The last entry is still pre-norm; the model applies norm separately)
        """
        # First get the reference hidden states from the text model directly
        _, hook = reference_and_captured
        text_model = model.language_model

        kwargs = {k: v for k, v in hook.kwargs.items()
                  if k not in ("output_hidden_states", "return_dict")}

        with torch.no_grad():
            ref = text_model(**kwargs, output_hidden_states=True, return_dict=True)
            ref_hidden_states = ref.hidden_states

        with torch.no_grad():
            inputs_embeds = hook.kwargs["inputs_embeds"]
            position_ids = hook.kwargs["position_ids"]
            attention_mask = hook.kwargs.get("attention_mask")
            cache_position = hook.kwargs.get("cache_position")

            # Step 1: The first hidden state is the input embeddings
            hidden_states = inputs_embeds
            assert torch.equal(hidden_states, ref_hidden_states[0]), "Input embeddings don't match"

            # Step 2: Prepare position embeddings (RoPE) and attention mask
            if position_ids.ndim == 3 and position_ids.shape[0] == 4:
                text_position_ids = position_ids[0]
                rope_position_ids = position_ids[1:]
            else:
                text_position_ids = None
                rope_position_ids = position_ids

            position_embeddings = text_model.rotary_emb(hidden_states, rope_position_ids)

            from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

            if cache_position is None:
                cache_position = torch.arange(
                    hidden_states.shape[1], device=hidden_states.device
                )

            mask_kwargs = {
                "config": text_model.config,
                "input_embeds": hidden_states,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": None,
                "position_ids": text_position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            if hasattr(text_model, "has_sliding_layers") and text_model.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(
                    **mask_kwargs
                )

            # Step 3: Run each decoder layer
            # ref_hidden_states: [input_embeds, layer_0_out, ..., layer_N-2_out, norm(layer_N-1_out)]
            # The last entry is post-final-norm, so we compare layers 0..N-2 directly
            # and verify layer N-1 after applying norm ourselves.
            num_layers = len(text_model.layers)
            mismatches = []
            for i, decoder_layer in enumerate(text_model.layers):
                layer_output = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                    position_ids=text_position_ids,
                    past_key_values=None,
                    output_attentions=False,
                    use_cache=False,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
                hidden_states = layer_output[0]

                if i < num_layers - 1:
                    # Pre-final layers: compare directly
                    if not torch.equal(hidden_states, ref_hidden_states[i + 1]):
                        diff = (hidden_states - ref_hidden_states[i + 1]).abs().max().item()
                        mismatches.append((i, diff))

            if mismatches:
                msg = "; ".join(f"layer {i}: max_diff={d}" for i, d in mismatches)
                pytest.fail(f"Mismatches at: {msg}")

            # Step 4: Final layer output + norm should match ref_hidden_states[-1]
            normed = text_model.norm(hidden_states)
            assert torch.equal(normed, ref_hidden_states[-1]), (
                f"Post-norm output differs. Max diff: "
                f"{(normed - ref_hidden_states[-1]).abs().max().item()}"
            )

        print(f"\n  All {num_layers} layers + final norm match exactly layer-by-layer.")
        print(f"  Sequence length: {inputs_embeds.shape[1]} tokens")
        print(f"  Position IDs shape: {position_ids.shape}")
