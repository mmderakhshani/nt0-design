import torch
import pytest

from unified_vla.utils import count_params, format_num, backbone_param_summary, print_param_summary

# Small test dims
D_VLM = 256
D_PC = 128
D_ACTION = 128
HEAD_DIM = 32
D_COND = 64
NUM_LAYERS = 2
B = 2


class TestCountParams:
    def test_all_params(self):
        m = torch.nn.Linear(10, 5)
        assert count_params(m) == 10 * 5 + 5  # weight + bias

    def test_trainable_only(self):
        m = torch.nn.Linear(10, 5)
        m.weight.requires_grad_(False)
        assert count_params(m, requires_grad=True) == 5  # bias only
        assert count_params(m, requires_grad=False) == 50  # weight only

    def test_frozen_only(self):
        m = torch.nn.Linear(10, 5, bias=False)
        m.weight.requires_grad_(False)
        assert count_params(m, requires_grad=False) == 50
        assert count_params(m, requires_grad=True) == 0


class TestFormatNum:
    def test_millions(self):
        assert format_num(1_500_000) == "1.50M"
        assert format_num(85_613_568) == "85.61M"

    def test_thousands(self):
        assert format_num(4_096) == "4.1K"

    def test_small(self):
        assert format_num(256) == "256"


class TestQwenParamSummary:
    """Test with real Qwen2.5-VL-3B weights."""

    @pytest.fixture(scope="class")
    def backbone(self):
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
            from unified_vla.surgery import create_backbone
        except ImportError:
            pytest.skip("transformers not installed")

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct", dtype=torch.float32,
        )
        return create_backbone(
            model, d_pc=1024, d_action=1024, d_cond=64, num_layers=2,
        )

    def test_print_qwen_summary(self, backbone, capsys):
        print_param_summary(backbone)
        captured = capsys.readouterr()
        print(captured.out)
        assert "VLM expert (frozen)" in captured.out
