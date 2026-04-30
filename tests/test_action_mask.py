"""Action-segment masking tests.

The Action segment is
    [sink_action, embodiment_action,
     <proprio_start>, proprio_tokens, <proprio_end>,
     <action_start>, action_tokens,  <action_end>]
(single-stream). At non-expert (self-only) layers the mask must be a single
causal triangle covering the entire segment so action tokens can attend to
sink + embodiment + all earlier proprio positions but not vice versa. At
expert (cross+self) layers the mask is
`[full_cross_to_VLM/PC | causal_self_over_segment]`.
"""

import torch

from unified_vla.backbone import (
    _build_action_self_causal_mask,
    _build_action_cross_causal_mask,
)


N_PROPRIO = 4
N_ACTION_TOKENS = 8
# Six non-content tokens in front of / between content:
#   sink_action, embodiment_action, <proprio_start>, <proprio_end>,
#   <action_start>, <action_end>.
N_DELIMITERS = 6
N_SEGMENT = N_PROPRIO + N_ACTION_TOKENS + N_DELIMITERS

# Layout, with absolute indices:
#   0                              sink_action
#   1                              embodiment_action
#   2                              <proprio_start>
#   [3, 3+N_PROPRIO)               proprio tokens
#   3 + N_PROPRIO                  <proprio_end>
#   4 + N_PROPRIO                  <action_start>
#   [5+N_PROPRIO, ...)             action tokens
#   5 + N_PROPRIO + N_ACTION_TOKENS    <action_end>
SINK_IDX = 0
EMBODIMENT_IDX = 1
PROPRIO_START_IDX = 2
PROPRIO_TOKEN_IDXS = list(range(3, 3 + N_PROPRIO))
PROPRIO_END_IDX = 3 + N_PROPRIO
ACTION_START_IDX = 4 + N_PROPRIO
ACTION_CHUNK_FIRST_IDX = 5 + N_PROPRIO
ACTION_END_IDX = N_SEGMENT - 1


def test_self_only_mask_is_unified_causal_triangle():
    """Self-only mask must be lower-triangular over the whole segment."""
    mask = _build_action_self_causal_mask(N_SEGMENT, device=torch.device("cpu"))
    assert mask.shape == (1, 1, N_SEGMENT, N_SEGMENT)

    expected = torch.tril(torch.ones(N_SEGMENT, N_SEGMENT, dtype=torch.bool))
    assert torch.equal(mask[0, 0], expected)


def test_action_tokens_can_attend_to_sink_embodiment_and_proprio_block():
    """Every action-chunk position can attend to sink + embodiment + every earlier proprio token + delimiters."""
    mask = _build_action_self_causal_mask(N_SEGMENT, device=torch.device("cpu"))[0, 0]
    earlier_idxs = (
        [SINK_IDX, EMBODIMENT_IDX, PROPRIO_START_IDX]
        + PROPRIO_TOKEN_IDXS
        + [PROPRIO_END_IDX, ACTION_START_IDX]
    )

    for q in range(ACTION_CHUNK_FIRST_IDX, N_SEGMENT):
        for k in earlier_idxs:
            assert mask[q, k].item() is True, (
                f"action position {q} cannot attend to earlier position {k}"
            )


def test_sink_at_position_zero_only_sees_itself():
    """Causal mask: sink_action at index 0 can only attend to itself in the self portion."""
    mask = _build_action_self_causal_mask(N_SEGMENT, device=torch.device("cpu"))[0, 0]
    assert mask[SINK_IDX, SINK_IDX].item() is True
    for k in range(1, N_SEGMENT):
        assert mask[SINK_IDX, k].item() is False, (
            f"sink[0] should not see future position {k}"
        )


def test_every_later_token_can_attend_to_sink():
    """The sink token's job is to be visible from every later query."""
    mask = _build_action_self_causal_mask(N_SEGMENT, device=torch.device("cpu"))[0, 0]
    for q in range(1, N_SEGMENT):
        assert mask[q, SINK_IDX].item() is True, (
            f"position {q} cannot reach the sink — sink is useless here"
        )


def test_proprio_cannot_attend_to_future_action_tokens():
    """Causality: a proprio token cannot peek at later action tokens."""
    mask = _build_action_self_causal_mask(N_SEGMENT, device=torch.device("cpu"))[0, 0]
    last_proprio_idx = PROPRIO_TOKEN_IDXS[-1]
    for k in range(ACTION_CHUNK_FIRST_IDX, N_SEGMENT):
        assert mask[last_proprio_idx, k].item() is False, (
            f"proprio[{last_proprio_idx}] should not see action[{k}]"
        )


def test_proprio_start_does_not_see_proprio_end():
    """The start delimiter sits before the end one (causal)."""
    mask = _build_action_self_causal_mask(N_SEGMENT, device=torch.device("cpu"))[0, 0]
    assert mask[PROPRIO_START_IDX, PROPRIO_END_IDX].item() is False
    assert mask[PROPRIO_END_IDX, PROPRIO_START_IDX].item() is True


def test_cross_mask_full_over_external_then_causal_over_segment():
    """Cross mask: full attention over external K/V, causal over the action segment itself."""
    n_ext = 13
    mask = _build_action_cross_causal_mask(N_SEGMENT, n_ext, device=torch.device("cpu"))
    assert mask.shape == (1, 1, N_SEGMENT, n_ext + N_SEGMENT)

    cross_part = mask[0, 0, :, :n_ext]
    self_part = mask[0, 0, :, n_ext:]

    assert torch.equal(cross_part, torch.ones(N_SEGMENT, n_ext, dtype=torch.bool))
    assert torch.equal(
        self_part, torch.tril(torch.ones(N_SEGMENT, N_SEGMENT, dtype=torch.bool))
    )
