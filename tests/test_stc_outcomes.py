"""Truth-table tests for the headline Safe Task Completion metric."""

from __future__ import annotations

import pytest

from evaluation.outcomes import reduce_stc


@pytest.mark.parametrize(
    (
        "reached_goal",
        "physical_collision",
        "semantic_violation",
        "timeout",
        "expected",
    ),
    (
        (True, False, False, False, True),
        (True, False, True, False, False),
        (False, True, False, False, False),
        (False, False, False, True, False),
        (True, True, True, False, False),
    ),
    ids=(
        "safe-completion",
        "goal-with-semantic-violation",
        "physical-collision",
        "timeout",
        "joint-failure",
    ),
)
def test_stc_truth_table(
    reached_goal: bool,
    physical_collision: bool,
    semantic_violation: bool,
    timeout: bool,
    expected: bool,
) -> None:
    components = reduce_stc(
        reached_goal=reached_goal,
        physical_collision=physical_collision,
        applicable_semantic_violation=semantic_violation,
        timeout=timeout,
    )
    assert components.safe_task_completion is expected
    assert components.to_dict()["STC"] is expected


def test_timeout_cannot_overlap_terminal_event() -> None:
    with pytest.raises(ValueError, match="timeout cannot coincide"):
        reduce_stc(
            reached_goal=True,
            physical_collision=False,
            applicable_semantic_violation=False,
            timeout=True,
        )
