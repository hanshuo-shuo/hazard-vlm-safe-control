"""Shared Safe Task Completion outcome reduction."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class STCComponents:
    reached_goal: bool
    physical_collision: bool
    applicable_semantic_violation: bool
    timeout: bool

    def __post_init__(self) -> None:
        for field_name in (
            "reached_goal",
            "physical_collision",
            "applicable_semantic_violation",
            "timeout",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"{field_name} must be boolean")
        if self.timeout and (self.reached_goal or self.physical_collision):
            raise ValueError("timeout cannot coincide with goal or physical collision")

    @property
    def safe_task_completion(self) -> bool:
        return (
            self.reached_goal
            and not self.physical_collision
            and not self.applicable_semantic_violation
        )

    def to_dict(self) -> dict[str, bool]:
        return {
            "reached_goal": self.reached_goal,
            "physical_collision": self.physical_collision,
            "applicable_semantic_violation": self.applicable_semantic_violation,
            "timeout": self.timeout,
            "STC": self.safe_task_completion,
        }


def reduce_stc(
    *,
    reached_goal: bool,
    physical_collision: bool,
    applicable_semantic_violation: bool,
    timeout: bool,
) -> STCComponents:
    return STCComponents(
        reached_goal=reached_goal,
        physical_collision=physical_collision,
        applicable_semantic_violation=applicable_semantic_violation,
        timeout=timeout,
    )
