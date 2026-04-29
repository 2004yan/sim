"""Offline validation helpers for Pure Pursuit tuning.

The simulator here uses the same kinematic equations as the algorithm report.
It is not a replacement for Isaac Sim, but it provides quick checks for path
geometry, signs, and controller smoothness.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np

from pure_pursuit_controller import (
    DEFAULT_TRACK_WIDTH,
    DEFAULT_WHEEL_RADIUS,
    DEFAULT_WHEELBASE,
    PurePursuitTracker,
    generate_lawnmower_path,
)


@dataclass(frozen=True)
class ValidationResult:
    name: str
    done: bool
    steps: int
    rmse_cte: float
    max_cte: float
    steer_sign_changes: int
    control_effort: float
    final_pose: tuple[float, float, float]


def simulate_path(
    path: Iterable[Sequence[float]],
    *,
    initial_pose: tuple[float, float, float],
    name: str = "path",
    v_mps: float = 0.8,
    dt: float = 0.05,
    max_steps: int = 600,
    alpha: float = 0.5,
    min_lookahead: float = 0.5,
    max_lookahead: float = 2.0,
    goal_tolerance: float = 0.2,
) -> ValidationResult:
    tracker = PurePursuitTracker(
        path,
        alpha=alpha,
        min_lookahead=min_lookahead,
        max_lookahead=max_lookahead,
        goal_tolerance=goal_tolerance,
    )
    x, y, theta = initial_pose
    cte_values: list[float] = []
    steer_values: list[float] = []
    control_effort = 0.0
    done = False
    steps = 0

    for steps in range(1, max_steps + 1):
        command = tracker.compute(x=x, y=y, theta=theta, v_mps=v_mps)
        done = command.done
        cte_values.append(_distance_to_path(tracker, np.array([x, y], dtype=float)))
        steer_values.append(command.steer_rad)

        if done:
            break

        left_mps = command.left_rad_s * DEFAULT_WHEEL_RADIUS
        right_mps = command.right_rad_s * DEFAULT_WHEEL_RADIUS
        v = (left_mps + right_mps) / 2.0
        omega_diff = (right_mps - left_mps) / DEFAULT_TRACK_WIDTH
        omega_steer = v * math.tan(command.steer_rad) / DEFAULT_WHEELBASE
        omega_total = omega_diff + omega_steer

        x += v * math.cos(theta) * dt
        y += v * math.sin(theta) * dt
        theta = _wrap_angle(theta + omega_total * dt)
        control_effort += (
            command.left_rad_s * command.left_rad_s
            + command.right_rad_s * command.right_rad_s
            + command.steer_rad * command.steer_rad
        ) * dt

    cte = np.asarray(cte_values, dtype=float)
    return ValidationResult(
        name=name,
        done=done,
        steps=steps,
        rmse_cte=float(np.sqrt(np.mean(cte * cte))) if len(cte) else 0.0,
        max_cte=float(np.max(np.abs(cte))) if len(cte) else 0.0,
        steer_sign_changes=_count_sign_changes(steer_values),
        control_effort=float(control_effort),
        final_pose=(float(x), float(y), float(theta)),
    )


def _distance_to_path(tracker: PurePursuitTracker, position: np.ndarray) -> float:
    progress = tracker._closest_progress(position)
    closest = tracker._point_at_progress(progress)
    return float(np.linalg.norm(position - closest))


def _count_sign_changes(values: Sequence[float], deadband: float = 1e-4) -> int:
    signs = [math.copysign(1.0, value) for value in values if abs(value) > deadband]
    return sum(1 for previous, current in zip(signs, signs[1:]) if previous != current)


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _arc_path(radius: float = 3.0, samples: int = 16) -> list[tuple[float, float]]:
    angles = np.linspace(0.0, math.pi / 2.0, samples)
    return [(float(radius * math.cos(a)), float(radius * math.sin(a))) for a in angles]


def main() -> None:
    cases = [
        (
            "straight",
            [(0.0, 0.0), (5.0, 0.0)],
            (0.0, 0.0, 0.0),
        ),
        (
            "arc",
            _arc_path(),
            (3.0, 0.0, math.pi / 2.0),
        ),
        (
            "field_30x3_nine_turns",
            generate_lawnmower_path(),
            (0.0, 3.0, 0.0),
        ),
    ]

    for name, path, pose in cases:
        if name == "field_30x3_nine_turns":
            result = simulate_path(
                path,
                initial_pose=pose,
                name=name,
                v_mps=0.5,
                dt=0.2,
                max_steps=4000,
                alpha=0.7,
                min_lookahead=0.25,
                max_lookahead=0.8,
                goal_tolerance=0.3,
            )
        else:
            result = simulate_path(path, initial_pose=pose, name=name)
        print(
            f"{result.name}: done={result.done} steps={result.steps} "
            f"rmse_cte={result.rmse_cte:.3f} max_cte={result.max_cte:.3f} "
            f"steer_sign_changes={result.steer_sign_changes} "
            f"control_effort={result.control_effort:.2f}"
        )


if __name__ == "__main__":
    main()
