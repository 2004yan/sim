"""Pure Pursuit path tracking for the 3-wheel paddy robot.

This module is intentionally Isaac-independent. It converts a 2D path and the
robot pose into front wheel angular velocities plus a rear steering target.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np


# Estimates from Sim_Robot_V2.urdf.
DEFAULT_TRACK_WIDTH = 1.2
DEFAULT_WHEELBASE = 1.2
DEFAULT_WHEEL_RADIUS = 0.235
DEFAULT_STEER_SIGN = 1.0
DEFAULT_FIELD_LENGTH = 30.0
DEFAULT_FIELD_WIDTH = 3.0
DEFAULT_SEMICIRCLE_COUNT = 9
DEFAULT_MAIN_LANE_COUNT = 2


@dataclass(frozen=True)
class PurePursuitCommand:
    left_rad_s: float
    right_rad_s: float
    steer_rad: float
    done: bool
    lookahead_point: tuple[float, float]
    curvature: float
    closest_progress: float


class PurePursuitTracker:
    def __init__(
        self,
        path: Iterable[Sequence[float]],
        *,
        wheelbase: float = DEFAULT_WHEELBASE,
        track_width: float = DEFAULT_TRACK_WIDTH,
        wheel_radius: float = DEFAULT_WHEEL_RADIUS,
        lookahead_gain: float = 1.0,
        min_lookahead: float = 0.5,
        max_lookahead: float = 2.0,
        alpha: float = 0.5,
        max_steer: float = math.pi / 2.0,
        goal_tolerance: float = 0.2,
        steer_sign: float = DEFAULT_STEER_SIGN,
    ) -> None:
        self.path = np.asarray(path, dtype=float)
        if self.path.ndim != 2 or self.path.shape[1] != 2 or len(self.path) < 2:
            raise ValueError("path must contain at least two 2D points")

        if wheelbase <= 0:
            raise ValueError("wheelbase must be positive")
        if track_width <= 0:
            raise ValueError("track_width must be positive")
        if wheel_radius <= 0:
            raise ValueError("wheel_radius must be positive")
        if min_lookahead <= 0 or max_lookahead < min_lookahead:
            raise ValueError("lookahead bounds must be positive and ordered")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if max_steer <= 0:
            raise ValueError("max_steer must be positive")
        if goal_tolerance < 0:
            raise ValueError("goal_tolerance must be non-negative")
        if steer_sign not in (-1.0, 1.0):
            raise ValueError("steer_sign must be 1.0 or -1.0")

        self.wheelbase = float(wheelbase)
        self.track_width = float(track_width)
        self.wheel_radius = float(wheel_radius)
        self.lookahead_gain = float(lookahead_gain)
        self.min_lookahead = float(min_lookahead)
        self.max_lookahead = float(max_lookahead)
        self.alpha = float(alpha)
        self.max_steer = float(max_steer)
        self.goal_tolerance = float(goal_tolerance)
        self.steer_sign = float(steer_sign)

        self._segments = self.path[1:] - self.path[:-1]
        self._segment_lengths = np.linalg.norm(self._segments, axis=1)
        if np.any(self._segment_lengths <= 0):
            raise ValueError("path segments must have non-zero length")
        self._cum_lengths = np.concatenate(([0.0], np.cumsum(self._segment_lengths)))
        self.total_length = float(self._cum_lengths[-1])
        self._last_progress = 0.0

    def compute(self, x: float, y: float, theta: float, v_mps: float) -> PurePursuitCommand:
        position = np.array([float(x), float(y)], dtype=float)
        goal = self.path[-1]
        closest_progress = max(self._closest_progress(position), self._last_progress)
        self._last_progress = closest_progress

        reached_goal_position = np.linalg.norm(position - goal) <= self.goal_tolerance
        reached_goal_progress = (self.total_length - closest_progress) <= self.goal_tolerance
        if reached_goal_position and reached_goal_progress:
            return PurePursuitCommand(
                left_rad_s=0.0,
                right_rad_s=0.0,
                steer_rad=0.0,
                done=True,
                lookahead_point=(float(goal[0]), float(goal[1])),
                curvature=0.0,
                closest_progress=closest_progress,
            )

        speed = float(v_mps)
        lookahead = float(
            np.clip(abs(speed) * self.lookahead_gain, self.min_lookahead, self.max_lookahead)
        )
        target_progress = min(closest_progress + lookahead, self.total_length)
        lookahead_point = self._point_at_progress(target_progress)

        dx, dy = lookahead_point - position
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        y_body = -sin_t * dx + cos_t * dy
        curvature = 2.0 * y_body / (lookahead * lookahead)

        omega_diff = self.alpha * curvature * speed
        left_mps = speed - omega_diff * self.track_width / 2.0
        right_mps = speed + omega_diff * self.track_width / 2.0
        steer = self.steer_sign * math.atan((1.0 - self.alpha) * curvature * self.wheelbase)
        steer = float(np.clip(steer, -self.max_steer, self.max_steer))

        return PurePursuitCommand(
            left_rad_s=left_mps / self.wheel_radius,
            right_rad_s=right_mps / self.wheel_radius,
            steer_rad=steer,
            done=False,
            lookahead_point=(float(lookahead_point[0]), float(lookahead_point[1])),
            curvature=float(curvature),
            closest_progress=closest_progress,
        )

    def _closest_progress(self, position: np.ndarray) -> float:
        best_dist_sq = math.inf
        best_progress = 0.0

        for index, segment in enumerate(self._segments):
            start = self.path[index]
            length = self._segment_lengths[index]
            t = float(np.dot(position - start, segment) / (length * length))
            t = float(np.clip(t, 0.0, 1.0))
            projection = start + t * segment
            dist_sq = float(np.dot(position - projection, position - projection))
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_progress = float(self._cum_lengths[index] + t * length)

        return best_progress

    def _point_at_progress(self, progress: float) -> np.ndarray:
        progress = float(np.clip(progress, 0.0, self.total_length))
        if progress >= self.total_length:
            return self.path[-1].copy()

        index = int(np.searchsorted(self._cum_lengths, progress, side="right") - 1)
        index = min(index, len(self._segments) - 1)
        segment_start = self._cum_lengths[index]
        t = (progress - segment_start) / self._segment_lengths[index]
        return self.path[index] + t * self._segments[index]


def yaw_from_quat(quat: Sequence[float]) -> float:
    """Return yaw from an Isaac-style quaternion tuple ordered as (w, x, y, z)."""
    w, x, y, z = quat
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def generate_lawnmower_path(
    *,
    field_length: float = DEFAULT_FIELD_LENGTH,
    field_width: float = DEFAULT_FIELD_WIDTH,
    semicircle_count: int = DEFAULT_SEMICIRCLE_COUNT,
    turn_samples: int = 13,
) -> list[tuple[float, float]]:
    """Generate a 30m x 3m-style coverage path with alternating semicircle turns.

    Coordinates follow the planning sketch: start at the upper-left corner,
    drive along the field length, then use headland semicircles to step down.
    """
    if field_length <= 0:
        raise ValueError("field_length must be positive")
    if field_width <= 0:
        raise ValueError("field_width must be positive")
    if semicircle_count <= 0:
        raise ValueError("semicircle_count must be positive")
    if turn_samples < 2:
        raise ValueError("turn_samples must be at least 2")

    lane_spacing = field_width / float(semicircle_count)
    turn_radius = lane_spacing / 2.0
    path: list[tuple[float, float]] = [(0.0, float(field_width))]

    for turn_index in range(semicircle_count):
        y_top = field_width - turn_index * lane_spacing
        y_bottom = y_top - lane_spacing
        going_right = turn_index % 2 == 0
        straight_end_x = field_length if going_right else 0.0
        side_sign = 1.0 if going_right else -1.0
        center_x = straight_end_x
        center_y = y_top - turn_radius

        _append_unique(path, (straight_end_x, y_top))
        for angle in np.linspace(math.pi / 2.0, -math.pi / 2.0, turn_samples):
            x = center_x + side_sign * turn_radius * math.cos(float(angle))
            y = center_y + turn_radius * math.sin(float(angle))
            _append_unique(path, (x, y))
        _append_unique(path, (straight_end_x, y_bottom))

    final_x = field_length if semicircle_count % 2 == 1 else 0.0
    _append_unique(path, (final_x, 0.0))
    return path


def generate_main_lane_path(
    *,
    field_length: float = DEFAULT_FIELD_LENGTH,
    field_width: float = DEFAULT_FIELD_WIDTH,
    lane_count: int = DEFAULT_MAIN_LANE_COUNT,
    turn_samples: int = 25,
) -> list[tuple[float, float]]:
    """Generate a feasible demo path with 2 or 3 main vehicle-center lanes."""
    if field_length <= 0:
        raise ValueError("field_length must be positive")
    if field_width <= 0:
        raise ValueError("field_width must be positive")
    if lane_count < 2:
        raise ValueError("lane_count must be at least 2")
    if turn_samples < 2:
        raise ValueError("turn_samples must be at least 2")

    lane_spacing = field_width / float(lane_count - 1)
    turn_radius = lane_spacing / 2.0
    path: list[tuple[float, float]] = [(0.0, float(field_width))]

    for turn_index in range(lane_count - 1):
        y_top = field_width - turn_index * lane_spacing
        y_bottom = y_top - lane_spacing
        going_right = turn_index % 2 == 0
        straight_end_x = field_length if going_right else 0.0
        side_sign = 1.0 if going_right else -1.0
        center_y = y_top - turn_radius

        _append_unique(path, (straight_end_x, y_top))
        for angle in np.linspace(math.pi / 2.0, -math.pi / 2.0, turn_samples):
            x = straight_end_x + side_sign * turn_radius * math.cos(float(angle))
            y = center_y + turn_radius * math.sin(float(angle))
            _append_unique(path, (x, y))
        _append_unique(path, (straight_end_x, y_bottom))

    final_x = field_length if lane_count % 2 == 1 else 0.0
    _append_unique(path, (final_x, 0.0))
    return path


def transform_path_to_pose(
    path: Iterable[Sequence[float]],
    *,
    x: float,
    y: float,
    theta: float,
) -> list[tuple[float, float]]:
    """Move a local path so its first point starts at the robot pose."""
    points = np.asarray(path, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("path must contain at least two 2D points")

    local_start = points[0].copy()
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    anchored: list[tuple[float, float]] = []
    for point in points:
        dx, dy = point - local_start
        world_x = x + cos_t * dx - sin_t * dy
        world_y = y + sin_t * dx + cos_t * dy
        anchored.append((float(world_x), float(world_y)))
    return anchored


def apply_command(bot, command: PurePursuitCommand) -> None:
    """Apply a computed command to PaddyRobotController-like objects."""
    bot.set_wheel_speeds(command.left_rad_s, command.right_rad_s)
    bot.set_steering_angle(command.steer_rad)


def _append_unique(path: list[tuple[float, float]], point: tuple[float, float]) -> None:
    if path and math.isclose(path[-1][0], point[0], abs_tol=1e-9) and math.isclose(
        path[-1][1], point[1], abs_tol=1e-9
    ):
        return
    path.append((float(point[0]), float(point[1])))


__all__ = [
    "DEFAULT_FIELD_LENGTH",
    "DEFAULT_MAIN_LANE_COUNT",
    "DEFAULT_STEER_SIGN",
    "DEFAULT_SEMICIRCLE_COUNT",
    "DEFAULT_FIELD_WIDTH",
    "DEFAULT_TRACK_WIDTH",
    "DEFAULT_WHEEL_RADIUS",
    "DEFAULT_WHEELBASE",
    "PurePursuitCommand",
    "PurePursuitTracker",
    "apply_command",
    "generate_main_lane_path",
    "generate_lawnmower_path",
    "transform_path_to_pose",
    "yaw_from_quat",
]
