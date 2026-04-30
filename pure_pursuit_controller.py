"""Pure Pursuit path tracking for the 3-wheel paddy robot.

This module is intentionally Isaac-independent for the core tracker, but also
includes small helpers to turn *full 3D* Isaac poses (position + wxyz quaternion)
into a planar (x, y, yaw) tracking frame. That matches common practice in Isaac
Sim and other simulators: the path lives on a ground plane, while the vehicle
state comes from a rigid body in SE(3).

IMPORTANT: Do not ``import pytest`` here. Isaac Sim's kit Python often does not
include pytest; tests live only under ``tests/`` and run with your dev venv.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Sequence

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
# Ground plane span for clamp_field_length / corner start pose. Must match
# ``ground_setup.py`` GroundPlane ``size`` so requested straights are not
# capped far below DEFAULT_FIELD_LENGTH on a small square.
DEFAULT_GROUND_SIZE = 40.0
# Reserve extra forward length in ``clamp_field_length_to_ground`` so the 180° apex stays
# on the square when the vehicle cuts wide vs the centreline polyline.
CLAMP_UTURN_APEX_HEADROOM_M = 0.45

# --- Physics / materials (keep in sync with ``ground_setup.py`` / ``robot_setup.py``) ---
GRAVITY_MPS2 = 9.81
GROUND_CONTACT_STIFFNESS = 800.0
GROUND_CONTACT_DAMPING = 1000.0
GROUND_STATIC_FRICTION = 0.6
GROUND_DYNAMIC_FRICTION = 0.45
TIRE_STATIC_FRICTION = 0.8
TIRE_DYNAMIC_FRICTION = 0.7
# Total sprung mass order-of-magnitude from URDF + robot_setup ``MASS_CONFIG`` (~375 kg).
DEFAULT_ROBOT_TOTAL_MASS_KG = 375.0


@dataclass(frozen=True)
class PaddySimPhysics:
    """Bundled contact/friction numbers documented in this repo's Isaac setup scripts."""

    ground_static_mu: float = GROUND_STATIC_FRICTION
    ground_dynamic_mu: float = GROUND_DYNAMIC_FRICTION
    tire_static_mu: float = TIRE_STATIC_FRICTION
    tire_dynamic_mu: float = TIRE_DYNAMIC_FRICTION
    ground_contact_stiffness: float = GROUND_CONTACT_STIFFNESS
    ground_contact_damping: float = GROUND_CONTACT_DAMPING
    total_mass_kg: float = DEFAULT_ROBOT_TOTAL_MASS_KG


def recommended_max_lateral_accel_mps2(
    physics: PaddySimPhysics | None = None,
    *,
    use_dynamic_mu: bool = True,
    friction_coupling: float = 0.5,
) -> float:
    """Conservative lateral acceleration cap: a ≈ friction_coupling * min(mu) * g.

    ``friction_coupling`` accounts for combined slip, uneven normal load, and mud
    compliance (see compliant contact stiffness in ``ground_setup.py``).
    """
    if not 0.0 < friction_coupling <= 1.0:
        raise ValueError("friction_coupling must be in (0, 1]")
    p = physics or PaddySimPhysics()
    mu_g = p.ground_dynamic_mu if use_dynamic_mu else p.ground_static_mu
    mu_t = p.tire_dynamic_mu if use_dynamic_mu else p.tire_static_mu
    mu = min(mu_g, mu_t)
    return float(friction_coupling * mu * GRAVITY_MPS2)


@dataclass(frozen=True)
class RecommendedPurePursuitProfile:
    """仓库维护的默认推荐：泥地 + 后桥转向 + PhysX 线速度观测。

    在「能跟住路径」与「少打滑、少抖舵」之间偏保守；调参只改这一处即可分叉版本。
    """

    version: str = "2026.05-mud-v8"
    friction_coupling: float = 0.48
    lookahead_gain: float = 1.0
    min_lookahead: float = 1.6
    max_lookahead: float = 2.4
    alpha: float = 0.55
    goal_tolerance: float = 0.3
    progress_search_behind: float = 1.0
    progress_search_ahead: float = 1.2
    progress_anchor_full_search_m: float = 0.0
    progress_tie_break_epsilon_m: float = 0.38
    progress_stuck_path_err_m: float = 1.05
    progress_stuck_escape_m: float = 0.42
    heading_association_far_scale_m: float = 1.15
    cte_gain_far_error_scale: float = 2.15
    cte_far_error_m: float = 1.05
    heading_gain: float = 0.0
    segment_jump_hysteresis_m: float = 0.35
    segment_penalty_m: float = 0.85
    progress_retrack_margin_m: float = 0.12
    progress_relocalize_cte_m: float = 1.05
    heading_association_weight: float = 0.25
    cte_softening_m: float = 0.42
    cte_gain: float = 0.0
    path_lookahead_gain: float = 0.0
    alpha_min: float = 0.2
    alpha_curvature_half: float = 0.085

    def lateral_accel_cap_mps2(self, physics: PaddySimPhysics | None = None) -> float:
        return recommended_max_lateral_accel_mps2(
            physics or PaddySimPhysics(),
            friction_coupling=self.friction_coupling,
        )

    def tracker_kwargs(
        self,
        *,
        max_steer: float,
        steer_sign: float,
        physics: PaddySimPhysics | None = None,
    ) -> dict[str, Any]:
        return {
            "lookahead_gain": self.lookahead_gain,
            "min_lookahead": self.min_lookahead,
            "max_lookahead": self.max_lookahead,
            "alpha": self.alpha,
            "max_steer": float(max_steer),
            "goal_tolerance": self.goal_tolerance,
            "steer_sign": float(steer_sign),
            "progress_search_behind": self.progress_search_behind,
            "progress_search_ahead": self.progress_search_ahead,
            "progress_anchor_full_search_m": self.progress_anchor_full_search_m,
            "progress_tie_break_epsilon_m": self.progress_tie_break_epsilon_m,
            "progress_stuck_path_err_m": self.progress_stuck_path_err_m,
            "progress_stuck_escape_m": self.progress_stuck_escape_m,
            "heading_association_far_scale_m": self.heading_association_far_scale_m,
            "cte_gain_far_error_scale": self.cte_gain_far_error_scale,
            "cte_far_error_m": self.cte_far_error_m,
            "heading_gain": self.heading_gain,
            "segment_jump_hysteresis_m": self.segment_jump_hysteresis_m,
            "segment_penalty_m": self.segment_penalty_m,
            "progress_retrack_margin_m": self.progress_retrack_margin_m,
            "progress_relocalize_cte_m": self.progress_relocalize_cte_m,
            "heading_association_weight": self.heading_association_weight,
            "max_lateral_accel_mps2": self.lateral_accel_cap_mps2(physics),
            "cte_gain": self.cte_gain,
            "cte_softening_m": self.cte_softening_m,
            "path_lookahead_gain": self.path_lookahead_gain,
            "alpha_min": self.alpha_min,
            "alpha_curvature_half": self.alpha_curvature_half,
        }


@dataclass(frozen=True)
class RecommendedIsaacPurePursuitScript:
    """Script Editor 层推荐：限速、爬升 / 执行器限幅、打滑启发式。"""

    version: str = "2026.05-mud-v8"
    cruise_speed_mps: float = 0.85
    turn_speed_mps: float = 0.15
    slowdown_curvature: float = 0.11
    straight_curvature_deadband: float = 0.03
    accel_limit_mps2: float = 0.55
    decel_limit_mps2: float = 0.55
    max_wheel_rad_s: float = 12.0
    max_wheel_accel_rad_s2: float = 8.0
    max_steer_deg: float = 50.0
    max_steer_rate_deg_s: float = 60.0
    dynamic_margin_m: float = 0.5
    debug_period_s: float = 0.5
    stall_speed_mps: float = 0.03
    stall_curvature: float = 0.5
    stall_crawl_speed_mps: float = 0.25
    stall_max_steer_deg: float = 15.0
    slip_speed_fraction: float = 0.4

    @property
    def max_steer_rad(self) -> float:
        return float(math.radians(self.max_steer_deg))

    @property
    def max_steer_rate_rad_s(self) -> float:
        return float(math.radians(self.max_steer_rate_deg_s))

    @property
    def stall_max_steer_rad(self) -> float:
        return float(math.radians(self.stall_max_steer_deg))


RECOMMENDED_PURE_PURSUIT = RecommendedPurePursuitProfile()
RECOMMENDED_SCRIPT = RecommendedIsaacPurePursuitScript()


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
        allow_reverse_wheels: bool = False,
        progress_search_behind: float = 1.0,
        progress_search_ahead: float = math.inf,
        heading_gain: float = 0.0,
        segment_jump_hysteresis_m: float = 0.35,
        segment_penalty_m: float = 0.85,
        progress_retrack_margin_m: float = 0.12,
        progress_relocalize_cte_m: float = 1.05,
        progress_anchor_full_search_m: float = 2.58,
        progress_tie_break_epsilon_m: float = 0.38,
        progress_stuck_path_err_m: float = 1.05,
        progress_stuck_escape_m: float = 0.42,
        heading_association_far_scale_m: float = 1.15,
        cte_gain_far_error_scale: float = 2.15,
        cte_far_error_m: float = 1.05,
        heading_association_weight: float = 0.25,
        max_lateral_accel_mps2: float = 0.0,
        cte_gain: float = 0.0,
        cte_softening_m: float = 0.5,
        path_lookahead_gain: float = 0.0,
        alpha_min: float | None = None,
        alpha_curvature_half: float = 0.1,
        phase_straight_end_progress: float | None = None,
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
        if progress_search_behind < 0:
            raise ValueError("progress_search_behind must be non-negative")
        if progress_search_ahead <= 0:
            raise ValueError("progress_search_ahead must be positive")
        if heading_gain < 0:
            raise ValueError("heading_gain must be non-negative")
        if segment_jump_hysteresis_m < 0:
            raise ValueError("segment_jump_hysteresis_m must be non-negative")
        if segment_penalty_m < 0:
            raise ValueError("segment_penalty_m must be non-negative")
        if progress_retrack_margin_m < 0:
            raise ValueError("progress_retrack_margin_m must be non-negative")
        if progress_relocalize_cte_m < 0:
            raise ValueError("progress_relocalize_cte_m must be non-negative")
        if progress_anchor_full_search_m < 0:
            raise ValueError("progress_anchor_full_search_m must be non-negative (0 disables)")
        if progress_tie_break_epsilon_m < 0:
            raise ValueError("progress_tie_break_epsilon_m must be non-negative")
        if progress_stuck_path_err_m <= 0 or progress_stuck_escape_m <= 0:
            raise ValueError("stuck-escape distances must be positive")
        if heading_association_far_scale_m <= 0:
            raise ValueError("heading_association_far_scale_m must be positive")
        if cte_gain_far_error_scale < 1.0:
            raise ValueError("cte_gain_far_error_scale must be >= 1")
        if cte_far_error_m <= 0:
            raise ValueError("cte_far_error_m must be positive")
        if heading_association_weight < 0:
            raise ValueError("heading_association_weight must be non-negative")
        if max_lateral_accel_mps2 < 0:
            raise ValueError("max_lateral_accel_mps2 must be non-negative")
        if cte_gain < 0:
            raise ValueError("cte_gain must be non-negative")
        if cte_softening_m < 0:
            raise ValueError("cte_softening_m must be non-negative")
        if path_lookahead_gain < 0:
            raise ValueError("path_lookahead_gain must be non-negative")
        if alpha_min is not None and not 0.0 <= float(alpha_min) <= 1.0:
            raise ValueError("alpha_min must be in [0, 1] when provided")
        if alpha_curvature_half <= 0:
            raise ValueError("alpha_curvature_half must be positive")
        if phase_straight_end_progress is not None and float(phase_straight_end_progress) < 0:
            raise ValueError("phase_straight_end_progress must be non-negative when provided")

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
        self.allow_reverse_wheels = bool(allow_reverse_wheels)
        self.progress_search_behind = float(progress_search_behind)
        self.progress_search_ahead = float(progress_search_ahead)
        self.heading_gain = float(heading_gain)
        self.segment_jump_hysteresis_m = float(segment_jump_hysteresis_m)
        self.segment_penalty_m = float(segment_penalty_m)
        self.progress_retrack_margin_m = float(progress_retrack_margin_m)
        self.progress_relocalize_cte_m = float(progress_relocalize_cte_m)
        self.progress_anchor_full_search_m = float(progress_anchor_full_search_m)
        self.progress_tie_break_epsilon_m = float(progress_tie_break_epsilon_m)
        self.progress_stuck_path_err_m = float(progress_stuck_path_err_m)
        self.progress_stuck_escape_m = float(progress_stuck_escape_m)
        self.heading_association_far_scale_m = float(heading_association_far_scale_m)
        self.cte_gain_far_error_scale = float(cte_gain_far_error_scale)
        self.cte_far_error_m = float(cte_far_error_m)
        self.heading_association_weight = float(heading_association_weight)
        self.max_lateral_accel_mps2 = float(max_lateral_accel_mps2)
        self.cte_gain = float(cte_gain)
        self.cte_softening_m = float(cte_softening_m)
        self.path_lookahead_gain = float(path_lookahead_gain)
        self.alpha_min = None if alpha_min is None else float(alpha_min)
        self.alpha_curvature_half = float(alpha_curvature_half)
        self.phase_straight_end_progress = (
            None if phase_straight_end_progress is None else float(phase_straight_end_progress)
        )

        self._segments = self.path[1:] - self.path[:-1]
        self._segment_lengths = np.linalg.norm(self._segments, axis=1)
        if np.any(self._segment_lengths <= 0):
            raise ValueError("path segments must have non-zero length")
        self._cum_lengths = np.concatenate(([0.0], np.cumsum(self._segment_lengths)))
        self.total_length = float(self._cum_lengths[-1])
        if self.phase_straight_end_progress is not None and self.phase_straight_end_progress > self.total_length + 1e-9:
            raise ValueError("phase_straight_end_progress exceeds path length")
        self._last_progress = 0.0
        self._has_progress = False
        self._last_segment_index = 0
        self._last_xy: np.ndarray | None = None
        self._prev_closest_for_stuck = -1.0e9

    def compute(self, x: float, y: float, theta: float, v_mps: float) -> PurePursuitCommand:
        position = np.array([float(x), float(y)], dtype=float)
        goal = self.path[-1]
        speed = float(v_mps)
        lookahead = float(
            np.clip(abs(speed) * self.lookahead_gain, self.min_lookahead, self.max_lookahead)
        )

        if self._has_progress:
            search_ahead = max(float(self.progress_search_ahead), lookahead + 0.5)
            measured_progress = self._closest_progress(
                position,
                min_progress=self._last_progress,
                max_progress=min(self.total_length, self._last_progress + search_ahead),
                continuity=False,
            )
            closest_progress = max(self._last_progress, measured_progress)
        else:
            closest_progress = self._closest_progress(position, continuity=False)
            self._has_progress = True

        closest_progress = float(np.clip(closest_progress, 0.0, self.total_length))
        self._last_xy = position.copy()
        self._last_progress = closest_progress
        self._last_segment_index = int(
            min(max(np.searchsorted(self._cum_lengths, closest_progress, side="right") - 1, 0), len(self._segments) - 1)
        )

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

        target_progress = min(closest_progress + lookahead, self.total_length)
        lookahead_point = self._point_at_progress(target_progress)

        dx, dy = lookahead_point - position
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        x_body = cos_t * dx + sin_t * dy
        y_body = -sin_t * dx + cos_t * dy
        dist_sq = x_body * x_body + y_body * y_body
        curvature = 0.0 if dist_sq <= 1e-9 else float(2.0 * y_body / dist_sq)
        max_curvature_geom = math.tan(self.max_steer) / self.wheelbase
        if self.max_lateral_accel_mps2 > 0.0 and abs(speed) > 0.08:
            max_curvature_lat = self.max_lateral_accel_mps2 / (speed * speed)
            max_curvature = min(max_curvature_geom, max_curvature_lat)
        else:
            max_curvature = max_curvature_geom
        curvature = float(np.clip(curvature, -max_curvature, max_curvature))

        # The robot is a rear-steering tricycle: front differential is only a
        # mild assist, never the primary turning mechanism in tight turns.
        omega_diff = self.alpha * curvature * speed
        left_mps = speed - omega_diff * self.track_width / 2.0
        right_mps = speed + omega_diff * self.track_width / 2.0
        if not self.allow_reverse_wheels and speed >= 0.0:
            left_mps = max(left_mps, 0.0)
            right_mps = max(right_mps, 0.0)
        steer = self.steer_sign * math.atan(curvature * self.wheelbase)
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

    def _closest_progress(
        self,
        position: np.ndarray,
        *,
        min_progress: float = 0.0,
        max_progress: float | None = None,
        continuity: bool = True,
        continuity_anchor: int | None = None,
        guide_progress: float | None = None,
        heading: float | None = None,
        heading_weight_scale: float = 1.0,
    ) -> float:
        min_progress = float(np.clip(min_progress, 0.0, self.total_length))
        max_progress = self.total_length if max_progress is None else float(np.clip(max_progress, 0.0, self.total_length))
        if max_progress < min_progress:
            min_progress, max_progress = max_progress, min_progress

        ref_progress = float(np.clip(self._last_progress, min_progress, max_progress))
        guide: float | None = None
        if guide_progress is not None:
            guide = float(np.clip(guide_progress, min_progress, max_progress))

        guide_point = self._point_at_progress(guide) if guide is not None else None
        guide_dist_sq = (
            float(np.dot(position - guide_point, position - guide_point)) if guide_point is not None else None
        )

        use_continuity = continuity and self._has_progress
        ref_index = self._last_segment_index
        if continuity_anchor is not None:
            ref_index = int(continuity_anchor)
        best_score = math.inf
        best_progress = min_progress
        best_delta_prog = math.inf
        best_geom_dist_sq = math.inf
        tie_band_sq = float(self.progress_tie_break_epsilon_m) ** 2
        tie_geom_min_sq = 0.88 * 0.88
        hgw = float(heading_weight_scale)

        for index, segment in enumerate(self._segments):
            segment_start = float(self._cum_lengths[index])
            segment_end = float(self._cum_lengths[index + 1])
            seg_len = float(self._segment_lengths[index])
            start = self.path[index]

            if segment_end < min_progress or segment_start > max_progress:
                continue

            t_min = float(np.clip((min_progress - segment_start) / seg_len, 0.0, 1.0))
            t_max = float(np.clip((max_progress - segment_start) / seg_len, 0.0, 1.0))
            if t_min > t_max:
                t_min, t_max = t_max, t_min

            t_raw = float(np.dot(position - start, segment) / (seg_len * seg_len))
            t = float(np.clip(t_raw, t_min, t_max))
            progress = float(segment_start + t * seg_len)
            projection = start + t * segment
            dist_sq = float(np.dot(position - projection, position - projection))

            jump = abs(index - ref_index)
            penalty_sq = (
                0.0
                if (not use_continuity or jump <= 1)
                else (float(self.segment_penalty_m) ** 2)
            )
            segment_heading = float(math.atan2(float(segment[1]), float(segment[0])))
            heading_cost = 0.0
            if heading is not None and self.heading_association_weight > 0.0:
                heading_err = self._wrap_angle(segment_heading - float(heading))
                heading_cost = (
                    float(self.heading_association_weight)
                    * hgw
                    * (1.0 - math.cos(heading_err))
                )

            loss_sq = dist_sq + penalty_sq + heading_cost
            delta_prog = abs(progress - ref_progress)
            if (
                guide_dist_sq is not None
                and guide is not None
                and use_continuity
                and (
                    dist_sq > guide_dist_sq + float(self.segment_jump_hysteresis_m) ** 2
                    or (
                        abs(progress - guide) > float(self.segment_jump_hysteresis_m)
                        and dist_sq > guide_dist_sq + 1e-6
                    )
                )
            ):
                loss_sq += float(self.segment_penalty_m) ** 2

            if loss_sq + 1e-9 < best_score:
                best_score = loss_sq
                best_progress = progress
                best_delta_prog = delta_prog
                best_geom_dist_sq = dist_sq
            elif (
                dist_sq <= best_geom_dist_sq + tie_band_sq
                and best_geom_dist_sq > tie_geom_min_sq
                and progress + 1e-6 > best_progress
            ):
                best_progress = progress
                best_score = loss_sq
                best_delta_prog = delta_prog
                best_geom_dist_sq = min(best_geom_dist_sq, dist_sq)

            for edge_t in (t_min, t_max):
                edge_progress = float(segment_start + edge_t * seg_len)
                edge_point = start + edge_t * segment
                edge_dist_sq = float(np.dot(position - edge_point, position - edge_point))

                jump = abs(index - ref_index)
                penalty_sq = (
                    0.0
                    if (not use_continuity or jump <= 1)
                    else (float(self.segment_penalty_m) ** 2)
                )
                segment_heading = float(math.atan2(float(segment[1]), float(segment[0])))
                heading_cost = 0.0
                if heading is not None and self.heading_association_weight > 0.0:
                    heading_err = self._wrap_angle(segment_heading - float(heading))
                    heading_cost = (
                        float(self.heading_association_weight)
                        * hgw
                        * (1.0 - math.cos(heading_err))
                    )

                loss_sq = edge_dist_sq + penalty_sq + heading_cost
                delta_prog = abs(edge_progress - ref_progress)
                if (
                    guide_dist_sq is not None
                    and guide is not None
                    and use_continuity
                    and (
                        edge_dist_sq > guide_dist_sq + float(self.segment_jump_hysteresis_m) ** 2
                        or (
                            abs(edge_progress - guide) > float(self.segment_jump_hysteresis_m)
                            and edge_dist_sq > guide_dist_sq + 1e-6
                        )
                    )
                ):
                    loss_sq += float(self.segment_penalty_m) ** 2

                if loss_sq + 1e-9 < best_score:
                    best_score = loss_sq
                    best_progress = edge_progress
                    best_delta_prog = delta_prog
                    best_geom_dist_sq = edge_dist_sq
                elif (
                    edge_dist_sq <= best_geom_dist_sq + tie_band_sq
                    and best_geom_dist_sq > tie_geom_min_sq
                    and edge_progress + 1e-6 > best_progress
                ):
                    best_progress = edge_progress
                    best_score = loss_sq
                    best_delta_prog = delta_prog
                    best_geom_dist_sq = min(best_geom_dist_sq, edge_dist_sq)

        return best_progress

    def _unit_tangent_at_progress(self, progress: float) -> np.ndarray:
        heading = self._path_heading_at_progress(progress)
        return np.array([math.cos(heading), math.sin(heading)], dtype=float)

    def _path_curvature_estimate(self, progress: float, window_m: float) -> float:
        window_m = float(max(window_m, 0.05))
        p1 = float(np.clip(progress + window_m, 0.0, self.total_length))
        if p1 <= progress + 1e-6:
            return 0.0
        h0 = self._path_heading_at_progress(progress)
        h1 = self._path_heading_at_progress(p1)
        return float(self._wrap_angle(h1 - h0) / (p1 - progress))

    def _path_heading_at_progress(self, progress: float) -> float:
        progress = float(np.clip(progress, 0.0, self.total_length))
        if progress >= self.total_length - 1e-9:
            tangent = self._segments[-1]
        else:
            index = int(np.searchsorted(self._cum_lengths, progress, side="right") - 1)
            index = min(index, len(self._segments) - 1)
            tangent = self._segments[index]
        return float(math.atan2(float(tangent[1]), float(tangent[0])))

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

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


def quat_from_yaw(theta: float) -> tuple[float, float, float, float]:
    """Return an Isaac-style quaternion tuple ordered as (w, x, y, z)."""
    half = theta / 2.0
    return (float(math.cos(half)), 0.0, 0.0, float(math.sin(half)))


def quat_wxyz_to_rotation_matrix(quat: Sequence[float]) -> np.ndarray:
    """Rotation R (body -> world) with v_world = R @ v_body.

    Quaternion order matches Isaac / USD style: (w, x, y, z).
    """
    w, x, y, z = (float(quat[i]) for i in range(4))
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def rotate_body_vector_to_world(quat_wxyz: Sequence[float], body_vec: Sequence[float]) -> np.ndarray:
    """Map a fixed vector defined in the robot body frame into world coordinates."""
    rotation = quat_wxyz_to_rotation_matrix(quat_wxyz)
    return rotation @ np.asarray(body_vec, dtype=float)


def tracking_point_world(
    position_world: Sequence[float],
    quat_wxyz: Sequence[float],
    offset_body: Sequence[float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """World position of a body-fixed tracking point (e.g. rear axle) under the current pose."""
    base = np.asarray(position_world, dtype=float).reshape(3)
    delta = rotate_body_vector_to_world(quat_wxyz, offset_body)
    return base + delta


def planar_yaw_from_pose(
    quat_wxyz: Sequence[float],
    *,
    body_forward: Sequence[float] = (1.0, 0.0, 0.0),
    world_up: Sequence[float] = (0.0, 0.0, 1.0),
) -> float:
    """Yaw on the horizontal plane: full orientation projected away from ``world_up``.

    If roll/pitch are small (typical flat-field sim), this matches ``yaw_from_quat``.
    For noticeable roll/pitch, this tracks the *horizontal* heading of the body x-axis
    (or ``body_forward``) instead of extracting yaw from the quaternion in isolation.
    """
    up = np.asarray(world_up, dtype=float).reshape(3)
    norm_up = float(np.linalg.norm(up))
    if norm_up < 1e-9:
        return yaw_from_quat(quat_wxyz)
    up = up / norm_up

    forward_world = rotate_body_vector_to_world(quat_wxyz, body_forward)
    forward_planar = forward_world - float(np.dot(forward_world, up)) * up
    norm_f = float(np.linalg.norm(forward_planar))
    if norm_f < 1e-9:
        return yaw_from_quat(quat_wxyz)

    forward_planar = forward_planar / norm_f
    reference = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(up, reference))) > 0.95:
        reference = np.array([1.0, 0.0, 0.0], dtype=float)
    east = np.cross(up, reference)
    east = east / float(np.linalg.norm(east))
    north = np.cross(east, up)
    north = north / float(np.linalg.norm(north))
    return float(math.atan2(float(np.dot(forward_planar, east)), float(np.dot(forward_planar, north))))


def planar_speed_from_linear_velocity(
    linear_velocity_world: Sequence[float],
    *,
    world_up: Sequence[float] = (0.0, 0.0, 1.0),
) -> float:
    """Magnitude of linear velocity after removing motion along ``world_up`` (climb/dive)."""
    velocity = np.asarray(linear_velocity_world, dtype=float).reshape(3)
    up = np.asarray(world_up, dtype=float).reshape(3)
    norm_up = float(np.linalg.norm(up))
    if norm_up < 1e-9:
        return float(np.linalg.norm(velocity))
    up = up / norm_up
    planar = velocity - float(np.dot(velocity, up)) * up
    return float(np.linalg.norm(planar))


def tracking_pose_for_planar_path(
    position_world: Sequence[float],
    quat_wxyz: Sequence[float],
    *,
    offset_body: Sequence[float] = (0.0, 0.0, 0.0),
    body_forward: Sequence[float] = (1.0, 0.0, 0.0),
    world_up: Sequence[float] = (0.0, 0.0, 1.0),
) -> tuple[float, float, float, float]:
    """Convenience: (track_x, track_y, track_z, yaw_planar) for 2D path tracking."""
    point = tracking_point_world(position_world, quat_wxyz, offset_body)
    yaw = planar_yaw_from_pose(quat_wxyz, body_forward=body_forward, world_up=world_up)
    return float(point[0]), float(point[1]), float(point[2]), yaw


def generate_lawnmower_path(
    *,
    field_length: float = DEFAULT_FIELD_LENGTH,
    field_width: float = DEFAULT_FIELD_WIDTH,
    semicircle_count: int = DEFAULT_SEMICIRCLE_COUNT,
    turn_samples: int = 13,
) -> list[tuple[float, float]]:
    """Boustrophedon path: long straights + 180° semicircle headlands.

    ``field_length`` is the straight run (e.g. 30 m in the field diagram).
    ``field_width`` is the total span stepped in *y* (``row_spacing * num_rows``);
    each step uses semicircle radius ``(field_width / semicircle_count) / 2`` (e.g.
    3 m row spacing ⇒ 1.5 m radius for a half-circle turn between passes).
    Start polyline vertex is the upper-left of the local patch, first segment toward +x.
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
    turn_samples: int = 40,
    first_straight_direction: str = "east",
) -> list[tuple[float, float]]:
    """Generate a feasible demo path with 2 or 3 main vehicle-center lanes.

    U-turns are a **single** semicircle per transition (continuous :math:`\\pi` rad in
    heading along the polyline), not two quarter-turn polylines.

    ``first_straight_direction``:
    - ``"east"``: start at ``(0, field_width)``, first run toward ``+x`` (legacy).
    - ``"west"``: start at ``(field_length, field_width)``, first run toward ``-x``.
      Use this with a **right-edge** spawn and :math:`\\theta=\\pi`; do **not** use
      :func:`mirror_path_along_field_length` on an east path — mirroring flips the
      U-turn bulge and drives the arc **off** the strip.
    """
    if field_length <= 0:
        raise ValueError("field_length must be positive")
    if field_width <= 0:
        raise ValueError("field_width must be positive")
    if lane_count < 2:
        raise ValueError("lane_count must be at least 2")
    if turn_samples < 2:
        raise ValueError("turn_samples must be at least 2")
    if first_straight_direction not in ("east", "west"):
        raise ValueError("first_straight_direction must be 'east' or 'west'")

    lane_spacing = field_width / float(lane_count - 1)
    turn_radius = lane_spacing / 2.0
    if first_straight_direction == "east":
        path: list[tuple[float, float]] = [(0.0, float(field_width))]
    else:
        path = [(float(field_length), float(field_width))]

    for turn_index in range(lane_count - 1):
        y_top = field_width - turn_index * lane_spacing
        y_bottom = y_top - lane_spacing
        going_right = turn_index % 2 == 0
        if first_straight_direction == "east":
            straight_end_x = field_length if going_right else 0.0
            side_sign = 1.0 if going_right else -1.0
        else:
            straight_end_x = 0.0 if going_right else field_length
            side_sign = -1.0 if going_right else 1.0
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


def _forward_ray_positive_exit_distance(
    *,
    x: float,
    y: float,
    cos_theta: float,
    sin_theta: float,
    half: float,
) -> float:
    distances: list[float] = []
    if cos_theta > 1e-9:
        distances.append((half - x) / cos_theta)
    elif cos_theta < -1e-9:
        distances.append((-half - x) / cos_theta)
    if sin_theta > 1e-9:
        distances.append((half - y) / sin_theta)
    elif sin_theta < -1e-9:
        distances.append((-half - y) / sin_theta)
    return min((distance for distance in distances if distance > 0.0), default=math.inf)


def straight_run_budget_on_square_ground(
    *,
    x: float,
    y: float,
    theta: float,
    requested_length: float,
    turn_radius: float,
    ground_size: float = DEFAULT_GROUND_SIZE,
    margin: float = 1.0,
    min_length: float = 2.0,
    dynamic_margin: float = 0.0,
    vehicle_half_length: float = 0.0,
) -> dict[str, float]:
    """Transparency helper: same numbers as inside ``clamp_field_length_to_ground``.

    Use this in Script Editor to print *why* ``FIELD_LENGTH`` is what it is, without
    trusting commentary. Keys are all in metres (or your planning length unit).
    """
    if math.isnan(requested_length):
        raise ValueError("requested_length must not be NaN")
    if math.isinf(requested_length):
        if requested_length < 0:
            raise ValueError("requested_length must not be -inf")
    elif requested_length <= 0:
        raise ValueError("requested_length must be positive")
    if turn_radius < 0:
        raise ValueError("turn_radius must be non-negative")
    if ground_size <= 0:
        raise ValueError("ground_size must be positive")
    if margin < 0:
        raise ValueError("margin must be non-negative")
    if min_length <= 0:
        raise ValueError("min_length must be positive")
    if dynamic_margin < 0:
        raise ValueError("dynamic_margin must be non-negative")
    if vehicle_half_length < 0:
        raise ValueError("vehicle_half_length must be non-negative")

    half = ground_size / 2.0
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    forward_to_edge = _forward_ray_positive_exit_distance(
        x=x, y=y, cos_theta=cos_t, sin_theta=sin_t, half=half
    )
    forward_extent = float(turn_radius) + 0.35 * float(vehicle_half_length) + CLAMP_UTURN_APEX_HEADROOM_M
    usable = float(forward_to_edge - margin - dynamic_margin - forward_extent)
    applied = float(np.clip(usable, min_length, requested_length))
    return {
      "forward_to_edge_m": float(forward_to_edge),
      "margin_m": float(margin),
      "dynamic_margin_m": float(dynamic_margin),
      "uturn_clearance_m": float(forward_extent),
      "usable_before_clip_m": usable,
      "requested_cap_m": float(requested_length) if math.isfinite(requested_length) else float("inf"),
      "field_length_applied_m": applied,
    }


def clamp_field_length_to_ground(
    *,
    x: float,
    y: float,
    theta: float,
    requested_length: float,
    turn_radius: float,
    ground_size: float = DEFAULT_GROUND_SIZE,
    margin: float = 1.0,
    min_length: float = 2.0,
    dynamic_margin: float = 0.0,
    vehicle_half_length: float = 0.0,
) -> float:
    """Clamp forward path length so the far turn stays inside a square ground plane.

    ``requested_length`` may be ``math.inf`` to use the maximum usable straight that
    still fits (given ``ground_size``, margins, and ``turn_radius``).
    """
    if math.isnan(requested_length):
        raise ValueError("requested_length must not be NaN")
    if math.isinf(requested_length):
        if requested_length < 0:
            raise ValueError("requested_length must not be -inf")
    elif requested_length <= 0:
        raise ValueError("requested_length must be positive")
    if turn_radius < 0:
        raise ValueError("turn_radius must be non-negative")
    if ground_size <= 0:
        raise ValueError("ground_size must be positive")
    if margin < 0:
        raise ValueError("margin must be non-negative")
    if min_length <= 0:
        raise ValueError("min_length must be positive")
    if dynamic_margin < 0:
        raise ValueError("dynamic_margin must be non-negative")
    if vehicle_half_length < 0:
        raise ValueError("vehicle_half_length must be non-negative")

    half = ground_size / 2.0
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    forward_to_edge = _forward_ray_positive_exit_distance(
        x=x, y=y, cos_theta=cos_t, sin_theta=sin_t, half=half
    )
    # ``turn_radius`` reserves space for the 180° arc apex beyond the lane corner (centerline).
    # Start pose already insets ``vehicle_half_length`` (or turn radius) from the platform edge,
    # so counting the **full** half-length again here over-shortens the first straight. Keep a
    # small coupling term for nose clearance in tight fits.
    forward_extent = float(turn_radius) + 0.35 * float(vehicle_half_length) + CLAMP_UTURN_APEX_HEADROOM_M
    usable_length = float(forward_to_edge - margin - dynamic_margin - forward_extent)
    return float(np.clip(usable_length, min_length, requested_length))


def compute_platform_corner_start_pose(
    *,
    ground_size: float = DEFAULT_GROUND_SIZE,
    field_width: float = DEFAULT_FIELD_WIDTH,
    lane_count: int = DEFAULT_MAIN_LANE_COUNT,
    track_width: float = DEFAULT_TRACK_WIDTH,
    wheelbase: float = DEFAULT_WHEELBASE,
    wheel_radius: float = DEFAULT_WHEEL_RADIUS,
    margin: float = 1.0,
    dynamic_margin: float = 0.0,
    theta: float = 0.0,
) -> tuple[float, float, float]:
    """Choose an upper-left platform start pose with vehicle and turn clearance."""
    if ground_size <= 0:
        raise ValueError("ground_size must be positive")
    if field_width <= 0:
        raise ValueError("field_width must be positive")
    if lane_count < 2:
        raise ValueError("lane_count must be at least 2")
    if track_width <= 0 or wheelbase <= 0 or wheel_radius <= 0:
        raise ValueError("vehicle dimensions must be positive")
    if margin < 0:
        raise ValueError("margin must be non-negative")
    if dynamic_margin < 0:
        raise ValueError("dynamic_margin must be non-negative")

    half = ground_size / 2.0
    lane_spacing = field_width / float(lane_count - 1)
    turn_radius = lane_spacing / 2.0
    vehicle_half_width = (track_width + 2.0 * wheel_radius) / 2.0
    vehicle_half_length = (wheelbase + 2.0 * wheel_radius) / 2.0

    x = -half + margin + dynamic_margin + max(vehicle_half_length, turn_radius)
    y = half - margin - dynamic_margin - vehicle_half_width
    return (float(x), float(y), float(theta))


def compute_platform_coverage_start_pose(
    *,
    ground_size: float = DEFAULT_GROUND_SIZE,
    turn_radius: float,
    track_width: float = DEFAULT_TRACK_WIDTH,
    wheelbase: float = DEFAULT_WHEELBASE,
    wheel_radius: float = DEFAULT_WHEEL_RADIUS,
    margin: float = 1.0,
    dynamic_margin: float = 0.0,
    start_edge: str = "upper_left_plus_x",
) -> tuple[float, float, float]:
    """Corner pose for lawnmower / boustrophedon paths (explicit ``turn_radius``).

    - ``upper_left_plus_x``: first straight runs toward +X (diagram start).
    - ``upper_right_minus_x``: first straight runs toward −X (common Isaac view).
    """
    if ground_size <= 0:
        raise ValueError("ground_size must be positive")
    if turn_radius < 0:
        raise ValueError("turn_radius must be non-negative")
    if track_width <= 0 or wheelbase <= 0 or wheel_radius <= 0:
        raise ValueError("vehicle dimensions must be positive")
    if margin < 0:
        raise ValueError("margin must be non-negative")
    if dynamic_margin < 0:
        raise ValueError("dynamic_margin must be non-negative")
    if start_edge not in ("upper_left_plus_x", "upper_right_minus_x"):
        raise ValueError("start_edge must be 'upper_left_plus_x' or 'upper_right_minus_x'")

    half = ground_size / 2.0
    vehicle_half_width = (track_width + 2.0 * wheel_radius) / 2.0
    vehicle_half_length = (wheelbase + 2.0 * wheel_radius) / 2.0
    inset = margin + dynamic_margin + max(vehicle_half_length, turn_radius)
    y = half - margin - dynamic_margin - vehicle_half_width
    if start_edge == "upper_left_plus_x":
        x = -half + inset
        theta = 0.0
    else:
        x = half - inset
        theta = math.pi
    return (float(x), float(y), float(theta))


def mirror_path_along_field_length(
    path: Iterable[Sequence[float]],
    *,
    field_length: float,
) -> list[tuple[float, float]]:
    """Reflect path in x about ``x = field_length / 2`` (same as ``x' = L - x``).

    .. warning::
        **Do not** use this to reverse an east-bound ``generate_main_lane_path``
        polyline. It swaps the ends of each straight but **flips** whether the
        U-turn bulge is toward ``+x`` or ``-x``, so the semicircle can point **off**
        the strip and drive the robot off the platform. Use
        ``generate_main_lane_path(..., first_straight_direction=\"west\")`` instead.
    """
    if field_length <= 0:
        raise ValueError("field_length must be positive")
    points = [(float(px), float(py)) for px, py in path]
    if not points:
        return []
    return [(float(field_length - px), py) for px, py in points]


def speed_from_curvature(
    curvature: float,
    *,
    cruise_speed: float = 1.2,
    turn_speed: float = 0.25,
    slowdown_curvature: float = 0.8,
    straight_curvature_deadband: float = 0.0,
) -> float:
    """Choose a target speed: fast on straight segments, slow in tight turns."""
    if cruise_speed <= 0:
        raise ValueError("cruise_speed must be positive")
    if turn_speed <= 0:
        raise ValueError("turn_speed must be positive")
    if turn_speed > cruise_speed:
        raise ValueError("turn_speed must not exceed cruise_speed")
    if slowdown_curvature <= 0:
        raise ValueError("slowdown_curvature must be positive")
    if straight_curvature_deadband < 0:
        raise ValueError("straight_curvature_deadband must be non-negative")

    effective_curvature = max(abs(curvature) - straight_curvature_deadband, 0.0)
    ratio = min(effective_curvature / slowdown_curvature, 1.0)
    return float(cruise_speed - (cruise_speed - turn_speed) * ratio)


def phase_speed_limit_for_progress(
    progress: float,
    *,
    u_turn_phases: Sequence[tuple[float, float]],
    cruise_speed: float,
    turn_speed: float,
    pre_turn_slowdown_m: float,
) -> float:
    """Limit speed by path phase: slow before each U and hold turn speed through it."""
    if cruise_speed <= 0.0:
        raise ValueError("cruise_speed must be positive")
    if turn_speed <= 0.0:
        raise ValueError("turn_speed must be positive")
    if turn_speed > cruise_speed:
        raise ValueError("turn_speed must not exceed cruise_speed")
    if pre_turn_slowdown_m <= 0.0:
        raise ValueError("pre_turn_slowdown_m must be positive")

    p = float(progress)
    limit = float(cruise_speed)
    for start, end in u_turn_phases:
        s0 = float(start)
        s1 = float(end)
        if s1 < s0:
            s0, s1 = s1, s0
        if s0 - pre_turn_slowdown_m <= p < s0:
            ratio = (p - (s0 - pre_turn_slowdown_m)) / pre_turn_slowdown_m
            phase_limit = float(cruise_speed + (turn_speed - cruise_speed) * ratio)
            limit = min(limit, phase_limit)
        elif s0 <= p <= s1:
            limit = min(limit, float(turn_speed))
    return float(limit)


def apply_command(bot, command: PurePursuitCommand) -> None:
    """Apply a computed command to PaddyRobotController-like objects."""
    bot.set_wheel_speeds(command.left_rad_s, command.right_rad_s)
    bot.set_steering_angle(command.steer_rad)


def limit_actuator_command(
    command: PurePursuitCommand,
    previous: PurePursuitCommand,
    *,
    step_size: float,
    max_wheel_rad_s: float,
    max_wheel_accel_rad_s2: float,
    max_steer_rad: float,
    max_steer_rate_rad_s: float,
) -> PurePursuitCommand:
    """Clamp wheel and steering commands like Isaac wheeled controllers do."""
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    if max_wheel_rad_s <= 0 or max_wheel_accel_rad_s2 <= 0:
        raise ValueError("wheel limits must be positive")
    if max_steer_rad <= 0 or max_steer_rate_rad_s <= 0:
        raise ValueError("steering limits must be positive")

    left = _rate_limit(
        current=previous.left_rad_s,
        target=float(np.clip(command.left_rad_s, -max_wheel_rad_s, max_wheel_rad_s)),
        max_delta=max_wheel_accel_rad_s2 * step_size,
    )
    right = _rate_limit(
        current=previous.right_rad_s,
        target=float(np.clip(command.right_rad_s, -max_wheel_rad_s, max_wheel_rad_s)),
        max_delta=max_wheel_accel_rad_s2 * step_size,
    )
    steer = _rate_limit(
        current=previous.steer_rad,
        target=float(np.clip(command.steer_rad, -max_steer_rad, max_steer_rad)),
        max_delta=max_steer_rate_rad_s * step_size,
    )

    return PurePursuitCommand(
        left_rad_s=left,
        right_rad_s=right,
        steer_rad=steer,
        done=command.done,
        lookahead_point=command.lookahead_point,
        curvature=command.curvature,
        closest_progress=command.closest_progress,
    )


def _rate_limit(*, current: float, target: float, max_delta: float) -> float:
    delta = target - current
    return float(current + np.clip(delta, -max_delta, max_delta))


def _append_unique(path: list[tuple[float, float]], point: tuple[float, float]) -> None:
    if path and math.isclose(path[-1][0], point[0], abs_tol=1e-9) and math.isclose(
        path[-1][1], point[1], abs_tol=1e-9
    ):
        return
    path.append((float(point[0]), float(point[1])))


__all__ = [
    "DEFAULT_FIELD_LENGTH",
    "DEFAULT_GROUND_SIZE",
    "DEFAULT_MAIN_LANE_COUNT",
    "DEFAULT_STEER_SIGN",
    "DEFAULT_SEMICIRCLE_COUNT",
    "DEFAULT_FIELD_WIDTH",
    "DEFAULT_TRACK_WIDTH",
    "DEFAULT_WHEEL_RADIUS",
    "DEFAULT_WHEELBASE",
    "CLAMP_UTURN_APEX_HEADROOM_M",
    "GROUND_CONTACT_DAMPING",
    "GROUND_CONTACT_STIFFNESS",
    "GROUND_DYNAMIC_FRICTION",
    "GROUND_STATIC_FRICTION",
    "GRAVITY_MPS2",
    "PaddySimPhysics",
    "PurePursuitCommand",
    "PurePursuitTracker",
    "TIRE_DYNAMIC_FRICTION",
    "TIRE_STATIC_FRICTION",
    "apply_command",
    "clamp_field_length_to_ground",
    "compute_platform_corner_start_pose",
    "compute_platform_coverage_start_pose",
    "mirror_path_along_field_length",
    "generate_main_lane_path",
    "generate_lawnmower_path",
    "limit_actuator_command",
    "phase_speed_limit_for_progress",
    "planar_speed_from_linear_velocity",
    "planar_yaw_from_pose",
    "quat_from_yaw",
    "quat_wxyz_to_rotation_matrix",
    "RECOMMENDED_PURE_PURSUIT",
    "RECOMMENDED_SCRIPT",
    "RecommendedIsaacPurePursuitScript",
    "RecommendedPurePursuitProfile",
    "rotate_body_vector_to_world",
    "speed_from_curvature",
    "tracking_point_world",
    "tracking_pose_for_planar_path",
    "straight_run_budget_on_square_ground",
    "transform_path_to_pose",
    "yaw_from_quat",
]
