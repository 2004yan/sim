"""Pure Pursuit runner for Isaac Sim Script Editor.

Use this inside an already-open Isaac Sim session:
1. Open Paddy_Sim_V2.usd.
2. Run robot_setup.py once.
3. Press Play.
4. Run this file in Script Editor.

Run this file again to replace the previous Pure Pursuit callback.
"""

import builtins
from dataclasses import replace
import math
import sys

import numpy as np
import omni.physx


# Keep this path fixed for Isaac Sim Script Editor imports.
PROJECT_DIR = "/workspace/sim"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from controller import PaddyRobotController
from pure_pursuit_controller import (
    DEFAULT_GROUND_SIZE,
    DEFAULT_TRACK_WIDTH,
    DEFAULT_WHEEL_RADIUS,
    DEFAULT_WHEELBASE,
    PurePursuitTracker,
    RECOMMENDED_PURE_PURSUIT,
    RECOMMENDED_SCRIPT,
    apply_command,
    clamp_field_length_to_ground,
    compute_platform_coverage_start_pose,
    generate_main_lane_path,
    limit_actuator_command,
    planar_speed_from_linear_velocity,
    quat_from_yaw,
    speed_from_curvature,
    straight_run_budget_on_square_ground,
    tracking_pose_for_planar_path,
    transform_path_to_pose,
    yaw_from_quat,
)


SUBSCRIPTION_ATTR = "_pure_pursuit_physics_subscription"

# --- 推荐预设（单一点维护；与 ``RECOMMENDED_PURE_PURSUIT`` / ``RECOMMENDED_SCRIPT`` 同步）---
PP = RECOMMENDED_PURE_PURSUIT
SC = RECOMMENDED_SCRIPT

CRUISE_SPEED_MPS = SC.cruise_speed_mps
TURN_SPEED_MPS = SC.turn_speed_mps
SLOWDOWN_CURVATURE = SC.slowdown_curvature
# Slightly lower than ``SC`` so mild curvature at the start of the 180° arc is not
# treated as “straight” and zeroed (which can look like a sharp 90°+90° jog).
STRAIGHT_CURVATURE_DEADBAND = min(SC.straight_curvature_deadband, 0.012)
ACCEL_LIMIT_MPS2 = SC.accel_limit_mps2
DECEL_LIMIT_MPS2 = SC.decel_limit_mps2
MAX_WHEEL_RAD_S = SC.max_wheel_rad_s
MAX_WHEEL_ACCEL_RAD_S2 = SC.max_wheel_accel_rad_s2
MAX_STEER_RAD = SC.max_steer_rad
MAX_STEER_RATE_RAD_S = SC.max_steer_rate_rad_s
# Rear_Link 关节：正角绕 +Z 时 IMU/前向与 Pure Pursuit 的曲率符号应对齐。
# 若仿真里首弯明显“朝反方向”削出台边，把下面改成 -1.0（二选一，无第三解）。
ISAAC_REAR_STEER_SIGN = 1.0
DYNAMIC_MARGIN_M = SC.dynamic_margin_m
LOOKAHEAD_MIN_M = PP.min_lookahead
LOOKAHEAD_MAX_M = PP.max_lookahead
HEADING_GAIN_PATH = PP.heading_gain
DEBUG_PERIOD_S = SC.debug_period_s
STALL_SPEED_MPS = SC.stall_speed_mps
STALL_CURVATURE = SC.stall_curvature
STALL_CRAWL_SPEED_MPS = SC.stall_crawl_speed_mps
STALL_MAX_STEER_RAD = SC.stall_max_steer_rad
SLIP_SPEED_FRACTION = SC.slip_speed_fraction
VEHICLE_HALF_LENGTH_M = (DEFAULT_WHEELBASE + 2.0 * DEFAULT_WHEEL_RADIUS) / 2.0
# Simple 回字 / U-turn demo: 三段长直道 + 两段 180° 半圆（``generate_main_lane_path``，``lane_count=3``）。
MAIN_LANE_COUNT = 3
# 三行在 *y* 方向的总跨度（规划用长度单位）；转弯半径 = (FIELD_WIDTH / (lane_count-1)) / 2。
FIELD_WIDTH = 5.0
# Isaac ``GroundPlane(size=...)`` 的数值**不一定**等于米；以下只做**路径与限幅**用的方形边长（与你标定的一致即可）。
# 若地台在 USD 里 40 但实际可跑 25 m，请把这里改成 25，不要把 ``ground_setup`` 的 40 死当成米。
PLANNING_GROUND_EXTENT_M = float(DEFAULT_GROUND_SIZE)
# 每一段「长直路」希望多长；``math.inf`` 表示在平地范围内尽量拉满（由 ``clamp_field_length_to_ground`` 决定）。
REQUESTED_STRAIGHT_RUN_M = math.inf
# ``upper_left_plus_x`` = 先朝 +X；``upper_right_minus_x`` = 先朝 −X（用 ``first_straight_direction=west``，勿镜像整条折线）。
PLATFORM_START_EDGE = "upper_right_minus_x"
PLATFORM_MARGIN = 1.25

# 3D pose -> planar tracking frame (Isaac ``get_world_pose`` is SE(3)):
# - offset is in the robot *body* frame (e.g. rear axle vs articulation root)
# - forward axis should match the USD/URDF +X (change if your asset differs)
TRACKING_OFFSET_BODY_M = (0.0, 0.0, 0.0)
BODY_FORWARD_AXIS = (1.0, 0.0, 0.0)
WORLD_UP_AXIS = (0.0, 0.0, 1.0)
# Prefer PhysX linear velocity (world frame) for speed estimate; falls back to finite-diff XY.
USE_LINEAR_VELOCITY_FOR_SPEED = True


def _clear_existing_subscription() -> None:
    subscription = getattr(builtins, SUBSCRIPTION_ATTR, None)
    if subscription is not None and hasattr(subscription, "unsubscribe"):
        subscription.unsubscribe()
    setattr(builtins, SUBSCRIPTION_ATTR, None)


def _reset_robot_state(bot: PaddyRobotController) -> None:
    """Reset physical joint state so reruns do not inherit the last steering angle."""
    bot.stop()
    bot.set_steering_angle(0.0)
    if bot.rear_link_idx is not None:
        bot.robot.set_joint_positions(
            positions=np.array([0.0], dtype=np.float32),
            joint_indices=np.array([bot.rear_link_idx], dtype=np.int32),
        )
    wheel_indices = [
        index
        for index in (bot.left_idx, bot.right_idx, bot.rear_wheel_idx)
        if index is not None
    ]
    if wheel_indices:
        bot.robot.set_joint_velocities(
            velocities=np.zeros(len(wheel_indices), dtype=np.float32),
            joint_indices=np.array(wheel_indices, dtype=np.int32),
        )


def _distance_to_path(tracker: PurePursuitTracker, xy: np.ndarray) -> float:
    progress = tracker._closest_progress(xy)
    closest = tracker._point_at_progress(progress)
    return float(np.linalg.norm(xy - closest))


def _joint_position_or_nan(bot: PaddyRobotController, joint_index: int | None) -> float:
    if joint_index is None:
        return math.nan
    try:
        values = bot.robot.get_joint_positions(
            joint_indices=np.array([joint_index], dtype=np.int32),
        )
    except Exception:
        return math.nan
    if values is None or len(values) == 0:
        return math.nan
    return float(values[0])


bot = PaddyRobotController()
current_pos, _current_quat = bot.robot.get_world_pose()
lane_spacing = FIELD_WIDTH / float(MAIN_LANE_COUNT - 1)
turn_radius = lane_spacing / 2.0
NOMINAL_U_SEMICIRCLE_ARC_M = math.pi * turn_radius
start_x, start_y, start_theta = compute_platform_coverage_start_pose(
    ground_size=PLANNING_GROUND_EXTENT_M,
    turn_radius=turn_radius,
    track_width=DEFAULT_TRACK_WIDTH,
    wheelbase=DEFAULT_WHEELBASE,
    wheel_radius=DEFAULT_WHEEL_RADIUS,
    margin=PLATFORM_MARGIN,
    dynamic_margin=DYNAMIC_MARGIN_M,
    start_edge=PLATFORM_START_EDGE,
)
start_pos = np.array([start_x, start_y, float(current_pos[2])], dtype=np.float32)
start_quat = np.array(quat_from_yaw(start_theta), dtype=np.float32)
bot.robot.set_world_pose(position=start_pos, orientation=start_quat)
_reset_robot_state(bot)
FIELD_LENGTH = clamp_field_length_to_ground(
    x=start_x,
    y=start_y,
    theta=start_theta,
    requested_length=REQUESTED_STRAIGHT_RUN_M,
    turn_radius=turn_radius,
    ground_size=PLANNING_GROUND_EXTENT_M,
    margin=PLATFORM_MARGIN,
    dynamic_margin=DYNAMIC_MARGIN_M,
    vehicle_half_length=VEHICLE_HALF_LENGTH_M,
)
STRAIGHT_BUDGET = straight_run_budget_on_square_ground(
    x=start_x,
    y=start_y,
    theta=start_theta,
    requested_length=REQUESTED_STRAIGHT_RUN_M,
    turn_radius=turn_radius,
    ground_size=PLANNING_GROUND_EXTENT_M,
    margin=PLATFORM_MARGIN,
    dynamic_margin=DYNAMIC_MARGIN_M,
    vehicle_half_length=VEHICLE_HALF_LENGTH_M,
)

PATH_FIRST_RUN = "west" if PLATFORM_START_EDGE == "upper_right_minus_x" else "east"
LOCAL_WAYPOINTS = generate_main_lane_path(
    field_length=FIELD_LENGTH,
    field_width=FIELD_WIDTH,
    lane_count=MAIN_LANE_COUNT,
    turn_samples=48,
    first_straight_direction=PATH_FIRST_RUN,
)
WAYPOINTS = transform_path_to_pose(
    LOCAL_WAYPOINTS,
    x=start_x,
    y=start_y,
    theta=0.0,
)
FIRST_STRAIGHT_LEN_M = float(np.linalg.norm(np.asarray(WAYPOINTS[1]) - np.asarray(WAYPOINTS[0])))
WORLD_STRAIGHT_CHECK_M = float(
    np.linalg.norm(np.asarray(WAYPOINTS[1], dtype=float)[:2] - np.asarray(WAYPOINTS[0], dtype=float)[:2])
)
_STRAIGHT_OK = math.isclose(WORLD_STRAIGHT_CHECK_M, FIELD_LENGTH, rel_tol=0.0, abs_tol=0.05)
tracker = PurePursuitTracker(
    WAYPOINTS,
    wheelbase=DEFAULT_WHEELBASE,
    track_width=DEFAULT_TRACK_WIDTH,
    wheel_radius=DEFAULT_WHEEL_RADIUS,
    phase_straight_end_progress=FIRST_STRAIGHT_LEN_M,
    **PP.tracker_kwargs(max_steer=MAX_STEER_RAD, steer_sign=ISAAC_REAR_STEER_SIGN),
)
speed_state = {"v_mps": TURN_SPEED_MPS}
command_state = {
    "command": tracker.compute(x=start_x, y=start_y, theta=start_theta, v_mps=0.0),
}
debug_state = {"elapsed_s": 0.0}
pose_state = {
    "last_xy": np.array([start_x, start_y], dtype=float),
    "v_measured": 0.0,
}


def _advance_speed(current_speed: float, target_speed: float, step_size: float) -> float:
    limit = ACCEL_LIMIT_MPS2 if target_speed > current_speed else DECEL_LIMIT_MPS2
    max_delta = limit * max(float(step_size), 1.0 / 60.0)
    delta = max(-max_delta, min(max_delta, target_speed - current_speed))
    return current_speed + delta


def pure_pursuit_step(_step_size: float) -> None:
    step_size = max(float(_step_size), 1.0 / 60.0)
    pos, quat = bot.robot.get_world_pose()
    tx, ty, _tz, yaw_planar = tracking_pose_for_planar_path(
        pos,
        quat,
        offset_body=TRACKING_OFFSET_BODY_M,
        body_forward=BODY_FORWARD_AXIS,
        world_up=WORLD_UP_AXIS,
    )
    track_xy = np.array([tx, ty], dtype=float)
    instant_v = float(np.linalg.norm(track_xy - pose_state["last_xy"]) / step_size)
    pose_state["last_xy"] = track_xy
    v_blend = instant_v
    if USE_LINEAR_VELOCITY_FOR_SPEED:
        try:
            lin_vel = bot.robot.get_linear_velocity()
            v_blend = planar_speed_from_linear_velocity(lin_vel, world_up=WORLD_UP_AXIS)
        except Exception:
            v_blend = instant_v
    pose_state["v_measured"] = 0.8 * pose_state["v_measured"] + 0.2 * v_blend
    desired_v = speed_state["v_mps"]
    v_mps = min(desired_v, max(pose_state["v_measured"] + 0.25, TURN_SPEED_MPS))
    raw_command = tracker.compute(
        x=tx,
        y=ty,
        theta=yaw_planar,
        v_mps=v_mps,
    )
    stalling = (
        pose_state["v_measured"] < STALL_SPEED_MPS
        and abs(raw_command.curvature) > STALL_CURVATURE
    )
    if abs(raw_command.curvature) < STRAIGHT_CURVATURE_DEADBAND:
        wheel_rad_s = (raw_command.left_rad_s + raw_command.right_rad_s) / 2.0
        raw_command = replace(
            raw_command,
            left_rad_s=wheel_rad_s,
            right_rad_s=wheel_rad_s,
            steer_rad=0.0,
            curvature=0.0,
        )
    elif stalling:
        wheel_rad_s = STALL_CRAWL_SPEED_MPS / DEFAULT_WHEEL_RADIUS
        raw_command = replace(
            raw_command,
            left_rad_s=wheel_rad_s,
            right_rad_s=wheel_rad_s,
            steer_rad=float(np.clip(raw_command.steer_rad, -STALL_MAX_STEER_RAD, STALL_MAX_STEER_RAD)),
        )
    command = limit_actuator_command(
        raw_command,
        command_state["command"],
        step_size=step_size,
        max_wheel_rad_s=MAX_WHEEL_RAD_S,
        max_wheel_accel_rad_s2=MAX_WHEEL_ACCEL_RAD_S2,
        max_steer_rad=MAX_STEER_RAD,
        max_steer_rate_rad_s=MAX_STEER_RATE_RAD_S,
    )
    command_state["command"] = command
    apply_command(bot, command)
    target_speed = speed_from_curvature(
        raw_command.curvature,
        cruise_speed=CRUISE_SPEED_MPS,
        turn_speed=TURN_SPEED_MPS,
        slowdown_curvature=SLOWDOWN_CURVATURE,
        straight_curvature_deadband=STRAIGHT_CURVATURE_DEADBAND,
    )
    try:
        v_cmd_rim = ((raw_command.left_rad_s + raw_command.right_rad_s) * 0.5) * DEFAULT_WHEEL_RADIUS
        if v_cmd_rim > 0.06 and pose_state["v_measured"] < SLIP_SPEED_FRACTION * v_cmd_rim:
            target_speed = min(target_speed, max(pose_state["v_measured"], 0.5 * TURN_SPEED_MPS))
    except Exception:
        pass
    speed_state["v_mps"] = _advance_speed(speed_state["v_mps"], target_speed, step_size)
    debug_state["elapsed_s"] += step_size
    if debug_state["elapsed_s"] >= DEBUG_PERIOD_S:
        debug_state["elapsed_s"] = 0.0
        cte = _distance_to_path(tracker, track_xy)
        actual_steer = _joint_position_or_nan(bot, bot.rear_link_idx)
        straight_left_m = max(0.0, FIRST_STRAIGHT_LEN_M - float(raw_command.closest_progress))
        pr = float(raw_command.closest_progress)
        if pr < FIRST_STRAIGHT_LEN_M - 0.02:
            nav_phase = (
                f"phase=S0_straight | U_turn_at_progress={FIRST_STRAIGHT_LEN_M:.2f}m "
                f"nominal_arc≈{NOMINAL_U_SEMICIRCLE_ARC_M:.2f}m"
            )
        elif pr < FIRST_STRAIGHT_LEN_M + NOMINAL_U_SEMICIRCLE_ARC_M + 2.0:
            nav_phase = (
                f"phase=U0_arc | s_into_turn={pr - FIRST_STRAIGHT_LEN_M:.2f}m "
                f"(~0…{NOMINAL_U_SEMICIRCLE_ARC_M:.2f}m)"
            )
        else:
            nav_phase = f"phase=post_U0 | progress={pr:.2f}m"
        print(
            "[pure_pursuit] "
            f"{nav_phase} | "
            f"progress={raw_command.closest_progress:.2f}/{tracker.total_length:.2f}m "
            f"straight_left={straight_left_m:.2f}/{FIRST_STRAIGHT_LEN_M:.2f}m "
            f"k={raw_command.curvature:.3f} "
            f"target_v={target_speed:.2f} cmd_v={speed_state['v_mps']:.2f} "
            f"meas_v={pose_state['v_measured']:.2f} "
            f"cte={cte:.2f} "
            f"stall={stalling} "
            f"L={command.left_rad_s:.2f} R={command.right_rad_s:.2f} "
            f"steer={math.degrees(command.steer_rad):.1f}deg "
            f"actual_steer={math.degrees(actual_steer):.1f}deg "
            f"lookahead=({raw_command.lookahead_point[0]:.2f}, "
            f"{raw_command.lookahead_point[1]:.2f})"
        )

    if command.done:
        bot.set_steering_angle(0.0)
        bot.stop()
        _clear_existing_subscription()
        print("[pure_pursuit] reached goal, subscription removed")


_clear_existing_subscription()
physx_interface = omni.physx.get_physx_interface()
subscription = physx_interface.subscribe_physics_step_events(pure_pursuit_step)
setattr(builtins, SUBSCRIPTION_ATTR, subscription)
print(
    "---- pure_pursuit STRAIGHT_CONTRACT (copy this if lengths look wrong) ----\n"
    f"  planning_extent_m={PLANNING_GROUND_EXTENT_M} start=({start_x:.3f},{start_y:.3f}) "
    f"heading_rad={start_theta:.4f}\n"
    f"  ray_exit_to_square_edge_m={STRAIGHT_BUDGET['forward_to_edge_m']:.3f}  "
    f"- margin - dynamic - uturn_clearance\n"
    f"  margin_m={STRAIGHT_BUDGET['margin_m']:.3f}  dynamic_m={STRAIGHT_BUDGET['dynamic_margin_m']:.3f}  "
    f"uturn_clearance_m={STRAIGHT_BUDGET['uturn_clearance_m']:.3f}\n"
    f"  usable_before_clip_m={STRAIGHT_BUDGET['usable_before_clip_m']:.3f}  "
    f"requested_cap={STRAIGHT_BUDGET['requested_cap_m']}\n"
    f"  FIELD_LENGTH_m (applied to polyline)={FIELD_LENGTH:.3f}\n"
    f"  world_first_seg_m={WORLD_STRAIGHT_CHECK_M:.3f}  matches_FIELD_LENGTH={_STRAIGHT_OK}\n"
    f"  first_U_turn: begin steering when path_progress approaches {FIRST_STRAIGHT_LEN_M:.3f}m "
    f"(nominal half-circle arc length ≈ {NOMINAL_U_SEMICIRCLE_ARC_M:.3f}m, polyline slightly longer)\n"
    "-------------------------------------------------------------------------",
)
print(
    "[pure_pursuit] PhysX subscription installed: "
    f"preset=({PP.version},{SC.version}) "
    f"{len(WAYPOINTS)} path points, start=({float(start_pos[0]):.2f}, "
    f"{float(start_pos[1]):.2f}), heading={start_theta:.2f}rad, "
    f"path=simple_Ux2 (lanes={MAIN_LANE_COUNT}) straight_run={FIELD_LENGTH} "
    f"field_W={FIELD_WIDTH} "
    f"edge={PLATFORM_START_EDGE} first_run={PATH_FIRST_RUN} "
    f"planning_extent={PLANNING_GROUND_EXTENT_M} "
    f"speed={TURN_SPEED_MPS}-{CRUISE_SPEED_MPS}m/s",
)
