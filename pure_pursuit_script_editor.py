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
    DEFAULT_FIELD_LENGTH,
    DEFAULT_FIELD_WIDTH,
    DEFAULT_GROUND_SIZE,
    DEFAULT_MAIN_LANE_COUNT,
    DEFAULT_SEMICIRCLE_COUNT,
    DEFAULT_TRACK_WIDTH,
    DEFAULT_WHEEL_RADIUS,
    DEFAULT_WHEELBASE,
    PurePursuitTracker,
    apply_command,
    clamp_field_length_to_ground,
    compute_platform_corner_start_pose,
    generate_main_lane_path,
    generate_lawnmower_path,
    limit_actuator_command,
    quat_from_yaw,
    speed_from_curvature,
    transform_path_to_pose,
    yaw_from_quat,
)


SUBSCRIPTION_ATTR = "_pure_pursuit_physics_subscription"

CRUISE_SPEED_MPS = 0.9
TURN_SPEED_MPS = 0.15
SLOWDOWN_CURVATURE = 0.12
STRAIGHT_CURVATURE_DEADBAND = 0.03
ACCEL_LIMIT_MPS2 = 0.6
DECEL_LIMIT_MPS2 = 0.6
MAX_WHEEL_RAD_S = 12.0
MAX_WHEEL_ACCEL_RAD_S2 = 8.0
MAX_STEER_RAD = math.radians(50.0)
MAX_STEER_RATE_RAD_S = math.radians(60.0)
ISAAC_REAR_STEER_SIGN = -1.0
DEBUG_PERIOD_S = 0.5
STALL_SPEED_MPS = 0.03
STALL_CURVATURE = 0.5
STALL_CRAWL_SPEED_MPS = 0.25
STALL_MAX_STEER_RAD = math.radians(15.0)
REQUESTED_FIELD_LENGTH = DEFAULT_FIELD_LENGTH
FIELD_WIDTH = DEFAULT_FIELD_WIDTH
GROUND_SIZE = DEFAULT_GROUND_SIZE
SEMICIRCLE_COUNT = DEFAULT_SEMICIRCLE_COUNT
MAIN_LANE_COUNT = DEFAULT_MAIN_LANE_COUNT
PLATFORM_MARGIN = 1.0


def _clear_existing_subscription() -> None:
    subscription = getattr(builtins, SUBSCRIPTION_ATTR, None)
    if subscription is not None and hasattr(subscription, "unsubscribe"):
        subscription.unsubscribe()
    setattr(builtins, SUBSCRIPTION_ATTR, None)

bot = PaddyRobotController()
current_pos, _current_quat = bot.robot.get_world_pose()
start_x, start_y, start_theta = compute_platform_corner_start_pose(
    ground_size=GROUND_SIZE,
    field_width=FIELD_WIDTH,
    lane_count=MAIN_LANE_COUNT,
    track_width=DEFAULT_TRACK_WIDTH,
    wheelbase=DEFAULT_WHEELBASE,
    wheel_radius=DEFAULT_WHEEL_RADIUS,
    margin=PLATFORM_MARGIN,
)
start_pos = np.array([start_x, start_y, float(current_pos[2])], dtype=np.float32)
start_quat = np.array(quat_from_yaw(start_theta), dtype=np.float32)
bot.stop()
bot.set_steering_angle(0.0)
bot.robot.set_world_pose(position=start_pos, orientation=start_quat)
lane_spacing = FIELD_WIDTH / float(MAIN_LANE_COUNT - 1)
turn_radius = lane_spacing / 2.0
FIELD_LENGTH = clamp_field_length_to_ground(
    x=start_x,
    y=start_y,
    theta=start_theta,
    requested_length=REQUESTED_FIELD_LENGTH,
    turn_radius=turn_radius,
    ground_size=GROUND_SIZE,
    margin=PLATFORM_MARGIN,
)

# Use the feasible demo path by default. Set MAIN_LANE_COUNT to 2 for a wider
# single U-turn, or 3 for a compact three-pass field demonstration.
LOCAL_WAYPOINTS = generate_main_lane_path(
    field_length=FIELD_LENGTH,
    field_width=FIELD_WIDTH,
    lane_count=MAIN_LANE_COUNT,
)
WAYPOINTS = transform_path_to_pose(
    LOCAL_WAYPOINTS,
    x=start_x,
    y=start_y,
    theta=start_theta,
)
tracker = PurePursuitTracker(
    WAYPOINTS,
    wheelbase=DEFAULT_WHEELBASE,
    track_width=DEFAULT_TRACK_WIDTH,
    wheel_radius=DEFAULT_WHEEL_RADIUS,
    lookahead_gain=1.0,
    min_lookahead=0.6,
    max_lookahead=1.5,
    alpha=0.0,
    max_steer=MAX_STEER_RAD,
    goal_tolerance=0.3,
    steer_sign=ISAAC_REAR_STEER_SIGN,
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
    theta = yaw_from_quat(quat)
    xy = np.array([float(pos[0]), float(pos[1])], dtype=float)
    instant_v = float(np.linalg.norm(xy - pose_state["last_xy"]) / step_size)
    pose_state["last_xy"] = xy
    pose_state["v_measured"] = 0.8 * pose_state["v_measured"] + 0.2 * instant_v
    desired_v = speed_state["v_mps"]
    v_mps = min(desired_v, max(pose_state["v_measured"] + 0.25, TURN_SPEED_MPS))
    raw_command = tracker.compute(
        x=float(pos[0]),
        y=float(pos[1]),
        theta=theta,
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
    speed_state["v_mps"] = _advance_speed(speed_state["v_mps"], target_speed, step_size)
    debug_state["elapsed_s"] += step_size
    if debug_state["elapsed_s"] >= DEBUG_PERIOD_S:
        debug_state["elapsed_s"] = 0.0
        print(
            "[pure_pursuit] "
            f"progress={raw_command.closest_progress:.2f}/{tracker.total_length:.2f}m "
            f"k={raw_command.curvature:.3f} "
            f"target_v={target_speed:.2f} cmd_v={speed_state['v_mps']:.2f} "
            f"meas_v={pose_state['v_measured']:.2f} "
            f"stall={stalling} "
            f"L={command.left_rad_s:.2f} R={command.right_rad_s:.2f} "
            f"steer={math.degrees(command.steer_rad):.1f}deg "
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
    "[pure_pursuit] PhysX subscription installed: "
    f"{len(WAYPOINTS)} path points, start=({float(start_pos[0]):.2f}, "
    f"{float(start_pos[1]):.2f}), heading={start_theta:.2f}rad, "
    f"field={FIELD_LENGTH}m x {FIELD_WIDTH}m, main_lanes={MAIN_LANE_COUNT}, "
    f"speed={TURN_SPEED_MPS}-{CRUISE_SPEED_MPS}m/s, ground={GROUND_SIZE}m"
)
