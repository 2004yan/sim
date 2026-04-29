"""Pure Pursuit runner for Isaac Sim Script Editor.

Use this inside an already-open Isaac Sim session:
1. Open Paddy_Sim_V2.usd.
2. Run robot_setup.py once.
3. Press Play.
4. Run this file in Script Editor.

Run this file again to replace the previous Pure Pursuit callback.
"""

import builtins
import math
import sys

import numpy as np
import omni.physx


PROJECT_DIR = "/data2/file_swap/yzy_space/Sim_Robot_V2"
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

CRUISE_SPEED_MPS = 1.4
TURN_SPEED_MPS = 0.25
SLOWDOWN_CURVATURE = 0.45
ACCEL_LIMIT_MPS2 = 0.55
DECEL_LIMIT_MPS2 = 1.2
MAX_WHEEL_RAD_S = 8.0
MAX_WHEEL_ACCEL_RAD_S2 = 10.0
MAX_STEER_RAD = math.radians(28.0)
MAX_STEER_RATE_RAD_S = math.radians(45.0)
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
    alpha=0.45,
    max_steer=MAX_STEER_RAD,
    goal_tolerance=0.3,
)
speed_state = {"v_mps": TURN_SPEED_MPS}
command_state = {
    "command": tracker.compute(x=start_x, y=start_y, theta=start_theta, v_mps=0.0),
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
    v_mps = speed_state["v_mps"]
    raw_command = tracker.compute(
        x=float(pos[0]),
        y=float(pos[1]),
        theta=theta,
        v_mps=v_mps,
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
    )
    speed_state["v_mps"] = _advance_speed(v_mps, target_speed, step_size)

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
