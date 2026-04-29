"""Pure Pursuit runner for Isaac Sim Script Editor.

Use this inside an already-open Isaac Sim session:
1. Open Paddy_Sim_V2.usd.
2. Run robot_setup.py once.
3. Press Play.
4. Run this file in Script Editor.

Run this file again to replace the previous Pure Pursuit callback.
"""

import builtins
import sys

import omni.physx


PROJECT_DIR = "/workspace/sim"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from controller import PaddyRobotController
from pure_pursuit_controller import (
    DEFAULT_FIELD_LENGTH,
    DEFAULT_FIELD_WIDTH,
    DEFAULT_MAIN_LANE_COUNT,
    DEFAULT_SEMICIRCLE_COUNT,
    DEFAULT_TRACK_WIDTH,
    DEFAULT_WHEEL_RADIUS,
    DEFAULT_WHEELBASE,
    PurePursuitTracker,
    apply_command,
    generate_main_lane_path,
    generate_lawnmower_path,
    transform_path_to_pose,
    yaw_from_quat,
)


SUBSCRIPTION_ATTR = "_pure_pursuit_physics_subscription"

V_MPS = 0.5
FIELD_LENGTH = DEFAULT_FIELD_LENGTH
FIELD_WIDTH = DEFAULT_FIELD_WIDTH
SEMICIRCLE_COUNT = DEFAULT_SEMICIRCLE_COUNT
MAIN_LANE_COUNT = DEFAULT_MAIN_LANE_COUNT

# Use the feasible demo path by default. Set MAIN_LANE_COUNT to 2 for a wider
# single U-turn, or 3 for a compact three-pass field demonstration.
LOCAL_WAYPOINTS = generate_main_lane_path(
    field_length=FIELD_LENGTH,
    field_width=FIELD_WIDTH,
    lane_count=MAIN_LANE_COUNT,
)


def _clear_existing_subscription() -> None:
    subscription = getattr(builtins, SUBSCRIPTION_ATTR, None)
    if subscription is not None and hasattr(subscription, "unsubscribe"):
        subscription.unsubscribe()
    setattr(builtins, SUBSCRIPTION_ATTR, None)

bot = PaddyRobotController()
start_pos, start_quat = bot.robot.get_world_pose()
start_theta = yaw_from_quat(start_quat)
WAYPOINTS = transform_path_to_pose(
    LOCAL_WAYPOINTS,
    x=float(start_pos[0]),
    y=float(start_pos[1]),
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
    alpha=0.7,
    goal_tolerance=0.3,
)


def pure_pursuit_step(_step_size: float) -> None:
    pos, quat = bot.robot.get_world_pose()
    theta = yaw_from_quat(quat)
    command = tracker.compute(
        x=float(pos[0]),
        y=float(pos[1]),
        theta=theta,
        v_mps=V_MPS,
    )
    apply_command(bot, command)

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
    f"speed={V_MPS}m/s"
)
