"""Pure Pursuit runner for Isaac Sim Script Editor.

Use this inside an already-open Isaac Sim session:
1. Open Paddy_Sim_V2.usd.
2. Run robot_setup.py once.
3. Press Play.
4. Run this file in Script Editor.

Run this file again to replace the previous Pure Pursuit callback.
"""

from isaacsim.core.api import World

from controller import PaddyRobotController
from pure_pursuit_controller import (
    DEFAULT_FIELD_LENGTH,
    DEFAULT_FIELD_WIDTH,
    DEFAULT_SEMICIRCLE_COUNT,
    DEFAULT_TRACK_WIDTH,
    DEFAULT_WHEEL_RADIUS,
    DEFAULT_WHEELBASE,
    PurePursuitTracker,
    apply_command,
    generate_lawnmower_path,
    yaw_from_quat,
)


CALLBACK_NAME = "pure_pursuit_field_tracker"

V_MPS = 0.5
FIELD_LENGTH = DEFAULT_FIELD_LENGTH
FIELD_WIDTH = DEFAULT_FIELD_WIDTH
SEMICIRCLE_COUNT = DEFAULT_SEMICIRCLE_COUNT

WAYPOINTS = generate_lawnmower_path(
    field_length=FIELD_LENGTH,
    field_width=FIELD_WIDTH,
    semicircle_count=SEMICIRCLE_COUNT,
)


def _get_world() -> World:
    world = World.instance()
    if world is None:
        world = World(stage_units_in_meters=1.0)
    return world


world = _get_world()

try:
    world.remove_physics_callback(CALLBACK_NAME)
except Exception:
    pass

bot = PaddyRobotController()
tracker = PurePursuitTracker(
    WAYPOINTS,
    wheelbase=DEFAULT_WHEELBASE,
    track_width=DEFAULT_TRACK_WIDTH,
    wheel_radius=DEFAULT_WHEEL_RADIUS,
    lookahead_gain=1.0,
    min_lookahead=0.25,
    max_lookahead=0.8,
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
        world.remove_physics_callback(CALLBACK_NAME)
        print("[pure_pursuit] reached goal, callback removed")


world.add_physics_callback(CALLBACK_NAME, pure_pursuit_step)
print(
    "[pure_pursuit] callback installed: "
    f"{len(WAYPOINTS)} path points, field={FIELD_LENGTH}m x {FIELD_WIDTH}m, "
    f"semicircles={SEMICIRCLE_COUNT}, speed={V_MPS}m/s"
)
