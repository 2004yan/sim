"""Standalone Pure Pursuit demo for the paddy robot in Isaac Sim 5.1.0.

Run this with Isaac Sim's Python after the USD asset is available. The script
opens the scene, applies the existing runtime robot setup, and drives the robot
through a waypoint path using Pure Pursuit.
"""

from isaacsim.simulation_app import SimulationApp

simulation_app = SimulationApp({"headless": False})

import omni.usd
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


USD_PATH = "/data2/file_swap/yzy_space/Sim_Robot_V2/Paddy_Sim_V2.usd"

FIELD_LENGTH = DEFAULT_FIELD_LENGTH
FIELD_WIDTH = DEFAULT_FIELD_WIDTH
SEMICIRCLE_COUNT = DEFAULT_SEMICIRCLE_COUNT
WAYPOINTS = generate_lawnmower_path(
    field_length=FIELD_LENGTH,
    field_width=FIELD_WIDTH,
    semicircle_count=SEMICIRCLE_COUNT,
)

V_MPS = 0.5

# Geometry defaults from Sim_Robot_V2.urdf:
# front wheels at y ~= +/-0.6 m, front axle x ~= 0.575 m, rear steer x ~= -0.625 m.
TRACK_WIDTH = DEFAULT_TRACK_WIDTH
WHEELBASE = DEFAULT_WHEELBASE
WHEEL_RADIUS = DEFAULT_WHEEL_RADIUS


def main() -> None:
    omni.usd.get_context().open_stage(USD_PATH)
    simulation_app.update()

    # Existing setup code configures drive modes and releases the imported fixed base.
    import robot_setup

    robot_setup.remove_root_fixed_joint()
    robot_setup.set_masses()
    robot_setup.fix_rear_link_joint()
    robot_setup.set_drives()
    robot_setup.set_wheel_friction()
    robot_setup.disable_fixed_base()

    world = World(stage_units_in_meters=1.0)
    world.reset()

    bot = PaddyRobotController()
    tracker = PurePursuitTracker(
        WAYPOINTS,
        wheelbase=WHEELBASE,
        track_width=TRACK_WIDTH,
        wheel_radius=WHEEL_RADIUS,
        lookahead_gain=1.0,
        min_lookahead=0.25,
        max_lookahead=0.8,
        alpha=0.7,
        goal_tolerance=0.3,
    )

    try:
        while simulation_app.is_running():
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
                bot.stop()
                break

            world.step(render=True)
    finally:
        bot.set_steering_angle(0.0)
        bot.stop()
        simulation_app.close()


if __name__ == "__main__":
    main()
