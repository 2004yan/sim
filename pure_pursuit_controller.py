import math
from pathlib import Path

import numpy as np
import pytest

from pure_pursuit_controller import (
    DEFAULT_STEER_SIGN,
    DEFAULT_TRACK_WIDTH,
    DEFAULT_WHEEL_RADIUS,
    DEFAULT_WHEELBASE,
    GRAVITY_MPS2,
    PurePursuitCommand,
    PurePursuitTracker,
    apply_command,
    clamp_field_length_to_ground,
    compute_platform_corner_start_pose,
    generate_main_lane_path,
    generate_lawnmower_path,
    limit_actuator_command,
    quat_from_yaw,
    recommended_max_lateral_accel_mps2,
    speed_from_curvature,
    transform_path_to_pose,
    yaw_from_quat,
)
from pure_pursuit_validation import simulate_path


def test_straight_path_outputs_equal_wheel_speeds_and_zero_steer():
    tracker = PurePursuitTracker(
        [(0.0, 0.0), (5.0, 0.0)],
        wheelbase=1.2,
        track_width=1.2,
        wheel_radius=0.25,
        lookahead_gain=1.0,
        min_lookahead=1.0,
        max_lookahead=1.0,
        alpha=0.5,
    )

    cmd = tracker.compute(x=0.0, y=0.0, theta=0.0, v_mps=1.0)

    assert cmd.done is False
    assert cmd.lookahead_point == pytest.approx((1.0, 0.0))
    assert cmd.curvature == pytest.approx(0.0)
    assert cmd.left_rad_s == pytest.approx(4.0)
    assert cmd.right_rad_s == pytest.approx(4.0)
    assert cmd.steer_rad == pytest.approx(0.0)


def test_lateral_offset_uses_body_frame_distance_and_rear_steering_geometry():
    tracker = PurePursuitTracker(
        np.array([(0.0, 0.0), (5.0, 0.0)]),
        wheelbase=1.2,
        track_width=1.2,
        wheel_radius=0.25,
        lookahead_gain=1.0,
        min_lookahead=1.0,
        max_lookahead=1.0,
        alpha=0.5,
    )

    cmd = tracker.compute(x=0.0, y=1.0, theta=0.0, v_mps=1.0)

    assert cmd.lookahead_point == pytest.approx((1.0, 0.0))
    assert cmd.curvature == pytest.approx(-1.0)
    assert cmd.left_rad_s == pytest.approx(5.2)
    assert cmd.right_rad_s == pytest.approx(2.8)
    assert cmd.steer_rad == pytest.approx(math.atan(-1.2))


def test_curvature_is_limited_by_rear_steering_capability():
    tracker = PurePursuitTracker(
        np.array([(0.0, 0.0), (5.0, 0.0)]),
        wheelbase=1.2,
        track_width=1.2,
        wheel_radius=0.25,
        min_lookahead=0.6,
        max_lookahead=0.6,
        max_steer=math.radians(50.0),
    )

    cmd = tracker.compute(x=0.0, y=1.0, theta=0.0, v_mps=0.15)
    max_curvature = math.tan(math.radians(50.0)) / 1.2

    assert abs(cmd.curvature) <= max_curvature
    assert cmd.left_rad_s >= 0.0
    assert cmd.right_rad_s >= 0.0


def test_finished_path_reports_done_and_zero_commands_near_goal():
    tracker = PurePursuitTracker(
        [(0.0, 0.0), (1.0, 0.0)],
        wheelbase=1.2,
        track_width=1.2,
        wheel_radius=0.25,
        goal_tolerance=0.2,
    )

    cmd = tracker.compute(x=0.95, y=0.0, theta=0.0, v_mps=1.0)

    assert cmd.done is True
    assert cmd.left_rad_s == pytest.approx(0.0)
    assert cmd.right_rad_s == pytest.approx(0.0)
    assert cmd.steer_rad == pytest.approx(0.0)


def test_near_goal_position_does_not_finish_until_path_progress_reaches_end():
    tracker = PurePursuitTracker(
        [(0.0, 0.0), (5.0, 0.0), (0.1, 0.0)],
        wheelbase=1.2,
        track_width=1.2,
        wheel_radius=0.25,
        min_lookahead=1.0,
        max_lookahead=1.0,
        goal_tolerance=0.2,
    )

    cmd = tracker.compute(x=0.0, y=0.0, theta=0.0, v_mps=1.0)

    assert cmd.done is False


def test_rejects_invalid_geometry_and_path_inputs():
    with pytest.raises(ValueError, match="at least two"):
        PurePursuitTracker([(0.0, 0.0)])

    with pytest.raises(ValueError, match="wheel_radius"):
        PurePursuitTracker([(0.0, 0.0), (1.0, 0.0)], wheel_radius=0.0)


def test_apply_command_sends_front_wheel_speeds_and_rear_steering_only():
    class FakeBot:
        def __init__(self):
            self.wheels = None
            self.steer = None
            self.rear = None

        def set_wheel_speeds(self, left_rad_s, right_rad_s):
            self.wheels = (left_rad_s, right_rad_s)

        def set_steering_angle(self, angle_rad):
            self.steer = angle_rad

        def set_rear_wheel_speed(self, rad_s):
            self.rear = rad_s

    bot = FakeBot()
    cmd = PurePursuitCommand(
        left_rad_s=1.2,
        right_rad_s=1.8,
        steer_rad=0.3,
        done=False,
        lookahead_point=(1.0, 0.0),
        curvature=0.2,
        closest_progress=0.0,
    )

    apply_command(bot, cmd)

    assert bot.wheels == pytest.approx((1.2, 1.8))
    assert bot.steer == pytest.approx(0.3)
    assert bot.rear is None


def test_robot_setup_leaves_rear_wheel_as_passive_free_rolling_joint():
    setup = Path(__file__).resolve().parents[1] / "robot_setup.py"
    text = setup.read_text(encoding="utf-8")

    assert "_disable_drive(REAR_WHEEL_JOINT)" in text
    assert "_set_drive(REAR_WHEEL_JOINT, stiffness=0,   damping=1e4)" not in text
    assert 'f"{BASE}/Rear_Link_Link": 10.0' in text
    assert "FRONT_WHEEL_DAMPING = 1500.0" in text
    assert "FRONT_WHEEL_MAX_FORCE = 300.0" in text
    assert "REAR_STEER_STIFFNESS = 2e4" in text
    assert "REAR_STEER_DAMPING = 800.0" in text
    assert "REAR_STEER_MAX_FORCE = 300.0" in text


def test_ground_setup_uses_validation_friction_for_turning_debug():
    ground = Path(__file__).resolve().parents[1] / "ground_setup.py"
    text = ground.read_text(encoding="utf-8")

    assert "static_friction=0.6" in text
    assert "dynamic_friction=0.45" in text


def test_runtime_controller_does_not_reapply_articulation_root():
    controller = Path(__file__).resolve().parents[1] / "controller.py"
    text = controller.read_text(encoding="utf-8")

    assert "UsdPhysics.ArticulationRootAPI.Apply(prim)" not in text


def test_limit_actuator_command_clamps_wheel_speed_and_steering_rate():
    previous = PurePursuitCommand(
        left_rad_s=0.0,
        right_rad_s=0.0,
        steer_rad=0.0,
        done=False,
        lookahead_point=(0.0, 0.0),
        curvature=0.0,
        closest_progress=0.0,
    )
    raw = PurePursuitCommand(
        left_rad_s=20.0,
        right_rad_s=-20.0,
        steer_rad=1.0,
        done=False,
        lookahead_point=(1.0, 0.0),
        curvature=2.0,
        closest_progress=1.0,
    )

    limited = limit_actuator_command(
        raw,
        previous,
        step_size=0.1,
        max_wheel_rad_s=6.0,
        max_wheel_accel_rad_s2=10.0,
        max_steer_rad=0.5,
        max_steer_rate_rad_s=1.0,
    )

    assert limited.left_rad_s == pytest.approx(1.0)
    assert limited.right_rad_s == pytest.approx(-1.0)
    assert limited.steer_rad == pytest.approx(0.1)
    assert limited.lookahead_point == raw.lookahead_point
    assert limited.curvature == raw.curvature


def test_default_geometry_matches_urdf_estimates():
    assert DEFAULT_TRACK_WIDTH == pytest.approx(1.2)
    assert DEFAULT_WHEELBASE == pytest.approx(1.2)
    assert DEFAULT_WHEEL_RADIUS == pytest.approx(0.235, abs=0.001)
    assert DEFAULT_STEER_SIGN == 1.0


def test_generate_lawnmower_path_matches_30_by_3_field_geometry():
    path = generate_lawnmower_path(
        field_length=30.0,
        field_width=3.0,
        semicircle_count=9,
        turn_samples=7,
    )

    assert len(path) == 64
    assert path[0] == pytest.approx((0.0, 3.0))
    assert path[-1] == pytest.approx((30.0, 0.0))
    assert path[1] == pytest.approx((30.0, 3.0))
    assert path[7] == pytest.approx((30.0, 3.0 - 1.0 / 3.0))
    assert path[8] == pytest.approx((0.0, 3.0 - 1.0 / 3.0))
    assert max(x for x, _ in path) == pytest.approx(30.0 + 1.0 / 6.0)
    assert min(x for x, _ in path) == pytest.approx(-1.0 / 6.0)


def test_generate_lawnmower_path_rejects_invalid_field_geometry():
    with pytest.raises(ValueError, match="semicircle_count"):
        generate_lawnmower_path(semicircle_count=0)

    with pytest.raises(ValueError, match="turn_samples"):
        generate_lawnmower_path(turn_samples=1)


def test_generate_main_lane_path_supports_two_or_three_feasible_lanes():
    two_lane = generate_main_lane_path(field_length=30.0, field_width=3.0, lane_count=2, turn_samples=7)
    three_lane = generate_main_lane_path(field_length=30.0, field_width=3.0, lane_count=3, turn_samples=7)

    assert two_lane[0] == pytest.approx((0.0, 3.0))
    assert two_lane[-1] == pytest.approx((0.0, 0.0))
    assert max(x for x, _ in two_lane) == pytest.approx(31.5)

    assert len(three_lane) == 16
    assert three_lane[0] == pytest.approx((0.0, 3.0))
    assert three_lane[1] == pytest.approx((30.0, 3.0))
    assert three_lane[7] == pytest.approx((30.0, 1.5))
    assert three_lane[8] == pytest.approx((0.0, 1.5))
    assert three_lane[-1] == pytest.approx((30.0, 0.0))
    assert max(x for x, _ in three_lane) == pytest.approx(30.75)
    assert min(x for x, _ in three_lane) == pytest.approx(-0.75)


def test_generate_main_lane_path_rejects_unusable_lane_counts():
    with pytest.raises(ValueError, match="lane_count"):
        generate_main_lane_path(lane_count=1)


def test_clamp_field_length_keeps_path_on_20m_ground_plane():
    assert clamp_field_length_to_ground(
        x=0.0,
        y=0.0,
        theta=0.0,
        requested_length=30.0,
        turn_radius=1.5,
        ground_size=20.0,
        margin=1.0,
    ) == pytest.approx(7.5)

    assert clamp_field_length_to_ground(
        x=-8.0,
        y=0.0,
        theta=0.0,
        requested_length=30.0,
        turn_radius=1.5,
        ground_size=20.0,
        margin=1.0,
    ) == pytest.approx(15.5)


def test_path_geometry_can_reserve_dynamic_vehicle_clearance():
    x, y, theta = compute_platform_corner_start_pose(
        ground_size=20.0,
        field_width=5.0,
        lane_count=2,
        track_width=1.2,
        wheelbase=1.2,
        wheel_radius=0.235,
        margin=1.0,
        dynamic_margin=0.5,
    )
    length = clamp_field_length_to_ground(
        x=x,
        y=y,
        theta=theta,
        requested_length=30.0,
        turn_radius=2.5,
        ground_size=20.0,
        margin=1.0,
        dynamic_margin=0.5,
        vehicle_half_length=(1.2 + 2.0 * 0.235) / 2.0,
    )

    assert x == pytest.approx(-6.0)
    assert y == pytest.approx(7.665)
    assert length == pytest.approx(11.165)


def test_speed_from_curvature_runs_fast_on_straights_and_slow_in_turns():
    assert speed_from_curvature(0.0, cruise_speed=1.2, turn_speed=0.25) == pytest.approx(1.2)
    assert speed_from_curvature(
        0.05,
        cruise_speed=1.2,
        turn_speed=0.25,
        straight_curvature_deadband=0.08,
    ) == pytest.approx(1.2)
    assert speed_from_curvature(2.0, cruise_speed=1.2, turn_speed=0.25) == pytest.approx(0.25)
    assert 0.25 < speed_from_curvature(0.4, cruise_speed=1.2, turn_speed=0.25) < 1.2


def test_compute_platform_corner_start_pose_respects_vehicle_and_turn_clearance():
    x, y, theta = compute_platform_corner_start_pose(
        ground_size=20.0,
        field_width=3.0,
        lane_count=2,
        track_width=1.2,
        wheelbase=1.2,
        wheel_radius=0.235,
        margin=1.0,
    )

    assert x == pytest.approx(-7.5)
    assert y == pytest.approx(8.165)
    assert theta == pytest.approx(0.0)


def test_quat_from_yaw_uses_isaac_wxyz_order():
    assert quat_from_yaw(0.0) == pytest.approx((1.0, 0.0, 0.0, 0.0))
    assert quat_from_yaw(math.pi / 2.0) == pytest.approx(
        (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))
    )


def test_planar_yaw_matches_z_yaw_when_roll_pitch_zero():
    from pure_pursuit_controller import planar_yaw_from_pose

    theta = math.radians(27.0)
    quat = quat_from_yaw(theta)
    assert planar_yaw_from_pose(quat) == pytest.approx(yaw_from_quat(quat))


def test_tracking_pose_applies_body_offset():
    from pure_pursuit_controller import tracking_pose_for_planar_path

    tx, ty, tz, yaw = tracking_pose_for_planar_path(
        (10.0, -5.0, 1.25),
        quat_from_yaw(0.0),
        offset_body=(1.0, 0.5, -0.25),
    )
    assert tx == pytest.approx(11.0)
    assert ty == pytest.approx(-4.5)
    assert tz == pytest.approx(1.0)
    assert yaw == pytest.approx(0.0)


def test_planar_speed_drops_vertical_motion():
    from pure_pursuit_controller import planar_speed_from_linear_velocity

    assert planar_speed_from_linear_velocity((3.0, 4.0, 12.0)) == pytest.approx(5.0)


def test_recommended_max_lateral_accel_matches_dynamic_ground_mu():
    a = recommended_max_lateral_accel_mps2(friction_coupling=1.0, use_dynamic_mu=True)
    assert a == pytest.approx(0.45 * GRAVITY_MPS2)


def test_recommended_max_lateral_accel_rejects_invalid_coupling():
    with pytest.raises(ValueError, match="friction_coupling"):
        recommended_max_lateral_accel_mps2(friction_coupling=0.0)


def test_recommended_profile_versions_match_and_tracker_builds():
    from pure_pursuit_controller import (
        DEFAULT_TRACK_WIDTH,
        DEFAULT_WHEEL_RADIUS,
        DEFAULT_WHEELBASE,
        RECOMMENDED_PURE_PURSUIT,
        RECOMMENDED_SCRIPT,
        PurePursuitTracker,
    )

    assert RECOMMENDED_PURE_PURSUIT.version == RECOMMENDED_SCRIPT.version
    kw = RECOMMENDED_PURE_PURSUIT.tracker_kwargs(
        max_steer=RECOMMENDED_SCRIPT.max_steer_rad,
        steer_sign=-1.0,
    )
    tracker = PurePursuitTracker(
        [(0.0, 0.0), (5.0, 0.0)],
        wheelbase=DEFAULT_WHEELBASE,
        track_width=DEFAULT_TRACK_WIDTH,
        wheel_radius=DEFAULT_WHEEL_RADIUS,
        **kw,
    )
    cmd = tracker.compute(0.0, 0.0, 0.0, 0.5)
    assert cmd.done is False


def test_lateral_accel_cap_limits_curvature_at_speed():
    tracker = PurePursuitTracker(
        [(0.0, 0.0), (20.0, 0.0)],
        wheelbase=1.2,
        track_width=1.2,
        wheel_radius=0.25,
        min_lookahead=1.0,
        max_lookahead=1.0,
        lookahead_gain=1.0,
        alpha=0.0,
        heading_gain=0.0,
        max_steer=math.radians(50.0),
        max_lateral_accel_mps2=0.5,
    )
    cmd = tracker.compute(x=0.0, y=1.5, theta=0.0, v_mps=2.0)
    k_lat = 0.5 / 4.0
    k_steer = math.tan(math.radians(50.0)) / 1.2
    assert abs(cmd.curvature) <= min(k_lat, k_steer) + 1e-5


def test_transform_path_to_pose_anchors_field_path_at_robot_start():
    path = generate_lawnmower_path(field_length=30.0, field_width=3.0, semicircle_count=9)

    anchored = transform_path_to_pose(path, x=10.0, y=20.0, theta=0.0)
    rotated = transform_path_to_pose(path, x=10.0, y=20.0, theta=math.pi / 2.0)

    assert anchored[0] == pytest.approx((10.0, 20.0))
    assert anchored[1] == pytest.approx((40.0, 20.0))
    assert anchored[-1] == pytest.approx((40.0, 17.0))
    assert rotated[0] == pytest.approx((10.0, 20.0))
    assert rotated[1] == pytest.approx((10.0, 50.0))
    assert rotated[-1] == pytest.approx((13.0, 50.0))


def test_tracker_progress_does_not_jump_backward_on_dense_parallel_lanes():
    tracker = PurePursuitTracker(generate_lawnmower_path(), min_lookahead=0.25, max_lookahead=0.8)

    first = tracker.compute(x=20.0, y=3.0, theta=0.0, v_mps=0.5)
    second = tracker.compute(x=20.0, y=2.6666666667, theta=math.pi, v_mps=0.5)

    assert second.closest_progress >= first.closest_progress


def test_heading_gain_reduces_spurious_curvature_on_straight_when_yaw_misaligned():
    tracker = PurePursuitTracker(
        [(0.0, 0.0), (10.0, 0.0)],
        wheelbase=1.2,
        track_width=1.2,
        wheel_radius=0.25,
        lookahead_gain=1.0,
        min_lookahead=1.0,
        max_lookahead=1.0,
        alpha=0.0,
        heading_gain=1.0,
    )
    no_heading = PurePursuitTracker(
        [(0.0, 0.0), (10.0, 0.0)],
        wheelbase=1.2,
        track_width=1.2,
        wheel_radius=0.25,
        lookahead_gain=1.0,
        min_lookahead=1.0,
        max_lookahead=1.0,
        alpha=0.0,
        heading_gain=0.0,
    )

    yaw_error = math.radians(12.0)
    cmd_none = no_heading.compute(x=1.0, y=0.0, theta=yaw_error, v_mps=0.15)
    cmd_fix = tracker.compute(x=1.0, y=0.0, theta=yaw_error, v_mps=0.15)

    assert abs(cmd_fix.curvature) + 1e-3 < abs(cmd_none.curvature)


def test_tracker_progress_uses_local_window_to_avoid_far_lane_jumps():
    tracker = PurePursuitTracker(
        [(0.0, 0.0), (10.0, 0.0), (10.0, 1.0), (0.0, 1.0)],
        min_lookahead=0.5,
        max_lookahead=0.5,
        progress_search_ahead=2.0,
    )

    tracker.compute(x=4.0, y=0.0, theta=0.0, v_mps=0.5)
    cmd = tracker.compute(x=8.0, y=1.0, theta=0.0, v_mps=0.5)

    assert cmd.closest_progress <= 6.0


def test_offline_straight_path_validation_finishes_with_low_error():
    result = simulate_path(
        [(0.0, 0.0), (4.0, 0.0)],
        initial_pose=(0.0, 0.0, 0.0),
        v_mps=0.8,
        dt=0.05,
        max_steps=300,
    )

    assert result.done is True
    assert result.rmse_cte < 0.05
    assert result.max_cte < 0.05
    assert result.steer_sign_changes == 0


def test_offline_arc_and_four_point_validation_have_bounded_error():
    arc = [(3.0 * math.cos(a), 3.0 * math.sin(a)) for a in np.linspace(0.0, math.pi / 2.0, 16)]
    arc_result = simulate_path(
        arc,
        initial_pose=(3.0, 0.0, math.pi / 2.0),
        v_mps=0.8,
        dt=0.05,
        max_steps=300,
    )
    four_point_result = simulate_path(
        [(2.0, 0.0), (4.0, 2.0), (2.0, 4.0), (0.0, 2.0)],
        initial_pose=(2.0, 0.0, 0.0),
        v_mps=0.8,
        dt=0.05,
        max_steps=400,
    )

    assert arc_result.done is True
    assert arc_result.rmse_cte < 0.05
    assert arc_result.max_cte < 0.1
    assert four_point_result.done is True
    assert four_point_result.rmse_cte < 0.2
    assert four_point_result.max_cte < 0.3


def test_offline_30_by_3_field_validation_reaches_end():
    result = simulate_path(
        generate_lawnmower_path(),
        initial_pose=(0.0, 3.0, 0.0),
        v_mps=0.5,
        dt=0.2,
        max_steps=4000,
        alpha=0.7,
        min_lookahead=0.25,
        max_lookahead=0.8,
        goal_tolerance=0.3,
    )

    assert result.done is True
    assert result.rmse_cte < 0.08
    assert result.max_cte < 0.45


def test_script_editor_version_uses_existing_isaac_session():
    script = Path(__file__).resolve().parents[1] / "pure_pursuit_script_editor.py"
    text = script.read_text(encoding="utf-8")

    assert "SimulationApp" not in text
    assert "open_stage" not in text
    assert 'PROJECT_DIR = "/workspace/sim"' in text
    assert "sys.path.insert(0, PROJECT_DIR)" in text
    assert "transform_path_to_pose" in text
    assert "generate_main_lane_path" in text
    assert "MAIN_LANE_COUNT = DEFAULT_MAIN_LANE_COUNT" in text
    assert "clamp_field_length_to_ground" in text
    assert "compute_platform_corner_start_pose" in text
    assert "limit_actuator_command" in text
    assert "PP = RECOMMENDED_PURE_PURSUIT" in text
    assert "SC = RECOMMENDED_SCRIPT" in text
    assert "CRUISE_SPEED_MPS = SC.cruise_speed_mps" in text
    assert "TURN_SPEED_MPS = SC.turn_speed_mps" in text
    assert "SLOWDOWN_CURVATURE = SC.slowdown_curvature" in text
    assert "STRAIGHT_CURVATURE_DEADBAND = SC.straight_curvature_deadband" in text
    assert "ACCEL_LIMIT_MPS2 = SC.accel_limit_mps2" in text
    assert "DECEL_LIMIT_MPS2 = SC.decel_limit_mps2" in text
    assert "MAX_WHEEL_ACCEL_RAD_S2 = SC.max_wheel_accel_rad_s2" in text
    assert "MAX_STEER_RAD = SC.max_steer_rad" in text
    assert "MAX_STEER_RATE_RAD_S = SC.max_steer_rate_rad_s" in text
    assert "DYNAMIC_MARGIN_M = SC.dynamic_margin_m" in text
    assert "VEHICLE_HALF_LENGTH_M" in text
    assert "vehicle_half_length=VEHICLE_HALF_LENGTH_M" in text
    assert "LOOKAHEAD_MIN_M = PP.min_lookahead" in text
    assert "LOOKAHEAD_MAX_M = PP.max_lookahead" in text
    assert "**PP.tracker_kwargs" in text
    assert "HEADING_GAIN_PATH = PP.heading_gain" in text
    assert "_reset_robot_state(bot)" in text
    assert "bot.robot.set_joint_positions" in text
    assert "bot.robot.set_joint_velocities" in text
    assert "STALL_SPEED_MPS = SC.stall_speed_mps" in text
    assert "STALL_CURVATURE = SC.stall_curvature" in text
    assert "STALL_CRAWL_SPEED_MPS = SC.stall_crawl_speed_mps" in text
    assert "STALL_MAX_STEER_RAD = SC.stall_max_steer_rad" in text
    assert "FIELD_WIDTH = 5.0" in text
    assert "dynamic_margin=DYNAMIC_MARGIN_M" in text
    assert "FIRST_STRAIGHT_LEN_M" in text
    assert "straight_left=" in text
    assert "SLIP_SPEED_FRACTION = SC.slip_speed_fraction" in text
    assert "ISAAC_REAR_STEER_SIGN = -1.0" in text
    assert "steer_sign=ISAAC_REAR_STEER_SIGN" in text
    assert "DEBUG_PERIOD_S = SC.debug_period_s" in text
    assert "debug_state" in text
    assert "pose_state" in text
    assert "v_measured" in text
    assert "stalling" in text
    assert "_distance_to_path" in text
    assert "_joint_position_or_nan" in text
    assert "cte=" in text
    assert "actual_steer=" in text
    assert "raw_command = replace(" in text
    assert "set_world_pose" in text
    assert "quat_from_yaw" in text
    assert "speed_from_curvature" in text
    assert "current_pos, _current_quat = bot.robot.get_world_pose()" in text
    assert "add_physics_callback" not in text
    assert "remove_physics_callback" not in text
    assert "omni.physx" in text
    assert "subscribe_physics_step_events" in text
    assert "preset=(" in text
    assert "tracking_pose_for_planar_path" in text
    assert "planar_speed_from_linear_velocity" in text
    assert "TRACKING_OFFSET_BODY_M" in text
    assert "USE_LINEAR_VELOCITY_FOR_SPEED" in text
    assert "get_linear_velocity()" in text
    assert "RECOMMENDED_PURE_PURSUIT" in text
    assert "RECOMMENDED_SCRIPT" in text
