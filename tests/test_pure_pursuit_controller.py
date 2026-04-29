import math
from pathlib import Path

import numpy as np
import pytest

from pure_pursuit_controller import (
    DEFAULT_STEER_SIGN,
    DEFAULT_TRACK_WIDTH,
    DEFAULT_WHEEL_RADIUS,
    DEFAULT_WHEELBASE,
    PurePursuitCommand,
    PurePursuitTracker,
    apply_command,
    generate_lawnmower_path,
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


def test_lateral_offset_generates_curvature_and_blended_actuator_commands():
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
    assert cmd.curvature == pytest.approx(-2.0)
    assert cmd.left_rad_s == pytest.approx(6.4)
    assert cmd.right_rad_s == pytest.approx(1.6)
    assert cmd.steer_rad == pytest.approx(math.atan(-1.2))


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


def test_apply_command_sends_wheel_speeds_and_steering_to_robot_controller():
    class FakeBot:
        def __init__(self):
            self.wheels = None
            self.steer = None

        def set_wheel_speeds(self, left_rad_s, right_rad_s):
            self.wheels = (left_rad_s, right_rad_s)

        def set_steering_angle(self, angle_rad):
            self.steer = angle_rad

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
    assert "add_physics_callback" in text
    assert "remove_physics_callback" in text
