"""
controller.py - Runtime controller for the 3-wheel paddy robot.

Run AFTER robot_setup.py has been executed and AFTER pressing Play.

Paddy robot drive model:
    - Front 2 wheels: powered, independent velocity (rad/s)
    - Rear wheel:     passive free spinning (no drive force)
    - Rear link:      rotates about vertical axis to steer (position target, rad)

Uses Isaac Sim 5.1.0 APIs (isaacsim.core.*). See
IsaacSim_5.1.0_API_Reference.md in this directory for full API details.

Replaces: Controller_1.py.
"""
import numpy as np
import omni.usd
from pxr import Sdf, UsdPhysics

from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction

# ─── Configuration ────────────────────────────────────────────────────────────
ROBOT_PRIM_PATH  = "/Sim_Robot_V2"
LEFT_JOINT       = "Front_Left_Joint"
RIGHT_JOINT      = "Front_Right_Joint"
REAR_WHEEL_JOINT = "Rear_Wheel_Joint"
REAR_LINK_JOINT  = "Rear_Link_Joint"

# Default commands (tune as needed)
LEFT_SPEED      = 10.0   # rad/s
RIGHT_SPEED     = 10.0   # rad/s
REAR_LINK_ANGLE = 0.0    # rad  (steering target)


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _resolve_dof_index(dof_names, name):
    """Exact match first; fall back to _y-suffix variants for rotary joints."""
    if name in dof_names:
        return dof_names.index(name)
    for i, n in enumerate(dof_names):
        nl = n.lower()
        if name in n and (nl.endswith("_y") or "roty" in nl):
            print(f"[ctrl] using '{n}' as fallback for '{name}'")
            return i
    print(f"[ctrl] joint '{name}' not found in DOF list")
    return None


def _disable_joint_drive(stage, joint_path):
    """Make a rolling joint passive even if an older setup script added a drive."""
    prim = stage.GetPrimAtPath(joint_path)
    if not prim.IsValid():
        return
    UsdPhysics.DriveAPI.Apply(prim, "angular")
    for attr, val in (
        ("drive:angular:physics:stiffness", 0.0),
        ("drive:angular:physics:damping", 0.0),
        ("drive:angular:physics:maxForce", 0.0),
    ):
        a = prim.GetAttribute(attr)
        if not a.IsValid():
            a = prim.CreateAttribute(attr, Sdf.ValueTypeNames.Float)
        a.Set(val)
    print(f"[ctrl] passive rear wheel drive disabled at {joint_path}")


# ─── Robot wrapper ────────────────────────────────────────────────────────────
class PaddyRobotController:
    def __init__(self, prim_path=ROBOT_PRIM_PATH):
        stage = omni.usd.get_context().get_stage()
        prim  = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"Robot prim not found at {prim_path}")

        UsdPhysics.ArticulationRootAPI.Apply(prim)
        _disable_joint_drive(stage, f"{prim_path}/joints/{REAR_WHEEL_JOINT}")
        self.robot = SingleArticulation(prim_path=prim_path)
        self.robot.initialize()

        self.dof_names  = list(self.robot.dof_names)
        self.controller = self.robot.get_articulation_controller()

        self.left_idx       = _resolve_dof_index(self.dof_names, LEFT_JOINT)
        self.right_idx      = _resolve_dof_index(self.dof_names, RIGHT_JOINT)
        self.rear_wheel_idx = _resolve_dof_index(self.dof_names, REAR_WHEEL_JOINT)
        self.rear_link_idx  = _resolve_dof_index(self.dof_names, REAR_LINK_JOINT)

        print(f"[ctrl] DOF names: {self.dof_names}")

    # ── Primary commands ─────────────────────────────────────────────────────
    def set_wheel_speeds(self, left_rad_s, right_rad_s):
        if self.left_idx is None or self.right_idx is None:
            return
        self.controller.apply_action(ArticulationAction(
            joint_velocities=np.array([left_rad_s, right_rad_s], dtype=np.float32),
            joint_indices=np.array([self.left_idx, self.right_idx], dtype=np.int32),
        ))

    def set_steering_angle(self, angle_rad):
        if self.rear_link_idx is None:
            return
        self.controller.apply_action(ArticulationAction(
            joint_positions=np.array([angle_rad], dtype=np.float32),
            joint_indices=np.array([self.rear_link_idx], dtype=np.int32),
        ))

    def set_rear_wheel_speed(self, rad_s):
        """No-op: the rear wheel is passive and must not be velocity-driven."""
        return

    # ── Convenience ──────────────────────────────────────────────────────────
    def drive(self, linear_speed_rad_s, steering_angle_rad=0.0):
        """Convenience: both front wheels same speed + steering angle."""
        self.set_wheel_speeds(linear_speed_rad_s, linear_speed_rad_s)
        self.set_steering_angle(steering_angle_rad)

    def stop(self):
        self.set_wheel_speeds(0.0, 0.0)


# ─── Entry ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    bot = PaddyRobotController()
    bot.set_wheel_speeds(LEFT_SPEED, RIGHT_SPEED)
    bot.set_steering_angle(REAR_LINK_ANGLE)
    print(f"[ctrl] commands sent: L={LEFT_SPEED}, R={RIGHT_SPEED}, "
          f"steer={np.degrees(REAR_LINK_ANGLE):.1f} deg")
