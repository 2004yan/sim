"""
robot_setup.py - One-shot physics setup for the 3-wheel paddy robot.

Run once in the Script Editor (NOT every frame), before pressing Play.

Configures /Sim_Robot_V2:
    - removes the world-to-base root_joint (URDF import adds a FixedJoint
      that pins the robot to the world - wheels spin but robot can't move)
    - masses on base + wheels + rear link
    - angular drives (position control on rear link, velocity control on wheels)
    - rear link joint limits + removes blocking JointStateAPI
    - rubber wheel physics materials
    - disables articulation fixed base (so gravity acts)

Combines and replaces: Setup_3.py, Fix.py (robot parts), and the schema/limit
patches from Controller_1.py.
"""
from pxr import UsdPhysics, UsdShade, Sdf, Gf, UsdGeom, Usd
import omni.usd

# ─── Configuration ────────────────────────────────────────────────────────────
BASE         = "/Sim_Robot_V2"
JOINTS       = f"{BASE}/joints"
BASE_LINK    = f"{BASE}/base_link"

REAR_LINK_JOINT  = f"{JOINTS}/Rear_Link_Joint"
FRONT_L_JOINT    = f"{JOINTS}/Front_Left_Joint"
FRONT_R_JOINT    = f"{JOINTS}/Front_Right_Joint"
REAR_WHEEL_JOINT = f"{JOINTS}/Rear_Wheel_Joint"

MASS_CONFIG = {
    BASE_LINK:            350.0,
    f"{BASE}/Rear_Link":   10.0,
    f"{BASE}/Rear_Wheel":   5.0,
    f"{BASE}/Front_Left":   5.0,
    f"{BASE}/Front_Right":  5.0,
}

# (μs static, μd dynamic) — rubber tyre grips initially, drags through mud
WHEEL_FRICTION = (0.8, 0.7)
WHEEL_PRIMS = [f"{BASE}/Front_Left", f"{BASE}/Front_Right", f"{BASE}/Rear_Wheel"]

# Rear link steering angle limits (degrees)
REAR_LINK_LIMIT_DEG = (-90.0, 90.0)

# Robot spawn Z offset above ground (ground is at Z=0 in 5.1.0 GroundPlane)
SPAWN_Z = 0.5

stage = omni.usd.get_context().get_stage()


# ─── 0. Remove world-to-base root_joint (if present) ──────────────────────────
def remove_root_fixed_joint():
    """
    URDF import sometimes adds a world -> base_link PhysicsFixedJoint named
    'root_joint' to anchor the robot. Keep it and the articulation is welded
    to the world: wheels spin, body doesn't translate. Delete any FixedJoint
    under the robot that connects to the world.
    """
    root = stage.GetPrimAtPath(BASE)
    if not root.IsValid():
        print(f"[robot] root missing at {BASE}")
        return

    to_delete = []
    for prim in Usd.PrimRange(root):
        if prim.GetTypeName() != "PhysicsFixedJoint":
            continue
        joint = UsdPhysics.Joint(prim)
        body0 = [str(t) for t in joint.GetBody0Rel().GetTargets()]
        body1 = [str(t) for t in joint.GetBody1Rel().GetTargets()]
        # FixedJoint is world-anchoring if body0 or body1 is empty/world
        is_world_anchor = (not body0) or (not body1) or any(
            t in ("", "/World", "/") for t in body0 + body1
        )
        if is_world_anchor or prim.GetName() == "root_joint":
            to_delete.append(prim.GetPath())

    if not to_delete:
        print("[robot] no world-anchoring FixedJoint found")
        return

    for path in to_delete:
        stage.RemovePrim(path)
        print(f"[robot] removed fixed joint at {path}")


# ─── 1. Masses ────────────────────────────────────────────────────────────────
def set_masses():
    for path, kg in MASS_CONFIG.items():
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            print(f"[robot] mass skipped - missing {path}")
            continue
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr().Set(kg)
        print(f"[robot] mass {path.split('/')[-1]} = {kg} kg")


# ─── 2. Rear link: clean schemas, set limits ──────────────────────────────────
def fix_rear_link_joint():
    prim = stage.GetPrimAtPath(REAR_LINK_JOINT)
    if not prim.IsValid():
        print(f"[robot] rear link joint missing at {REAR_LINK_JOINT}")
        return
    # PhysicsJointStateAPI:angular silently blocks drive commands - strip it
    if "PhysicsJointStateAPI:angular" in prim.GetAppliedSchemas():
        prim.RemoveAppliedSchema("PhysicsJointStateAPI:angular")
        print("[robot] removed PhysicsJointStateAPI:angular from rear link")
    # Widen limits (were 0,0 = fully locked)
    lower, upper = REAR_LINK_LIMIT_DEG
    for attr_name, val in (("physics:lowerLimit", lower), ("physics:upperLimit", upper)):
        attr = prim.GetAttribute(attr_name)
        if attr.IsValid():
            attr.Set(val)
    print(f"[robot] rear link limits = {REAR_LINK_LIMIT_DEG} deg")


# ─── 3. Joint drives ──────────────────────────────────────────────────────────
def _set_drive(path, stiffness, damping, max_force=1e8):
    """Position control: stiffness>0. Velocity control: stiffness=0, damping>0."""
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        print(f"[robot] drive skipped - missing {path}")
        return
    UsdPhysics.DriveAPI.Apply(prim, "angular")
    for attr, val in (
        ("drive:angular:physics:stiffness", stiffness),
        ("drive:angular:physics:damping",   damping),
        ("drive:angular:physics:maxForce",  max_force),
    ):
        a = prim.GetAttribute(attr)
        if not a.IsValid():
            a = prim.CreateAttribute(attr, Sdf.ValueTypeNames.Float)
        a.Set(val)
    mode = "position" if stiffness > 0 else "velocity"
    print(f"[robot] {path.split('/')[-1]} drive: {mode} (k={stiffness}, b={damping})")


def set_drives():
    _set_drive(REAR_LINK_JOINT,  stiffness=1e6, damping=1e4)   # rear steering
    _set_drive(FRONT_L_JOINT,    stiffness=0,   damping=1e4)   # left front wheel
    _set_drive(FRONT_R_JOINT,    stiffness=0,   damping=1e4)   # right front wheel
    _set_drive(REAR_WHEEL_JOINT, stiffness=0,   damping=1e4)   # rear free wheel


# ─── 4. Wheel friction materials ──────────────────────────────────────────────
def _set_friction(prim_path, mu_s, mu_d, restitution=0.0):
    mat_path = prim_path + "/PhysMat"
    mat      = UsdShade.Material.Define(stage, mat_path)
    phys_api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    phys_api.CreateStaticFrictionAttr().Set(mu_s)
    phys_api.CreateDynamicFrictionAttr().Set(mu_d)
    phys_api.CreateRestitutionAttr().Set(restitution)

    target = stage.GetPrimAtPath(prim_path)
    if not target.IsValid():
        print(f"[robot] friction skipped - missing {prim_path}")
        return
    UsdShade.MaterialBindingAPI.Apply(target)
    UsdShade.MaterialBindingAPI(target).Bind(
        mat, UsdShade.Tokens.strongerThanDescendants, "physics",
    )
    print(f"[robot] {prim_path.split('/')[-1]} friction: us={mu_s}, ud={mu_d}")


def set_wheel_friction():
    mu_s, mu_d = WHEEL_FRICTION
    for wheel in WHEEL_PRIMS:
        _set_friction(wheel, mu_s, mu_d)


# ─── 5. Articulation root flags ───────────────────────────────────────────────
def disable_fixed_base():
    root = stage.GetPrimAtPath(BASE)
    if not root.IsValid():
        print(f"[robot] root missing at {BASE}")
        return
    attr = root.GetAttribute("physxArticulation:fixedBase")
    if attr.IsValid() and attr.Get():
        attr.Set(False)
        print("[robot] fixedBase disabled - gravity now acts")


# ─── 6. Spawn Z (optional) ────────────────────────────────────────────────────
def lift_robot_above_ground(z=SPAWN_Z):
    prim = stage.GetPrimAtPath(BASE)
    if not prim.IsValid():
        return
    for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
        if "translate" in op.GetOpName().lower():
            cur = op.Get()
            op.Set(Gf.Vec3d(cur[0], cur[1], z))
            print(f"[robot] lifted to Z={z}")
            return


# ─── Entry ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    remove_root_fixed_joint()   # must happen BEFORE Articulation.initialize()
    set_masses()
    fix_rear_link_joint()
    set_drives()
    set_wheel_friction()
    disable_fixed_base()
    # lift_robot_above_ground()   # uncomment if the robot spawns inside the ground
    print("[robot] Physics setup complete.")
