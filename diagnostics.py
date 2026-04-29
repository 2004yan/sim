"""
diagnostics.py - Sanity-check tools for the paddy robot scene.

Run in the Script Editor any time the robot is misbehaving.
Reports: fixed-base status, collision geometry, ground collision,
and per-joint drive/limit/schema state.

Combines and replaces: diagnose.py and final.py.
"""
from pxr import UsdPhysics, Usd
import omni.usd

ROBOT_PRIM_PATH = "/Sim_Robot_V2"
GROUND_COL_PATH = "/World/GroundPlane/CollisionPlane"

stage = omni.usd.get_context().get_stage()


# ─── 1. Articulation root ─────────────────────────────────────────────────────
def check_articulation_root():
    print("── Articulation Root ──")
    root = stage.GetPrimAtPath(ROBOT_PRIM_PATH)
    if not root.IsValid():
        print(f"  [!] missing root at {ROBOT_PRIM_PATH}")
        return
    fixed_attr = root.GetAttribute("physxArticulation:fixedBase")
    sc_attr    = root.GetAttribute("physxArticulation:enabledSelfCollisions")
    print(f"  fixedBase              = {fixed_attr.Get() if fixed_attr.IsValid() else 'N/A'}")
    print(f"  enabledSelfCollisions  = {sc_attr.Get()    if sc_attr.IsValid()    else 'N/A'}")


# ─── 2. Collision geometry ────────────────────────────────────────────────────
def check_collision_geometry():
    print("\n── Collision Geometry ──")
    root = stage.GetPrimAtPath(ROBOT_PRIM_PATH)
    if not root.IsValid():
        return
    found = False
    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        found = True
        enabled = prim.GetAttribute("physics:collisionEnabled").Get()
        print(f"  {prim.GetName():30s} → {prim.GetTypeName():18s} enabled={enabled}")
    if not found:
        print("  [!] no collision geometry under robot - robot will float")


# ─── 3. Ground ────────────────────────────────────────────────────────────────
def check_ground_plane():
    print("\n── Ground Plane ──")
    gp = stage.GetPrimAtPath(GROUND_COL_PATH)
    if not gp.IsValid():
        print(f"  [!] missing {GROUND_COL_PATH}")
        return
    print(f"  type             = {gp.GetTypeName()}")
    print(f"  has CollisionAPI = {gp.HasAPI(UsdPhysics.CollisionAPI)}")
    print(f"  collisionEnabled = {gp.GetAttribute('physics:collisionEnabled').Get()}")


# ─── 4. World-anchoring fixed joints ──────────────────────────────────────────
def check_world_anchor_joints():
    print("\n── Fixed-Joint (world anchor) Check ──")
    root = stage.GetPrimAtPath(ROBOT_PRIM_PATH)
    if not root.IsValid():
        return
    found = False
    for prim in Usd.PrimRange(root):
        if prim.GetTypeName() != "PhysicsFixedJoint":
            continue
        joint = UsdPhysics.Joint(prim)
        body0 = [str(t) for t in joint.GetBody0Rel().GetTargets()]
        body1 = [str(t) for t in joint.GetBody1Rel().GetTargets()]
        anchors_world = (not body0) or (not body1) or any(
            t in ("", "/World", "/") for t in body0 + body1
        )
        if anchors_world:
            found = True
            print(f"  [!] {prim.GetPath()} anchors robot to world - robot will NOT move")
    if not found:
        print("  ok: no world-anchoring fixed joints")


# ─── 5. Joints ────────────────────────────────────────────────────────────────
JOINT_TYPES = ("PhysicsRevoluteJoint", "PhysicsPrismaticJoint", "PhysicsJoint")


def check_joints():
    print("\n── Joint Drive/Limit Status ──")
    root = stage.GetPrimAtPath(ROBOT_PRIM_PATH)
    if not root.IsValid():
        return
    for prim in Usd.PrimRange(root):
        if prim.GetTypeName() not in JOINT_TYPES:
            continue
        ang_drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        lin_drive = UsdPhysics.DriveAPI.Get(prim, "linear")
        limit_api = UsdPhysics.LimitAPI.Get(prim, "angular")

        has_ang = bool(ang_drive and ang_drive.GetStiffnessAttr().IsValid())
        has_lin = bool(lin_drive and lin_drive.GetStiffnessAttr().IsValid())
        low     = limit_api.GetLowAttr().Get()  if limit_api else "N/A"
        high    = limit_api.GetHighAttr().Get() if limit_api else "N/A"

        print(f"\n  {prim.GetName()}")
        print(f"    path          : {prim.GetPath()}")
        print(f"    type          : {prim.GetTypeName()}")
        print(f"    angular drive : {has_ang}")
        print(f"    linear drive  : {has_lin}")
        print(f"    limits (deg)  : [{low}, {high}]")
        print(f"    schemas       : {prim.GetAppliedSchemas()}")


# ─── Entry ────────────────────────────────────────────────────────────────────
def run_all():
    check_articulation_root()
    check_collision_geometry()
    check_ground_plane()
    check_world_anchor_joints()
    check_joints()


if __name__ == "__main__":
    run_all()
