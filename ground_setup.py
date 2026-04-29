"""
ground_setup.py - Paddy field ground plane with muddy physics + wet-mud visual.

Run once in Isaac Sim's Script Editor before pressing Play.
Creates /World/GroundPlane with:
    - validation mud physics material (moderate grip, compliant contact)
    - OmniPBR visual material (dark olive wet soil)
    - PhysX contact offsets so wheels "sink" slightly

Combines and replaces: Paddy_Ground_Creation.py, Groundmesh.py,
                       and the ground sections of Setup_3.py.
"""
import numpy as np
import omni.usd
import omni.kit.commands
from pxr import UsdShade, Sdf, Gf, PhysxSchema

from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.core.api.materials.physics_material import PhysicsMaterial
from isaacsim.core.api.physics_context import PhysicsContext

# ─── Paths ────────────────────────────────────────────────────────────────────
GROUND_PATH   = "/World/GroundPlane"
PHYS_MAT_PATH = "/World/Looks/PaddyPhysicsMaterial"
VIS_MAT_PATH  = "/World/Looks/PaddyField_Surface"

stage = omni.usd.get_context().get_stage()


# ─── 1. Create the ground plane ───────────────────────────────────────────────
def create_ground_plane():
    PhysicsContext()
    GroundPlane(
        prim_path=GROUND_PATH,
        size=20.0,
        color=np.array([0.35, 0.25, 0.10]),
    )
    print(f"[ground] GroundPlane created at {GROUND_PATH}")


# ─── 2. Muddy physics material ────────────────────────────────────────────────
def create_physics_material():
    PhysicsMaterial(
        prim_path=PHYS_MAT_PATH,
        static_friction=0.6,     # validation grip; lower after steering is stable
        dynamic_friction=0.45,   # keep dynamic friction above hard-turn demand
        restitution=0.0,         # zero bounce - mud absorbs impact
    )
    # PhysX compliant contact: low stiffness + damping = soft mushy surface
    phys_prim = stage.GetPrimAtPath(PHYS_MAT_PATH)
    physx_api = PhysxSchema.PhysxMaterialAPI.Apply(phys_prim)
    physx_api.CreateCompliantContactStiffnessAttr(800)
    physx_api.CreateCompliantContactDampingAttr(1000.0)
    print(f"[ground] Mud physics material at {PHYS_MAT_PATH}")


def bind_physics_material():
    ground_prim   = stage.GetPrimAtPath(GROUND_PATH)
    phys_mat_prim = stage.GetPrimAtPath(PHYS_MAT_PATH)
    if not (ground_prim.IsValid() and phys_mat_prim.IsValid()):
        print("[ground] physics-material bind skipped (prim missing)")
        return
    shade   = UsdShade.Material(phys_mat_prim)
    binding = UsdShade.MaterialBindingAPI.Apply(ground_prim)
    # purpose="physics" leaves the visual material alone
    binding.Bind(shade, UsdShade.Tokens.weakerThanDescendants, "physics")
    print(f"[ground] Physics material bound to {GROUND_PATH}")


# ─── 3. OmniPBR visual material (dark wet mud) ────────────────────────────────
def create_visual_material():
    omni.kit.commands.execute(
        "CreateMdlMaterialPrim",
        mtl_url="OmniPBR.mdl",
        mtl_name="PaddyField_Surface",
        mtl_path=VIS_MAT_PATH,
    )
    vis_prim = stage.GetPrimAtPath(VIS_MAT_PATH)
    if not vis_prim.IsValid():
        print("[ground] Visual material creation failed (Nucleus unreachable?)")
        return

    inputs = [
        ("diffuse_color_constant", Gf.Vec3f(0.18, 0.22, 0.08), Sdf.ValueTypeNames.Color3f),
        ("roughness_constant",     0.95,                        Sdf.ValueTypeNames.Float),
        ("metallic_constant",      0.0,                         Sdf.ValueTypeNames.Float),
        ("enable_emission",        False,                       Sdf.ValueTypeNames.Bool),
    ]
    for name, value, type_ in inputs:
        omni.usd.create_material_input(vis_prim, name, value, type_)

    _bind_visual_material(vis_prim)


def _find_mesh(prim):
    for child in prim.GetChildren():
        if child.GetTypeName() == "Mesh":
            return child
        found = _find_mesh(child)
        if found:
            return found
    return None


def _bind_visual_material(vis_prim):
    ground_prim = stage.GetPrimAtPath(GROUND_PATH)
    target = _find_mesh(ground_prim) if ground_prim.IsValid() else None
    target = target or ground_prim
    if not (target and target.IsValid()):
        print("[ground] no mesh to bind visual material")
        return
    shade = UsdShade.Material(vis_prim)
    UsdShade.MaterialBindingAPI(target).Bind(
        shade, UsdShade.Tokens.strongerThanDescendants,
    )
    print(f"[ground] Visual OmniPBR bound to {target.GetPath()}")


# ─── 4. PhysX contact offsets (wheel sinking) ─────────────────────────────────
def apply_contact_offsets():
    ground_prim = stage.GetPrimAtPath(GROUND_PATH)
    if not ground_prim.IsValid():
        return
    PhysxSchema.PhysxCollisionAPI.Apply(ground_prim)
    for name, value in (
        ("physxCollision:contactOffset", 0.02),   # 2 cm soft-mud contact zone
        ("physxCollision:restOffset",    0.005),  # 5 mm sink depth
    ):
        attr = ground_prim.GetAttribute(name)
        if not attr.IsValid():
            attr = ground_prim.CreateAttribute(name, Sdf.ValueTypeNames.Float)
        attr.Set(value)
    print("[ground] contactOffset=0.02 m, restOffset=0.005 m")


# ─── Entry ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    create_ground_plane()
    create_physics_material()
    bind_physics_material()
    create_visual_material()
    apply_contact_offsets()
    print("[ground] Paddy field ready.")
