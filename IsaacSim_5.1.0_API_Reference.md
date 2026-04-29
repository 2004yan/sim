# NVIDIA Isaac Sim 5.1.0 — Python API Reference

Source: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/
Compiled: 2026-04-24

This document compiles the main Python API calls, classes, and methods available in Isaac Sim 5.1.0. Use it as a quick lookup reference when scripting extensions, standalone apps, or tasks within this workspace.

---

## Table of Contents

1. [Full Extension Module List](#1-full-extension-module-list)
2. [isaacsim.simulation_app — Launching Isaac Sim](#2-isaacsimsimulation_app--launching-isaac-sim)
3. [isaacsim.core.api — Core Scene/World API](#3-isaacsimcoreapi--core-sceneworld-api)
4. [isaacsim.core.prims — Prim Wrappers](#4-isaacsimcoreprims--prim-wrappers)
5. [isaacsim.core.utils — Utilities](#5-isaacsimcoreutils--utilities)
6. [isaacsim.core.simulation_manager](#6-isaacsimcoresimulation_manager)
7. [isaacsim.core.cloner — Environment Cloning](#7-isaacsimcorecloner--environment-cloning)
8. [isaacsim.sensors.camera](#8-isaacsimsensorscamera)
9. [isaacsim.sensors.physics — Contact / IMU / Effort](#9-isaacsimsensorsphysics--contact--imu--effort)
10. [isaacsim.sensors.rtx — RTX Lidar / Radar / IDS](#10-isaacsimsensorsrtx--rtx-lidar--radar--ids)
11. [isaacsim.robot.manipulators](#11-isaacsimrobotmanipulators)
12. [isaacsim.robot.wheeled_robots](#12-isaacsimrobotwheeled_robots)
13. [isaacsim.robot_motion.motion_generation](#13-isaacsimrobot_motionmotion_generation)
14. [isaacsim.asset.importer.urdf — URDF Import](#14-isaacsimassetimporterurdf--urdf-import)
15. [isaacsim.storage.native — Nucleus / Assets](#15-isaacsimstoragenative--nucleus--assets)
16. [isaacsim.replicator.writers](#16-isaacsimreplicatorwriters)

---

## 1. Full Extension Module List

All Python extensions shipped with Isaac Sim 5.1.0, grouped by category:

### App
- `isaacsim.app.about`
- `isaacsim.app.selector`
- `isaacsim.app.setup`

### Asset (Importers / Exporters / Generators)
- `isaacsim.asset.browser`
- `isaacsim.asset.exporter.urdf`
- `isaacsim.asset.gen.conveyor`
- `isaacsim.asset.gen.conveyor.ui`
- `isaacsim.asset.gen.omap`
- `isaacsim.asset.gen.omap.ui`
- `isaacsim.asset.importer.heightmap`
- `isaacsim.asset.importer.urdf`
- `isaacsim.asset.importer.mjcf`

### Benchmark
- `isaacsim.benchmark.examples`
- `isaacsim.benchmark.services`

### Code Editor
- `isaacsim.code_editor.jupyter`
- `isaacsim.code_editor.vscode`

### Core
- `isaacsim.core.api`
- `isaacsim.core.cloner`
- `isaacsim.core.deprecation_manager`
- `isaacsim.core.experimental.materials`
- `isaacsim.core.experimental.objects`
- `isaacsim.core.experimental.prims`
- `isaacsim.core.experimental.utils`
- `isaacsim.core.includes`
- `isaacsim.core.nodes`
- `isaacsim.core.prims`
- `isaacsim.core.simulation_manager`
- `isaacsim.core.throttling`
- `isaacsim.core.utils`
- `isaacsim.core.version`

### Cortex (Task / Behavior framework)
- `isaacsim.cortex.behaviors`
- `isaacsim.cortex.framework`

### Examples
- `isaacsim.examples.browser`
- `isaacsim.examples.extension`
- `isaacsim.examples.interactive`
- `isaacsim.examples.ui`

### GUI
- `isaacsim.gui.components`
- `isaacsim.gui.menu`
- `isaacsim.gui.property`

### Replicator (Synthetic Data Generation)
- `isaacsim.replicator.behavior`
- `isaacsim.replicator.behavior.ui`
- `isaacsim.replicator.domain_randomization`
- `isaacsim.replicator.examples`
- `isaacsim.replicator.grasping`
- `isaacsim.replicator.mobility_gen`
- `isaacsim.replicator.synthetic_recorder`
- `isaacsim.replicator.writers`

### Robot Motion & Setup
- `isaacsim.robot_motion.lula`
- `isaacsim.robot_motion.lula_test_widget`
- `isaacsim.robot_motion.motion_generation`
- `isaacsim.robot_setup.assembler`
- `isaacsim.robot_setup.gain_tuner`
- `isaacsim.robot_setup.grasp_editor`
- `isaacsim.robot_setup.wizard`
- `isaacsim.robot_setup.xrdf_editor`

### Robot
- `isaacsim.robot.manipulators`
- `isaacsim.robot.manipulators.examples`
- `isaacsim.robot.manipulators.ui`
- `isaacsim.robot.policy.examples`
- `isaacsim.robot.schema`
- `isaacsim.robot.surface_gripper`
- `isaacsim.robot.surface_gripper.ui`
- `isaacsim.robot.wheeled_robots`
- `isaacsim.robot.wheeled_robots.ui`

### ROS 2
- `isaacsim.ros2.bridge`
- `isaacsim.ros2.tf_viewer`
- `isaacsim.ros2.urdf`

### Sensors
- `isaacsim.sensors.camera` (+ `.ui`)
- `isaacsim.sensors.physics` (+ `.examples`, `.ui`)
- `isaacsim.sensors.physx` (+ `.examples`, `.ui`)
- `isaacsim.sensors.rtx` (+ `.ui`)

### Simulation
- `isaacsim.simulation_app`

### Storage
- `isaacsim.storage.native`

### Test
- `isaacsim.test.collection`
- `isaacsim.test.docstring`

### Util
- `isaacsim.util.camera_inspector`
- `isaacsim.util.debug_draw`
- `isaacsim.util.merge_mesh`
- `isaacsim.util.physics`

### Deprecated (kept for backward compatibility)
- `omni.isaac.dynamic_control`
- `omni.kit.loop-isaac`

---

## 2. isaacsim.simulation_app — Launching Isaac Sim

### SimulationApp
Top-level helper for launching the Omniverse Toolkit from a standalone Python script. Must be instantiated **before** importing any `isaacsim.*` or `omni.*` module.

```python
from isaacsim.simulation_app import SimulationApp
simulation_app = SimulationApp({"headless": False})
```

**Methods:**
- `update()` — Step the app forward one frame.
- `close(wait_for_replicator=True, skip_cleanup=False)` — Shut down the app.
- `is_running()` — `True` if app is running with a valid stage.
- `is_exiting()` — `True` once `close()` has been called.
- `set_setting(setting, value)` — Update a Carbonite setting.
- `reset_render_settings()` — Reapply initial render settings.
- `run_coroutine(coroutine, run_until_complete=True)` — Run a coroutine in Kit's event loop.

**Properties:**
- `app` — underlying Omniverse Kit application object
- `context` — current USD context
- `DEFAULT_LAUNCHER_CONFIG` — dict of default launcher settings (headless, GPU, rendering, window size)

### AppFramework
Minimal launcher without app config.
- `update()`, `close()`
- `app`, `framework`

---

## 3. isaacsim.core.api — Core Scene/World API

### SimulationContext
Central class providing physics and render stepping, time management, and simulation control.

### World  (extends SimulationContext)
High-level interface for managing simulation scenes, objects, and tasks.

```python
from isaacsim.core.api import World
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
world.reset()
while simulation_app.is_running():
    world.step(render=True)
```

### Controllers

**ArticulationController**
- `initialize(articulation_view)`
- `apply_action(control_actions, indices=None)`
- `get_applied_action() -> ArticulationAction`
- `switch_control_mode(mode)`
- `switch_dof_control_mode(dof_index, mode)`
- `set_gains(kps=None, kds=None, save_to_usd=False)`
- `get_gains() -> Tuple[ndarray, ndarray]`
- `set_max_efforts(values, joint_indices=None)`
- `get_max_efforts() -> ndarray`
- `get_joint_limits() -> Tuple[ndarray, ndarray]`
- `set_effort_modes(mode, joint_indices=None)`
- `get_effort_modes() -> List[str]`

**BaseController** (abstract)
- `forward(*args, **kwargs) -> ArticulationAction`
- `reset()`

**BaseGripperController** (abstract)
- `forward(action, current_joint_positions) -> ArticulationAction`
- `open(current_joint_positions) -> ArticulationAction`
- `close(current_joint_positions) -> ArticulationAction`
- `reset()`

### Objects
- **Visual**: `VisualCapsule`, `VisualCone`, `VisualCuboid`, `VisualCylinder`, `VisualSphere`
- **Fixed (static collider)**: `FixedCapsule`, `FixedCone`, `FixedCuboid`, `FixedCylinder`, `FixedSphere`
- **Dynamic (rigid body)**: `DynamicCapsule`, `DynamicCone`, `DynamicCuboid`, `DynamicCylinder`, `DynamicSphere`
- `GroundPlane` — High-level wrapper for a ground plane.

### Robots
- **Robot** — single-articulation wrapper
- **RobotView** — batched multi-articulation wrapper

### Scenes
- **Scene**: `add(obj)`, `get_object(name)`, `reset_default_state()`, access registered objects/properties
- **SceneRegistry** — tracks different object types added to scenes

### Tasks
- **BaseTask** (abstract) — modular task definition (setup objects, observations, metrics)
- Pre-built: **FollowTarget**, **PickPlace**, **Stacking**

### Materials
- **VisualMaterial** (base) + **PreviewSurface**, **OmniPBR**, **OmniGlass**
- **PhysicsMaterial**: `get/set_static_friction()`, `get/set_dynamic_friction()`, `get/set_restitution()`
- **ParticleMaterial**, **DeformableMaterial**

### Sensors / Physics
- **BaseSensor** — common sensor interface
- **RigidContactView** — contact tracking for rigid prims
- **PhysicsContext** — scene physics configuration

### Loggers
**DataLogger**
- `start()`, `pause()`, `reset()`
- `add_data(data, current_time_step, current_time)`
- `add_data_frame_logging_func(func)`
- `save(log_path)`, `load(log_path)`
- `get_data_frame(index) -> DataFrame`
- `get_num_of_data_frames() -> int`
- `is_started() -> bool`

---

## 4. isaacsim.core.prims — Prim Wrappers

High-level wrappers for managing USD prims. Multi-instance classes accept a regex path and operate on one or many prims at once.

### Multi-instance (preferred)
- **Articulation** — prims with Root Articulation API (articulated robots)
- **RigidPrim** — rigid body physics
- **XFormPrim** — transforms (translation, orientation, scale)
- **GeometryPrim** — geometry attributes
- **ClothPrim** — cloth simulation
- **DeformablePrim** — deformable physics
- **ParticleSystem** — particle simulation
- **SdfShapePrim** — signed distance field geometry

### Single-instance (legacy; prefer multi-instance)
- **SingleArticulation**, **SingleRigidPrim**, **SingleXFormPrim**, **SingleGeometryPrim**, **SingleClothPrim**, **SingleDeformablePrim**, **SingleParticleSystem**

### Common Methods

**Transforms:**
- `set_world_pose()` / `get_world_pose()` — world-frame pose
- `set_local_pose()` / `get_local_pose()` — parent-frame pose

**Physics (Articulation):**
- `apply_action()` — apply joint positions / velocities / efforts
- `set_joint_position_targets()` / `set_joint_velocity_targets()` / `set_joint_efforts()`

**State queries:**
- `get_joint_positions()` / `get_joint_velocities()`
- `get_linear_velocities()` / `get_angular_velocities()`
- `get_body_masses()` / `get_body_inertias()`

**Visual:**
- `apply_visual_materials()` / `get_applied_visual_materials()`

**Lifecycle:**
- `initialize()` — required before operating on the view

---

## 5. isaacsim.core.utils — Utilities

Organized into submodules. Import as `from isaacsim.core.utils.<submodule> import <fn>`.

### Submodules
`articulations`, `bounds`, `carb`, `collisions`, `commands`, `extensions`, `interops`, `math`, `stage`, `prims`, `rotations`, `transformations`, `nucleus`, `viewports`, `render_product`, `semantics`, `string`, `types`.

### Selected Functions

**articulations**
- `add_articulation_root(prim)`
- `find_all_articulation_base_paths()`
- `move_articulation_root(src_prim, dst_prim)`
- `remove_articulation_root(prim)`

**bounds**
- `compute_aabb(bbox_cache, prim_path, include_children)`
- `compute_obb(bbox_cache, prim_path)` — returns centroid, axes, half-extent
- `compute_obb_corners(bbox_cache, prim_path)` — 8 corner vertices
- `create_bbox_cache(time, use_extents_hint)`

**carb**
- `get_carb_setting(carb_settings, setting)`
- `set_carb_setting(carb_settings, setting, value)`

**collisions**
- `ray_cast(position, orientation, offset, max_dist)` — cast ray forward

**commands** (USD prim manipulation)
- `IsaacSimSpawnPrim`, `IsaacSimTeleportPrim`, `IsaacSimScalePrim`, `IsaacSimDestroyPrim`

**extensions**
- `enable_extension(extension_name)`
- `disable_extension(extension_name)`
- `get_extension_path_from_name(extension_name)`

**interops** (framework conversions)
- `torch2numpy()`, `numpy2torch()`
- `jax2numpy()`, `numpy2jax()`
- `tensorflow2torch()`, `torch2tensorflow()`
- `warp2numpy()`, `numpy2warp()`

**math**
- `cross(a, b)`, `normalize(v)` / `normalized(v)`
- `radians_to_degrees(rad_angles)`

---

## 6. isaacsim.core.simulation_manager

### SimulationManager
Handles time-based events (warm-starting, physics stepping, etc.).

**Queries:**
- `assets_loading()`, `get_backend()`, `get_broadphase_type()`
- `get_default_physics_scene()`, `get_num_physics_steps()`, `get_physics_dt()`
- `get_physics_sim_device()`, `get_physics_sim_view()`, `get_simulation_time()`
- `get_solver_type()`, `get_default_callback_status()`

**Enable/Disable:**
- `enable_all_default_callbacks()`, `enable_ccd()`, `enable_fabric()`
- `enable_gpu_dynamics()`, `enable_stablization()`
- `enable_warm_start_callback()`, `enable_on_stop_callback()`
- `enable_post_warm_start_callback()`, `enable_stage_open_callback()`

**Status:**
- `is_ccd_enabled()`, `is_gpu_dynamics_enabled()`, `is_stablization_enabled()`
- `is_paused()`, `is_simulating()`, `is_fabric_enabled()`

**Configuration:**
- `set_backend()`, `set_broadphase_type()`, `set_physics_dt()`
- `set_solver_type()` — `"TGS"` or `"PGS"`
- `set_default_physics_scene()`, `set_physics_sim_device()`

**Callbacks:**
- `register_callback()`, `deregister_callback()`

**Lifecycle:**
- `initialize_physics()`, `step()`

### IsaacEvents (enum)
- `PHYSICS_READY`, `PHYSICS_WARMUP`
- `PRE_PHYSICS_STEP`, `POST_PHYSICS_STEP`
- `POST_RESET`, `PRIM_DELETION`
- `SIMULATION_VIEW_CREATED`, `TIMELINE_STOP`

---

## 7. isaacsim.core.cloner — Environment Cloning

Used for creating parallel training environments (reinforcement learning).

### Cloner
- `clone()` — duplicate a source prim at destination paths with poses
- `define_base_env()` — create parent USD Scope
- `generate_paths()` — generate path list
- `replicate_physics()` — replicate physics in `omni.physics`
- `filter_collisions()` — filter collisions between clones
- `enable_change_listener()` / `disable_change_listener()`

### GridCloner (extends Cloner)
- `clone()` — grid-arranged clones with auto-computed positions
- `get_clone_transforms()` — compute grid transforms
- (all inherited Cloner methods)

---

## 8. isaacsim.sensors.camera

### Camera
Primary camera wrapper.

```python
from isaacsim.sensors.camera import Camera
cam = Camera(prim_path="/World/Camera", resolution=(1280, 720))
cam.initialize()
cam.add_distance_to_image_plane_to_frame()
rgba = cam.get_rgba()
```

**Data Acquisition:**
- `get_rgba()`, `get_rgb()`, `get_depth()`, `get_pointcloud()`
- `get_current_frame()`

**Camera Properties:**
- `get_resolution()`, `get_focal_length()`
- `get_horizontal_fov()`, `get_vertical_fov()`
- `get_clipping_range()`, `get_aspect_ratio()`, `get_projection_mode()`

**Pose & Projection:**
- `get_world_pose()`, `get_local_pose()`
- `get_world_scale()`, `get_local_scale()`
- `get_image_coords_from_world_points()` — 3D→2D
- `get_world_points_from_image_coords()` — 2D→3D

**Annotators:**
- `add_rgb_to_frame()` / `add_rgba_to_frame()`
- `add_depth_to_frame()` / `add_distance_to_image_plane_to_frame()`
- `add_pointcloud_to_frame()`
- `add_bounding_box_2d_loose_to_frame()` / `add_bounding_box_2d_tight_to_frame()`
- `add_bounding_box_3d_to_frame()`
- `add_semantic_segmentation_to_frame()`
- `add_instance_id_segmentation_to_frame()` / `add_instance_segmentation_to_frame()`
- `add_normals_to_frame()` / `add_motion_vectors_to_frame()`
- `add_occlusion_to_frame()`
- `attach_annotator()`, `detach_annotator()`

**Lens Distortion Models:**
- `get_lens_distortion_model()`
- `get_opencv_pinhole_properties()`, `get_opencv_fisheye_properties()`
- `get_ftheta_properties()`, `get_kannala_brandt_k3_properties()`
- `get_fisheye_polynomial_properties()`, `get_rad_tan_thin_prism_properties()`
- `get_lut_properties()`

**Calibration:**
- `get_intrinsics_matrix()`
- `get_view_matrix_ros()`

**State:**
- `initialize()`, `post_reset()`, `pause()`, `is_paused()`
- `get_default_state()`
- `is_valid()`, `get_visibility()`, `destroy()`
- `get_render_product_path()`

### CameraView
Handles tiled/batched data from multiple cameras.

---

## 9. isaacsim.sensors.physics — Contact / IMU / Effort

### ContactSensor (extends BaseSensor)
Configurable contact reporting with thresholds, radius, and sampling frequency.
- Pose getters/setters, threshold config, radius config, raw contact frame data.

### IMUSensor (extends BaseSensor)
Inertial Measurement Unit.
- Acceleration, angular velocity (gyro), magnetic field
- Real-time readings with timestamps, configurable measurement frequency

### EffortSensor (extends SingleArticulation)
Joint effort measurement.
- Joint position/velocity/effort readings
- Applied vs measured efforts
- Joint forces / torques across links
- Physics solver residuals
- Gravity enable/disable

### EsSensorReading (data class)
- `validity`, `timestamp`, `value`

### Commands
- `IsaacSensorCreatePrim`
- `IsaacSensorCreateContactSensor`
- `IsaacSensorCreateImuSensor`

---

## 10. isaacsim.sensors.rtx — RTX Lidar / Radar / IDS

### Commands
- `IsaacSensorCreateRtxLidar`
- `IsaacSensorCreateRtxIDS` (Idealized Depth Sensor)
- `IsaacSensorCreateRtxRadar`

### LidarRtx

**Data Management (per annotator):**
- `add_point_cloud_data_to_frame()` / `remove_point_cloud_data_to_frame()`
- `add_linear_depth_data_to_frame()` / `remove_linear_depth_data_to_frame()`
- `add_intensities_data_to_frame()` / `remove_intensities_data_to_frame()`
- `add_azimuth_range_to_frame()` / `remove_azimuth_range_to_frame()`
- `add_horizontal_resolution_to_frame()` / `remove_horizontal_resolution_to_frame()`

**Control:**
- `initialize()`, `pause()`, `resume()`, `post_reset()`
- `is_paused()`, `is_valid()`

**Configuration / State:**
- `get_current_frame()`, `get_render_product_path()`
- `set_world_pose()` / `get_world_pose()`
- `set_local_pose()` / `get_local_pose()`
- `set_default_state()` / `get_default_state()`

**Annotators / Writers:**
- `attach_annotator()`, `detach_annotator()`, `detach_all_annotators()`
- `attach_writer()`, `detach_writer()`, `detach_all_writers()`
- `get_annotators()`, `get_writers()`

**Visualization:**
- `enable_visualization()`, `disable_visualization()`

**Properties:** `prim_path`, `name`, `prim`, `non_root_articulation_link`

---

## 11. isaacsim.robot.manipulators

### SingleManipulator
High-level wrapper for an articulated manipulator with a single end-effector and optional gripper.
- `apply_action(control_action)`
- `get_joint_positions()` / `set_joint_positions(positions)`
- `get_joint_velocities()` / `set_joint_velocities(velocities)`
- `get_measured_joint_efforts()`, `set_joint_efforts(efforts)`
- `get_world_pose()`, `set_local_pose()`
- `enable_gravity()` / `disable_gravity()`
- `post_reset()`, `initialize()`

### Gripper (base)
### ParallelGripper
Two-finger gripper (e.g. Franka Panda gripper).
### SurfaceGripper
Suction-cup style gripper.

### Controllers
- **PickPlaceController** — pick-and-place state machine
- **StackingController** — stacking task controller

---

## 12. isaacsim.robot.wheeled_robots

### WheeledRobot
- `apply_action()`, `apply_wheel_actions()`
- `get_joint_positions()`, `get_joint_velocities()`
- `get_linear_velocity()`, `get_angular_velocity()`
- `enable_gravity()` / `disable_gravity()`
- `get_applied_action()`

### DifferentialController
Unicycle/differential-drive kinematic model.
- `forward(command)` — `command = [lin_vel, ang_vel]`
- `reset()`

### HolonomicController
Omnidirectional platforms.
- `forward(command)` — `command = [vx, vy, wz]`
- `build_base()`, `reset()`

### WheelBasePoseController
Drives toward a desired pose.
- `forward()`, `reset()`

### HolonomicRobotUsdSetup
Configures USD wheel attributes for holonomic robots.

### Utility Functions
- `quintic_polynomials_planner()` — path planning
- `stanley_control()` — Stanley steering
- `pid_control()` — PID speed control
- `QuinticPolynomial` — trajectory class

---

## 13. isaacsim.robot_motion.motion_generation

### WorldInterface
Obstacle world for motion planners.
- `add_capsule()`, `add_cone()`, `add_cuboid()`, `add_cylinder()`, `add_ground_plane()`
- `add_obstacle(obstacle)`, `remove_obstacle(obstacle)`
- `enable_obstacle(obstacle)`, `disable_obstacle(obstacle)`
- `update_world()`, `reset()`

### MotionPolicy (extends WorldInterface)
- `compute_joint_targets()` — position/velocity for next frame
- `set_cspace_target(target)` — c-space goal
- `set_end_effector_target(position, orientation)` — task-space goal
- `set_robot_base_pose(position, orientation)`
- `get_active_joints()`, `get_watched_joints()`

### RmpFlow (extends MotionPolicy)
Reactive motion policy with obstacle avoidance.
- `get_default_cspace_position_target()` (from YAML)
- `visualize_collision_spheres()`
- `delete_collision_sphere_prims()`, `delete_end_effector_prim()`

### Kinematics
- **KinematicsSolver** (interface)
- **LulaKinematicsSolver** — Lula-based IK/FK
- **ArticulationKinematicsSolver** — wrapper for simulated articulations

### Path Planning
- **PathPlanner** (interface)
- **RRT** — rapidly-exploring random tree

### Trajectories
- **Trajectory** (interface)
- **LulaTrajectory** — Lula trajectory
- **ArticulationTrajectory** — converts trajectory → discrete articulation actions
- **LulaCSpaceTrajectoryGenerator** — time-optimal c-space trajectories
- **LulaTaskSpaceTrajectoryGenerator** — task-space trajectories

### Bridging Classes
- **ArticulationMotionPolicy** — apply a motion policy to a simulated robot
- **MotionPolicyController** — controller implementation over any MotionPolicy

---

## 14. isaacsim.asset.importer.urdf — URDF Import

### Commands
- **URDFCreateImportConfig** — returns an `ImportConfig` object.
- **URDFParseFile** — parse URDF file → `UrdfRobot`.
- **URDFParseText** — parse URDF from text → `UrdfRobot`.
- **URDFParseAndImportFile** — parse + import in one step; returns USD stage path.
- **URDFImportRobot** — parse + import; returns base path or articulation root prim path.

### Urdf Class Methods
- `parse_urdf()` — parse URDF file into internal structures
- `parse_string_urdf()` — parse URDF string
- `import_robot()` — import parsed data onto the USD stage
- `get_kinematic_chain()` — kinematic chain (for display)
- `compute_natural_stiffness()` — joint stiffness from natural frequency

### ImportConfig (selected properties)
- `merge_fixed_joints`
- `convex_decomp`
- `import_inertia_tensor`
- `fix_base`
- `collision_from_visuals`
- `density`
- `distance_scale`
- `default_drive_type`
- `replace_cylinders_with_capsules`

---

## 15. isaacsim.storage.native — Nucleus / Assets

### Nucleus Server
- `build_server_list()`
- `check_server()` / `check_server_async()`
- `find_nucleus_server()`
- `get_assets_server()`
- `get_assets_root_path()` / `get_assets_root_path_async()`
- `get_isaac_asset_root_path()`
- `get_nvidia_asset_root_path()`
- `get_server_path()` / `get_server_path_async()`
- `get_url_root()`

### File / Folder Operations
- `create_folder()`, `delete_folder()`
- `is_dir()` / `is_dir_async()`
- `is_file()` / `is_file_async()`
- `list_folder()`, `recursive_list_folder()`

### Asset Management
- `get_full_asset_path()` / `get_full_asset_path_async()`
- `download_assets_async()` — from S3
- `verify_asset_root_path()`

---

## 16. isaacsim.replicator.writers

### DataVisualizationWriter
Visualizes annotator data incl. bounding boxes.
- Methods: `detach()`, `write()`
- Constants: `BB_2D_LOOSE`, `BB_2D_TIGHT`, `BB_3D`, `SUPPORTED_BACKGROUNDS`

### DOPEWriter
Built-in annotator ground truth for DOPE pose estimation.
- `is_last_frame_valid()`, `register_pose_annotator()`, `setup_writer()`, `write()`
- Properties: `output_dir`, `semantic_types`, `image_output_format`, `use_s3`

### PoseWriter
Pose estimation data with optional debug viz.
- `detach()`, `get_current_frame_id()`, `write()`
- Constants: annotation/keypoint order definitions

### PytorchListener
- `get_rgb_data()`, `write_data()` — retrieves data as PyTorch tensors

### PytorchWriter
- `write()` — batched PyTorch tensor output

### YCBVideoWriter
YCB Video Dataset format output.
- `is_last_frame_valid()`, `register_pose_annotator()`, `save_mesh_vertices()`, `setup_writer()`, `write()`
- Properties: `output_dir`, `num_frames`, `semantic_types`, `rgb`, `pose`, …

---

## Quick-Start Skeleton (Standalone App)

```python
from isaacsim.simulation_app import SimulationApp
simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
cube = world.scene.add(DynamicCuboid(
    prim_path="/World/cube",
    name="cube",
    position=[0, 0, 1.0],
    size=0.3,
    color=[1, 0, 0],
))

world.reset()
for _ in range(600):
    world.step(render=True)

simulation_app.close()
```

---

*Pulled from the official docs at `docs.isaacsim.omniverse.nvidia.com/5.1.0`. For complete signatures and edge cases, navigate to the per-extension pages under `py/source/extensions/<module>/docs/index.html`.*
