# CLAUDE.md — Sim2 (Paddy Field Robot Simulation)

## Project

Isaac Sim simulation of a 3-wheeled paddy field robot.

**Drive layout:**
- **Front 2 wheels** — powered, independent speed control (differential-drive style: left/right wheel speeds set forward motion and yaw).
- **Rear wheel** — unpowered, free-spinning around its wheel axle.
- **Rear link (castor/steering link)** — rotates about a vertical axis to steer the rear wheel and thus influence overall robot heading / turning geometry.

Robot assets and URDF live in `Sim_Robot_V2/` (urdf, meshes, config, launch, CMakeLists.txt, package.xml).

Scene: `Paddy_Sim_V2.usd`. Ground generation: `Paddy_Ground_Creation.py`, `Groundmesh.py`. Main scripts: `Setup_3.py`, `Controller_1.py`, `final.py`, `Check.py`, `Fix.py`, `diagnose.py`.

## Isaac Sim Version

**Isaac Sim 5.1.0.** Before writing or modifying any Isaac Sim code, read `IsaacSim_5.1.0_API_Reference.md` in this directory to confirm class names, module paths, and method signatures for 5.1.0. Prefer APIs listed there over older `omni.isaac.*` equivalents.

## Coding Conventions for This Project

- Use the **new** `isaacsim.*` module namespace (not the deprecated `omni.isaac.*`), matching the reference file.
- Standalone scripts start with `SimulationApp` before any other Isaac Sim / omni imports.
- Prefer multi-instance prim classes (`Articulation`, `RigidPrim`, `XFormPrim`) over the legacy `Single*` variants.
- For the robot's drivetrain, model the front wheels via `ArticulationController` joint velocity targets; the rear wheel and rear steering link are passive / articulated joints — do **not** command them directly unless explicitly controlling the rear-link steering angle.
- When writing controllers that look like a differential drive, reuse `isaacsim.robot.wheeled_robots.DifferentialController` only if the rear castor is treated as a passive follower; otherwise write a custom controller that sets left/right front wheel speeds and (optionally) a rear-link target angle.

## Workflow Expectations

- **Always** consult `IsaacSim_5.1.0_API_Reference.md` before writing Isaac Sim code. If the API you need isn't listed, fetch the corresponding page under `https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/<module>/docs/index.html` and update the reference file.
- Don't introduce new abstractions or refactor surrounding code when fixing a specific issue.
- Don't add speculative error handling — only validate at real boundaries (user input, file I/O, external configs).
