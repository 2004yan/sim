# Using `controller.py`

Quick guide to driving the paddy robot's joints from Python.

`controller.py` serves two purposes:

- **As a script** — run it directly to send a one-shot set of default commands (edit the constants at the top of the file — `LEFT_SPEED`, `RIGHT_SPEED`, `REAR_LINK_ANGLE`, `REAR_WHEEL_SPEED` — then execute it). The `if __name__ == "__main__"` block creates a `PaddyRobotController` and applies those values.
- **As a library** — import `PaddyRobotController` from it to get the movement-control functions (`set_wheel_speeds`, `set_steering_angle`, `set_rear_wheel_speed`, `drive`, `stop`) in your own scripts, loops, or higher-level controllers.

## Setup

Before using the controller:
1. Open `Paddy_Sim_V2.usd`
2. Run `ground_setup.py` once
3. Run `robot_setup.py` once
4. Press **Play**
5. Now you can use `controller.py`

## Import and create

```python
from controller import PaddyRobotController
import numpy as np

bot = PaddyRobotController()   # create once, after Play
```

## The functions

| Function | What it does |
|---|---|
| `bot.set_wheel_speeds(left, right)` | Front wheel velocities, rad/s |
| `bot.set_steering_angle(angle)` | Rear-link angle, rad (±π/2) |
| `bot.set_rear_wheel_speed(v)` | Rear wheel velocity (keep at 0) |
| `bot.drive(speed, steer=0)` | Both front wheels same speed + steering |
| `bot.stop()` | Zero all wheel speeds |

## Examples

**Drive straight forward:**
```python
bot.set_wheel_speeds(10.0, 10.0)
```

**Turn right while moving:**
```python
bot.set_wheel_speeds(10.0, 10.0)
bot.set_steering_angle(np.radians(30))
```

**Spin in place (differential):**
```python
bot.set_wheel_speeds(-5.0, 5.0)
```

**Stop:**
```python
bot.stop()
```

## Reading robot state

```python
pos, quat = bot.robot.get_world_pose()         # base pose
lin_v  = bot.robot.get_linear_velocity()       # world linear velocity
q      = bot.robot.get_joint_positions()       # all joint angles
qdot   = bot.robot.get_joint_velocities()      # all joint speeds

current_steer = q[bot.rear_link_idx]
```

## Simple path follower example

```python
import numpy as np
from controller import PaddyRobotController

WAYPOINTS = [(2.0, 0.0), (4.0, 2.0), (2.0, 4.0), (0.0, 2.0)]
FORWARD   = 8.0      # rad/s
TOL       = 0.2      # m

def yaw_from_quat(q):
    w, x, y, z = q
    return np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

bot = PaddyRobotController()
idx = 0

while simulation_app.is_running():
    pos, quat = bot.robot.get_world_pose()
    tx, ty = WAYPOINTS[idx]

    # advance to next waypoint when close
    if np.hypot(tx - pos[0], ty - pos[1]) < TOL:
        idx += 1
        if idx >= len(WAYPOINTS):
            bot.stop()
            break
        continue

    # heading error -> steering angle
    yaw     = yaw_from_quat(quat)
    bearing = np.arctan2(ty - pos[1], tx - pos[0])
    steer   = np.clip(bearing - yaw, -np.pi/2, np.pi/2)

    bot.set_wheel_speeds(FORWARD, FORWARD)
    bot.set_steering_angle(steer)

    world.step(render=True)
```

That's it — the robot will drive through the waypoints in order and stop at the last one.

## Rules of thumb

- Create `PaddyRobotController()` **once**, not every frame.
- Wheel commands are **rad/s**, not m/s. Convert with `omega = v / wheel_radius`.
- Steering is a **position target**, not a rate. Clamped to ±90°.
- Rear wheel is passive — leave its speed at 0.
