# Extending robots_realtime

## Adding a new agent

Implement the `Agent` protocol — just `act(obs: dict) -> dict`:

```python
# robots_realtime/agents/my_agent.py
class MyAgent:
    def reset(self) -> None: ...
    def act(self, obs: dict) -> dict:
        # obs contains whatever state_topics / image_topics you subscribed to
        return {"pos": joint_positions}          # single arm
        # or {"left": {"pos": ...}, "right": {"pos": ...}}  # multi-arm
```

Reference it in YAML — no node code needed:

```yaml
- type: AgentNode
  name: my_agent
  agent_class: robots_realtime.agents.my_agent:MyAgent
  agent_kwargs:
    checkpoint: /path/to/weights.pt
  loop_mode: subscriber_driven
  state_topics:
    left: yam_left/joint_state
    right: yam_right/joint_state
  image_topics:
    top: camera_top/rgb
```

## Adding a new robot

Implement two methods:

```python
class MyRobot:
    def command_joint_pos(self, joint_pos: np.ndarray) -> None: ...
    def get_observations(self) -> dict: ...  # must contain "joint_pos"
```

## Adding a new camera

Implement `read() -> CameraData` from `robots_realtime.sensors.cameras.camera`.

## Session config reference

```yaml
version: "1"

session:
  save_root: recordings
  record_topic: teleop_left/record  # bus topic that triggers record start/stop
  auto_record_duration: 10.0        # auto-record for N seconds then exit

nodes:
  - type: AgentNode
    name: teleop_left
    agent_class: robots_realtime.agents.teleoperation.franka_pyroki_viser_agent:FrankaPyrokiViserAgent
    agent_kwargs:
      robotiq_gripper: true
    arm_key: left
    loop_mode: flat_out

  - type: RobotNode
    name: franka_left
    robot_config: robot_configs/franka/franka_default_panda_gripper.yaml
    cmd_topic: teleop_left/joint_pos

  - type: CameraNode
    name: camera_top
    fps: 30
```
