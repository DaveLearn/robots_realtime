# robots_realtime

A research codebase for real-time robot teleoperation, data collection, and policy deployment.

### Why robots_realtime?
- **Unified Pipeline:** Collect data in simulation or on real hardware platforms, and deploy learned policies with the same infrastructure.
- **Modular Stack:** Switch between IK gizmos, Franka Panda hardware and simulated arms via runtime YAML configs.
- **High Frequency:** Built with ZeroMQ nodes for asynchronous, low-latency real-time control.

<table>
<tr>
<td><img src="media/real_yams_rr.gif" width="360"></td>
<td><img src="media/franka_realtime2.gif" width="360"></td>
</tr>
<tr>
<td><img src="media/yam_active_leader_dagger.gif" width="360"></td>
<td><img src="media/rr_vr_support.gif" width="360"></td>
</tr>
</table>

## Other Documentation
[Architecture & recording format](docs/architecture.md) 

[Extending (new agents, robots, cameras)](docs/extending.md) 

[VR streaming MuJoCo sim to Quest](docs/vr_streaming.md)

---

## Installation

Environments are managed with [pixi](https://pixi.sh); the manifest lives in
`pyproject.toml` under `[tool.pixi.*]`. Conda packages cover the native
libraries (Python, OpenGL, a C toolchain), everything else resolves from PyPI.

The OpenGL that conda installs is libglvnd, the vendor-neutral *dispatch*
layer. It contains no driver code: at context creation it dlopens the machine's
own `libGLX_<vendor>.so.0`, which has to be present and to match the running
kernel driver.

MuJoCo then resolves GL entry points with glad, and the environment sets
`MUJOCO_GL=egl` so it does that through EGL rather than GLX. The GLX loader
fails on some machines in a way that is easy to misread: the context is created
fine and ordinary GL calls work, but glad reports `gladLoadGL error`, so the
viewer window appears and vanishes — and since that exits from C, nothing lands
in the node's Python log. EGL drives the same driver and the same on-screen
context, and it's the better backend for headless offscreen rendering anyway.
Override with `MUJOCO_GL=glfw` if a machine ever needs GLX.

To see how GL resolves here, and to compare backends in a subprocess the way a
session node runs one:

```bash
pixi run gl-check
pixi run gl-check --subprocess
```

```bash
git clone --recurse-submodules https://github.com/uynitsuj/robots_realtime.git
cd robots_realtime
# if already cloned, or some of the submodules are incompletely cloned, run
git submodule update --init --recursive
pixi install
```

| Environment | Contents |
| --- | --- |
| `default` | core stack + Franka Panda (`panda_py` / libfranka 0.10.0) + RealSense + sim |
| `sensors` | the above plus the ZED bindings (`pixi run -e sensors ...`; needs the ZED SDK 5.1 installed on the machine) |
| `mjlab` | core stack with `mjlab` for mjlab-backed sim |

Everything resolves cleanly with no dependency overrides, which is why the
openpi websocket client is vendored in-tree at `robots_realtime/policy_client`
rather than installed from PyPI — the published `openpi-client` wheel pins
`numpy<2`, and that pin is the only thing that held the whole stack (and the
ZED bindings, which need numpy≥2) back on numpy 1.26.

---

## Usage / Quickstart

### Run a Franka teleop session with Viser IK gizmos

```bash
pixi run rr-session configs/franka/franka_robotiq_viser_teleop.yaml
```

Upon running any of the configs, you should see the terminal populate with a rich TUI session:
```
╭─────────────────────────────── robots_realtime ────────────────────────────────╮
│   NODE                STATUS             HZ    TOPICS                          │
│   viser_left          ● live          255.8    joint_pos                       │
│   viser_right         ● live          255.8    joint_pos                       │
│   arm                 ● live           29.6    left_state, right_state         │
│ http://localhost:8765  (viser)  http://localhost:8012  (vr)                    │
│ ────────────────────────────────────────────────────────────────────────────── │
│ ○  idle                                      [r] record  [d] discard  [q] quit │
│ ────────────────────────────────────────────────────────────────────────────── │
│ [arm] ╭────── viser (listening *:8765) ───────╮                                │
│ [arm] │             ╷                         │                                │
│ [arm] │   HTTP      │ http://localhost:8765   │                                │
│ [arm] │   Websocket │ ws://localhost:8765     │                                │
│ [arm] │             ╵                         │                                │
│ [arm] ╰───────────────────────────────────────╯                                │
│   logs: /tmp/rr_logs_7hhz62am                                                  │
╰────────────────────────────────────────────────────────────────────────────────╯
```
Look under `/configs` for other existing configs

### Replay an episode

```bash
pixi run rr-replay recordings/20260323/episode_175805_0473b1bc/
```

Opens a Viser viewer at `http://localhost:8080`. For sim episodes you get two modes: **qpos** (exact, restores recorded state) and **physics** (re-simulates from actions). For real data, you get viser visualization of joint angles replayed on urdfs and other sensor streams (e.g. rgb).

# TODOS / Roadmap
* [ ] Test + verify policy deploy pipeline
* [ ] DAgger on-policy intervention data collection
