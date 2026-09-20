"""FpsTeleopNode — first-person-shooter style mouse+keyboard teleop for a Panda.

The gripper always points at the floor; the operator drives a 4-DOF target
(x, y, z, yaw) in the hand frame from a pygame window that shows the wrist
and third-person cameras. PyRoKi IK turns the target into joint positions.

Controls (window focused, cursor captured):
    mouse        translate in the hand's x/y plane (image up = hand +x)
    W / S        along the gripper axis: W = toward the fingertips (down)
    Q / E        yaw about the gripper axis
    left click   toggle gripper open/closed
    R            start / stop recording an episode
    N            new scene (sim reset) — only while not recording
    Esc          release / capture the cursor

Published topics:
    {name}/joint_target  {"joint_pos": [q0..q6, gripper]}  gripper 1 = open, 0 = closed
    {name}/reset         {}                                 request a scene reset
    {name}/record        {"record": bool}                   the session's record_topic

Session YAML example::

    - type: robots_realtime.agents.teleoperation.fps_teleop_node:FpsTeleopNode
      name: teleop
      state_topic: sim/state
      image_topics: [sim/wrist_rgb, sim/third_person_rgb]
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from robots_realtime.runtime.node import Node, NodeRole

logger = logging.getLogger(__name__)

PANDA_HOME_Q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.02])
TARGET_LINK = "panda_hand"
# 180° about x: hand z points at the floor and hand x is world x, so at yaw 0
# the wrist image's "up" is the robot's forward direction.
HAND_DOWN_WXYZ = np.array([0.0, 1.0, 0.0, 0.0])


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


@dataclass
class FpsIntegrator:
    """Integrates operator inputs into a hand-frame pose target. Pure numpy."""

    home_pos: np.ndarray
    workspace_lo: np.ndarray
    workspace_hi: np.ndarray
    mouse_gain: float = 0.0005  # metres per mouse count
    z_speed: float = 0.15       # m/s while W/S held
    yaw_speed: float = 1.2      # rad/s while Q/E held

    pos: np.ndarray = field(init=False)
    yaw: float = field(init=False, default=0.0)
    grip_open: bool = field(init=False, default=True)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.pos = self.home_pos.copy()
        self.yaw = 0.0
        self.grip_open = True

    @property
    def orientation_wxyz(self) -> np.ndarray:
        h = self.yaw / 2.0
        return _quat_mul(np.array([np.cos(h), 0.0, 0.0, np.sin(h)]), HAND_DOWN_WXYZ)

    def update(self, mouse_dx: float, mouse_dy: float, along_axis: float, yaw_rate: float, dt: float) -> None:
        """mouse_*: counts this tick; along_axis, yaw_rate: in [-1, 1]."""
        delta_hand = np.array([
            -mouse_dy * self.mouse_gain,   # image up  = hand +x
            mouse_dx * self.mouse_gain,    # image right = hand +y
            along_axis * self.z_speed * dt,  # hand +z = toward the fingertips
        ])
        self.pos = np.clip(
            self.pos + _quat_to_mat(self.orientation_wxyz) @ delta_hand,
            self.workspace_lo,
            self.workspace_hi,
        )
        self.yaw += yaw_rate * self.yaw_speed * dt


class FpsTeleopNode(Node):
    role = NodeRole.CONTROLLER
    published_topics: list[str] = ["joint_target", "reset", "record"]

    def __init__(
        self,
        name: str = "teleop",
        state_topic: str | None = None,
        image_topics: list[str] | None = None,
        rate_hz: float = 20.0,
        panel_size: tuple[int, int] = (480, 360),
        mouse_gain: float = 0.0005,
        z_speed: float = 0.15,
        yaw_speed: float = 1.2,
        workspace_lo: tuple[float, float, float] = (0.25, -0.35, 0.115),
        workspace_hi: tuple[float, float, float] = (0.70, 0.35, 0.65),
        writer=None,
        **kwargs,
    ) -> None:
        self._state_topic = state_topic
        self._image_topics = list(image_topics or [])
        self.subscribed_topics = [t for t in [state_topic, *self._image_topics] if t]
        self.poll_freq = rate_hz
        super().__init__(name=name, writer=writer, **kwargs)

        self._dt = 1.0 / rate_hz
        self._panel_size = tuple(panel_size)
        self._gains = dict(mouse_gain=mouse_gain, z_speed=z_speed, yaw_speed=yaw_speed)
        self._workspace = (np.asarray(workspace_lo, float), np.asarray(workspace_hi, float))

        # Created in setup() (pygame, jax and the URDF are not picklable).
        self._pg = None
        self._screen = None
        self._font = None
        self._robot = None
        self._integrator: FpsIntegrator | None = None
        self._cfg = PANDA_HOME_Q.copy()
        self._want_record = False
        self._grabbed = True

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def setup(self) -> None:
        import pygame
        import pyroki as pk
        from robot_descriptions.loaders.yourdfpy import load_robot_description

        self._robot = pk.Robot.from_urdf(load_robot_description("panda_description"))
        fk = np.asarray(self._robot.forward_kinematics(PANDA_HOME_Q))
        home_pos = fk[self._robot.links.names.index(TARGET_LINK)][4:7]
        self._integrator = FpsIntegrator(home_pos, *self._workspace, **self._gains)
        self._solve_ik()  # JIT warm-up so the first teleop tick doesn't stall

        self._pg = pygame
        pygame.init()
        w, h = self._panel_size
        self._screen = pygame.display.set_mode((w * max(1, len(self._image_topics)), h + 72))
        pygame.display.set_caption(f"{self.name} — FPS teleop")
        self._font = pygame.font.SysFont("monospace", 15)
        self._set_grab(True)

    def step(self) -> None:
        pg = self._pg
        assert pg is not None and self._integrator is not None
        for event in pg.event.get():
            if event.type == pg.KEYDOWN:
                self._on_key(event.key)
            elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1 and self._grabbed:
                self._integrator.grip_open = not self._integrator.grip_open

        keys = pg.key.get_pressed()
        dx, dy = pg.mouse.get_rel() if self._grabbed else (0, 0)
        self._integrator.update(
            dx, dy,
            along_axis=float(keys[pg.K_w]) - float(keys[pg.K_s]),
            yaw_rate=float(keys[pg.K_q]) - float(keys[pg.K_e]),
            dt=self._dt,
        )

        target = np.concatenate([self._solve_ik()[:7], [1.0 if self._integrator.grip_open else 0.0]])
        ts = time.time()
        self.publish("joint_target", {"joint_pos": target}, ts=ts)
        self.publish("record", {"record": self._want_record}, ts=ts, record=False)
        self._draw()

    def cleanup(self) -> None:
        if self._pg is not None:
            self._pg.quit()

    # ── Input ──────────────────────────────────────────────────────────────────

    def _on_key(self, key: int) -> None:
        pg = self._pg
        if key == pg.K_ESCAPE:
            self._set_grab(not self._grabbed)
        elif key == pg.K_r:
            self._want_record = not self._want_record
        elif key == pg.K_n:
            if self._recording or self._want_record:
                logger.warning("[%s] stop recording before resetting the scene", self.name)
                return
            self.publish("reset", {}, ts=time.time(), record=False)
            self._integrator.reset()
            self._cfg = PANDA_HOME_Q.copy()

    def _set_grab(self, grab: bool) -> None:
        self._grabbed = grab
        self._pg.event.set_grab(grab)
        self._pg.mouse.set_visible(not grab)
        self._pg.mouse.get_rel()  # discard the jump from re-centering the cursor

    # ── IK ─────────────────────────────────────────────────────────────────────

    def _solve_ik(self) -> np.ndarray:
        from robots_realtime.robots.inverse_kinematics.pyroki_snippets._solve_ik_vel_cost import solve_ik_vel_cost

        self._cfg = solve_ik_vel_cost(
            robot=self._robot,
            target_link_name=TARGET_LINK,
            target_wxyz=self._integrator.orientation_wxyz,
            target_position=self._integrator.pos,
            prev_cfg=self._cfg,
        )
        return self._cfg

    # ── Display ────────────────────────────────────────────────────────────────

    def _draw(self) -> None:
        pg = self._pg
        w, h = self._panel_size
        self._screen.fill((20, 20, 20))
        for i, topic in enumerate(self._image_topics):
            msg = self.get_latest(topic)
            frame = msg.get("frame") if msg else None
            if frame is None:
                continue
            surf = pg.surfarray.make_surface(np.ascontiguousarray(np.swapaxes(frame, 0, 1)))
            self._screen.blit(pg.transform.smoothscale(surf, (w, h)), (i * w, 0))

        it = self._integrator
        rec = "REC" if self._recording else ("rec pending" if self._want_record else "idle")
        lines = [
            f"[{rec}]  grip {'open' if it.grip_open else 'CLOSED'}  "
            f"pos {it.pos[0]:+.3f} {it.pos[1]:+.3f} {it.pos[2]:+.3f}  yaw {np.degrees(it.yaw):+.0f} deg"
            f"{'' if self._grabbed else '   [cursor released - Esc to capture]'}",
            "mouse: xy   W/S: down/up   Q/E: yaw   click: gripper   R: record   N: new scene   Esc: cursor",
        ]
        for j, text in enumerate(lines):
            color = (255, 80, 80) if (j == 0 and self._recording) else (220, 220, 220)
            self._screen.blit(self._font.render(text, True, color), (8, h + 10 + 24 * j))
        pg.display.flip()

    @classmethod
    def build_kwargs(cls, params: dict) -> dict:
        kw = {"name": params["name"]}
        for key in (
            "state_topic", "image_topics", "rate_hz", "panel_size",
            "mouse_gain", "z_speed", "yaw_speed", "workspace_lo", "workspace_hi",
        ):
            if key in params:
                kw[key] = params[key]
        return kw
