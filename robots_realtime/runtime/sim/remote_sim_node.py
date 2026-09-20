"""RemoteSimNode — bridge a simulator process onto the bus over a lockstep RPC.

The simulator lives in its own process (and its own Python environment); this
node knows nothing about its physics. Each step() sends one request and the
sim advances exactly one policy step, so the node's poll rate is the sim clock.

RPC (ZMQ REQ/REP, msgpack + msgpack_numpy, same encoding as the bus):

    {"cmd": "info"}                    -> {"rate_hz", "action_dim", "cameras", ...}
    {"cmd": "reset"}                   -> {"state": {...}, "images": {cam: uint8[H,W,3]}}
    {"cmd": "step", "action": [...]}   -> same as reset

Published topics:
    {name}/state        the sim's state dict, verbatim (recorded to MCAP as JSON)
    {name}/{cam}_rgb    {"frame": ndarray} per camera (recorded to MP4)

Subscribed topics (from YAML):
    cmd_topic    {"joint_pos": [...]}  action for the next step; held if absent
    reset_topic  any message           re-randomize the scene

Session YAML example::

    - type: robots_realtime.runtime.sim.remote_sim_node:RemoteSimNode
      name: sim
      endpoint: tcp://127.0.0.1:5600
      launch_cmd: "pixi run --manifest-path ../../control_sufficient_physics/pyproject.toml sim-server"
      cmd_topic: teleop/joint_target
      reset_topic: teleop/reset
"""

from __future__ import annotations

import logging
import shlex
import subprocess
import threading
import time

import numpy as np
import zmq

from robots_realtime.runtime.node import Node, NodeRole
from robots_realtime.runtime.recording import AsyncMp4Writer
from robots_realtime.runtime.transport.serialization import pack, unpack

logger = logging.getLogger(__name__)


class RemoteSimNode(Node):
    role = NodeRole.ROBOT
    published_topics: list[str] = ["state"]

    def __init__(
        self,
        name: str = "sim",
        endpoint: str = "tcp://127.0.0.1:5600",
        launch_cmd: str | None = None,
        launch_cwd: str | None = None,
        cmd_topic: str | None = None,
        reset_topic: str | None = None,
        connect_timeout: float = 60.0,
        writer=None,
        **kwargs,
    ) -> None:
        self._endpoint = endpoint
        self._launch_cmd = launch_cmd
        self._launch_cwd = launch_cwd
        self._cmd_topic = cmd_topic
        self._reset_topic = reset_topic
        self._connect_timeout = connect_timeout
        self.subscribed_topics = [t for t in (cmd_topic, reset_topic) if t]
        super().__init__(name=name, writer=writer, **kwargs)

        self._proc: subprocess.Popen | None = None
        self._sock: zmq.Socket | None = None
        self._cameras: list[str] = []
        self._fps: float = 20.0
        self._action: np.ndarray | None = None
        self._last_reset_ts: float | None = None
        self._cam_writers: dict[str, AsyncMp4Writer] = {}
        self._save_dir: str = ""
        # start/stop_recording arrive on the control thread while step() runs on
        # the main thread; the lock keeps every state+frames write atomic with
        # respect to opening/closing the writers so the streams stay aligned.
        self._write_lock = threading.Lock()

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def setup(self) -> None:
        if self._launch_cmd:
            logger.info("[%s] launching sim: %s", self.name, self._launch_cmd)
            self._proc = subprocess.Popen(shlex.split(self._launch_cmd), cwd=self._launch_cwd)

        info = self._wait_for_sim()
        self._fps = float(info["rate_hz"])
        self.poll_freq = self._fps
        self._cameras = list(info.get("cameras", []))
        self.published_topics = ["state"] + [f"{c}_rgb" for c in self._cameras]
        logger.info("[%s] sim ready: %s", self.name, info)

        # Ignore any reset request that predates this node.
        if self._reset_topic:
            self._last_reset_ts = self.get_timestamp(self._reset_topic)
        self._absorb(self._rpc({"cmd": "reset"}))

    def step(self) -> None:
        if self._paused:
            return
        if self._reset_requested():
            obs = self._rpc({"cmd": "reset"})
        else:
            cmd = self.get_latest(self._cmd_topic) if self._cmd_topic else None
            if cmd is not None and cmd.get("joint_pos") is not None:
                self._action = np.asarray(cmd["joint_pos"], dtype=np.float64)
            obs = self._rpc({"cmd": "step", "action": self._action})
        self._absorb(obs)

    def cleanup(self) -> None:
        self.stop_recording()
        if self._sock is not None:
            self._sock.close(linger=0)
            self._sock = None
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None

    # ── Recording: state via the injected McapWriter, frames via MP4 writers ──

    def start_recording(self, save_dir: str) -> None:
        with self._write_lock:
            super().start_recording(save_dir)
            self._save_dir = save_dir
            self._cam_writers = {}
            for cam in self._cameras:
                w = AsyncMp4Writer(fps=self._fps)
                w.open(save_dir, f"{self.name}-{cam}")
                self._cam_writers[cam] = w

    def stop_recording(self) -> str:
        with self._write_lock:
            super().stop_recording()
            for w in self._cam_writers.values():
                if w.is_open:
                    w.close()
            self._cam_writers = {}
        return self._save_dir

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _absorb(self, obs: dict) -> None:
        """Publish one observation; the action the sim consumed becomes the held action."""
        ts = time.time()
        state = obs["state"]
        self._action = np.asarray(state["applied_action"], dtype=np.float64)
        with self._write_lock:
            self.publish("state", state, ts=ts)
            for cam, frame in obs.get("images", {}).items():
                self.publish(f"{cam}_rgb", {"frame": frame}, ts=ts, record=False)
                w = self._cam_writers.get(cam)
                if w is not None and w.is_open:
                    w.write("rgb", ts, {"frame": frame})

    def _reset_requested(self) -> bool:
        if not self._reset_topic:
            return False
        ts = self.get_timestamp(self._reset_topic)
        if ts is None or ts == self._last_reset_ts:
            return False
        self._last_reset_ts = ts
        return True

    def _connect(self) -> None:
        if self._sock is not None:
            self._sock.close(linger=0)
        self._sock = zmq.Context.instance().socket(zmq.REQ)
        self._sock.connect(self._endpoint)

    def _rpc(self, req: dict, timeout_s: float = 5.0) -> dict:
        assert self._sock is not None
        self._sock.send(pack(req))
        if not self._sock.poll(int(timeout_s * 1000)):
            self._connect()  # a REQ socket cannot resend after a lost reply
            raise TimeoutError(f"[{self.name}] no reply from sim at {self._endpoint}")
        reply = unpack(self._sock.recv())
        if "error" in reply:
            raise RuntimeError(f"[{self.name}] sim error: {reply['error']}")
        return reply

    def _wait_for_sim(self) -> dict:
        self._connect()
        deadline = time.monotonic() + self._connect_timeout
        while True:
            try:
                return self._rpc({"cmd": "info"}, timeout_s=1.0)
            except TimeoutError:
                if self._proc is not None and self._proc.poll() is not None:
                    raise RuntimeError(f"[{self.name}] sim process exited with {self._proc.returncode}")
                if time.monotonic() > deadline:
                    raise

    @classmethod
    def build_kwargs(cls, params: dict) -> dict:
        return {
            "name": params["name"],
            "endpoint": params.get("endpoint", "tcp://127.0.0.1:5600"),
            "launch_cmd": params.get("launch_cmd"),
            "launch_cwd": params.get("launch_cwd"),
            "cmd_topic": params.get("cmd_topic"),
            "reset_topic": params.get("reset_topic"),
            "connect_timeout": params.get("connect_timeout", 60.0),
        }
