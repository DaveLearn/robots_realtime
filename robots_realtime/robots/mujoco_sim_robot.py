"""MuJoCo simulation robots implementing the Robot protocol.

Wraps a MuJoCo model and provides forward-kinematics visualization
driven by joint position commands. Optionally launches a passive viewer.
"""

import logging
import os
from typing import Dict, Optional

import mujoco
import mujoco.viewer
import numpy as np

from robots_realtime.robots.protocol import Robot

logger = logging.getLogger(__name__)


class MujocoSimRobot(Robot):
    """A simulated robot backed by a MuJoCo model.

    Accepts joint-position commands (in radians), updates the model state
    via ``mj_kinematics``, and optionally renders in a passive MuJoCo
    viewer window.

    Args:
        xml_path: Path to the MuJoCo XML model file.
        render: Whether to launch a passive viewer window.
        gripper_index: If set, the last DOF in ``command_joint_pos``
            is treated as a virtual gripper value (not part of the
            MuJoCo model's qpos).
    """

    def __init__(
        self,
        xml_path: str,
        render: bool = True,
        gripper_index: Optional[int] = None,
    ) -> None:
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self._gripper_index = gripper_index
        self._gripper_pos = np.array([0.0])

        # Total DOFs exposed to the control system
        self._nq = self.model.nq
        self._num_dofs = self._nq + (1 if gripper_index is not None else 0)

        # Optionally launch viewer. MuJoCo's viewer calls exit() from C if GLFW
        # can't initialize, which kills the whole node process and leaves the
        # session running with a robot that never publishes state — so check for
        # a display up front rather than letting it fail. Same headless test the
        # ViserMonitorNode uses for browser auto-open.
        self.viewer = None
        if render and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            logger.warning(
                "No DISPLAY/WAYLAND_DISPLAY detected — running the MuJoCo sim headless. "
                "The robot still runs and publishes state; use the Viser view to see it."
            )
            render = False
        if render:
            self.viewer = mujoco.viewer.launch_passive(
                model=self.model,
                data=self.data,
                show_left_ui=False,
                show_right_ui=False,
            )
            mujoco.mjv_defaultFreeCamera(self.model, self.viewer.cam)

    # ------------------------------------------------------------------ #
    # Robot protocol
    # ------------------------------------------------------------------ #

    def num_dofs(self) -> int:
        return self._num_dofs

    def get_joint_pos(self) -> np.ndarray:
        qpos = self.data.qpos[: self._nq].copy()
        if self._gripper_index is not None:
            return np.concatenate([qpos, self._gripper_pos])
        return qpos

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        """Set the sim model's qpos and update kinematics + viewer.

        Args:
            joint_pos: Joint positions in radians.  If ``gripper_index``
                was provided, the last element is the gripper value and
                the preceding elements map to MuJoCo joints.
        """
        self.data.qpos[: self._nq] = joint_pos[: self._nq]
        if self._gripper_index is not None and len(joint_pos) > self._nq:
            self._gripper_pos[0] = joint_pos[self._nq]

        mujoco.mj_kinematics(self.model, self.data)

        if self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()

    def get_observations(self) -> Dict[str, np.ndarray]:
        obs: Dict[str, np.ndarray] = {
            "joint_pos": self.data.qpos[: self._nq].copy(),
            "joint_vel": self.data.qvel[: self._nq].copy(),
        }
        if self._gripper_index is not None:
            obs["gripper_pos"] = self._gripper_pos.copy()
        return obs

    # ------------------------------------------------------------------ #
    # Viewer helpers
    # ------------------------------------------------------------------ #

    def is_viewer_running(self) -> bool:
        """Return True if the viewer window is still open."""
        if self.viewer is None:
            return True  # headless mode — never "closes"
        return self.viewer.is_running()

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()


class PandaMujocoSimRobot(MujocoSimRobot):
    """A Franka Panda in MuJoCo, driven by the 8-vector the Franka agents send.

    The MuJoCo Menagerie Panda has 9 position DOFs — 7 arm joints plus two
    independently actuated fingers — while ``FrankaPyrokiViserAgent`` (and the
    ``panda_description`` URDF the Viser overlay renders) speaks 8: the arm
    joints plus one gripper value. This maps between the two by driving both
    fingers from that single value, so the sim can stand in for the real arm in
    any session config without the agent knowing the difference.

    Args:
        xml_path: MJCF to load. Defaults to the Menagerie Panda that
            ``robot_descriptions`` downloads and caches.
        render: Launch a passive MuJoCo viewer window. Needs a display —
            set false when running headless.
    """

    _N_ARM_JOINTS = 7
    _FINGER_JOINTS = ("finger_joint1", "finger_joint2")

    def __init__(self, xml_path: Optional[str] = None, render: bool = True) -> None:
        if xml_path is None:
            from robot_descriptions import panda_mj_description  # noqa: PLC0415

            xml_path = panda_mj_description.MJCF_PATH
        super().__init__(xml_path=xml_path, render=render, gripper_index=None)

        self._arm_qpos_adr = np.array(
            [self.model.jnt_qposadr[i] for i in range(self._N_ARM_JOINTS)], dtype=int
        )
        self._finger_qpos_adr = []
        for name in self._FINGER_JOINTS:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"{xml_path} has no joint {name!r}; is this the Menagerie Panda?")
            self._finger_qpos_adr.append(self.model.jnt_qposadr[jid])
            self._finger_range = self.model.jnt_range[jid]

    def num_dofs(self) -> int:
        return self._N_ARM_JOINTS + 1

    def get_joint_pos(self) -> np.ndarray:
        return np.concatenate(
            [self.data.qpos[self._arm_qpos_adr], self.data.qpos[self._finger_qpos_adr[:1]]]
        )

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        joint_pos = np.asarray(joint_pos, dtype=np.float64)
        if joint_pos.shape != (self.num_dofs(),):
            raise ValueError(f"expected {self.num_dofs()} joint values, got {joint_pos.shape}")
        self.data.qpos[self._arm_qpos_adr] = joint_pos[: self._N_ARM_JOINTS]
        # One commanded width drives both fingers; the agent's gripper slider is
        # not clamped to the model's finger travel, so clip rather than let
        # MuJoCo render an impossible pose.
        width = float(np.clip(joint_pos[-1], self._finger_range[0], self._finger_range[1]))
        for adr in self._finger_qpos_adr:
            self.data.qpos[adr] = width

        mujoco.mj_kinematics(self.model, self.data)
        if self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()

    def get_observations(self) -> Dict[str, np.ndarray]:
        return {
            "joint_pos": self.get_joint_pos(),
            "joint_vel": np.concatenate(
                [self.data.qvel[: self._N_ARM_JOINTS], np.zeros(1)]
            ),
        }
