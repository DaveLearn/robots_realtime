"""Robot protocol and joint-space remapping helpers.

Vendored from i2rt (MIT License, Copyright (c) I2RT Robotics) so that the
non-YAM parts of this package do not depend on the i2rt SDK:
https://github.com/i2rt-robotics/i2rt
"""

import enum
from abc import abstractmethod
from typing import Any, Dict, Protocol, Tuple, Union, runtime_checkable

import numpy as np
from dm_env.specs import Array

ActionSpec = Union[Array, Dict[str, "ActionSpec"]]
"""Action specification for the agent/robot. It also includes the action space for the gripper."""


@enum.unique
class RobotType(enum.Enum):
    ARM = "arm"
    MOBILE_BASE = "mobile_base"


@runtime_checkable
class Robot(Protocol):
    """A generic Robot protocol."""

    @abstractmethod
    def num_dofs(self) -> int:
        """Get the number of controllable degrees of freedom of the robot.

        Returns:
            int: The number of controllable degrees of freedom of the robot.
        """
        raise NotImplementedError

    def get_joint_pos(self) -> np.ndarray:
        """Get the current joint positions of the robot in radians.

        Returns:
            np.ndarray: The current joint positions of the robot in radians.
        """
        pass

    def get_joint_state(self) -> Dict[str, np.ndarray]:
        """Get the current joint positions and velocities of the robot in radians.

        Returns:
            Dict[str, np.ndarray]: A dictionary containing the current joint positions and velocities of the robot in radians.
        """
        pass

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        """Command the leader robot to a given state.

        Args:
            joint_pos (np.ndarray): The state to command the leader robot to.
        """
        pass

    def command_target_vel(self, joint_vel: np.ndarray) -> None:
        """Command the leader robot to a given state.

        Args:
            joint_vel (np.ndarray): The state to command the leader robot to.
        """
        pass

    def command_joint_state(self, joint_state: Dict[str, np.ndarray]) -> None:
        """Command the leader robot to a given state.

        Args:
            joint_state (Dict[str, np.ndarray]): The state to command the leader robot to.
        """
        pass

    @abstractmethod
    def get_observations(self) -> Dict[str, np.ndarray]:
        """Get the current observations of the robot.

        This is to extract all the information that is available from the robot,
        such as joint positions, joint velocities, etc. This may also include
        information from additional sensors, such as cameras, force sensors, etc.

        Returns:
            Dict[str, np.ndarray]: A dictionary of observations.
        """
        raise NotImplementedError

    def joint_pos_spec(self) -> ActionSpec:
        """Return the action specification for the robot, which includes the gripper."""
        return Array(
            shape=(self.num_dofs(),),
            dtype=np.float32,
        )

    def joint_state_spec(self) -> ActionSpec:
        """Return the action specification for the robot, which includes the gripper."""
        return dict(
            {
                "pos": Array(
                    shape=(self.num_dofs(),),
                    dtype=np.float32,
                ),
                "vel": Array(
                    shape=(self.num_dofs(),),
                    dtype=np.float32,
                ),
            }
        )

    def get_robot_info(self) -> Dict[str, Any]:
        """Get the robot information, such as kp, kd, joint limits, gripper limits, etc."""
        return {}

    def get_robot_type(self) -> RobotType:
        """Get the robot type."""
        return RobotType.ARM

    def reinit(self) -> None:
        """Reinitialize the robot."""
        pass

    def close(self) -> None:
        """Close the robot."""
        pass


class JointMapper:
    """Maps joint positions between the normalized command space and robot joint space."""

    def __init__(self, index_range_map: Dict[int, Tuple[float, float]], total_dofs: int):
        """Args:
        index_range_map (Dict[int, Tuple[float, float]]): 0 indexed
        total_dofs (int): num of joints in the robot including the gripper if the gripper is the second robot
        """
        self.empty = len(index_range_map) == 0
        if not self.empty:
            self.joints_one_hot = np.zeros(total_dofs).astype(bool)
            self.joint_limits = []
            for idx, (start, end) in index_range_map.items():
                self.joints_one_hot[idx] = True
                self.joint_limits.append((start, end))
            self.joint_limits = np.array(self.joint_limits)
            self.joint_range = self.joint_limits[:, 1] - self.joint_limits[:, 0]

    def to_robot_joint_pos_space(self, command_joint_pos: np.ndarray) -> np.ndarray:
        if self.empty:
            return command_joint_pos
        command_joint_pos = np.asarray(command_joint_pos, order="C")
        result = command_joint_pos.copy()
        needs_remapping = command_joint_pos[self.joints_one_hot]
        needs_remapping = needs_remapping * self.joint_range + self.joint_limits[:, 0]
        result[self.joints_one_hot] = needs_remapping
        return result

    def to_robot_joint_vel_space(self, command_joint_vel: np.ndarray) -> np.ndarray:
        if self.empty:
            return command_joint_vel
        result = command_joint_vel.copy()
        needs_remapping = command_joint_vel[self.joints_one_hot]
        needs_remapping = needs_remapping * self.joint_range
        result[self.joints_one_hot] = needs_remapping
        return result

    def to_command_joint_vel_space(self, robot_joint_vel: np.ndarray) -> np.ndarray:
        if self.empty:
            return robot_joint_vel
        result = robot_joint_vel.copy()
        needs_remapping = robot_joint_vel[self.joints_one_hot]
        needs_remapping = needs_remapping / self.joint_range
        result[self.joints_one_hot] = needs_remapping
        return result

    def to_command_joint_pos_space(self, robot_joint_pos: np.ndarray) -> np.ndarray:
        if self.empty:
            return robot_joint_pos
        result = robot_joint_pos.copy()
        needs_remapping = robot_joint_pos[self.joints_one_hot]
        needs_remapping = (needs_remapping - self.joint_limits[:, 0]) / self.joint_range
        result[self.joints_one_hot] = needs_remapping
        return result
