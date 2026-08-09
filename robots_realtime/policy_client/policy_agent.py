"""Vendored from openpi-client (Apache License 2.0) — see the package docstring."""

import abc

from robots_realtime.policy_client import base_policy as _base_policy


class Agent(abc.ABC):
    """An Agent is the thing with agency, i.e. the entity that makes decisions.

    Agents receive observations about the state of the world, and return actions
    to take in response.
    """

    @abc.abstractmethod
    def get_action(self, observation: dict) -> dict:
        """Query the agent for the next action."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset the agent to its initial state."""


class PolicyAgent(Agent):
    """An agent that uses a policy to determine actions."""

    def __init__(self, policy: _base_policy.BasePolicy) -> None:
        self._policy = policy

    def get_action(self, observation: dict) -> dict:
        return self._policy.infer(observation)

    def reset(self) -> None:
        self._policy.reset()
