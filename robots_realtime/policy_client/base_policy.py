"""Vendored from openpi-client (Apache License 2.0) — see the package docstring."""

import abc
from typing import Dict


class BasePolicy(abc.ABC):
    @abc.abstractmethod
    def infer(self, obs: Dict) -> Dict:
        """Infer actions from observations."""

    def reset(self) -> None:
        """Reset the policy to its initial state."""
        pass
