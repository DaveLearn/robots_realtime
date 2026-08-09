"""Vendored from openpi-client (Apache License 2.0) — see the package docstring.

Upstream slices the inference result with ``dm_tree.map_structure``; the local
``_map_structure`` below does the same over dicts/lists/tuples so we don't pull
in dm-tree for one call.
"""

from typing import Dict

import numpy as np

from robots_realtime.policy_client import base_policy as _base_policy


def _map_structure(fn, structure):
    """Apply ``fn`` to every leaf of a nested dict/list/tuple structure."""
    if isinstance(structure, dict):
        return {k: _map_structure(fn, v) for k, v in structure.items()}
    if isinstance(structure, (list, tuple)):
        mapped = [_map_structure(fn, v) for v in structure]
        return type(structure)(mapped) if not isinstance(structure, tuple) else tuple(mapped)
    return fn(structure)


class ActionChunkBroker(_base_policy.BasePolicy):
    """Wraps a policy to return action chunks one-at-a-time.

    Assumes that the first dimension of all action fields is the chunk size.

    A new inference call to the inner policy is only made when the current
    list of chunks is exhausted.
    """

    def __init__(self, policy: _base_policy.BasePolicy, action_horizon: int):
        self._policy = policy
        self._action_horizon = action_horizon
        self._cur_step: int = 0

        self._last_results: Dict[str, np.ndarray] | None = None

    def infer(self, obs: Dict) -> Dict:
        if self._last_results is None:
            self._last_results = self._policy.infer(obs)
            self._cur_step = 0

        def slicer(x):
            if isinstance(x, np.ndarray):
                return x[self._cur_step, ...]
            else:
                return x

        results = _map_structure(slicer, self._last_results)
        self._cur_step += 1

        if self._cur_step >= self._action_horizon:
            self._last_results = None

        return results

    def reset(self) -> None:
        self._policy.reset()
        self._last_results = None
        self._cur_step = 0
