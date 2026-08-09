"""Client half of the openpi websocket policy protocol.

Vendored from openpi-client (Apache License 2.0, Copyright Physical Intelligence):
https://github.com/Physical-Intelligence/openpi/tree/main/packages/openpi-client

We carry these ~250 lines in-tree rather than depending on the published
`openpi-client` wheel because that wheel pins `numpy<2.0.0`, which is
incompatible with the rest of this stack (notably the ZED SDK bindings, which
require numpy>=2). The code itself has no numpy-1-specific behaviour.

The wire format is unchanged, so this still talks to any openpi
``WebsocketPolicyServer`` — see docs/rtc.md for the server side.
"""

from robots_realtime.policy_client import image_tools, msgpack_numpy
from robots_realtime.policy_client.action_chunk_broker import ActionChunkBroker
from robots_realtime.policy_client.base_policy import BasePolicy
from robots_realtime.policy_client.policy_agent import Agent, PolicyAgent
from robots_realtime.policy_client.websocket_client_policy import WebsocketClientPolicy

__all__ = [
    "ActionChunkBroker",
    "Agent",
    "BasePolicy",
    "PolicyAgent",
    "WebsocketClientPolicy",
    "image_tools",
    "msgpack_numpy",
]
