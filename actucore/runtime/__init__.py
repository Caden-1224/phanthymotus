"""ActuCore's model runtime.

The public surface is deliberately small: the gateway the card talks to, and
the config translation that decides whether the card talks to it at all.
"""

from __future__ import annotations

from .config import base_url, capabilities, enabled, vla_config
from .gateway import RuntimeGateway

__all__ = ["RuntimeGateway", "base_url", "capabilities", "enabled", "vla_config"]
