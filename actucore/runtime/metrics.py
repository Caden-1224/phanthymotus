"""Small numeric helpers shared by runtime observability."""

from __future__ import annotations

import math


def p95(samples: list[float]) -> float:
    ordered = sorted(samples)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]
