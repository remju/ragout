"""Vector helpers - numpy if present, stdlib otherwise."""

from __future__ import annotations

import math
import struct
from typing import List, Sequence

try:
    import numpy as np
except ImportError:  # numpy is optional; we fall back to pure python
    np = None


def normalize(vec: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def pack(vec: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


def unpack(blob: bytes) -> Sequence[float]:
    if np is not None:
        return np.frombuffer(blob, dtype="<f4")
    return struct.unpack(f"<{len(blob) // 4}f", blob)


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    if np is not None:
        return float(np.dot(np.asarray(a, dtype="f4"), np.asarray(b, dtype="f4")))
    return sum(x * y for x, y in zip(a, b))
