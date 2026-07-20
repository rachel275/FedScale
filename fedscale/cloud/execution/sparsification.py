from __future__ import annotations

from typing import Any

import numpy as np


def topk_compress(
    delta_state: dict[str, np.ndarray],
    ratio: float,
) -> dict[str, dict[str, Any]]:
    """Compress model deltas using global per-tensor Top-K sparsification."""

    if not 0.0 < ratio <= 1.0:
        raise ValueError(
            f"topk ratio must be in (0, 1], got {ratio}"
        )

    compressed = {}

    for name, array in delta_state.items():
        array = np.asarray(array)

        flat = array.reshape(-1)

        if flat.size == 0:
            continue

        k = max(
            1,
            int(np.ceil(flat.size * ratio)),
        )

        if k >= flat.size:
            indices = np.arange(
                flat.size,
                dtype=np.int64,
            )
        else:
            indices = np.argpartition(
                np.abs(flat),
                -k,
            )[-k:].astype(np.int64)

        values = flat[indices].astype(
            np.float32,
            copy=False,
        )

        compressed[name] = {
            "shape": array.shape,
            "indices": indices,
            "values": values,
        }

    return compressed


def topk_decompress(
    compressed: dict[str, dict[str, Any]],
) -> dict[str, np.ndarray]:
    """Reconstruct dense model deltas from Top-K payload."""

    dense = {}

    for name, payload in compressed.items():
        shape = tuple(payload["shape"])

        indices = np.asarray(
            payload["indices"],
            dtype=np.int64,
        )

        values = np.asarray(
            payload["values"],
            dtype=np.float32,
        )

        flat = np.zeros(
            int(np.prod(shape)),
            dtype=np.float32,
        )

        flat[indices] = values

        dense[name] = flat.reshape(shape)

    return dense
