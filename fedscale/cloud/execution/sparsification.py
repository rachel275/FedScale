from __future__ import annotations

from typing import Any

import numpy as np


def topk_compress(
    delta_state: dict[str, np.ndarray],
    ratio: float,
) -> dict[str, dict[str, Any]]:
    """Compress model deltas using per-tensor Top-K sparsification."""

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
    """Reconstruct dense model deltas from a Top-K payload."""

    dense = {}

    for name, payload in compressed.items():
        shape = tuple(
            payload["shape"]
        )

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

        dense[name] = flat.reshape(
            shape
        )

    return dense


def apply_topk_delta(
    base_state: dict[str, np.ndarray],
    compressed_delta: dict[str, dict[str, Any]],
) -> dict[str, np.ndarray]:
    """
    Apply a compressed Top-K delta to an existing model state.

    This is intended for bidirectional Top-K, where an executor already
    holds a cached copy of the previous global model and receives only
    the sparse change required to update that model.

    Args:
        base_state:
            Existing model state, typically the previous global model.

        compressed_delta:
            Sparse Top-K delta produced by topk_compress().

    Returns:
        A new state dictionary containing:

            base_state + decompressed_topk_delta

        Parameters that do not appear in compressed_delta are copied
        unchanged from base_state.
    """

    updated_state = {}

    for name, base_array in base_state.items():
        base_array = np.asarray(
            base_array
        )

        updated = base_array.copy()

        payload = compressed_delta.get(
            name
        )

        if payload is None:
            updated_state[name] = updated
            continue

        expected_shape = tuple(
            payload["shape"]
        )

        if updated.shape != expected_shape:
            raise ValueError(
                f"Shape mismatch for {name}: "
                f"base shape={updated.shape}, "
                f"delta shape={expected_shape}"
            )

        indices = np.asarray(
            payload["indices"],
            dtype=np.int64,
        )

        values = np.asarray(
            payload["values"],
            dtype=np.float32,
        )

        flat = updated.reshape(-1)

        if np.any(indices < 0) or np.any(
            indices >= flat.size
        ):
            raise ValueError(
                f"Invalid Top-K indices for {name}"
            )

        flat[indices] += values

        updated_state[name] = flat.reshape(
            updated.shape
        )

    return updated_state