"""Communication accounting for FedScale simulation experiments."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch


def raw_tensor_bytes(value: Any) -> int:
    """Count bytes occupied by tensors/arrays in a nested object."""
    if isinstance(value, np.ndarray):
        return int(value.nbytes)

    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())

    if isinstance(value, Mapping):
        return sum(raw_tensor_bytes(item) for item in value.values())

    if isinstance(value, (list, tuple)):
        return sum(raw_tensor_bytes(item) for item in value)

    return 0


def append_communication_record(
    output_path: str,
    record: dict[str, Any],
) -> None:
    """Append one communication event as JSONL."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    completed_record = {
        "timestamp": time.time(),
        **record,
    }

    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(completed_record, sort_keys=True) + "\n")
