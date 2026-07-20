"""
Lightweight GEMM tracer for PyTorch/FedScale experiments.

The tracer instruments torch.nn.Linear modules and records the matrix
multiplications performed during:

    - forward propagation
    - backward propagation for input gradients
    - backward propagation for weight gradients

This is useful for comparing the compute workload of full fine-tuning
and LoRA.

Environment variables:

    GEMM_TRACE_DIR
        Directory where trace files are written.
        Default: /tmp

    GEMM_TRACE_METHOD
        Experiment method, e.g. "full" or "lora".
        Default: unknown

Output:

    gemm-executor-<executor_id>.jsonl

Each JSONL record contains fields such as:

    {
        "method": "lora",
        "executor_id": "1",
        "client_id": 42,
        "phase": "forward",
        "module": "...attention.query",
        "gemm_M": 128,
        "gemm_N": 768,
        "gemm_K": 768,
        "flops": 150994944,
        "duration_us": 1234,
        "timestamp": ...,
        "relative_time": ...,
        "pid": ...,
        "thread_id": ...
    }

For one GEMM C[M,N] = A[M,K] @ B[K,N]:

    FLOPs ~= 2 * M * N * K

Notes
-----
This tracer observes nn.Linear operations. Transformer dense/attention
projections are represented as Linear modules, including PEFT LoRA
A/B projections.

It therefore provides a useful workload-level comparison of GEMM
compute between full fine-tuning and LoRA, but it is not a replacement
for a low-level BLAS/kernel profiler.
"""

from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn


# ---------------------------------------------------------------------
# Global tracer state
# ---------------------------------------------------------------------

_lock = threading.Lock()

_trace_file: Optional[Path] = None
_method: str = "unknown"
_executor_id: str = "unknown"

_start_time = time.perf_counter()

_current_client_id: Optional[int] = None

_hook_handles: list[Any] = []

# Each Linear module can be invoked multiple times before backward,
# especially for architectures that share parameters (e.g. ALBERT).
#
# We therefore retain a stack of forward contexts for each module.
_module_contexts: dict[int, list[dict[str, Any]]] = {}

# Timing for backward hooks.
_backward_start_times: dict[tuple[int, int], float] = {}


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

def configure(
    executor_id: int | str,
    output_dir: Optional[str] = None,
    method: Optional[str] = None,
) -> Path:
    """
    Configure the GEMM tracer.

    Parameters
    ----------
    executor_id:
        FedScale executor/rank identifier.

    output_dir:
        Directory for JSONL output. If omitted, GEMM_TRACE_DIR is used.

    method:
        "full", "lora", etc. If omitted, GEMM_TRACE_METHOD is used.

    Returns
    -------
    pathlib.Path
        Path to the JSONL trace file.
    """

    global _trace_file
    global _method
    global _executor_id
    global _start_time

    _executor_id = str(executor_id)

    _method = (
        method
        if method is not None
        else os.environ.get("GEMM_TRACE_METHOD", "unknown")
    )

    trace_dir = (
        output_dir
        if output_dir is not None
        else os.environ.get("GEMM_TRACE_DIR", "/tmp")
    )

    path = Path(trace_dir)
    path.mkdir(parents=True, exist_ok=True)

    _trace_file = path / f"gemm-executor-{_executor_id}.jsonl"

    _start_time = time.perf_counter()

    return _trace_file


def set_client_id(client_id: Optional[int]) -> None:
    """
    Associate subsequent GEMM records with a logical FL client.
    """

    global _current_client_id

    if client_id is None:
        _current_client_id = None
    else:
        _current_client_id = int(client_id)
    print(
        f"[GEMM TRACE] set_client_id={_current_client_id} "
        f"module_id={id(sys.modules[__name__])}"
    )


def clear_client_id() -> None:
    """Clear the currently active logical client."""

    global _current_client_id
    _current_client_id = None


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _tensor_from_output(output: Any) -> Optional[torch.Tensor]:
    """
    Find the first Tensor in a module output.
    """

    if isinstance(output, torch.Tensor):
        return output

    if isinstance(output, (tuple, list)):
        for value in output:
            tensor = _tensor_from_output(value)
            if tensor is not None:
                return tensor

    if isinstance(output, dict):
        for value in output.values():
            tensor = _tensor_from_output(value)
            if tensor is not None:
                return tensor

    return None


def _calculate_linear_shape(
    input_tensor: torch.Tensor,
    module: nn.Linear,
) -> tuple[int, int, int]:
    """
    Convert a Linear input into GEMM dimensions.

    nn.Linear takes:

        (..., K)

    and computes:

        (..., N)

    We flatten all leading dimensions into M:

        M x K  @  K x N
    """

    shape = tuple(input_tensor.shape)

    if len(shape) == 0:
        raise ValueError("Linear input has no dimensions")

    k = int(shape[-1])
    n = int(module.out_features)

    if len(shape) == 1:
        m = 1
    else:
        m = math.prod(shape[:-1])

    return int(m), int(n), int(k)


def _write_record(
    *,
    phase: str,
    module_name: str,
    m: int,
    n: int,
    k: int,
    duration_us: float,
) -> None:
    """
    Write a single GEMM record to JSONL.
    """

    if _trace_file is None:
        return

    # Only record actual FL client training.
    if _current_client_id is None:
        return

    flops = 2 * m * n * k

    record = {
        "method": _method,
        "executor_id": _executor_id,
        "client_id": _current_client_id,
        "phase": phase,
        "module": module_name,
        "gemm_M": m,
        "gemm_N": n,
        "gemm_K": k,
        "flops": flops,
        "duration_us": duration_us,
        "timestamp": time.time(),
        "relative_time": time.perf_counter() - _start_time,
        "pid": os.getpid(),
        "thread_id": threading.get_ident(),
    }

    with _lock:
        with _trace_file.open("a", encoding="utf-8") as output:
            output.write(
                json.dumps(record, sort_keys=True)
                + "\n"
            )


# ---------------------------------------------------------------------
# Linear hooks
# ---------------------------------------------------------------------

def _make_forward_pre_hook(module_name: str):
    def hook(
        module: nn.Module,
        inputs: tuple[Any, ...],
    ) -> None:

        if not isinstance(module, nn.Linear):
            return

        if not inputs:
            return

        input_tensor = inputs[0]

        if not isinstance(input_tensor, torch.Tensor):
            return

        try:
            m, n, k = _calculate_linear_shape(
                input_tensor,
                module,
            )
        except Exception:
            return

        context = {
            "M": m,
            "N": n,
            "K": k,
            "start": time.perf_counter(),
            "input_requires_grad":
                bool(input_tensor.requires_grad),
            "weight_requires_grad":
                bool(module.weight.requires_grad),
        }

        module_id = id(module)

        with _lock:
            _module_contexts.setdefault(
                module_id,
                [],
            ).append(context)

    return hook


def _make_forward_hook(module_name: str):
    def hook(
        module: nn.Module,
        inputs: tuple[Any, ...],
        output: Any,
    ) -> None:

        module_id = id(module)

        with _lock:
            contexts = _module_contexts.get(
                module_id,
                [],
            )

            if not contexts:
                return

            context = contexts[-1]

        duration_us = (
            time.perf_counter()
            - context["start"]
        ) * 1e6

        _write_record(
            phase="forward",
            module_name=module_name,
            m=context["M"],
            n=context["N"],
            k=context["K"],
            duration_us=duration_us,
        )

        output_tensor = _tensor_from_output(output)

        needs_backward = (
            torch.is_grad_enabled()
            and output_tensor is not None
            and output_tensor.requires_grad
        )

        # Evaluation/no_grad paths have no matching backward call,
        # so remove their context immediately.
        if not needs_backward:
            with _lock:
                contexts = _module_contexts.get(
                    module_id,
                    [],
                )

                if contexts:
                    contexts.pop()

    return hook

def _make_backward_hook(module_name: str):
    def hook(
        module: nn.Module,
        grad_input: tuple[Any, ...],
        grad_output: tuple[Any, ...],
    ) -> None:

        module_id = id(module)

        with _lock:
            contexts = _module_contexts.get(
                module_id,
                [],
            )

            if not contexts:
                return

            context = contexts.pop()

        backward_ops: list[str] = []

        if context["input_requires_grad"]:
            backward_ops.append(
                "backward_input"
            )

        if context["weight_requires_grad"]:
            backward_ops.append(
                "backward_weight"
            )

        for phase in backward_ops:
            _write_record(
                phase=phase,
                module_name=module_name,
                m=context["M"],
                n=context["N"],
                k=context["K"],
                duration_us=0.0,
            )

    return hook


# ---------------------------------------------------------------------
# Public instrumentation API
# ---------------------------------------------------------------------

def attach(
    model: nn.Module,
) -> int:
    """
    Attach GEMM tracing hooks to all torch.nn.Linear modules.

    This includes:

        - attention query/key/value projections
        - feed-forward layers
        - output projections
        - PEFT LoRA A/B Linear modules

    Returns
    -------
    int
        Number of Linear modules instrumented.
    """

    global _hook_handles

    detach()

    count = 0

    for module_name, module in model.named_modules():

        if not isinstance(module, nn.Linear):
            continue

        _hook_handles.append(
            module.register_forward_pre_hook(
                _make_forward_pre_hook(
                    module_name
                )
            )
        )

        _hook_handles.append(
            module.register_forward_hook(
                _make_forward_hook(
                    module_name
                )
            )
        )

        _hook_handles.append(
            module.register_full_backward_hook(
                _make_backward_hook(
                    module_name
                )
            )
        )

        count += 1

    return count


def detach() -> None:
    """
    Remove all installed hooks.
    """

    global _hook_handles

    for handle in _hook_handles:
        try:
            handle.remove()
        except Exception:
            pass

    _hook_handles = []


def reset() -> None:
    """
    Reset transient tracing state.

    Does not delete the output JSONL file.
    """

    global _current_client_id

    with _lock:
        _module_contexts.clear()
        _backward_start_times.clear()

    _current_client_id = None


def trace_path() -> Optional[str]:
    """
    Return the current trace file path.
    """

    if _trace_file is None:
        return None

    return str(_trace_file)
