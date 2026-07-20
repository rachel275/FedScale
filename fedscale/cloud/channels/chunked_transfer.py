from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Iterator

import fedscale.cloud.channels.job_api_pb2 as job_api_pb2


CHUNK_SIZE = 32 * 1024 * 1024


def sha256_bytes(payload: bytes) -> str:
    """Return the SHA-256 checksum of a byte payload."""
    return hashlib.sha256(payload).hexdigest()


def iter_chunks(
    payload: bytes,
    transfer_id: str,
    chunk_size: int = CHUNK_SIZE,
    client_id: str = "",
    executor_id: str = "",
    event: str = "",
):
    if not transfer_id:
        raise ValueError(
            "transfer_id must be non-empty"
        )

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be positive"
        )

    total_bytes = len(payload)

    total_chunks = max(
        1,
        math.ceil(
            total_bytes / chunk_size
        ),
    )

    for chunk_index in range(
        total_chunks
    ):
        start = (
            chunk_index * chunk_size
        )

        end = min(
            start + chunk_size,
            total_bytes,
        )

        yield job_api_pb2.DataChunk(
            transfer_id=transfer_id,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            total_bytes=total_bytes,
            data=payload[start:end],
            client_id=client_id,
            executor_id=executor_id,
            event=event,
        )

def reassemble_chunks(
    chunks: Iterable[job_api_pb2.DataChunk],
) -> bytes:
    """Validate and reassemble an ordered DataChunk stream."""
    parts: list[bytes] = []
    transfer_id: str | None = None
    expected_index = 0
    expected_total_chunks: int | None = None
    expected_total_bytes: int | None = None

    for chunk in chunks:
        if transfer_id is None:
            transfer_id = chunk.transfer_id
            expected_total_chunks = int(chunk.total_chunks)
            expected_total_bytes = int(chunk.total_bytes)

            if not transfer_id:
                raise RuntimeError("Received chunk with empty transfer_id")
            if expected_total_chunks <= 0:
                raise RuntimeError(
                    f"Invalid total_chunks={expected_total_chunks}"
                )
            if expected_total_bytes < 0:
                raise RuntimeError(
                    f"Invalid total_bytes={expected_total_bytes}"
                )

        if chunk.transfer_id != transfer_id:
            raise RuntimeError(
                f"Transfer ID changed: expected {transfer_id}, "
                f"got {chunk.transfer_id}"
            )

        if int(chunk.total_chunks) != expected_total_chunks:
            raise RuntimeError("total_chunks changed during stream")

        if int(chunk.total_bytes) != expected_total_bytes:
            raise RuntimeError("total_bytes changed during stream")

        if int(chunk.chunk_index) != expected_index:
            raise RuntimeError(
                f"Unexpected chunk order: expected {expected_index}, "
                f"got {chunk.chunk_index}"
            )

        parts.append(bytes(chunk.data))
        expected_index += 1

    if transfer_id is None:
        raise RuntimeError("Cannot reassemble an empty chunk stream")

    if expected_index != expected_total_chunks:
        raise RuntimeError(
            f"Incomplete chunk stream: expected {expected_total_chunks} "
            f"chunks, received {expected_index}"
        )

    payload = b"".join(parts)

    if len(payload) != expected_total_bytes:
        raise RuntimeError(
            f"Payload size mismatch: expected {expected_total_bytes} "
            f"bytes, got {len(payload)} bytes"
        )

    return payload
