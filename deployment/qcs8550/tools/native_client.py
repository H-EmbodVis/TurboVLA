#!/usr/bin/env python3
"""Send a frozen TurboVLA request to the persistent QCS8550 QNN service."""

from __future__ import annotations

import argparse
import json
import socket
import struct
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def recv_all(connection: socket.socket, size: int) -> bytes:
    result = bytearray()
    while len(result) < size:
        chunk = connection.recv(size - len(result))
        if not chunk:
            raise RuntimeError("native service closed the connection")
        result.extend(chunk)
    return bytes(result)


def request_arrays(bundle: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "pixels_view0": np.ascontiguousarray(bundle["pixels"][:, 0], dtype=np.float32),
        "pixels_view1": np.ascontiguousarray(bundle["pixels"][:, 1], dtype=np.float32),
        "input_ids": np.ascontiguousarray(bundle["input_ids"], dtype=np.int32),
        "token_type_ids": np.ascontiguousarray(bundle["token_type_ids"], dtype=np.int32),
        "bert_attention_mask": np.ascontiguousarray(bundle["bert_attention_mask"], dtype=np.bool_),
        "position_ids": np.ascontiguousarray(bundle["position_ids"], dtype=np.int32),
        "text_key_padding_mask": np.ascontiguousarray(bundle["text_key_padding_mask"], dtype=np.bool_),
        "text_self_attention_mask": np.ascontiguousarray(bundle["text_self_attention_mask"], dtype=np.bool_),
        "state": np.ascontiguousarray(bundle["state"], dtype=np.float32),
    }


def run_request(host: str, port: int, arrays: dict[str, np.ndarray], timeout: float) -> tuple[np.ndarray, dict[str, object]]:
    rows = []
    for name, array in arrays.items():
        encoded = name.encode("ascii")
        value = np.ascontiguousarray(array)
        rows.append(struct.pack(">H", len(encoded)) + encoded + struct.pack(">Q", value.nbytes) + value.tobytes())
    payload = struct.pack(">I", len(rows)) + b"".join(rows)
    with socket.create_connection((host, port), timeout=timeout) as connection:
        connection.sendall(struct.pack(">I", len(payload)) + payload)
        response = recv_all(connection, struct.unpack(">I", recv_all(connection, 4))[0])
    if len(response) < 12:
        raise RuntimeError("native service response is truncated")
    status, action_size = struct.unpack(">II", response[:8])
    offset = 8
    if offset + action_size + 4 > len(response):
        raise RuntimeError("native service action payload is malformed")
    action = np.frombuffer(response[offset : offset + action_size], dtype=np.float32).copy()
    offset += action_size
    metrics_size = struct.unpack(">I", response[offset : offset + 4])[0]
    metrics_raw = response[offset + 4 : offset + 4 + metrics_size]
    try:
        metrics: dict[str, object] = json.loads(metrics_raw)
    except json.JSONDecodeError:
        metrics = {"raw": metrics_raw.decode("utf-8", errors="replace")}
    if status != 0:
        raise RuntimeError(f"native service rejected request: {metrics}")
    if action.size != 12 * 7:
        raise RuntimeError(f"native service returned {action.size} floats, expected 84")
    return action.reshape(1, 12, 7), metrics


def metric(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    diff = np.asarray(actual, dtype=np.float64) - np.asarray(expected, dtype=np.float64)
    return {
        "relative_l2": float(np.linalg.norm(diff) / max(np.linalg.norm(expected), 1e-12)),
        "max_abs": float(np.abs(diff).max()),
        "rmse": float(np.sqrt(np.mean(np.square(diff)))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=10092)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--requests", type=int, default=1)
    parser.add_argument("--expected", type=Path)
    parser.add_argument("--expected-key", default="board_action")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/native_client_output.npz")
    args = parser.parse_args()

    with np.load(args.bundle, allow_pickle=False) as archive:
        bundle = {name: archive[name] for name in archive.files}
    if args.requests < 1:
        raise ValueError("--requests must be positive")
    actions = []
    service_metrics = []
    arrays = request_arrays(bundle)
    for _ in range(args.requests):
        action, metrics = run_request(args.host, args.port, arrays, args.timeout)
        actions.append(action)
        service_metrics.append(metrics)
    action = actions[-1]
    request_us = np.asarray([row.get("request_us", np.nan) for row in service_metrics], dtype=np.float64)
    stability = np.asarray(actions, dtype=np.float32)
    report: dict[str, object] = {
        "service_last": service_metrics[-1],
        "action_shape": list(action.shape),
        "requests": args.requests,
        "action_max_abs_across_requests": float(np.max(np.abs(stability - stability[0]))),
        "service_request_us": {
            "p50": float(np.nanpercentile(request_us, 50)),
            "p95": float(np.nanpercentile(request_us, 95)),
            "min": float(np.nanmin(request_us)),
            "max": float(np.nanmax(request_us)),
        },
    }
    if args.expected:
        with np.load(args.expected, allow_pickle=False) as archive:
            report["vs_expected"] = metric(action, archive[args.expected_key])
    if "full_action" in bundle:
        report["vs_reference"] = metric(action, bundle["full_action"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, action=action)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
