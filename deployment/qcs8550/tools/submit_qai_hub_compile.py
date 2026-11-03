#!/usr/bin/env python3
"""Submit the static TurboVLA ONNX graph set to QCS8550 QAIRT 2.48."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import qai_hub as hub


PORT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ONNX_DIR = PORT_ROOT / "artifacts" / "object" / "onnx"
DEFAULT_JOBS = PORT_ROOT / "artifacts" / "object" / "qcs8550_compile_jobs.json"


def job_id(job) -> str:
    value = getattr(job, "job_id", None)
    if value:
        return str(value)
    return str(job.url).rstrip("/").split("/")[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-dir", type=Path, default=DEFAULT_ONNX_DIR)
    parser.add_argument("--jobs", type=Path, default=DEFAULT_JOBS)
    parser.add_argument("--device", default="QCS8550 (Proxy)")
    parser.add_argument("--qairt-version", default="2.48")
    parser.add_argument("--only", nargs="*", choices=["dinov3", "bert_l11", "bert_l14", "bert_l21", "policy_core"])
    args = parser.parse_args()

    graphs = {
        "dinov3": "dinov3_one_view_fp32.onnx",
        "bert_l11": "bert_l11_fp32.onnx",
        "bert_l14": "bert_l14_fp32.onnx",
        "bert_l21": "bert_l21_fp32.onnx",
        "policy_core": "policy_core_l21_fp32.onnx",
    }
    selected = args.only or list(graphs)
    args.jobs.parent.mkdir(parents=True, exist_ok=True)
    saved = json.loads(args.jobs.read_text(encoding="utf-8")) if args.jobs.is_file() else {}
    saved.setdefault("target", {"device": args.device, "qairt_version": args.qairt_version})
    saved.setdefault("graphs", {})

    client = hub.Client()
    device = hub.Device(args.device)
    options = f"--target_runtime qnn_context_binary --qairt_version {args.qairt_version} --truncate_64bit_io"
    for name in selected:
        model = (args.onnx_dir / graphs[name]).resolve()
        if not model.is_file():
            raise FileNotFoundError(model)
        prior = saved["graphs"].get(name)
        if prior and prior.get("model_sha256") == _sha256(model):
            print(f"reuse {name}: {prior['compile_job_id']}")
            continue
        job = client.submit_compile_job(
            model=str(model),
            device=device,
            name=f"turbovla_object_{name}_qcs8550_qairt{args.qairt_version.replace('.', '')}",
            options=options,
        )
        saved["graphs"][name] = {
            "compile_job_id": job_id(job),
            "url": job.url,
            "model": str(model),
            "model_sha256": _sha256(model),
            "options": options,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
        }
        args.jobs.write_text(json.dumps(saved, indent=2), encoding="utf-8")
        print(f"submitted {name}: {job.url}")
    args.jobs.write_text(json.dumps(saved, indent=2), encoding="utf-8")


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
