#!/usr/bin/env python3
"""Download the fixed TurboVLA QCS8550 contexts with a local manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import qai_hub as hub


PORT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PORT_ROOT / "artifacts" / "object"
DEFAULT_COMPILE_JOBS = ARTIFACT_DIR / "qcs8550_compile_jobs.json"
DEFAULT_OUTPUT_DIR = ARTIFACT_DIR / "qcs8550_contexts"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile-jobs", type=Path, default=DEFAULT_COMPILE_JOBS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    jobs = json.loads(args.compile_jobs.read_text(encoding="utf-8"))
    graphs: dict[str, Any] = jobs["graphs"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    manifest["target"] = jobs["target"]
    manifest["graphs"] = {}

    for name, entry in graphs.items():
        compile_job = hub.get_job(entry["compile_job_id"])
        status = compile_job.get_status()
        if status.code != "SUCCESS":
            raise RuntimeError(f"{name} compile job {entry['compile_job_id']} is {status.code}: {status.message}")
        target_model = compile_job.get_target_model()
        if target_model is None:
            raise RuntimeError(f"{name} compile job has no target model")
        output = args.output_dir / f"{name}.bin"
        previous = manifest["graphs"].get(name, {})
        if output.is_file() and previous.get("model_id") == target_model.model_id and previous.get("sha256") == sha256(output):
            print(f"reuse {name}: {output}")
        else:
            print(f"download {name}: {target_model.model_id}")
            downloaded = Path(target_model.download(str(output)))
            if downloaded.resolve() != output.resolve():
                downloaded.replace(output)
        manifest["graphs"][name] = {
            "compile_job_id": entry["compile_job_id"],
            "model_id": target_model.model_id,
            "filename": output.name,
            "bytes": output.stat().st_size,
            "sha256": sha256(output),
            "input_spec": str(target_model.input_spec),
            "output_spec": str(target_model.output_spec),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
