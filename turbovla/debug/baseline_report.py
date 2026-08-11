from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


def git_commit(repo: Path) -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def environment_metadata() -> dict[str, Any]:
    result = {
        "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(), "platform": platform.platform(),
        "command": sys.argv, "cwd": os.getcwd(),
    }
    if torch.cuda.is_available():
        result["gpu"] = torch.cuda.get_device_name()
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_baseline_summary(path: Path, metadata: dict, status: dict, details: dict) -> None:
    coverage = details.get("coverage", {})
    comparison = details.get("deterministic_compare", {})
    weights = details.get("weights", {})
    manifests = details.get("manifests", {})
    lines = [
        f"# PyTorch reference baseline: {metadata['baseline_id']}", "",
        "## 1. Baseline identity", "", f"- ID: `{metadata['baseline_id']}`",
        f"- Complete: `{status.get('complete', False)}`", f"- Repository commit: `{metadata.get('repository_commit')}`", "",
        "## 2. Model and checkpoint hashes", "",
        f"- Checkpoint SHA256: `{metadata.get('checkpoint_sha256')}`",
        f"- Normalized config SHA256: `{metadata.get('config_sha256')}`", "",
        "## 3. Environment", "", "```json", json.dumps(metadata.get("environment", {}), indent=2), "```", "",
        "## 4. Fixture source", "", "```json", json.dumps(metadata.get("fixture_source", {}), indent=2), "```", "",
        "## 5. Preprocessing details", "", "```json", json.dumps(metadata.get("preprocessing", {}), indent=2), "```", "",
        "## 6. Model configuration", "", "```json", json.dumps(metadata.get("model_config", {}), indent=2), "```", "",
        "## 7. Tensor coverage count by stage", "", f"- Expected: {coverage.get('expected_count', 0)}",
        f"- Actual semantic names: {coverage.get('actual_count', 0)}", f"- Missing: `{coverage.get('missing', [])}`", "",
        "## 8. Trace size by stage", "", "```json", json.dumps(details.get("trace_size_by_stage", {}), indent=2), "```", "",
        "## 9. Deterministic replay result", "", f"- Status: `{comparison.get('status')}`",
        f"- Status counts: `{comparison.get('status_counts', {})}`", "",
        "## 10. Policy vs exact replay result", "", "```json", json.dumps(details.get("policy_vs_exact", {}), indent=2), "```", "",
        "## 11. NaN/Inf summary", "", f"- NaN/Inf tensors: `{details.get('non_finite_count', 0)}`", "",
        "## 12. Weights manifest summary", "", "```json", json.dumps(weights, indent=2), "```", "",
        "## 13. All command lines", "", "```text", " ".join(metadata.get("environment", {}).get("command", [])), "```", "",
        "## 14. Test results", "", "```text", details.get("tests", "not run"), "```", "",
        "## 15. Manifest validation", "", "```json", json.dumps(manifests, indent=2), "```", "",
        "## 16. Unresolved warnings", "", "```json", json.dumps(details.get("warnings", []), indent=2), "```", "",
        "## 17. Archive checksum", "", f"- SHA256: `{details.get('archive_sha256', 'pending')}`", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
