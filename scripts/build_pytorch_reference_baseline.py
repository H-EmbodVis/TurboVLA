from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from scripts.compare_tensor_traces import compare_traces
from scripts.validate_trace_manifest import validate_manifest
from turbovla.debug import FixtureReader, FixtureWriter, TraceConfig, TraceContext, deterministic_context
from turbovla.debug.baseline_report import environment_metadata, git_commit, sha256_file, write_baseline_summary
from turbovla.debug.coverage_validator import validate_coverage, write_coverage_report
from turbovla.debug.exhaustive_coverage_spec import expected_semantic_names
from turbovla.debug.fixture_validator import validate_fixture, write_fixture_validation
from turbovla.debug.weights_writer import write_weights_and_buffers
from turbovla.evaluation.model_loader import load_turbovla_for_inference
from turbovla.evaluation.policy import build_dinov3_manual_processor, load_preprocessor_config, rotate_libero_image
from turbovla.evaluation.replay import (
    ExactInputs, derive_exact_inputs, load_suite_statistics, run_exact_replay,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the authoritative exhaustive PyTorch reference baseline")
    parser.add_argument("--task-suite", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dinov3-path", required=True)
    parser.add_argument("--bert-path", required=True)
    parser.add_argument("--stats-path", type=Path, required=True)
    parser.add_argument("--stats-key", required=True)
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), required=True)
    parser.add_argument("--precision", choices=("bf16", "fp32"), required=True)
    parser.add_argument("--trace-level", choices=("boundary", "layer", "op", "exhaustive"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--baseline-id", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-trace-bytes", type=int, default=0)
    parser.add_argument("--skip-tests", action="store_true", help="Developer-only; makes tests_pass false")
    return parser.parse_args()


def sha_model_files(root: Path) -> dict[str, str]:
    result = {}
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.txt", "config.json"):
        path = root / name
        if path.is_file():
            result[name] = sha256_file(path)
    return result


def write_fixture(work: Path, exact: ExactInputs, loaded, args, source: dict, processor_config: dict, statistics) -> Path:
    writer = FixtureWriter(work, "fixture", overwrite=True)
    writer.write_instruction(exact.instruction)
    for index in range(2):
        writer.write_image(f"raw/view_{index}_rgb_u8", exact.raw_views[index])
    writer.write_tensor("raw/state_raw_f32", exact.state_raw_f32, "f32", "state_raw_f32le.bin")
    writer.write_tensor("exact_inputs/pixel_values_f32", exact.pixel_values_f32, "f32", "pixel_values_f32le.bin")
    dtype_suffix = "bf16" if loaded.model_dtype == torch.bfloat16 else "f32"
    dtype_binary = "pixel_values_bf16le.bin" if dtype_suffix == "bf16" else "pixel_values_f32le.bin"
    writer.write_tensor("exact_inputs/pixel_values_model_dtype", exact.pixel_values_model_dtype, dtype_suffix, dtype_binary)

    text = loaded.model.text_encoder
    padding = text.config.padding_length_by_instruction.get(exact.instruction, text.config.padding_length)
    tokenized, self_mask, position_ids = text._tokenize_group([exact.instruction], torch.device("cpu"), padding)
    writer.write_tensor("exact_inputs/input_ids_i64", tokenized.input_ids, "i64", "input_ids_i64le.bin")
    writer.write_tensor("exact_inputs/attention_mask_u8", tokenized.attention_mask.bool(), "u8", "attention_mask_u8.bin")
    writer.write_tensor("exact_inputs/text_self_attention_mask_u8", self_mask, "u8", "text_self_attention_mask_u8.bin")
    writer.write_tensor("exact_inputs/position_ids_i64", position_ids, "i64", "position_ids_i64le.bin")
    writer.write_tensor("exact_inputs/state_normalized_f32", exact.state_normalized_f32, "f32", "state_normalized_f32le.bin")
    state_binary = "state_bf16le.bin" if dtype_suffix == "bf16" else "state_f32le.bin"
    writer.write_tensor("exact_inputs/state_model_dtype", exact.state_model_dtype, dtype_suffix, state_binary)
    writer.write_metadata({
        "source": source, "instruction": exact.instruction, "seed": args.seed,
        "model": {"checkpoint_sha256": loaded.checkpoint_sha256, "config_sha256": loaded.config_sha256,
                  "precision": args.precision},
        "preprocessing": processor_config,
        "state_normalization": {"mean": statistics.state_mean.tolist(), "std": statistics.state_std.tolist()},
        "tokenizer": {
            "class": type(text.tokenizer).__name__, "padding_side": text.tokenizer.padding_side,
            "pad_token_id": text.tokenizer.pad_token_id, "cls_token_id": text.tokenizer.cls_token_id,
            "sep_token_id": text.tokenizer.sep_token_id, "unpadded_length": int(tokenized.attention_mask.sum()),
            "model_text_length": int(tokenized.input_ids.shape[1]), "instruction_padding_length": padding,
            "files": sha_model_files(Path(args.bert_path)),
        },
    })
    return writer.finalize()


def load_exact_fixture(fixture: Path, instruction: str, model_dtype: torch.dtype, device: torch.device) -> ExactInputs:
    reader = FixtureReader(fixture)
    raw = np.stack([reader.load_numpy(f"raw/view_{index}_rgb_u8.npy") for index in range(2)])
    pixel_f32 = reader.load_tensor("exact_inputs/pixel_values_f32.npy").float()
    if model_dtype == torch.bfloat16:
        pixel_model = reader.load_tensor("exact_inputs/pixel_values_model_dtype.npy", bf16_bits=True)
        state_model = reader.load_tensor("exact_inputs/state_model_dtype.npy", bf16_bits=True)
    else:
        pixel_model = reader.load_tensor("exact_inputs/pixel_values_model_dtype.npy").float()
        state_model = reader.load_tensor("exact_inputs/state_model_dtype.npy").float()
    return ExactInputs(
        raw_views=raw, rotated_views=np.stack([rotate_libero_image(item) for item in raw]),
        pixel_values_f32=pixel_f32, pixel_values_model_dtype=pixel_model.to(device),
        state_raw_f32=reader.load_tensor("raw/state_raw_f32.npy").float(),
        state_normalized_f32=reader.load_tensor("exact_inputs/state_normalized_f32.npy").float(),
        state_model_dtype=state_model.to(device), instruction=instruction,
    )


def trace_config(work: Path, profile: str, args) -> TraceConfig:
    return TraceConfig(
        enabled=True, root_dir=work / "traces" / profile, level=args.trace_level, max_forwards=1,
        save_pt=False, save_f32=True, save_native=True, save_raw_bf16=True, save_stats=True,
        fail_on_nan=True, overwrite=True, max_bytes=args.max_trace_bytes, atomic=True,
    )


def save_actions(directory: Path, prefix: str, result) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / f"{prefix}_normalized_action.npy", result.normalized_action, allow_pickle=False)
    np.save(directory / f"{prefix}_env_action.npy", result.env_action, allow_pickle=False)


def trace_stage_sizes(trace_dir: Path) -> dict[str, int]:
    result: dict[str, int] = defaultdict(int)
    for line in (trace_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        for field in ("file", "native_file"):
            if record.get(field):
                result[record["stage"]] += (trace_dir / record[field]).stat().st_size
    return dict(sorted(result.items()))


def compare_actions(policy, exact) -> dict:
    fields = ("normalized_action", "env_action", "first_step")
    result = {"status": "PASS"}
    for field in fields:
        left, right = getattr(policy, field), getattr(exact, field)
        maximum = float(np.max(np.abs(left - right)))
        passed = bool(np.array_equal(left, right))
        result[field] = {"pass": passed, "max_abs": maximum}
        if not passed:
            result["status"] = "FAIL"
    return result


def preflight(work: Path, checkpoint: Path) -> dict:
    # Conservative upper bound: three exhaustive activation traces plus native/F32 weights and archive workspace.
    checkpoint_bytes = checkpoint.stat().st_size
    estimated = max(24 * 1024 ** 3, checkpoint_bytes * 20)
    free = shutil.disk_usage(work.parent).free
    margin = 5 * 1024 ** 3
    if free < estimated + margin:
        raise OSError(f"insufficient free space: free={free}, estimated={estimated}, safety_margin={margin}")
    return {"estimated_bytes": estimated, "free_bytes": free, "safety_margin_bytes": margin}


def capture_isolated(repo: Path, work: Path, args) -> tuple[dict, str, dict]:
    capture_dir = work / "capture_process"
    command = [
        sys.executable, str(repo / "scripts/capture_libero_parity_fixture.py"),
        "--libero-root", str(args.libero_root), "--task-suite", args.task_suite,
        "--task-id", str(args.task_id), "--episode", str(args.episode), "--step", str(args.step),
        "--seed", str(args.seed), "--output", str(capture_dir),
    ]
    env = dict(os.environ, MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa")
    result = subprocess.run(command, cwd=repo, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (work / "logs/libero_capture.log").write_text("$ " + " ".join(command) + "\n" + result.stdout, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"isolated LIBERO capture failed with exit {result.returncode}; see logs/libero_capture.log")
    obs = {name: np.load(capture_dir / f"{name}.npy", allow_pickle=False) for name in (
        "agentview_image", "robot0_eye_in_hand_image", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")}
    instruction = (capture_dir / "instruction.txt").read_text(encoding="utf-8")
    metadata = json.loads((capture_dir / "metadata.json").read_text(encoding="utf-8"))
    shutil.rmtree(capture_dir)
    return obs, instruction, metadata


def run_tests(repo: Path, log_path: Path) -> tuple[bool, str]:
    command = [sys.executable, "-m", "pytest", "-q", "tests/debug"]
    result = subprocess.run(command, cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_path.write_text("$ " + " ".join(command) + "\n" + result.stdout, encoding="utf-8")
    return result.returncode == 0, result.stdout[-8000:]


def create_archive(work: Path, baseline_id: str) -> tuple[Path, str]:
    outside = work.parent / f"{baseline_id}.tar.zst.tmp.{os.getpid()}"
    command = [
        "tar", "--zstd", "-cf", str(outside), "--exclude=*.tar.zst",
        f"--transform=s,^{work.name},{baseline_id},", "-C", str(work.parent), work.name,
    ]
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"archive creation failed: {' '.join(command)}\n{result.stderr}")
    archive = work / f"{baseline_id}.tar.zst"
    shutil.move(outside, archive)
    return archive, sha256_file(archive)


def main() -> int:
    args = parse_args()
    if args.trace_level != "exhaustive":
        raise ValueError("authoritative baseline requires --trace-level exhaustive")
    repo = Path(__file__).resolve().parents[1]
    target = args.output_root / args.baseline_id
    if target.exists() and not args.overwrite:
        raise FileExistsError(f"baseline already exists: {target}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    work = args.output_root / f"{args.baseline_id}.tmp.{os.getpid()}"
    if work.exists():
        shutil.rmtree(work)
    for name in ("actions", "validation", "logs", "traces", "weights"):
        (work / name).mkdir(parents=True, exist_ok=True)
    status = {"baseline_id": args.baseline_id, "complete": False, "gates": {}, "metrics": {}}
    metadata = {"baseline_id": args.baseline_id, "repository_commit": git_commit(repo), "environment": environment_metadata()}
    details = {"warnings": []}
    try:
        budget = preflight(work, args.checkpoint)
        tests_pass, tests_output = (False, "SKIPPED by --skip-tests") if args.skip_tests else run_tests(repo, work / "logs/tests.log")
        details["tests"] = tests_output
        statistics = load_suite_statistics(args.stats_path, args.stats_key)
        obs, instruction, fixture_source = capture_isolated(repo, work, args)
        loaded = load_turbovla_for_inference(
            checkpoint_path=args.checkpoint, dinov3_path=args.dinov3_path, bert_path=args.bert_path,
            device=args.device, precision=args.precision, strict=True, deterministic=False,
        )
        (work / "logs/model_load.log").write_text(
            "strict=True\n"
            f"checkpoint={args.checkpoint.resolve()}\n"
            f"checkpoint_sha256={loaded.checkpoint_sha256}\n"
            f"config_sha256={loaded.config_sha256}\n"
            f"state_tensor_count={loaded.state_tensor_count}\n"
            f"device={loaded.device}\nmodel_dtype={loaded.model_dtype}\n",
            encoding="utf-8",
        )
        processor_config = load_preprocessor_config(args.dinov3_path)
        processor = build_dinov3_manual_processor(args.dinov3_path)
        exact = derive_exact_inputs(obs=obs, instruction=instruction, processor=processor, statistics=statistics,
                                    model_dtype=loaded.model_dtype, device=loaded.device)
        fixture = write_fixture(work, exact, loaded, args, fixture_source, processor_config, statistics)

        def reproduce(reader):
            raw = [reader.load_numpy(f"raw/view_{index}_rgb_u8.npy") for index in range(2)]
            pixels = processor([rotate_libero_image(item) for item in raw])["pixel_values"].view(1, 2, 3, 256, 256).numpy()
            state = reader.load_numpy("raw/state_raw_f32.npy")
            normalized = (state - statistics.state_mean) / (statistics.state_std + 1e-6)
            text = loaded.model.text_encoder
            padding = text.config.padding_length_by_instruction.get(instruction, text.config.padding_length)
            tokenized, self_mask, positions = text._tokenize_group([instruction], torch.device("cpu"), padding)
            return {
                "exact_inputs/pixel_values_f32.npy": pixels,
                "exact_inputs/state_normalized_f32.npy": normalized,
                "exact_inputs/input_ids_i64.npy": tokenized.input_ids.numpy(),
                "exact_inputs/attention_mask_u8.npy": tokenized.attention_mask.bool().numpy(),
                "exact_inputs/text_self_attention_mask_u8.npy": self_mask.numpy(),
                "exact_inputs/position_ids_i64.npy": positions.numpy(),
            }

        fixture_validation = validate_fixture(
            fixture, requested_fixture_id="fixture",
            expected_model={"checkpoint_sha256": loaded.checkpoint_sha256, "config_sha256": loaded.config_sha256,
                            "precision": args.precision}, reproduce_preprocessing=reproduce,
        )
        write_fixture_validation(work / "validation/fixture_validation.json", fixture_validation)
        metadata.update(checkpoint_sha256=loaded.checkpoint_sha256, config_sha256=loaded.config_sha256,
                        fixture_source=fixture_source, preprocessing=processor_config,
                        model_config=loaded.config.to_dict())
        (work / "baseline_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        weights = write_weights_and_buffers(loaded.model, work / "weights")
        details["weights"] = weights

        production_tracer = TraceContext(trace_config(work, "production_bf16", args))
        production = run_exact_replay(loaded=loaded, exact=exact, statistics=statistics, tracer=production_tracer,
                                      processor_config=processor_config)
        save_actions(work / "actions", "production", production)
        fixture_exact = load_exact_fixture(fixture, instruction, loaded.model_dtype, loaded.device)
        exact_replay = run_exact_replay(loaded=loaded, exact=fixture_exact, statistics=statistics)
        policy_vs_exact = compare_actions(production, exact_replay)
        details["policy_vs_exact"] = policy_vs_exact
        (work / "validation/policy_vs_exact_replay.json").write_text(json.dumps(policy_vs_exact, indent=2) + "\n", encoding="utf-8")

        del loaded
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        deterministic_loaded = load_turbovla_for_inference(
            checkpoint_path=args.checkpoint, dinov3_path=args.dinov3_path, bert_path=args.bert_path,
            device=args.device, precision=args.precision, strict=True, deterministic=True,
        )
        fixture_exact = load_exact_fixture(fixture, instruction, deterministic_loaded.model_dtype, deterministic_loaded.device)
        deterministic_results = []
        for suffix in ("a", "b"):
            tracer = TraceContext(trace_config(work, f"deterministic_bf16_run_{suffix}", args))
            with deterministic_context(args.seed):
                result = run_exact_replay(loaded=deterministic_loaded, exact=fixture_exact, statistics=statistics,
                                          tracer=tracer, processor_config=processor_config)
            save_actions(work / "actions", f"deterministic_{suffix}", result)
            deterministic_results.append(result)

        trace_dirs = {
            "production": work / "traces/production_bf16/forward_000000_rank_0",
            "deterministic_a": work / "traces/deterministic_bf16_run_a/forward_000000_rank_0",
            "deterministic_b": work / "traces/deterministic_bf16_run_b/forward_000000_rank_0",
        }
        manifests = {name: validate_manifest(path, strict=True) for name, path in trace_dirs.items()}
        details["manifests"] = manifests
        for name, result in manifests.items():
            (work / f"validation/{name}_manifest_validation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        compare_dir = work / "validation/deterministic_compare"
        deterministic_compare = compare_traces(trace_dirs["deterministic_a"], trace_dirs["deterministic_b"], compare_dir)
        details["deterministic_compare"] = deterministic_compare
        for source, destination in (("compare.csv", "deterministic_replay_compare.csv"),
                                    ("compare.json", "deterministic_replay_compare.json"),
                                    ("first_divergence.txt", "first_divergence.txt")):
            shutil.copy2(compare_dir / source, work / "validation" / destination)
        expected = expected_semantic_names(deterministic_loaded.config, deterministic_loaded.model)
        coverage = validate_coverage(trace_dirs["deterministic_a"], expected)
        details["coverage"] = coverage
        write_coverage_report(work / "validation/coverage_report.json", coverage)
        details["trace_size_by_stage"] = trace_stage_sizes(trace_dirs["deterministic_a"])
        records = [json.loads(line) for line in (trace_dirs["deterministic_a"] / "manifest.jsonl").read_text().splitlines()]
        stage_counts: dict[str, int] = defaultdict(int)
        for record in records:
            stage_counts[record["stage"]] += 1
        stage_lines = ["stage,tensor_count,bytes"] + [
            f"{stage},{stage_counts[stage]},{details['trace_size_by_stage'][stage]}"
            for stage in sorted(stage_counts)
        ]
        (work / "validation/stage_summary.csv").write_text("\n".join(stage_lines) + "\n", encoding="utf-8")
        structural_suffixes = ("masked_logits", "softmax.shifted")
        structural_inf = sum(bool(record["inf_count"]) and record["semantic_name"].endswith(structural_suffixes) for record in records)
        non_finite = sum(bool(record["nan_count"] or (record["inf_count"] and not record["semantic_name"].endswith(structural_suffixes))) for record in records)
        details["non_finite_count"] = non_finite
        details["warnings"].append({"structural_masked_logits_with_infinity": structural_inf})

        gates = {
            "real_fixture_captured": bool(obs), "fixture_validation_pass": fixture_validation["status"] == "PASS",
            "strict_checkpoint_load_pass": True,
            "policy_vs_exact_normalized_pass": policy_vs_exact["normalized_action"]["pass"],
            "policy_vs_exact_env_action_pass": policy_vs_exact["env_action"]["pass"],
            "production_trace_manifest_pass": manifests["production"]["status"] == "PASS",
            "deterministic_a_manifest_pass": manifests["deterministic_a"]["status"] == "PASS",
            "deterministic_b_manifest_pass": manifests["deterministic_b"]["status"] == "PASS",
            "deterministic_trace_compare_pass": deterministic_compare["status"] == "PASS",
            "coverage_pass": coverage["status"] == "PASS", "weights_manifest_pass": weights["status"] == "PASS",
            "no_nan_inf": non_finite == 0, "tests_pass": tests_pass, "archive_created": True,
        }
        status["gates"] = gates
        status["metrics"] = {
            "policy_vs_exact_normalized_max_abs": policy_vs_exact["normalized_action"]["max_abs"],
            "policy_vs_exact_env_action_max_abs": policy_vs_exact["env_action"]["max_abs"],
            "deterministic_trace_failures": sum(count for key, count in deterministic_compare["status_counts"].items() if not key.startswith("PASS")),
            "trace_tensor_count": manifests["deterministic_a"]["tensor_count"],
            "weights_tensor_count": weights["tensor_count"], "estimated_bytes": budget["estimated_bytes"],
            "actual_bytes": sum(path.stat().st_size for path in work.rglob("*") if path.is_file()),
        }
        status["complete"] = all(gates.values())
        (work / "baseline_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        write_baseline_summary(work / "baseline_summary.md", metadata, status, details)
        if not status["complete"]:
            raise RuntimeError(f"baseline gates failed: {[key for key, value in gates.items() if not value]}")
        archive, archive_sha = create_archive(work, args.baseline_id)
        details["archive_sha256"] = archive_sha
        status["metrics"]["total_bytes"] = sum(path.stat().st_size for path in work.rglob("*") if path.is_file())
        (work / "baseline_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        (work / "logs/baseline_build.log").write_text(
            "status=PASS\n" + json.dumps(status, indent=2) + "\n" + f"archive_sha256={archive_sha}\n",
            encoding="utf-8",
        )
        write_baseline_summary(work / "baseline_summary.md", metadata, status, details)
        if target.exists():
            old = args.output_root / f"{args.baseline_id}.old.{os.getpid()}"
            target.rename(old)
            work.rename(target)
            shutil.rmtree(old)
        else:
            work.rename(target)
        print(json.dumps(status, indent=2))
        print(f"archive={target / archive.name} sha256={archive_sha}")
        return 0
    except Exception as error:
        status["error"] = f"{type(error).__name__}: {error}"
        status["complete"] = False
        (work / "baseline_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        (work / "logs/baseline_build.log").write_text(status["error"] + "\n", encoding="utf-8")
        incomplete = args.output_root / f"{args.baseline_id}.incomplete.{os.getpid()}"
        if incomplete.exists():
            shutil.rmtree(incomplete)
        work.rename(incomplete)
        print(f"baseline build failed; diagnostics preserved at {incomplete}: {status['error']}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
