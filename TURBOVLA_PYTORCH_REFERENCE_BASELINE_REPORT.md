# TurboVLA Exhaustive PyTorch Reference Baseline Report

## Result

The exhaustive PyTorch reference baseline was implemented and built successfully from a real,
deterministic LIBERO observation. The final `baseline_status.json` has `complete: true`; all 14
required gates pass.

Final artifact:

`outputs/pytorch_reference_baselines/libero_object_task0_ep0_step0_bf16`

Archive:

`outputs/pytorch_reference_baselines/libero_object_task0_ep0_step0_bf16/libero_object_task0_ep0_step0_bf16.tar.zst`

Archive SHA256 at completion:

`bf03597d5b85158fa5a2ce35543a32f3cc1bae14c608852ae9e675e3fd969005`

## Implemented architecture

- Added one strict shared loader in `turbovla/evaluation/model_loader.py`. It requires a checkpoint
  and checkpoint-owned model config, explicit local BERT/DINO paths, `strict=True`, exact device and
  precision, eval/no-grad state, parameter and buffer dtype verification, and checkpoint/config
  hashes. There is no default/random or non-strict fallback.
- Moved authoritative trace ownership outside `TurboVLA.forward()`. Model code only emits records;
  replay/policy owns begin, action postprocessing, and finish.
- Kept `EmbeddingDumper` as a warning-emitting legacy tool and renamed the sample utility to
  `scripts/legacy_dump_sample_trace.py`.
- Rebuilt the trace schema around execution-order `trace_id`, real per-semantic `call_index`, and
  the four levels `boundary`, `layer`, `op`, and `exhaustive`.
- Added canonical little-endian F32 storage for floating tensors, exact native BF16/F16 files, native
  int64 IDs, and uint8 boolean masks. Per-tensor `.pt`/`.npy` output is not used by the baseline.
- Added atomic trace and fixture publication. Byte/tensor limits raise an incomplete-trace error
  instead of silently skipping data.
- Added real fixture reader/writer/validator, including hashes, byte lengths, shape/dtype checks,
  stale-file rejection, preprocessing/tokenizer/state reproduction, and model identity checks.
- Added deterministic backend context with seed/state restoration, deterministic algorithms, TF32
  disablement, cuDNN controls, and eager/manual attention for deterministic reference runs.
- Instrumented preprocessing, tokenizer/layout, all 12 BERT layers, both DINOv3 views and all 12
  blocks, vision projection, six fusion layers, six text enhancers, state projection, all three
  action decoder layers, action MLP/tanh, and environment-action postprocessing.
- Added dynamic exhaustive coverage spec/validator. The built model requires and emits exactly
  2,968 unique semantic tensor names with no missing, unexpected, or duplicate semantic keys.
- Added weights/buffers writer for every persistent model tensor with native/F32 payloads, layouts,
  statistics, and hashes.
- Reworked trace comparison to match by `(semantic_name, call_index)`, determine first divergence in
  reference execution order, use `np.allclose` semantics, and report exact/error/correlation/first
  bad coordinate data.
- Added isolated LIBERO capture subprocess. This avoids the local LLVM conflict between
  MuJoCo/OSMesa and Transformers/CUDA while retaining a single user-facing baseline command.
- Added the one-shot orchestrator, human report, machine status, validations, free-space preflight,
  logs, stage summary, and `.tar.zst` packaging.

## Tests

Command:

```text
/home/linh/anaconda3/envs/turbovla-libero/bin/python -m pytest -q tests/debug
```

Result:

```text
35 passed, 1 warning in 2.55s
```

The warning is the existing timm namespace deprecation warning. Tests cover trace levels and call
indices, native storage, byte-limit failure, fixture hashes and atomic overwrite, strict/no-random
load behavior, lifecycle through postprocessing, compare order/allclose/perturbation, coverage
expansion, and traced-refactor equivalence for vision projection, fusion, text enhancer, state
projection, decoder, and action MLP.

Final full-repository verification:

```text
/home/linh/anaconda3/envs/turbovla-libero/bin/python -m pytest -q
35 passed, 1 warning in 4.78s
```

## Exact baseline command

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
/home/linh/anaconda3/envs/turbovla-libero/bin/python \
scripts/build_pytorch_reference_baseline.py \
  --task-suite libero_object \
  --task-id 0 --episode 0 --step 0 --seed 7 \
  --checkpoint pretrained/TurboVLA/checkpoints/libero/object.pth \
  --dinov3-path pretrained/dinov3/dinov3-vitb16 \
  --bert-path pretrained/bert-base-uncased \
  --stats-path pretrained/TurboVLA/libero_all4_stats.json \
  --stats-key libero_all4_no_noops \
  --libero-root /home/linh/Desktop/LIBERO \
  --device cuda --precision bf16 --trace-level exhaustive \
  --output-root outputs/pytorch_reference_baselines \
  --baseline-id libero_object_task0_ep0_step0_bf16 \
  --overwrite
```

The final execution returned exit code 0.

## Validation results

- Strict checkpoint load: PASS, 672 state tensors.
- Checkpoint SHA256: `787c01bd8b328a5948b756aab92f8058a1e0802845a0e1f24506291b9cda59cf`.
- Normalized production config SHA256:
  `42b9dc0c07b60e199ed0626e063462aba6b6924d5ede5ac6ffc7a90f16ec4b8c`.
- Fixture validation: PASS; 23 listed payload files; preprocessing, token tensors, masks, positions,
  and normalized state reproduce exactly.
- Policy vs exact normalized action maximum absolute error: `0.0`.
- Policy vs exact environment action maximum absolute error: `0.0`.
- Deterministic A vs B: 2,968 `PASS_EXACT`, zero failures.
- Exhaustive coverage: expected 2,968, actual 2,968, missing `[]`, unexpected `[]`, duplicates `[]`.
- Production/deterministic A/deterministic B manifests: PASS, 2,968 tensors each.
- Weights and persistent buffers: PASS, 672 tensors.
- Unexpected NaN/Inf count: 0. Masked-logit and shifted-softmax records retain the model's expected
  structural `-inf` mask values and are explicitly classified rather than treated as unexplained
  numerical failure.

## Deterministic trace size by stage

| Stage | Tensors | Bytes |
|---|---:|---:|
| action | 250 | 20,361,184 |
| condition | 4 | 1,637,384 |
| input | 13 | 7,864,348 |
| interaction | 738 | 310,903,086 |
| state | 6 | 208 |
| state_projection | 26 | 40,230 |
| text | 693 | 41,409,227 |
| vision | 1,209 | 2,056,104,240 |
| vision_projection | 29 | 33,051,648 |

Each deterministic trace contains 2,471,371,555 tensor bytes. Total packaged baseline size recorded
by the final artifact is 13,322,753,891 bytes, including the archive.

## Important generated reports

- `baseline_status.json` and `baseline_summary.md`
- `validation/fixture_validation.json`
- `validation/coverage_report.json`
- `validation/policy_vs_exact_replay.json`
- `validation/deterministic_replay_compare.{json,csv}` and `first_divergence.txt`
- `validation/*_manifest_validation.json` and `validation/stage_summary.csv`
- `weights/weights_manifest.jsonl` and `weights/buffers_manifest.jsonl`
- `logs/model_load.log`, `logs/libero_capture.log`, `logs/tests.log`, and
  `logs/baseline_build.log`

Generated runtime output remains ignored by Git.
