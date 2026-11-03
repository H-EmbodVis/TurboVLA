# QCS8550 Deployment Instructions For AI Agents

Use this runbook when the task is to deploy the released TurboVLA **LIBERO
Object** checkpoint on a Qualcomm QCS8550 board. Follow the phases in order.
Do not substitute model outputs, change graph topology, or claim FP32-equivalent
results: this target is an experimental QAIRT 2.48 QNN deployment.

## Safety And Scope

- Work from the TurboVLA repository root and keep generated files below
  `deployment/qcs8550/artifacts/`; this directory is intentionally ignored.
- Do not commit, print, copy into source files, or request any API token. Run
  `qai-hub configure` only in an interactive user terminal. It stores credentials
  in the user's local configuration and is not part of this repository.
- Use an isolated board root such as `/opt/turbovla-qcs8550` and port `10092`.
  Do not stop, overwrite, or reconfigure an unrelated board service. The build
  helper refuses to start if the selected port is already occupied.
- Never use `last_hidden_state` for DINO features. Upstream issue
  [#4](https://github.com/H-EmbodVis/TurboVLA/issues/4) requires
  `outputs.hidden_states[-1]`. Pin `transformers==4.56.*` for reference/export.
- Stop and report the blocker rather than guessing if the checkpoint, QAIRT
  version, QCS8550 target, QNN headers, runtime, board access, or AI Hub access
  is unavailable.

## Inputs To Obtain From The User

Ask only for values that cannot be discovered locally:

| Variable | Meaning |
| --- | --- |
| `BOARD_HOST` | SSH host/IP for the QCS8550 board; key-based SSH must work. |
| `QAIRT_INCLUDE` | Host QAIRT 2.48 SDK `include` directory (or its `include/QNN` child). |
| `QAIRT_RUNTIME` | Board QAIRT 2.48 runtime root. |
| `LIBERO_ROOT` | Local LIBERO installation, required for rollout. |

The agent must verify these local assets before doing expensive work:

```bash
test -f pretrained/TurboVLA/checkpoints/libero/object.pth
test -d pretrained/bert-base-uncased
test -d "$QAIRT_INCLUDE"
ssh -o BatchMode=yes "$BOARD_HOST" true
ssh -o BatchMode=yes "$BOARD_HOST" "test -d '$QAIRT_RUNTIME'"
python -c 'import transformers; assert transformers.__version__.startswith("4.56.")'
```

The release checkpoint supplies the DINO weights used by static export. The
shared LIBERO command line still accepts a local DINO path for its upstream
configuration; retain a licensed local DINOv3 ViT-B asset for rollout.

## Phase 1: Prepare The Host

Use a Python 3.10 environment with TurboVLA's LIBERO dependencies, then install
the export and cloud client dependencies:

```bash
pip install -e '.[libero]'
pip install 'transformers==4.56.*' onnx onnxruntime qai-hub
```

Do not proceed if installing `transformers` changes the pin. Review the source
tree before continuing; do not stage local model or deployment assets:

```bash
git status --short
python -m py_compile deployment/qcs8550/tools/*.py deployment/qcs8550/qnn_policy.py
```

## Phase 2: Export Static ONNX

Export the exact fixed-shape graph set. This command also checks that the
eager-attention split reference preserves the original normalized action:

```bash
python deployment/qcs8550/tools/extract_checkpoint_contract.py
python deployment/qcs8550/tools/export_static_onnx.py
```

Require all of the following before going on:

```text
deployment/qcs8550/artifacts/object/onnx/dinov3_one_view_fp32.onnx
deployment/qcs8550/artifacts/object/onnx/bert_l11_fp32.onnx
deployment/qcs8550/artifacts/object/onnx/bert_l14_fp32.onnx
deployment/qcs8550/artifacts/object/onnx/bert_l21_fp32.onnx
deployment/qcs8550/artifacts/object/onnx/policy_core_l21_fp32.onnx
deployment/qcs8550/artifacts/object/export_report.json
deployment/qcs8550/artifacts/object/reference_bundle_l11.npz
deployment/qcs8550/artifacts/checkpoint_contract.json
```

Read `export_report.json`; preserve it locally with the generated artifacts.
Any failed exact split-reference check is a hard stop. Do not work around it by
changing DINO feature selection, operators, shapes, tokenizer behavior, or
normalization.

## Phase 3: Compile And Download Contexts

This phase needs an authorized Qualcomm AI Hub account. The user must complete
interactive login; an agent must never handle the token.

```bash
qai-hub configure
python deployment/qcs8550/tools/submit_qai_hub_compile.py --qairt-version 2.48
python deployment/qcs8550/tools/download_contexts.py
```

Do not target another SoC or silently use a different QAIRT version. Verify the
following six files exist after every job reaches `SUCCESS`:

```text
deployment/qcs8550/artifacts/object/qcs8550_contexts/dinov3.bin
deployment/qcs8550/artifacts/object/qcs8550_contexts/bert_l11.bin
deployment/qcs8550/artifacts/object/qcs8550_contexts/bert_l14.bin
deployment/qcs8550/artifacts/object/qcs8550_contexts/bert_l21.bin
deployment/qcs8550/artifacts/object/qcs8550_contexts/policy_core.bin
deployment/qcs8550/artifacts/object/qcs8550_contexts/manifest.json
```

Keep the manifest with the `.bin` files. They are deployment assets, not Git
assets: do not add them to a commit or upload them without confirming QAIRT and
model-license redistribution terms.

## Phase 4: Install And Start The Board Service

Set the discovered values and copy only the generated context directory to the
isolated board root. The `rsync --delete` below is safe only because its target
is the dedicated `contexts/` directory; do not point it at a shared location.

```bash
export BOARD_HOST=<board-host>
export QAIRT_INCLUDE=<host-qairt-2.48-include>
export QAIRT_RUNTIME=<board-qairt-2.48-runtime-root>
export REMOTE_ROOT=/opt/turbovla-qcs8550

ssh -o BatchMode=yes "$BOARD_HOST" "mkdir -p '$REMOTE_ROOT/contexts'"
rsync -a --delete deployment/qcs8550/artifacts/object/qcs8550_contexts/ \
  "$BOARD_HOST:$REMOTE_ROOT/contexts/"

python deployment/qcs8550/tools/build_native_server.py \
  --host "$BOARD_HOST" \
  --remote-root "$REMOTE_ROOT" \
  --qairt-include "$QAIRT_INCLUDE" \
  --runtime "$QAIRT_RUNTIME" \
  --port 10092 \
  --start
```

The helper compiles the native service against the copied QAIRT headers and
starts it with the board runtime's aarch64 and Hexagon v73 library paths. It
does not create, alter, or stop any other service. If port `10092` is already
occupied, stop here and ask the user whether it belongs to this deployment or
choose a different unused port explicitly.

Check the board process and log without exposing secrets:

```bash
ssh -o BatchMode=yes "$BOARD_HOST" \
  "ss -ltn | grep ':10092 ' && tail -n 80 '$REMOTE_ROOT/logs/turbovla_qnn_server.log'"
```

## Phase 5: Native Replay And LIBERO Smoke Test

First send the frozen reference input repeatedly. The output must be stable
across warm requests. This checks persistent context loading and the wire
protocol; it does not imply FP32 equivalence.

```bash
python deployment/qcs8550/tools/native_client.py \
  --bundle deployment/qcs8550/artifacts/object/reference_bundle_l11.npz \
  --host "$BOARD_HOST" --port 10092 --requests 20
```

Then perform a one-trial-per-task LIBERO Object smoke test using the board
service. Set the environment variables so the host adapter connects to the
same service:

```bash
export TURBOVLA_QNN_HOST="$BOARD_HOST"
export TURBOVLA_QNN_PORT=10092
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="$PWD/deployment/qcs8550:$PWD:$PWD/third_party/vla_adapter"

python deployment/qcs8550/tools/run_libero_rollout.py \
  --ckpt_path pretrained/TurboVLA/checkpoints/libero/object.pth \
  --dinov3_path pretrained/dinov3-vitb16 \
  --bert_path pretrained/bert-base-uncased \
  --stats_path experiments/libero/configs/libero_all4_stats.json \
  --stats_key libero_all4_no_noops \
  --libero_root "$LIBERO_ROOT" \
  --task_suite_name libero_object \
  --num_trials_per_task 1
```

Report the service median/p95 latency, native replay stability, task-level
results, exact QAIRT version, and context manifest hashes. Do not describe a
one-trial smoke test as an upstream 50-trial evaluation. On the original
bring-up setup, board service P50/P95 was 60.1/67.6 ms and one episode for each
of the ten LIBERO Object tasks succeeded; those numbers are reference evidence,
not guaranteed results on another board or build.

## Failure Triage

| Symptom | Required action |
| --- | --- |
| Export split-reference check fails | Check the Transformers 4.56 pin and DINO `hidden_states[-1]`; do not modify the graph to force a pass. |
| AI Hub compile fails | Save the job URL/status and stop; do not compile with a different target or QAIRT version without approval. |
| Native service cannot load a context | Confirm context manifest target/version, board runtime, and Hexagon v73 libraries match. |
| Port is in use | Do not kill its owner. Ask the user or select an approved unused port. |
| Replay is unstable or rollout fails | Preserve local logs/results, report the first failing phase and manifest hashes, and do not claim deployment success. |
