# QCS8550 QNN Deployment

This experimental deployment targets the released **LIBERO Object** checkpoint
on Qualcomm QCS8550 HTP v73 with QAIRT 2.48. It keeps TurboVLA's two 256x256
DINOv3 views, 8-D normalized state, static BERT lengths (`11`, `14`, `21`),
and `[1,12,7]` normalized action chunk.

The graph partition is:

```text
host normalization + tokenizer/masks
  -> DINOv3 one-view context, executed twice
  -> BERT context selected by instruction length
  -> host zero-pad BERT hidden state to 21 tokens
  -> policy-core context
  -> host action denormalization
```

The board-native server loads these five contexts once and uses direct QNN API
execution, avoiding per-request process launch, context creation, and raw-file
I/O.

For an AI agent that must carry out the complete deployment, including explicit
validation and safety stop conditions, read [AGENT_DEPLOYMENT.md](AGENT_DEPLOYMENT.md).

## Important Compatibility Rule

The DINO wrapper deliberately uses `outputs.hidden_states[-1]`, not
`last_hidden_state`. This preserves the behavior documented in upstream issue
[#4](https://github.com/H-EmbodVis/TurboVLA/issues/4). Export and numerical
validation must use `transformers==4.56.*`; loading the checkpoint alone is
not sufficient to establish equivalent actions.

## Requirements

- Python 3.10 with the TurboVLA dependencies, `onnx`, `onnxruntime`, and
  `qai-hub` for export and cloud compilation.
- A QCS8550 board with a QAIRT 2.48 runtime and Hexagon v73 skeletons.
- The QAIRT 2.48 Linux development SDK on the host for QNN headers.
- The released Object checkpoint under `pretrained/TurboVLA` and local BERT
  assets under `pretrained/bert-base-uncased`.

Model weights and context binaries are intentionally not committed.

## Export And Compile

Export the fixed-shape ONNX graphs and validate their CPU ONNX Runtime outputs
against the pinned PyTorch reference:

```bash
python deployment/qcs8550/tools/export_static_onnx.py
```

Authenticate with Qualcomm AI Hub, compile for the target, then download the
context binaries:

```bash
qai-hub configure
python deployment/qcs8550/tools/submit_qai_hub_compile.py --qairt-version 2.48
python deployment/qcs8550/tools/download_contexts.py
```

The compile command uses `qnn_context_binary`, QAIRT 2.48, and native 64-bit
I/O. The resulting `artifacts/object/qcs8550_contexts` directory must contain
`dinov3.bin`, `bert_l11.bin`, `bert_l14.bin`, `bert_l21.bin`, `policy_core.bin`,
and `manifest.json`.

## Board Service

Copy the contexts to an isolated board root, then compile and start the native
server. The commands below use an SSH host alias and deliberately do not
manage any unrelated service or port.

```bash
export QAIRT_INCLUDE=/path/to/qairt-2.48/include
export QAIRT_RUNTIME=/path/on/board/qairt-2.48.0.260626-linux

rsync -a deployment/qcs8550/artifacts/object/qcs8550_contexts/ \
  qcs8550:/opt/turbovla-qcs8550/contexts/

python deployment/qcs8550/tools/build_native_server.py \
  --host qcs8550 \
  --remote-root /opt/turbovla-qcs8550 \
  --qairt-include "$QAIRT_INCLUDE" \
  --runtime "$QAIRT_RUNTIME" \
  --start
```

The server listens on port `10092` by default. It accepts a framed named-array
request containing two normalized pixel tensors, BERT inputs/masks, and state;
it returns one normalized action chunk plus timing JSON. `native_client.py`
contains the reference client implementation.

## LIBERO Rollout

Extract the instruction-length contract once from the released checkpoint:

```bash
python deployment/qcs8550/tools/extract_checkpoint_contract.py
```

The adapter performs no PyTorch model forward pass. It reproduces image
preprocessing, tokenizer special-token masks, state normalization, and action
decoding on the host, then sends the request to the board service.

```bash
export TURBOVLA_QNN_HOST=<board-ip-or-hostname>
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
  --libero_root /path/to/LIBERO \
  --task_suite_name libero_object --num_trials_per_task 1
```

## Numerical Validation

Do not assume HTP outputs are exactly FP32-equivalent. The native server must
first match the same compiled QNN context through `qnn-net-run`; then compare
full normalized actions and rollout success against the pinned reference.
On the contributor's QCS8550/QAIRT 2.48 setup, the HTP service matched board
replay exactly across 20 warm requests and completed a one-episode smoke for
each LIBERO Object task. This is a bring-up result, not a replacement for the
upstream 50-trial-per-task evaluation.
