# TurboVLA Exhaustive PyTorch Reference Baseline — One-Shot Implementation Plan

**Repository:** `kamusarj/TurboVLA`
**Audited branch:** `main`
**Audited commit:** `c999c05a4e0a2b62df03a11a6d99936798013371`
**Scope:** PyTorch reference baseline only.
**Excluded:** CUDA-kernel tracing, `vla.cpp`, C++, GGUF and quantization.

---

# 0. Final objective

Implement one authoritative PyTorch baseline command that performs the complete workflow in one execution:

```text
real LIBERO observation
→ capture raw fixture
→ derive exact model inputs
→ strict production-equivalent model load
→ exhaustive PyTorch forward trace
→ normalized and environment actions
→ second deterministic replay
→ tensor-by-tensor self-compare
→ manifest and fixture validation
→ weights manifest
→ final baseline report
→ packaged reference artifact
```

The final baseline must be usable later by any implementation, including `vla.cpp`, without changing the PyTorch reference format.

The implementation agent must not stop after intermediate phases. It must implement the complete plan, run the tests and end-to-end command, fix failures, and only finish when the required artifacts and reports exist or when a real external blocker is documented with exact evidence.

---

# 1. What “dump everything” means

Dump every **model-semantic tensor boundary and mathematical operation** involved in the selected PyTorch forward:

- preprocessing;
- tokenizer and masks;
- BERT embeddings and all 12 transformer blocks;
- DINOv3 embeddings and all 12 blocks for both views;
- vision projection;
- six bidirectional vision-language interaction layers;
- six text-enhancer layers;
- condition concatenation;
- state projection;
- three action-decoder layers;
- action MLP;
- `tanh`;
- action denormalization and executed action selection;
- model parameters and buffers.

Do not trace individual CUDA kernels.

CUDA kernels are backend implementation details that can be fused, split, reordered, or replaced by PyTorch/CUDA versions. They do not provide a stable semantic interface for a future implementation diff. The stable comparison level is the tensor output of mathematical operations such as:

```text
linear
matmul
reshape
transpose
LayerNorm
softmax
residual add
concat
activation
tanh
denormalization
```

---

# 2. Audit of the current repository

## 2.1. Current useful components

The repository already contains:

```text
turbovla/debug/
├── trace_config.py
├── trace_context.py
├── tensor_record.py
├── tensor_writer.py
├── fixture_writer.py
├── hook_manager.py
└── trace_utils.py

scripts/
├── export_parity_fixture.py
├── run_fixture_inference.py
├── compare_tensor_traces.py
├── validate_trace_manifest.py
└── inspect_trace.py
```

These form a useful skeleton and should be refactored rather than discarded.

## 2.2. Current critical problems

### Two dump systems exist

The model contains both:

```text
EmbeddingDumper
TraceContext + TensorWriter
```

They have different schemas and lifecycles. `TurboVLA.forward()` couples the parity tracer’s forward index to `EmbeddingDumper`.

For the authoritative baseline:

```text
TraceContext + TensorWriter = only authoritative writer
EmbeddingDumper = legacy-only
```

### Current fixture is synthetic and internally inconsistent

The current fixture exporter independently generates:

- random raw RGB images;
- random preprocessed pixel values;
- random token IDs;
- artificial masks;
- fixed raw state;
- zero normalized state.

It cannot validate the real preprocessing pipeline.

### Current fixture replay is not production-equivalent

The replay path can:

- run without a checkpoint;
- build a default configuration;
- load with `strict=False`;
- fall back to random tensors;
- use different device/precision behavior than `TurboVLAPolicy`.

### Tracer finishes too early

The model finishes tracing inside `forward()`. Environment-action denormalization happens afterward in the policy, so `action.denormalized` cannot be part of the same authoritative trace.

### Trace levels are not implemented as real coverage levels

`boundary`, `layer`, and `op` exist in configuration but do not reliably control instrumentation coverage.

### Native integer and mask types are lost

Token IDs, position IDs and masks are currently written as canonical float32 in the new trace format. Exact reference storage must retain native integer/boolean representation.

### Call index is not real

`call_index` is effectively fixed to zero, although the same semantic operation can execute multiple times.

### Existing tracing is not exhaustive

Current tracing is largely at block boundaries. Q/K/V, logits, softmax intermediates, LayerNorm internals, residual operands, per-decoder-layer outputs and many MLP internals are missing.

### Existing sample dump script is not authoritative

`scripts/dump_sample_trace.py` monkey-patches model loaders, builds configuration-only backbones, forces FP32 settings and uses the legacy dump path. Preserve it as legacy tooling only.

---

# 3. Required final command

Create:

```text
scripts/build_pytorch_reference_baseline.py
```

Required usage:

```bash
python scripts/build_pytorch_reference_baseline.py \
  --task-suite libero_object \
  --task-id 0 \
  --episode 0 \
  --step 0 \
  --seed 7 \
  --checkpoint pretrained/TurboVLA/checkpoints/libero/object.pth \
  --dinov3-path pretrained/dinov3/dinov3-vitb16 \
  --bert-path pretrained/bert-base-uncased \
  --stats-path pretrained/TurboVLA/libero_all4_stats.json \
  --stats-key libero_all4_no_noops \
  --libero-root "$HOME/Desktop/LIBERO" \
  --device cuda \
  --precision bf16 \
  --trace-level exhaustive \
  --output-root outputs/pytorch_reference_baselines \
  --baseline-id libero_object_task0_ep0_step0_bf16 \
  --overwrite
```

This single command must:

1. load the selected LIBERO task and deterministic initial state;
2. obtain the exact selected observation;
3. capture a raw fixture;
4. derive and store exact model inputs;
5. strictly load the exact model;
6. run a production-compatible exhaustive trace;
7. run deterministic replay A;
8. run deterministic replay B;
9. compare A and B;
10. validate the fixture;
11. validate every trace manifest;
12. dump a weights and buffers manifest;
13. generate reports;
14. package the final baseline;
15. return exit code 0 only when every required gate passes.

No hidden random fallback is allowed.

---

# 4. Final artifact structure

```text
outputs/pytorch_reference_baselines/
└── libero_object_task0_ep0_step0_bf16/
    ├── baseline_metadata.json
    ├── baseline_summary.md
    ├── baseline_status.json
    │
    ├── fixture/
    │   ├── metadata.json
    │   ├── instruction.txt
    │   ├── raw/
    │   │   ├── view_0_rgb_u8.npy
    │   │   ├── view_0_rgb_u8.bin
    │   │   ├── view_1_rgb_u8.npy
    │   │   ├── view_1_rgb_u8.bin
    │   │   ├── state_raw_f32.npy
    │   │   └── state_raw_f32le.bin
    │   └── exact_inputs/
    │       ├── pixel_values_f32.npy
    │       ├── pixel_values_f32le.bin
    │       ├── pixel_values_model_dtype.npy
    │       ├── pixel_values_bf16le.bin
    │       ├── input_ids_i64.npy
    │       ├── input_ids_i64le.bin
    │       ├── attention_mask_u8.npy
    │       ├── attention_mask_u8.bin
    │       ├── text_self_attention_mask_u8.npy
    │       ├── text_self_attention_mask_u8.bin
    │       ├── position_ids_i64.npy
    │       ├── position_ids_i64le.bin
    │       ├── state_normalized_f32.npy
    │       ├── state_normalized_f32le.bin
    │       ├── state_model_dtype.npy
    │       └── state_bf16le.bin
    │
    ├── weights/
    │   ├── weights_manifest.jsonl
    │   ├── buffers_manifest.jsonl
    │   └── tensors/
    │
    ├── traces/
    │   ├── production_bf16/
    │   │   └── forward_000000_rank_0/
    │   ├── deterministic_bf16_run_a/
    │   │   └── forward_000000_rank_0/
    │   └── deterministic_bf16_run_b/
    │       └── forward_000000_rank_0/
    │
    ├── actions/
    │   ├── production_normalized_action.npy
    │   ├── production_env_action.npy
    │   ├── deterministic_a_normalized_action.npy
    │   ├── deterministic_a_env_action.npy
    │   ├── deterministic_b_normalized_action.npy
    │   └── deterministic_b_env_action.npy
    │
    ├── validation/
    │   ├── fixture_validation.json
    │   ├── production_manifest_validation.json
    │   ├── deterministic_a_manifest_validation.json
    │   ├── deterministic_b_manifest_validation.json
    │   ├── policy_vs_exact_replay.json
    │   ├── deterministic_replay_compare.csv
    │   ├── deterministic_replay_compare.json
    │   ├── first_divergence.txt
    │   ├── stage_summary.csv
    │   └── coverage_report.json
    │
    ├── logs/
    │   ├── model_load.log
    │   ├── baseline_build.log
    │   └── tests.log
    │
    └── libero_object_task0_ep0_step0_bf16.tar.zst
```

Do not commit this runtime output to Git.

---

# 5. Repository architecture changes

## 5.1. Add a shared strict inference loader

Create:

```text
turbovla/evaluation/model_loader.py
```

Suggested API:

```python
@dataclass
class LoadedTurboVLA:
    model: TurboVLA
    config: TurboVLAConfig
    checkpoint_sha256: str
    config_sha256: str
    device: torch.device
    model_dtype: torch.dtype


def load_turbovla_for_inference(
    *,
    checkpoint_path: Path,
    dinov3_path: str,
    bert_path: str,
    device: str | torch.device,
    precision: Literal["bf16", "fp32"],
    strict: bool = True,
    deterministic: bool = False,
) -> LoadedTurboVLA:
    ...
```

Required behavior:

- checkpoint must exist;
- checkpoint must contain the expected model state;
- checkpoint configuration must be used;
- DINO and BERT paths must be explicit;
- load state dict with `strict=True`;
- report all keys and tensor count;
- set exact model precision;
- move to exact device;
- call `eval()`;
- call `requires_grad_(False)`;
- verify every floating parameter dtype;
- hash checkpoint and normalized model configuration;
- no default random model;
- no `strict=False`;
- no silent backend fallback in deterministic reference mode.

Use this loader from:

```text
turbovla/evaluation/policy.py
scripts/run_fixture_inference.py
scripts/build_pytorch_reference_baseline.py
integration tests
```

## 5.2. Make `EmbeddingDumper` legacy-only

Keep:

```text
turbovla/models/embedding_dump.py
```

for legacy inspection, but authoritative baseline code must not activate it.

Add a deprecation warning when used for parity output.

Rename:

```text
scripts/dump_sample_trace.py
→ scripts/legacy_dump_sample_trace.py
```

The model must no longer use `EmbeddingDumper.forward_index` to drive `TraceContext`.

## 5.3. Move trace lifecycle outside `TurboVLA.forward()`

Remove tracer session ownership from:

```text
turbovla/models/turbovla.py
```

Runner/policy owns:

```python
tracer.begin(...)

try:
    normalized_action = model(...)
    env_action = convert_action(normalized_action)

    tracer.tensor("action.normalized", ...)
    tracer.tensor("action.denormalized", ...)
    tracer.tensor("action.first_step", ...)
    tracer.tensor("action.executed_steps", ...)
finally:
    tracer.finish()
```

Model code only emits trace records when an active tracer is attached.

## 5.4. Add fixture reader and validator

Create:

```text
turbovla/debug/fixture_reader.py
turbovla/debug/fixture_validator.py
```

The validator must check:

- every listed file exists;
- shape and dtype match metadata;
- file byte length is exact;
- SHA256 matches;
- image preprocessing reproduces stored pixel values;
- tokenizer reproduces all stored token tensors;
- state normalization reproduces stored normalized state;
- no stale files exist outside the metadata file list;
- fixture ID and model metadata match the requested build.

## 5.5. Add deterministic context

Create:

```text
turbovla/debug/deterministic.py
```

Context manager must:

- save current backend settings;
- set fixed Python/NumPy/Torch/CUDA seeds;
- disable TF32;
- disable cuDNN benchmark;
- enable deterministic algorithms with explicit warning handling;
- use eager/manual attention where required for full operation tracing;
- restore settings after use.

Keep two profiles:

```text
production_bf16
deterministic_bf16
```

Production trace reflects the real runtime path. Deterministic traces A/B prove reproducibility.

---

# 6. Trace data model

## 6.1. Replace `layer/op` with four explicit levels

```text
boundary
layer
op
exhaustive
```

Order:

```python
TRACE_LEVEL_ORDER = {
    "boundary": 0,
    "layer": 1,
    "op": 2,
    "exhaustive": 3,
}
```

The final baseline command always uses:

```text
exhaustive
```

Other levels remain for development and tests.

Every trace call declares:

```python
required_level="boundary" | "layer" | "op" | "exhaustive"
```

The level check must occur before copying the tensor to CPU.

## 6.2. Semantic identity

A tensor is uniquely identified by:

```text
semantic_name + call_index
```

`trace_id` records execution order.

Call index must be generated by `TraceContext`, not hard-coded.

## 6.3. Required manifest record

```json
{
  "trace_id": 42,
  "semantic_name": "interaction.layer_03.cross.visual_to_text.probs",
  "call_index": 0,
  "module_path": "vision_language_interaction.fusion_layers.3.attn",
  "operation": "softmax",
  "io": "intermediate",
  "stage": "interaction",
  "required_level": "op",

  "shape": [1, 4, 512, 21],
  "layout": "B,H,V,L",
  "strides": [43008, 10752, 21, 1],
  "source_dtype": "torch.bfloat16",
  "storage_dtype": "float32",
  "native_storage_dtype": "bfloat16",
  "endianness": "little",
  "contiguous": true,
  "numel": 43008,

  "file": "tensors/40_interaction/layer_03/...f32le.bin",
  "native_file": "tensors_native/40_interaction/layer_03/...bf16le.bin",

  "min": 0.0,
  "max": 0.71,
  "mean": 0.047619,
  "std": 0.062,
  "abs_max": 0.71,
  "l1_norm": 2048.0,
  "l2_norm": 15.4,
  "zero_count": 0,
  "zero_ratio": 0.0,
  "nan_count": 0,
  "inf_count": 0,

  "first_values": [0.01, 0.03],
  "last_values": [0.02, 0.01],

  "sha256_f32": "...",
  "sha256_native": "..."
}
```

## 6.4. Native storage rules

Floating tensors:

```text
canonical file: float32 little-endian
native file: BF16/F16 raw bits when source uses that dtype
```

Integer IDs:

```text
int64 little-endian
```

Masks:

```text
uint8
```

Do not use float32 as the canonical file for integer/mask tensors.

## 6.5. File naming

```text
<trace_id>__<semantic_name>__call_<call_index>.<dtype>.bin
```

Example:

```text
001942__vision.view_1.block_07.attn.probs__call_00.f32le.bin
```

Sanitize only filesystem-invalid characters. Preserve the exact semantic name in the manifest.

## 6.6. CSV correctness

Use `csv.DictWriter`.

Do not construct CSV rows by joining strings, because layout fields contain commas.

## 6.7. Storage policy

For the final exhaustive trace:

```text
save canonical binary: yes
save native binary: yes
save .pt per tensor: no
save .npy per tensor: no
save preview/stats/hash: yes
```

`.pt`/`.npy` are allowed only for:

- fixture inputs;
- final actions;
- explicitly selected debug tensors.

## 6.8. Atomic output

Write each fixture/trace into:

```text
<target>.tmp.<pid>
```

Validate, then atomically rename to the final directory.

`--overwrite` must remove or replace all old files atomically. No stale file may survive.

## 6.9. Byte budget

Implement preflight estimation and runtime limit.

The exhaustive baseline command must:

1. estimate required free space;
2. check filesystem free space;
3. refuse to run if estimated space plus safety margin is unavailable;
4. record estimated and actual bytes in `baseline_status.json`.

A trace must never silently truncate. If a configured byte limit is reached, the command fails and marks the baseline incomplete.

---

# 7. Exhaustive tensor coverage

# 7.1. Raw input and preprocessing

Dump:

```text
input.raw.view_0_rgb_u8
input.raw.view_1_rgb_u8
input.rotated.view_0_rgb_u8
input.rotated.view_1_rgb_u8

input.rescale.factor
input.rescaled.view_0
input.rescaled.view_1

input.normalize.mean
input.normalize.std
input.normalized.view_0
input.normalized.view_1

input.pixel_values_f32
input.pixel_values_model_dtype

state.raw
state.normalize.mean
state.normalize.std
state.centered
state.normalized_f32
state.normalized_model_dtype
```

Capture the actual LIBERO rotation and exact DINO preprocessor configuration.

# 7.2. Tokenizer and text layout

Dump:

```text
text.instruction_utf8_bytes
text.tokenizer.input_ids_unpadded
text.tokenizer.token_type_ids_unpadded
text.tokenizer.attention_mask_unpadded
text.tokenizer.self_attention_mask_unpadded
text.tokenizer.position_ids_unpadded

text.layout.input_ids_padded
text.layout.attention_mask_padded
text.layout.key_padding_mask_padded
text.layout.self_attention_mask_padded
text.layout.position_ids_padded
```

Metadata must include:

```text
tokenizer class
tokenizer files and hashes
padding side
pad token ID
CLS/SEP IDs
unpadded length
model text length
instruction-specific padding rule
```

# 7.3. BERT embeddings

Dump:

```text
text.bert.embeddings.input_ids
text.bert.embeddings.word
text.bert.embeddings.token_type
text.bert.embeddings.position
text.bert.embeddings.sum_before_norm
text.bert.embeddings.norm.mean
text.bert.embeddings.norm.variance
text.bert.embeddings.norm.inv_std
text.bert.embeddings.norm.normalized
text.bert.embeddings.norm.weight
text.bert.embeddings.norm.bias
text.bert.embeddings.output
```

# 7.4. Every BERT block 00–11

For each block:

```text
text.bert.layer_XX.input

text.bert.layer_XX.attn.q_linear.matmul
text.bert.layer_XX.attn.q_linear.bias_add
text.bert.layer_XX.attn.k_linear.matmul
text.bert.layer_XX.attn.k_linear.bias_add
text.bert.layer_XX.attn.v_linear.matmul
text.bert.layer_XX.attn.v_linear.bias_add

text.bert.layer_XX.attn.q.reshape
text.bert.layer_XX.attn.q.transpose
text.bert.layer_XX.attn.k.reshape
text.bert.layer_XX.attn.k.transpose
text.bert.layer_XX.attn.v.reshape
text.bert.layer_XX.attn.v.transpose

text.bert.layer_XX.attn.k_transposed
text.bert.layer_XX.attn.qk_matmul
text.bert.layer_XX.attn.scale
text.bert.layer_XX.attn.scaled_logits
text.bert.layer_XX.attn.mask
text.bert.layer_XX.attn.masked_logits

text.bert.layer_XX.attn.softmax.max
text.bert.layer_XX.attn.softmax.shifted
text.bert.layer_XX.attn.softmax.exp
text.bert.layer_XX.attn.softmax.sum
text.bert.layer_XX.attn.softmax.probs

text.bert.layer_XX.attn.context_heads
text.bert.layer_XX.attn.context_transpose
text.bert.layer_XX.attn.context_merged

text.bert.layer_XX.attn.output.matmul
text.bert.layer_XX.attn.output.bias_add
text.bert.layer_XX.attn.output

text.bert.layer_XX.residual_1.left
text.bert.layer_XX.residual_1.right
text.bert.layer_XX.residual_1.sum

text.bert.layer_XX.norm_1.input
text.bert.layer_XX.norm_1.mean
text.bert.layer_XX.norm_1.variance
text.bert.layer_XX.norm_1.inv_std
text.bert.layer_XX.norm_1.normalized
text.bert.layer_XX.norm_1.output

text.bert.layer_XX.ffn.linear_1.matmul
text.bert.layer_XX.ffn.linear_1.bias_add
text.bert.layer_XX.ffn.gelu.input
text.bert.layer_XX.ffn.gelu.output
text.bert.layer_XX.ffn.linear_2.matmul
text.bert.layer_XX.ffn.linear_2.bias_add

text.bert.layer_XX.residual_2.left
text.bert.layer_XX.residual_2.right
text.bert.layer_XX.residual_2.sum

text.bert.layer_XX.norm_2.input
text.bert.layer_XX.norm_2.mean
text.bert.layer_XX.norm_2.variance
text.bert.layer_XX.norm_2.inv_std
text.bert.layer_XX.norm_2.normalized
text.bert.layer_XX.norm_2.output

text.bert.layer_XX.output
```

Do not rely only on module hooks. Instrument or wrap the actual BERT forward path while preserving numerical behavior.

# 7.5. BERT output projection

Dump:

```text
text.bert.last_hidden_state_unpadded
text.bert.hidden_padded
text.projection.input
text.projection.matmul
text.projection.bias_add
text.projection.before_zero_fill
text.projection.zero_fill_mask
text.projection.output
```

# 7.6. DINOv3 preprocessing boundary per view

```text
vision.view_0.input
vision.view_1.input
```

# 7.7. DINOv3 embeddings per view

For each view:

```text
vision.view_X.patch_embed.input
vision.view_X.patch_embed.conv.weight
vision.view_X.patch_embed.conv.bias
vision.view_X.patch_embed.conv.output
vision.view_X.patch_embed.flatten
vision.view_X.patch_embed.transpose

vision.view_X.cls_token
vision.view_X.register_tokens
vision.view_X.position_embedding
vision.view_X.tokens_before_position
vision.view_X.tokens_after_position
vision.view_X.tokens_with_prefix
```

# 7.8. Every DINOv3 block 00–11 for both views

For each view and block:

```text
vision.view_X.block_XX.input

vision.view_X.block_XX.norm_1.input
vision.view_X.block_XX.norm_1.mean
vision.view_X.block_XX.norm_1.variance
vision.view_X.block_XX.norm_1.inv_std
vision.view_X.block_XX.norm_1.output

vision.view_X.block_XX.attn.qkv_linear.matmul
vision.view_X.block_XX.attn.qkv_linear.bias_add
vision.view_X.block_XX.attn.q
vision.view_X.block_XX.attn.k
vision.view_X.block_XX.attn.v
vision.view_X.block_XX.attn.q_heads
vision.view_X.block_XX.attn.k_heads
vision.view_X.block_XX.attn.v_heads
vision.view_X.block_XX.attn.logits
vision.view_X.block_XX.attn.scale
vision.view_X.block_XX.attn.scaled_logits
vision.view_X.block_XX.attn.softmax.max
vision.view_X.block_XX.attn.softmax.shifted
vision.view_X.block_XX.attn.softmax.exp
vision.view_X.block_XX.attn.softmax.sum
vision.view_X.block_XX.attn.probs
vision.view_X.block_XX.attn.context_heads
vision.view_X.block_XX.attn.context_merged
vision.view_X.block_XX.attn.output_projection

vision.view_X.block_XX.layer_scale_1.input
vision.view_X.block_XX.layer_scale_1.output
vision.view_X.block_XX.residual_1.left
vision.view_X.block_XX.residual_1.right
vision.view_X.block_XX.residual_1.sum

vision.view_X.block_XX.norm_2.input
vision.view_X.block_XX.norm_2.mean
vision.view_X.block_XX.norm_2.variance
vision.view_X.block_XX.norm_2.inv_std
vision.view_X.block_XX.norm_2.output

vision.view_X.block_XX.mlp.linear_1.matmul
vision.view_X.block_XX.mlp.linear_1.bias_add
vision.view_X.block_XX.mlp.activation.input
vision.view_X.block_XX.mlp.activation.output
vision.view_X.block_XX.mlp.linear_2.matmul
vision.view_X.block_XX.mlp.linear_2.bias_add

vision.view_X.block_XX.layer_scale_2.input
vision.view_X.block_XX.layer_scale_2.output
vision.view_X.block_XX.residual_2.left
vision.view_X.block_XX.residual_2.right
vision.view_X.block_XX.residual_2.sum

vision.view_X.block_XX.output
```

# 7.9. DINO prefix removal and view stacking

```text
vision.view_0.tokens_with_prefix_final
vision.view_0.prefix_tokens
vision.view_0.patch_tokens

vision.view_1.tokens_with_prefix_final
vision.view_1.prefix_tokens
vision.view_1.patch_tokens

vision.patch_tokens_stacked
```

Record prefix token count and patch ordering.

# 7.10. Vision projection

Refactor the one-line expression into numerically equivalent named operations:

```text
vision_projection.input
vision_projection.input_norm.input
vision_projection.input_norm.mean
vision_projection.input_norm.variance
vision_projection.input_norm.inv_std
vision_projection.input_norm.normalized
vision_projection.input_norm.output

vision_projection.skip.matmul
vision_projection.skip.output

vision_projection.mlp.linear_1.matmul
vision_projection.mlp.linear_1.bias_add
vision_projection.mlp.gelu.input
vision_projection.mlp.gelu.output
vision_projection.mlp.linear_2.matmul
vision_projection.mlp.linear_2.bias_add

vision_projection.residual.left
vision_projection.residual.right
vision_projection.residual.sum

vision_projection.output_norm.input
vision_projection.output_norm.mean
vision_projection.output_norm.variance
vision_projection.output_norm.inv_std
vision_projection.output_norm.normalized
vision_projection.output_norm.output

vision_projection.view_embedding
vision_projection.position.before_add
vision_projection.position.after_add

vision_projection.before_flatten
vision_projection.flattened
```

# 7.11. Six bidirectional fusion layers

For every `interaction.layer_00` through `interaction.layer_05`:

```text
interaction.layer_XX.visual.input
interaction.layer_XX.text.input

interaction.layer_XX.visual.norm.*
interaction.layer_XX.text.norm.*

interaction.layer_XX.cross.q_visual.matmul
interaction.layer_XX.cross.q_visual.bias_add
interaction.layer_XX.cross.k_text.matmul
interaction.layer_XX.cross.k_text.bias_add
interaction.layer_XX.cross.v_visual.matmul
interaction.layer_XX.cross.v_visual.bias_add
interaction.layer_XX.cross.v_text.matmul
interaction.layer_XX.cross.v_text.bias_add

interaction.layer_XX.cross.q_visual.reshape
interaction.layer_XX.cross.q_visual.transpose
interaction.layer_XX.cross.k_text.reshape
interaction.layer_XX.cross.k_text.transpose
interaction.layer_XX.cross.v_visual.reshape
interaction.layer_XX.cross.v_visual.transpose
interaction.layer_XX.cross.v_text.reshape
interaction.layer_XX.cross.v_text.transpose
```

Visual-to-text direction:

```text
interaction.layer_XX.cross.visual_to_text.qk_matmul
interaction.layer_XX.cross.visual_to_text.logits_raw
interaction.layer_XX.cross.visual_to_text.global_max
interaction.layer_XX.cross.visual_to_text.logits_global_shifted
interaction.layer_XX.cross.visual_to_text.logits_clamped
interaction.layer_XX.cross.visual_to_text.mask
interaction.layer_XX.cross.visual_to_text.masked_logits

interaction.layer_XX.cross.visual_to_text.softmax.max
interaction.layer_XX.cross.visual_to_text.softmax.shifted
interaction.layer_XX.cross.visual_to_text.softmax.exp
interaction.layer_XX.cross.visual_to_text.softmax.sum
interaction.layer_XX.cross.visual_to_text.probs

interaction.layer_XX.cross.visual_to_text.context_heads
interaction.layer_XX.cross.visual_to_text.context_transpose
interaction.layer_XX.cross.visual_to_text.context_merged
interaction.layer_XX.cross.visual_to_text.output.matmul
interaction.layer_XX.cross.visual_to_text.output.bias_add
interaction.layer_XX.cross.visual_to_text.output
```

Text-to-visual direction:

```text
interaction.layer_XX.cross.text_to_visual.logits_transposed
interaction.layer_XX.cross.text_to_visual.row_max
interaction.layer_XX.cross.text_to_visual.logits_shifted
interaction.layer_XX.cross.text_to_visual.logits_clamped
interaction.layer_XX.cross.text_to_visual.mask
interaction.layer_XX.cross.text_to_visual.masked_logits

interaction.layer_XX.cross.text_to_visual.softmax.max
interaction.layer_XX.cross.text_to_visual.softmax.shifted
interaction.layer_XX.cross.text_to_visual.softmax.exp
interaction.layer_XX.cross.text_to_visual.softmax.sum
interaction.layer_XX.cross.text_to_visual.probs

interaction.layer_XX.cross.text_to_visual.context_heads
interaction.layer_XX.cross.text_to_visual.context_transpose
interaction.layer_XX.cross.text_to_visual.context_merged
interaction.layer_XX.cross.text_to_visual.output.matmul
interaction.layer_XX.cross.text_to_visual.output.bias_add
interaction.layer_XX.cross.text_to_visual.output
```

Residuals:

```text
interaction.layer_XX.visual.residual_source
interaction.layer_XX.visual.gamma
interaction.layer_XX.visual.delta
interaction.layer_XX.visual.scaled_delta
interaction.layer_XX.visual.drop_path_output
interaction.layer_XX.visual.residual_sum
interaction.layer_XX.visual.after_fusion

interaction.layer_XX.text.residual_source
interaction.layer_XX.text.gamma
interaction.layer_XX.text.delta
interaction.layer_XX.text.scaled_delta
interaction.layer_XX.text.drop_path_output
interaction.layer_XX.text.residual_sum
interaction.layer_XX.text.after_fusion
```

Preserve the repository’s exact stabilization, clamp, mask and residual behavior. Do not “simplify” it during tracing.

# 7.12. Six text-enhancer layers

For every interaction layer:

```text
interaction.layer_XX.text_enhancer.input
interaction.layer_XX.text_enhancer.mask.original
interaction.layer_XX.text_enhancer.mask.repeated

interaction.layer_XX.text_enhancer.attn.q_linear
interaction.layer_XX.text_enhancer.attn.k_linear
interaction.layer_XX.text_enhancer.attn.v_linear
interaction.layer_XX.text_enhancer.attn.q_heads
interaction.layer_XX.text_enhancer.attn.k_heads
interaction.layer_XX.text_enhancer.attn.v_heads
interaction.layer_XX.text_enhancer.attn.logits
interaction.layer_XX.text_enhancer.attn.masked_logits
interaction.layer_XX.text_enhancer.attn.softmax.max
interaction.layer_XX.text_enhancer.attn.softmax.shifted
interaction.layer_XX.text_enhancer.attn.softmax.exp
interaction.layer_XX.text_enhancer.attn.softmax.sum
interaction.layer_XX.text_enhancer.attn.probs
interaction.layer_XX.text_enhancer.attn.context_heads
interaction.layer_XX.text_enhancer.attn.context_merged
interaction.layer_XX.text_enhancer.attn.output_projection

interaction.layer_XX.text_enhancer.residual_1.left
interaction.layer_XX.text_enhancer.residual_1.right
interaction.layer_XX.text_enhancer.residual_1.sum
interaction.layer_XX.text_enhancer.norm_1.*

interaction.layer_XX.text_enhancer.ffn.linear_1.matmul
interaction.layer_XX.text_enhancer.ffn.linear_1.bias_add
interaction.layer_XX.text_enhancer.ffn.activation.input
interaction.layer_XX.text_enhancer.ffn.activation.output
interaction.layer_XX.text_enhancer.ffn.linear_2.matmul
interaction.layer_XX.text_enhancer.ffn.linear_2.bias_add

interaction.layer_XX.text_enhancer.residual_2.left
interaction.layer_XX.text_enhancer.residual_2.right
interaction.layer_XX.text_enhancer.residual_2.sum
interaction.layer_XX.text_enhancer.norm_2.*
interaction.layer_XX.text_enhancer.output
```

Fix the existing `src_mask is None` guard without changing the existing attention semantics.

# 7.13. Condition

```text
condition.visual_tokens
condition.text_tokens
condition.concat_axis
condition.concatenated
```

# 7.14. State projection

```text
state_projection.input_raw
state_projection.input_normalized

state_projection.norm.input
state_projection.norm.mean
state_projection.norm.variance
state_projection.norm.inv_std
state_projection.norm.normalized
state_projection.norm.output

state_projection.linear_1.matmul
state_projection.linear_1.bias_add
state_projection.gelu.input
state_projection.gelu.output
state_projection.linear_2.matmul
state_projection.linear_2.bias_add

state_projection.before_reshape
state_projection.after_reshape
state_projection.position_embedding
state_projection.position.before_add
state_projection.position.after_add

state_projection.output_norm.*
state_projection.output
```

# 7.15. Action memory and query construction

```text
action.memory.condition
action.memory.state_tokens
action.memory.concat_axis
action.memory.concatenated

action.queries.weight
action.queries.expanded
```

# 7.16. Three action-decoder layers

For layer 00–02:

```text
action.decoder.layer_XX.input
```

Self-attention:

```text
action.decoder.layer_XX.self_attn.norm_input
action.decoder.layer_XX.self_attn.norm.*
action.decoder.layer_XX.self_attn.q_linear
action.decoder.layer_XX.self_attn.k_linear
action.decoder.layer_XX.self_attn.v_linear
action.decoder.layer_XX.self_attn.q_heads
action.decoder.layer_XX.self_attn.k_heads
action.decoder.layer_XX.self_attn.v_heads
action.decoder.layer_XX.self_attn.logits
action.decoder.layer_XX.self_attn.mask
action.decoder.layer_XX.self_attn.masked_logits
action.decoder.layer_XX.self_attn.softmax.max
action.decoder.layer_XX.self_attn.softmax.shifted
action.decoder.layer_XX.self_attn.softmax.exp
action.decoder.layer_XX.self_attn.softmax.sum
action.decoder.layer_XX.self_attn.probs
action.decoder.layer_XX.self_attn.context_heads
action.decoder.layer_XX.self_attn.context_merged
action.decoder.layer_XX.self_attn.output_projection
action.decoder.layer_XX.self_attn.residual.left
action.decoder.layer_XX.self_attn.residual.right
action.decoder.layer_XX.self_attn.residual.sum
```

Cross-attention:

```text
action.decoder.layer_XX.cross_attn.norm_input
action.decoder.layer_XX.cross_attn.norm.*
action.decoder.layer_XX.cross_attn.q_linear
action.decoder.layer_XX.cross_attn.k_linear
action.decoder.layer_XX.cross_attn.v_linear
action.decoder.layer_XX.cross_attn.q_heads
action.decoder.layer_XX.cross_attn.k_heads
action.decoder.layer_XX.cross_attn.v_heads
action.decoder.layer_XX.cross_attn.logits
action.decoder.layer_XX.cross_attn.mask
action.decoder.layer_XX.cross_attn.masked_logits
action.decoder.layer_XX.cross_attn.softmax.max
action.decoder.layer_XX.cross_attn.softmax.shifted
action.decoder.layer_XX.cross_attn.softmax.exp
action.decoder.layer_XX.cross_attn.softmax.sum
action.decoder.layer_XX.cross_attn.probs
action.decoder.layer_XX.cross_attn.context_heads
action.decoder.layer_XX.cross_attn.context_merged
action.decoder.layer_XX.cross_attn.output_projection
action.decoder.layer_XX.cross_attn.residual.left
action.decoder.layer_XX.cross_attn.residual.right
action.decoder.layer_XX.cross_attn.residual.sum
```

FFN:

```text
action.decoder.layer_XX.ffn.norm_input
action.decoder.layer_XX.ffn.norm.*
action.decoder.layer_XX.ffn.linear_1.matmul
action.decoder.layer_XX.ffn.linear_1.bias_add
action.decoder.layer_XX.ffn.activation.input
action.decoder.layer_XX.ffn.activation.output
action.decoder.layer_XX.ffn.linear_2.matmul
action.decoder.layer_XX.ffn.linear_2.bias_add
action.decoder.layer_XX.ffn.residual.left
action.decoder.layer_XX.ffn.residual.right
action.decoder.layer_XX.ffn.residual.sum

action.decoder.layer_XX.output
```

The traced implementation must be numerically tested against the original stock `nn.TransformerDecoderLayer` path before becoming authoritative.

# 7.17. Action MLP and output

```text
action.mlp.input

action.mlp.linear_0.matmul
action.mlp.linear_0.bias_add
action.mlp.relu_0.input
action.mlp.relu_0.output

action.mlp.linear_1.matmul
action.mlp.linear_1.bias_add
action.mlp.relu_1.input
action.mlp.relu_1.output

action.mlp.linear_2.matmul
action.mlp.linear_2.bias_add

action.before_tanh
action.normalized
```

Do not decompose `tanh` into exponential kernels unless the model explicitly implements it that way. The semantic reference is:

```text
before_tanh
after_tanh
```

# 7.18. Environment-action postprocessing

```text
action.denormalize.action_min
action.denormalize.action_max
action.denormalize.input
action.denormalize.plus_one
action.denormalize.range
action.denormalize.scaled
action.denormalize.arm_output

action.gripper.source
action.gripper.deadband
action.gripper.positive_mask
action.gripper.negative_mask
action.gripper.output

action.denormalized
action.first_step
action.executed_steps
```

---

# 8. Weights and buffers baseline

Dump every model parameter and persistent buffer once.

Required fields:

```text
semantic name
checkpoint key
module path
parameter/buffer kind
shape
layout
source dtype
native file
canonical F32 file
SHA256 native
SHA256 F32
min
max
mean
std
numel
```

Explicit layout conventions:

```text
Linear weight: OUT,IN
Conv2d weight: OUT,IN,KH,KW
Embedding weight: VOCAB,D
LayerNorm weight/bias: D
attention packed QKV: document exact packing
```

Include:

- BERT weights;
- DINO weights;
- fusion projections;
- gamma parameters;
- view/position embeddings;
- state projection;
- action queries;
- action decoder;
- action MLP.

---

# 9. Comparison and self-validation

# 9.1. Fix compare ordering

Match tensors by:

```text
semantic_name + call_index
```

Determine first divergence by:

```text
reference trace_id execution order
```

Do not sort alphabetically.

# 9.2. Tolerance semantics

Use:

```python
np.allclose(candidate, reference, rtol=rtol, atol=atol, equal_nan=False)
```

Still report:

```text
exact match
max absolute error
mean absolute error
median absolute error
RMSE
max relative error
mean relative error
cosine similarity
Pearson correlation
first bad flat index
first bad coordinate
reference value
candidate value
```

# 9.3. Required self-comparisons

### Production policy vs exact-input replay

Compare:

```text
normalized action
environment action
first executed action
```

### Deterministic run A vs deterministic run B

Compare every trace tensor.

Expected:

```text
PASS_EXACT
```

If an operation is not bitwise deterministic on the chosen backend, it must:

1. be documented;
2. use a declared per-stage tolerance;
3. still pass the numerical gate;
4. not contain unexplained divergence.

### Deliberate perturbation test

Create a copy of a tiny test trace, perturb one tensor, and prove that the compare script reports the correct first divergence and coordinate.

---

# 10. Coverage validation

Create:

```text
turbovla/debug/coverage_validator.py
```

The baseline must not rely on “some tensors were dumped.”

Define a machine-readable expected semantic coverage specification:

```text
turbovla/debug/exhaustive_coverage_spec.py
```

The validator expands expected names based on actual config:

```text
BERT layer count
DINO layer count
view count
interaction layer count
action decoder layer count
```

`coverage_report.json` must contain:

```json
{
  "expected_count": 0,
  "actual_count": 0,
  "missing": [],
  "unexpected": [],
  "duplicate_semantic_keys": [],
  "status": "PASS"
}
```

The one-shot command fails if any required tensor is missing.

This is essential: the agent cannot claim “exhaustive” based only on a successful forward.

---

# 11. Tests required before the one-shot build may pass

Add:

```text
tests/debug/test_real_fixture_consistency.py
tests/debug/test_fixture_hashes.py
tests/debug/test_atomic_fixture_overwrite.py
tests/debug/test_strict_model_loader.py
tests/debug/test_no_random_fallback.py
tests/debug/test_policy_vs_exact_replay.py
tests/debug/test_trace_disabled_output_unchanged.py
tests/debug/test_trace_levels.py
tests/debug/test_trace_call_index.py
tests/debug/test_native_integer_storage.py
tests/debug/test_native_bf16_storage.py
tests/debug/test_trace_max_bytes.py
tests/debug/test_trace_lifecycle_postprocessing.py
tests/debug/test_compare_execution_order.py
tests/debug/test_compare_allclose_semantics.py
tests/debug/test_compare_deliberate_perturbation.py
tests/debug/test_exhaustive_coverage_spec.py
tests/debug/test_vision_projection_traced_equivalence.py
tests/debug/test_fusion_traced_equivalence.py
tests/debug/test_text_enhancer_traced_equivalence.py
tests/debug/test_state_projection_traced_equivalence.py
tests/debug/test_action_decoder_traced_equivalence.py
tests/debug/test_action_mlp_traced_equivalence.py
```

For traced refactors, test:

```text
original output
vs
traced refactored output with tracing disabled
```

The output must be exact or pass a documented tiny tolerance before the refactor is accepted.

---

# 12. Required status gates

`baseline_status.json`:

```json
{
  "baseline_id": "...",
  "complete": true,

  "gates": {
    "real_fixture_captured": true,
    "fixture_validation_pass": true,
    "strict_checkpoint_load_pass": true,
    "policy_vs_exact_normalized_pass": true,
    "policy_vs_exact_env_action_pass": true,
    "production_trace_manifest_pass": true,
    "deterministic_a_manifest_pass": true,
    "deterministic_b_manifest_pass": true,
    "deterministic_trace_compare_pass": true,
    "coverage_pass": true,
    "weights_manifest_pass": true,
    "no_nan_inf": true,
    "tests_pass": true,
    "archive_created": true
  },

  "metrics": {
    "policy_vs_exact_normalized_max_abs": 0.0,
    "policy_vs_exact_env_action_max_abs": 0.0,
    "deterministic_trace_failures": 0,
    "trace_tensor_count": 0,
    "weights_tensor_count": 0,
    "total_bytes": 0
  }
}
```

The command exits non-zero when any gate is false.

---

# 13. Final human-readable report

Generate:

```text
baseline_summary.md
```

Required sections:

1. baseline identity;
2. repository commit;
3. model/checkpoint hashes;
4. environment;
5. fixture source;
6. preprocessing details;
7. model configuration;
8. tensor coverage count by stage;
9. trace size by stage;
10. deterministic replay result;
11. policy-vs-exact replay result;
12. NaN/Inf summary;
13. weights manifest summary;
14. all command lines;
15. all test results;
16. unresolved warnings;
17. archive checksum.

---

# 14. Files expected to be modified

## Existing

```text
turbovla/evaluation/policy.py

turbovla/models/turbovla.py
turbovla/models/text_encoder.py
turbovla/models/vision_encoder.py
turbovla/models/action_head.py
turbovla/models/components/fusion.py
turbovla/models/components/transformer.py
turbovla/models/configuration.py

turbovla/debug/trace_config.py
turbovla/debug/trace_context.py
turbovla/debug/tensor_record.py
turbovla/debug/tensor_writer.py
turbovla/debug/fixture_writer.py
turbovla/debug/hook_manager.py
turbovla/debug/trace_utils.py

scripts/run_fixture_inference.py
scripts/compare_tensor_traces.py
scripts/validate_trace_manifest.py

tests/debug/*
.gitignore
README.md
```

## New

```text
turbovla/evaluation/model_loader.py
turbovla/evaluation/replay.py

turbovla/debug/fixture_reader.py
turbovla/debug/fixture_validator.py
turbovla/debug/deterministic.py
turbovla/debug/coverage_validator.py
turbovla/debug/exhaustive_coverage_spec.py
turbovla/debug/weights_writer.py
turbovla/debug/baseline_report.py

scripts/build_pytorch_reference_baseline.py
scripts/capture_libero_parity_fixture.py
scripts/validate_fixture.py

tests/debug/test_real_fixture_consistency.py
tests/debug/test_fixture_hashes.py
tests/debug/test_atomic_fixture_overwrite.py
tests/debug/test_strict_model_loader.py
tests/debug/test_no_random_fallback.py
tests/debug/test_policy_vs_exact_replay.py
tests/debug/test_trace_disabled_output_unchanged.py
tests/debug/test_trace_levels.py
tests/debug/test_trace_call_index.py
tests/debug/test_native_integer_storage.py
tests/debug/test_native_bf16_storage.py
tests/debug/test_trace_max_bytes.py
tests/debug/test_trace_lifecycle_postprocessing.py
tests/debug/test_compare_execution_order.py
tests/debug/test_compare_allclose_semantics.py
tests/debug/test_compare_deliberate_perturbation.py
tests/debug/test_exhaustive_coverage_spec.py
tests/debug/test_vision_projection_traced_equivalence.py
tests/debug/test_fusion_traced_equivalence.py
tests/debug/test_text_enhancer_traced_equivalence.py
tests/debug/test_state_projection_traced_equivalence.py
tests/debug/test_action_decoder_traced_equivalence.py
tests/debug/test_action_mlp_traced_equivalence.py
```

Rename:

```text
scripts/dump_sample_trace.py
→ scripts/legacy_dump_sample_trace.py
```

---

# 15. Repository hygiene

Generated output must remain local:

```gitignore
outputs/embedding_dumps/
outputs/embedding_dumps_legacy_*/
outputs/parity_traces/
outputs/parity_baseline/
outputs/pytorch_reference_baselines/
outputs/*.tar.zst
```

If generated outputs are already tracked:

```bash
git rm -r --cached outputs/parity_traces || true
git rm -r --cached outputs/pytorch_reference_baselines || true
```

Do not delete local legacy output automatically.

---

# 16. One-shot coding-agent prompt

Use this prompt exactly:

```text
Read TURBOVLA_PYTORCH_REFERENCE_BASELINE_PLAN.md completely before modifying code.

Your task is to implement the complete exhaustive PyTorch reference baseline in one continuous run. Do not implement C++, vla.cpp, GGUF, quantization or CUDA-kernel tracing.

Do not stop after phases and do not ask for review between phases. Inspect the actual current repository and installed Transformers model classes first, then implement the entire plan, run tests, run the real end-to-end baseline build, diagnose failures, fix them and rerun until all required gates pass or a genuine external blocker is proven.

Authoritative requirements:

1. Use TraceContext + TensorWriter as the only authoritative baseline writer. Keep EmbeddingDumper legacy-only and remove lifecycle coupling.
2. Capture one real deterministic LIBERO observation. Do not generate independent random images, pixel values, tokens, masks or states.
3. Store both raw fixture inputs and exact model inputs derived from the actual production preprocessing/tokenizer/state-normalization path.
4. Add one shared strict model loader used by policy, replay and tests.
5. Load the checkpoint with strict=True. No default model, strict=False, missing-file fallback or random fallback is allowed.
6. Move trace begin/finish ownership out of model.forward into the policy/runner so postprocessing tensors are included.
7. Implement trace levels boundary/layer/op/exhaustive. The final artifact must use exhaustive.
8. Dump every semantic operation specified in the plan across preprocessing, all BERT blocks, both DINO views and all DINO blocks, vision projection, six fusion layers, six text-enhancer layers, state projection, three action-decoder layers, action MLP and environment-action postprocessing.
9. Do not trace individual CUDA kernels.
10. Preserve native int64 and uint8/bool storage. Store floating tensors as canonical F32 plus native BF16/F16 when applicable.
11. Implement real semantic call_index handling and execution-order trace_id.
12. Use atomic output and reject stale/partial baselines.
13. Dump every model parameter and persistent buffer once with native and F32 hashes.
14. Add an expected exhaustive semantic coverage specification and fail when any required tensor is missing.
15. Run production BF16 trace plus deterministic BF16 runs A and B.
16. Prove policy output equals exact-input replay output for normalized and environment actions.
17. Compare deterministic traces A and B tensor by tensor in reference execution order.
18. Fix compare tolerance to np.allclose semantics and add deliberate perturbation tests.
19. Generate baseline_status.json, baseline_summary.md, coverage_report.json and a tar.zst archive.
20. Exit successfully only when all gates in baseline_status.json are true.

Implementation constraints:

- Do not change model numerical behavior merely to make tracing easier.
- Any refactored traced forward must have an equivalence test against the original path.
- Preserve the repository’s exact attention stabilization, clamp, mask, residual, padding and denormalization behavior.
- Do not save .pt for every activation; use canonical and native raw binary files.
- Check free disk space before the exhaustive run.
- Never silently truncate an exhaustive trace.
- Do not commit generated runtime output.

At the end, provide:

1. all modified and created files;
2. focused and full pytest output;
3. the exact baseline command executed;
4. strict model-load log;
5. fixture validation result;
6. policy-vs-exact normalized max absolute error;
7. policy-vs-exact environment-action max absolute error;
8. deterministic A-vs-B tensor comparison summary;
9. exhaustive expected/actual tensor count and missing list;
10. NaN/Inf count;
11. trace size by stage;
12. final baseline_status.json;
13. final artifact tree;
14. archive path and SHA256;
15. any genuine external blocker with exact command, stack trace and evidence.

Do not declare completion if the fixture is synthetic, any required semantic tensor is missing, the tracer closes before denormalization, the checkpoint is not loaded strictly, the deterministic self-compare fails, or the coverage validator does not pass.
```

---

# 17. Expected one-shot completion checks

After the agent finishes, run only these checks:

```bash
cd ~/Desktop/TurboVLA

cat outputs/pytorch_reference_baselines/\
libero_object_task0_ep0_step0_bf16/\
baseline_status.json

python -m json.tool \
  outputs/pytorch_reference_baselines/\
libero_object_task0_ep0_step0_bf16/\
validation/coverage_report.json

sed -n '1,240p' \
  outputs/pytorch_reference_baselines/\
libero_object_task0_ep0_step0_bf16/\
baseline_summary.md

find \
  outputs/pytorch_reference_baselines/\
libero_object_task0_ep0_step0_bf16 \
  -maxdepth 3 -type f | sort

sha256sum \
  outputs/pytorch_reference_baselines/\
libero_object_task0_ep0_step0_bf16/\
libero_object_task0_ep0_step0_bf16.tar.zst
```

Expected final state:

```text
baseline_status.complete = true
all gates = true
coverage missing = []
coverage duplicates = []
manifest validators = PASS
policy vs exact action = PASS
deterministic A vs B = PASS
NaN/Inf = 0
archive exists and has SHA256
```

---

# 18. Honest limitation

This plan is designed so the coding agent can implement and validate the complete baseline in one continuous run.

A one-shot success cannot be mathematically guaranteed because real external blockers may exist, such as:

- insufficient disk space;
- unavailable local BERT/DINO files;
- missing LIBERO assets;
- GPU out-of-memory;
- incompatible installed Transformers internals;
- corrupted checkpoint.

The required orchestrator must detect these before or during execution, fail loudly, preserve logs, and identify the exact blocker. It must not silently produce an incomplete baseline.

---

# 19. Final deliverable

The required deliverable is one frozen PyTorch reference package containing:

```text
real fixture
exact model inputs
full exhaustive tensor trace
full weights/buffers manifest
normalized and environment actions
deterministic duplicate run
self-comparison
coverage validation
environment and model hashes
human-readable summary
machine-readable status
compressed archive
```

After this artifact passes all gates, future `vla.cpp` work only needs to consume the same fixture, emit the same semantic names, and run the existing diff tool.
