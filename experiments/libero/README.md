# LIBERO

This directory contains the released LIBERO recipe and public entry points for the root `turbovla` package. The configuration uses two DINOv3 views, an online BERT encoder, 7-D actions, and a chunk size of 12. The task and episode rollout protocol is the VLA-Adapter-derived implementation under `third_party/vla_adapter/`.

## Layout

```text
../../turbovla/             model, RLDS data, training, and policy code
configs/                    released mixed-suite normalization statistics
train.py                    public mixed-suite training entry point
evaluate.py                 public single-checkpoint evaluation entry point
../../scripts/libero/       data, statistics, and evaluation utilities
../../third_party/          VLA-Adapter rollout protocol and license
```

## Environment

The reference environment used PyTorch 2.3.1, torchvision 0.18.1, transformers 4.56, TensorFlow 2.20, and tensorflow-datasets 4.9.3.

```bash
pip install -e ".[libero]"
```

Install LIBERO separately. DINOv3 and GroundingDINO weights are not included.

## Data

```text
data/libero/
|-- libero_10_no_noops/1.0.0/
|-- libero_goal_no_noops/1.0.0/
|-- libero_object_no_noops/1.0.0/
`-- libero_spatial_no_noops/1.0.0/
```

The repository does not redistribute demonstrations. Use `scripts/libero/regenerate_libero_no_noops.py` with the official LIBERO data, then convert the derived trajectories to TFDS/RLDS. Released normalization statistics are in `experiments/libero/configs/libero_all4_stats.json`.

## Training

The released recipe uses full DINOv3 ViT-B unfreezing, global batch 128 on four GPUs (per-device batch size 8 with 4 gradient-accumulation steps), 40k optimizer steps, 10k warmup steps, learning rate `5e-5`, seed 42, FP32 policy parameters, and BF16 autocast for DINOv3.

```bash
torchrun --nproc_per_node=4 experiments/libero/train.py \
  --dataset_dir data/libero/libero_10_no_noops/1.0.0 \
  --dataset_dirs "data/libero/libero_10_no_noops/1.0.0,data/libero/libero_goal_no_noops/1.0.0,data/libero/libero_object_no_noops/1.0.0,data/libero/libero_spatial_no_noops/1.0.0" \
  --stats_path experiments/libero/configs/libero_all4_stats.json \
  --stats_key libero_all4_no_noops \
  --dinov3_path facebook/dinov3-vitb16-pretrain-lvd1689m \
  --bert_path google-bert/bert-base-uncased \
  --allow_hf_download \
  --pretrained_init_ckpt /path/to/groundingdino_swint_ogc.pth \
  --batch_size 8 \
  --checkpoint_dir outputs/checkpoints
```

## Evaluation

One invocation evaluates one checkpoint on one suite. Checkpoint selection and multi-GPU scheduling are intentionally outside the evaluator.

```bash
python experiments/libero/evaluate.py \
  --ckpt_path pretrained/TurboVLA/checkpoints/libero/turbovla_libero.pth \
  --dinov3_path /path/to/dinov3-vitb \
  --bert_path /path/to/bert-base-uncased \
  --stats_path experiments/libero/configs/libero_all4_stats.json \
  --stats_key libero_all4_no_noops \
  --task_suite_name libero_object \
  --num_trials_per_task 50 \
  --chunk_size 12 \
  --num_open_loop_steps 12 \
  --seed 7 \
  --precision bf16 \
  --result_json_path outputs/evaluation/object.json
```


Use `--dry_run_model_load true` to validate checkpoint compatibility without a simulator rollout. Complete release checkpoints contain BERT, DINOv3, the vision-language interaction module, and the action head in one strict-loadable state dict. `configs/online_text_layout.json` stores tokenizer padding metadata for the 40 standard instructions; it contains no embeddings or cached model outputs.

The rollout protocol is derived from [VLA-Adapter](https://github.com/OpenHelix-Team/VLA-Adapter/commit/23fa0c9c159e2aa04341cdd3e924f44061311060). See `third_party/licenses/VLA-Adapter.txt` for the retained license and attribution.
