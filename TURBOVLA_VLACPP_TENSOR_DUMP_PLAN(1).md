# TurboVLA ↔ vla.cpp Tensor Dump & Parity Debugging Plan

## 0. Mục tiêu

Tài liệu này mô tả cách xây dựng hệ thống dump tensor cho TurboVLA phía PyTorch và phía `vla.cpp` để:

1. Dùng cùng một input fixture cố định.
2. Dump tensor theo đúng thứ tự thực thi.
3. Có tên tensor mang ý nghĩa ngữ nghĩa, không chỉ là số thứ tự.
4. Ghi rõ shape, layout, dtype, stride và thống kê.
5. So sánh từng tensor giữa PyTorch và C++.
6. Tìm được **tensor đầu tiên bắt đầu lệch**.
7. Thu hẹp phạm vi debug từ block lớn xuống Q/K/V, logits, softmax, residual và các phép reshape/transpose.
8. Kiểm tra parity ở BF16/FP32 trước, sau đó mới đánh giá sai số do quantization.

Hệ thống dump hiện tại có thể giữ lại, nhưng cần chuẩn hóa lại naming, manifest, fixture, binary format và các điểm trace thủ công.


# 0.1. Xử lý output dump cũ trước khi bắt đầu

## Không xoá output cũ ngay

Output hiện tại phải được giữ lại làm:

- dữ liệu tham chiếu;
- bằng chứng rằng collector cũ từng chạy;
- nguồn để kiểm tra migration;
- fixture tạm thời nếu implementation mới gặp lỗi;
- dữ liệu để so sánh hành vi trước/sau khi refactor.

Không để output cũ nằm chung đường dẫn với format mới.

## Bước chuẩn bị bắt buộc

Từ root của TurboVLA:

```bash
cd ~/Desktop/TurboVLA

timestamp=$(date +%Y%m%d_%H%M%S)

if [ -d outputs/embedding_dumps ]; then
  mv outputs/embedding_dumps \
     "outputs/embedding_dumps_legacy_${timestamp}"
fi

mkdir -p outputs/parity_traces
```

Kết quả mong muốn:

```text
outputs/
├── embedding_dumps_legacy_YYYYMMDD_HHMMSS/
└── parity_traces/
```

## Không được overwrite legacy output

Implementation mới:

- không được ghi vào `outputs/embedding_dumps_legacy_*`;
- không được tự động xoá output cũ;
- không được dùng lại cùng output folder nếu chưa có `--trace-overwrite`;
- phải fail rõ ràng nếu target directory đã tồn tại.

## Tuỳ chọn archive để tiết kiệm dung lượng

Chỉ archive sau khi xác nhận folder legacy đọc được:

```bash
cd ~/Desktop/TurboVLA/outputs

legacy_dir=$(find . -maxdepth 1 -type d \
  -name 'embedding_dumps_legacy_*' \
  | sort | tail -n 1)

tar -I 'zstd -10' \
  -cf "${legacy_dir#./}.tar.zst" \
  "${legacy_dir#./}"

tar -I zstd \
  -tf "${legacy_dir#./}.tar.zst" \
  | head
```

Sau khi lệnh `tar -tf` đọc thành công, có thể chọn giữ:

- folder legacy;
- hoặc chỉ file `.tar.zst`;
- hoặc cả hai trong thời gian triển khai.

Không xoá cả folder và archive cùng lúc khi pipeline mới chưa được xác nhận.

## Điều kiện bắt buộc trước khi xoá legacy output

Chỉ được xoá output cũ sau khi toàn bộ điều kiện sau đạt:

1. Fixture mới được export đầy đủ.
2. Inference trực tiếp và replay từ fixture cho cùng action trong tolerance.
3. `manifest.jsonl` qua validator.
4. Mọi binary có file size khớp shape và dtype.
5. Boundary trace có đầy đủ các stage quan trọng.
6. Semantic names ổn định giữa nhiều lần chạy.
7. Compare script chạy được với ít nhất một trace mock hoặc trace C++.
8. PyTorch trace và C++ trace match được theo `semantic_name`.
9. Trace disabled không làm thay đổi output model.
10. Không còn dữ liệu duy nhất chỉ tồn tại trong legacy folder.

Khi đạt đủ:

```bash
rm -rf \
  ~/Desktop/TurboVLA/outputs/embedding_dumps_legacy_YYYYMMDD_HHMMSS
```

Nếu đã archive:

```bash
rm -rf \
  ~/Desktop/TurboVLA/outputs/embedding_dumps_legacy_YYYYMMDD_HHMMSS
```

và tiếp tục giữ:

```text
embedding_dumps_legacy_YYYYMMDD_HHMMSS.tar.zst
```

cho tới khi parity BF16 hoàn tất.

## `.gitignore`

Thêm:

```gitignore
outputs/embedding_dumps/
outputs/embedding_dumps_legacy_*/
outputs/parity_traces/
outputs/*.tar.zst
```

Nếu repository cần commit một fixture nhỏ cho unit test, đặt fixture đó ở đường dẫn riêng:

```text
tests/fixtures/parity/minimal_fixture/
```

Không đưa full runtime dumps vào Git.

## Yêu cầu migration

Collector mới nên hỗ trợ một script kiểm tra output cũ:

```text
scripts/inspect_legacy_trace.py
```

Script này không cần convert toàn bộ legacy format sang format mới, nhưng nên:

- đọc `manifest.json` cũ;
- liệt kê số tensor;
- kiểm tra file thiếu;
- hiển thị shape/dtype nếu có;
- map được những tensor có module name rõ ràng;
- xuất báo cáo migration;
- không sửa file cũ tại chỗ.

Nếu viết converter:

```text
scripts/migrate_legacy_trace.py
```

converter phải:

- đọc source legacy read-only;
- ghi sang output folder mới;
- không đoán semantic name khi thiếu thông tin;
- đánh dấu tensor không map được là `legacy.unresolved.<id>`;
- ghi warning vào migration report;
- không tuyên bố output migrated đạt parity nếu thiếu layout hoặc operation metadata.

---

# 1. Đánh giá hệ thống dump hiện tại

Cấu trúc hiện tại:

```text
outputs/
└── embedding_dumps/
    └── object_real_sample/
        └── forward_000000_rank_0/
            ├── tensors/
            │   ├── tensor_0001913.bin
            │   ├── tensor_0001913.pt
            │   ├── tensor_0001914.bin
            │   ├── tensor_0001914.pt
            │   └── ...
            ├── manifest.json
            ├── sample_actions_f32.bin
            ├── sample_actions.pt
            ├── sample_state_f32.bin
            ├── sample_state.pt
            ├── sample_view_0.png
            ├── sample_view_1.png
            ├── sample.json
            └── trace_index.pt
```

## 1.1. Những phần có thể giữ lại

- Phân tách theo `forward_xxxxxx_rank_y`.
- Lưu cả `.pt` và binary.
- Có `manifest`.
- Có ảnh camera mẫu.
- Có state và action mẫu.
- Có trace index.
- Có output root riêng cho từng sample.

## 1.2. Những điểm chưa phù hợp để debug parity

### Tên tensor không có ý nghĩa

Ví dụ:

```text
tensor_0001913.bin
```

không cho biết đây là:

- output của DINO block nào;
- Q/K/V của attention nào;
- attention logits;
- softmax probabilities;
- residual output;
- output của action decoder;
- hay action sau denormalization.

Numeric ID vẫn hữu ích để giữ thứ tự, nhưng phải đi kèm semantic name.

### Chưa chắc đã trace được functional operations

`forward_hook` chỉ thấy input/output của `nn.Module`.

Hook thông thường không tự bắt được:

- `view`;
- `reshape`;
- `flatten`;
- `permute`;
- `transpose`;
- `torch.cat`;
- position embedding addition;
- Q/K/V sau khi split head;
- attention logits;
- mask cộng vào logits;
- softmax;
- residual addition;
- gamma scaling;
- `tanh`;
- action denormalization.

Các điểm này thường là nguyên nhân chính gây sai parity giữa PyTorch và ggml/C++.

### Input fixture chưa đầy đủ

Ảnh PNG chỉ dùng để nhìn bằng mắt. Để compare chính xác cần lưu cả:

- ảnh RGB `uint8` trước normalization;
- tensor sau resize và preprocessing;
- token IDs;
- attention mask;
- custom text self-attention mask;
- position IDs;
- raw state;
- normalized state;
- action trước `tanh`;
- normalized action;
- denormalized action.

### Binary chưa tự mô tả

Extension `.bin` không nói được:

- dtype;
- endianness;
- layout;
- có header hay không;
- tensor đã contiguous chưa.

### Folder phẳng quá lớn

Hàng nghìn file trong một folder gây khó:

- duyệt bằng VS Code;
- grep;
- tìm tensor theo block;
- đối chiếu với graph C++;
- xóa hoặc rerun một phạm vi nhỏ.

---

# 2. Nguyên tắc thiết kế

## 2.1. Một fixture, hai runtime

PyTorch và `vla.cpp` phải đọc **cùng một fixture**.

Không để cả hai runtime tự lấy observation riêng từ simulator, vì khi output lệch sẽ không biết nguyên nhân đến từ:

- camera;
- rotation;
- RGB/BGR;
- preprocessing;
- tokenizer;
- simulator state;
- hay model.

## 2.2. Semantic name là khóa so sánh chính

Mỗi tensor có:

- `trace_id`: thứ tự thực thi;
- `semantic_name`: tên ổn định dùng để match Python ↔ C++;
- `module_path`: đường dẫn module PyTorch;
- `operation`: loại phép toán;
- `call_index`: phân biệt module bị gọi nhiều lần.

`semantic_name` phải giống tuyệt đối ở Python và C++.

## 2.3. Layout phải được ghi rõ

Shape giống nhau không có nghĩa layout giống nhau.

Ví dụ:

```text
[B, H, N, D]
[B, N, H, D]
```

có cùng số phần tử nhưng ý nghĩa khác hoàn toàn.

Mỗi tensor phải có trường `layout`, ví dụ:

```text
B,V,C,H,W
B,N,D
B,H,N,Dh
B,H,Q,K
B,T,A
```

## 2.4. Lưu một representation canonical

Để compare độc lập backend, mọi activation floating-point cần có một bản canonical:

```text
dtype: float32
endianness: little-endian
memory: contiguous row-major
header: none
```

Ngoài ra có thể lưu raw BF16 bits để kiểm tra chính xác quá trình cast.

## 2.5. Trace theo nhiều mức

Không bật dump op-level cho toàn model ngay từ đầu.

Quy trình:

```text
boundary toàn model
→ layer ở block đầu tiên bị lệch
→ op trong submodule đầu tiên bị lệch
```

---

# 3. Cấu trúc output chuẩn

```text
outputs/
└── parity_traces/
    ├── fixtures/
    │   └── object_task0_episode0_step0/
    │       ├── metadata.json
    │       ├── instruction.txt
    │       ├── view_0_rgb_u8.npy
    │       ├── view_0_rgb_u8.bin
    │       ├── view_1_rgb_u8.npy
    │       ├── view_1_rgb_u8.bin
    │       ├── pixel_values_f32.npy
    │       ├── pixel_values_f32le.bin
    │       ├── input_ids_i64.npy
    │       ├── input_ids_i64le.bin
    │       ├── attention_mask_u8.npy
    │       ├── attention_mask_u8.bin
    │       ├── text_self_attention_mask_u8.npy
    │       ├── text_self_attention_mask_u8.bin
    │       ├── position_ids_i64.npy
    │       ├── position_ids_i64le.bin
    │       ├── state_raw_f32.npy
    │       ├── state_raw_f32le.bin
    │       ├── state_normalized_f32.npy
    │       └── state_normalized_f32le.bin
    │
    ├── python/
    │   └── object_task0_episode0_step0/
    │       └── forward_000000_rank_0/
    │           ├── tensors/
    │           │   ├── 00_input/
    │           │   ├── 10_text_encoder/
    │           │   ├── 20_vision_encoder/
    │           │   │   ├── view_0/
    │           │   │   └── view_1/
    │           │   ├── 30_vision_projection/
    │           │   ├── 40_interaction/
    │           │   │   ├── layer_00/
    │           │   │   ├── layer_01/
    │           │   │   └── ...
    │           │   ├── 50_state_projection/
    │           │   ├── 60_action_decoder/
    │           │   │   ├── layer_00/
    │           │   │   ├── layer_01/
    │           │   │   └── layer_02/
    │           │   └── 70_output/
    │           ├── manifest.jsonl
    │           ├── summary.csv
    │           ├── trace_tree.txt
    │           ├── model_config.json
    │           ├── environment.json
    │           └── trace.log
    │
    ├── cpp/
    │   └── object_task0_episode0_step0/
    │       └── forward_000000_rank_0/
    │           └── ...
    │
    └── compare/
        └── object_task0_episode0_step0/
            ├── compare.csv
            ├── compare.json
            ├── first_divergence.txt
            ├── missing_in_python.txt
            ├── missing_in_cpp.txt
            ├── shape_mismatches.txt
            ├── layout_mismatches.txt
            ├── worst_tensors.txt
            └── diff_tensors/
```

---

# 4. Quy tắc đặt tên tensor

## 4.1. Tên file

```text
<trace_id>__<semantic_name>.<storage_dtype>.bin
```

Ví dụ:

```text
0001913__interaction.layer_02.cross.visual_to_text.probs.f32le.bin
0001914__interaction.layer_02.cross.visual_to_text.context.f32le.bin
0001915__interaction.layer_02.visual.after_residual.f32le.bin
```

Trong filesystem có thể thay dấu `.` bằng `__` nếu cần:

```text
0001913__interaction__layer_02__cross__visual_to_text__probs.f32le.bin
```

Trong manifest vẫn giữ semantic name dạng dấu chấm.

## 4.2. Dtype suffix

Dùng suffix rõ ràng:

```text
.f32le.bin
.bf16le.bin
.f16le.bin
.i64le.bin
.i32le.bin
.u16le.bin
.u8.bin
.bool_u8.bin
```

Không dùng `.bin` chung cho mọi dtype.

## 4.3. Semantic path

Quy tắc:

```text
<stage>.<subsystem>.<layer>.<operation>.<variant>
```

Ví dụ:

```text
input.pixel_values
text.bert.layer_00.attn.q_linear
text.bert.layer_00.attn.q_heads
vision.view_0.block_03.attn.logits
vision_projection.residual_sum
interaction.layer_02.cross.visual_to_text.probs
state_projection.position_add
action.decoder.layer_01.cross_attn.context
action.before_tanh
action.normalized
action.denormalized
```

---

# 5. Fixture chuẩn

## 5.1. Metadata

`metadata.json`:

```json
{
  "fixture_version": 1,
  "fixture_id": "object_task0_episode0_step0",
  "task_suite": "libero_object",
  "task_id": 0,
  "episode": 0,
  "step": 0,
  "seed": 7,
  "instruction": "pick up the alphabet soup and place it in the basket",
  "image_layout": "B,V,C,H,W",
  "state_layout": "B,D",
  "action_layout": "B,T,A",
  "model_precision": "bf16",
  "checkpoint_path": "pretrained/TurboVLA/checkpoints/libero/object.pth",
  "checkpoint_sha256": "<sha256>",
  "stats_path": "pretrained/TurboVLA/libero_all4_stats.json",
  "stats_key": "libero_all4_no_noops",
  "python_version": "<version>",
  "torch_version": "<version>",
  "transformers_version": "<version>",
  "numpy_version": "<version>",
  "cuda_version": "<version>",
  "gpu_name": "<name>"
}
```

## 5.2. Input files bắt buộc

```text
instruction.txt

view_0_rgb_u8.npy
view_0_rgb_u8.bin
view_1_rgb_u8.npy
view_1_rgb_u8.bin

pixel_values_f32.npy
pixel_values_f32le.bin

input_ids_i64.npy
input_ids_i64le.bin

attention_mask_u8.npy
attention_mask_u8.bin

text_self_attention_mask_u8.npy
text_self_attention_mask_u8.bin

position_ids_i64.npy
position_ids_i64le.bin

state_raw_f32.npy
state_raw_f32le.bin

state_normalized_f32.npy
state_normalized_f32le.bin
```

## 5.3. Nguyên tắc tạo fixture

- Chỉ tạo một lần.
- Dùng `model.eval()`.
- Chụp tại policy forward đầu tiên.
- Không overwrite fixture nếu file đã tồn tại, trừ khi có cờ `--overwrite-fixture`.
- Ghi hash từng file.
- C++ đọc fixture trực tiếp.
- Fixture không phụ thuộc simulator sau khi đã được tạo.

---

# 6. Ba mức trace

## 6.1. `boundary`

Mục tiêu: tìm block lớn đầu tiên bị lệch.

Dump:

```text
input
text encoder output
vision encoder output
vision projection output
interaction layer 0 output
interaction layer 1 output
...
interaction layer 5 output
state projection output
action decoder layer 0 output
action decoder layer 1 output
action decoder layer 2 output
action before tanh
action normalized
action denormalized
```

## 6.2. `layer`

Mục tiêu: tìm module nhỏ đầu tiên bị lệch.

Dump input/output của:

```text
Conv2d
Linear
LayerNorm
Embedding
GELU
ReLU
Dropout
attention module
transformer block
MLP block
projection block
decoder block
```

## 6.3. `op`

Mục tiêu: debug chính xác implementation.

Dump:

```text
linear Q/K/V output
head split
transpose
attention scale
QK^T
raw logits
mask
masked logits
softmax probs
dropout output
V aggregation
head merge
output projection
residual input
residual delta
residual output
LayerNorm statistics nếu cần
concat
position add
tanh
denormalization
```

## 6.4. Filter

Hỗ trợ filter để chỉ trace một phạm vi:

```bash
--trace-include "interaction.layer_02"
--trace-exclude "*.dropout*"
--trace-level op
```

Hoặc qua biến môi trường:

```bash
TURBOVLA_TRACE_INCLUDE="interaction.layer_02"
TURBOVLA_TRACE_EXCLUDE="*.dropout*"
```

---

# 7. Tensor checklist theo pipeline

Các shape bên dưới là shape kỳ vọng cho cấu hình LIBERO phổ biến. Implementation phải lấy shape động từ tensor thật, không hard-code.

Ví dụ cấu hình thường gặp:

```text
batch size B = 1
number of views V = 2
image size = 256 × 256
patch size = 16
patches per view = 256
DINO hidden = 768
TurboVLA hidden = 256
text length = L
action horizon = 12
action dim = 7
state dim = 8
state tokens = 2
```

---

## 7.1. Input và preprocessing

Bắt buộc dump:

```text
input.view_0.rgb_u8
input.view_1.rgb_u8
input.pixel_values
input.instruction
input.input_ids
input.attention_mask
input.text_self_attention_mask
input.position_ids
input.state_raw
input.state_normalized
```

Metadata cần ghi:

```text
RGB hay BGR
camera rotation
resize method
interpolation mode
input range
normalization mean
normalization std
layout
contiguous
```

---

## 7.2. Text encoder / BERT

### Boundary

```text
text.input_ids
text.attention_mask
text.self_attention_mask
text.position_ids
text.bert.embeddings
text.bert.last_hidden_state
text.projection.output
text.key_padding_mask
```

### Mỗi BERT layer

```text
text.bert.layer_00.input

text.bert.layer_00.attn.q_linear
text.bert.layer_00.attn.k_linear
text.bert.layer_00.attn.v_linear

text.bert.layer_00.attn.q_heads
text.bert.layer_00.attn.k_heads
text.bert.layer_00.attn.v_heads

text.bert.layer_00.attn.logits
text.bert.layer_00.attn.mask
text.bert.layer_00.attn.masked_logits
text.bert.layer_00.attn.probs
text.bert.layer_00.attn.context_heads
text.bert.layer_00.attn.context_merged
text.bert.layer_00.attn.output_projection

text.bert.layer_00.residual_1
text.bert.layer_00.norm_1

text.bert.layer_00.ffn.linear_1
text.bert.layer_00.ffn.activation
text.bert.layer_00.ffn.linear_2

text.bert.layer_00.residual_2
text.bert.layer_00.output
```

Lặp cho toàn bộ BERT layers.

### Điểm cần đặc biệt chú ý

- position IDs;
- custom self-attention mask;
- mask convention: `0/1`, `bool`, hay additive `-inf`;
- scale `1/sqrt(head_dim)`;
- thứ tự reshape và transpose;
- GELU variant;
- LayerNorm epsilon.

---

## 7.3. Vision encoder / DINOv3

Tách riêng từng camera:

```text
vision.view_0.*
vision.view_1.*
```

### Input

```text
vision.view_0.input
vision.view_1.input
```

### Patch embedding và prefix

```text
vision.view_0.patch_embedding
vision.view_0.cls_token
vision.view_0.register_tokens
vision.view_0.position_embedding
vision.view_0.tokens_with_prefix
```

### Mỗi vision transformer block

```text
vision.view_0.block_00.input
vision.view_0.block_00.norm_1

vision.view_0.block_00.attn.q_linear
vision.view_0.block_00.attn.k_linear
vision.view_0.block_00.attn.v_linear

vision.view_0.block_00.attn.q_heads
vision.view_0.block_00.attn.k_heads
vision.view_0.block_00.attn.v_heads

vision.view_0.block_00.attn.logits
vision.view_0.block_00.attn.probs
vision.view_0.block_00.attn.context_heads
vision.view_0.block_00.attn.context_merged
vision.view_0.block_00.attn.output_projection

vision.view_0.block_00.residual_1
vision.view_0.block_00.norm_2

vision.view_0.block_00.mlp.linear_1
vision.view_0.block_00.mlp.activation
vision.view_0.block_00.mlp.linear_2

vision.view_0.block_00.residual_2
vision.view_0.block_00.output
```

Lặp cho toàn bộ blocks và view 1.

### Output slicing

```text
vision.tokens_with_prefix
vision.prefix_tokens
vision.patch_tokens
vision.output
```

Điểm dễ lỗi ở C++:

- số prefix token;
- register token ordering;
- patch ordering;
- position interpolation;
- flatten order H/W;
- view ordering;
- BF16 cast position.

---

## 7.4. Vision projection

Dump:

```text
vision_projection.input
vision_projection.input_norm

vision_projection.skip.input
vision_projection.skip.output

vision_projection.mlp.linear_1
vision_projection.mlp.activation
vision_projection.mlp.linear_2

vision_projection.residual_sum
vision_projection.output_norm

vision_projection.view_embedding
vision_projection.position_add
vision_projection.flattened
vision_projection.output
```

Cần phân biệt:

```text
skip_output
mlp_output
skip_plus_mlp
output_norm
```

Không chỉ dump output cuối.

---

## 7.5. Vision-language interaction layers

Với mỗi layer:

```text
interaction.layer_00.visual.input
interaction.layer_00.text.input
```

### Pre-normalization

```text
interaction.layer_00.visual.norm
interaction.layer_00.text.norm
```

### Cross attention

```text
interaction.layer_00.cross.q_visual_linear
interaction.layer_00.cross.k_text_linear
interaction.layer_00.cross.v_visual_linear
interaction.layer_00.cross.v_text_linear

interaction.layer_00.cross.q_visual_heads
interaction.layer_00.cross.k_text_heads
interaction.layer_00.cross.v_visual_heads
interaction.layer_00.cross.v_text_heads
```

### Visual-to-text direction

```text
interaction.layer_00.cross.visual_to_text.logits
interaction.layer_00.cross.visual_to_text.mask
interaction.layer_00.cross.visual_to_text.masked_logits
interaction.layer_00.cross.visual_to_text.probs
interaction.layer_00.cross.visual_to_text.context_heads
interaction.layer_00.cross.visual_to_text.context_merged
interaction.layer_00.cross.visual_to_text.output_projection
```

### Text-to-visual direction

```text
interaction.layer_00.cross.text_to_visual.logits
interaction.layer_00.cross.text_to_visual.mask
interaction.layer_00.cross.text_to_visual.masked_logits
interaction.layer_00.cross.text_to_visual.probs
interaction.layer_00.cross.text_to_visual.context_heads
interaction.layer_00.cross.text_to_visual.context_merged
interaction.layer_00.cross.text_to_visual.output_projection
```

### Gamma và residual

```text
interaction.layer_00.visual.delta
interaction.layer_00.visual.gamma
interaction.layer_00.visual.scaled_delta
interaction.layer_00.visual.after_residual

interaction.layer_00.text.delta
interaction.layer_00.text.gamma
interaction.layer_00.text.scaled_delta
interaction.layer_00.text.after_residual
```

### Text enhancer

```text
interaction.layer_00.text_enhancer.input
interaction.layer_00.text_enhancer.norm_1

interaction.layer_00.text_enhancer.self_attn.q_linear
interaction.layer_00.text_enhancer.self_attn.k_linear
interaction.layer_00.text_enhancer.self_attn.v_linear
interaction.layer_00.text_enhancer.self_attn.q_heads
interaction.layer_00.text_enhancer.self_attn.k_heads
interaction.layer_00.text_enhancer.self_attn.v_heads
interaction.layer_00.text_enhancer.self_attn.logits
interaction.layer_00.text_enhancer.self_attn.mask
interaction.layer_00.text_enhancer.self_attn.masked_logits
interaction.layer_00.text_enhancer.self_attn.probs
interaction.layer_00.text_enhancer.self_attn.context_heads
interaction.layer_00.text_enhancer.self_attn.context_merged
interaction.layer_00.text_enhancer.self_attn.output_projection

interaction.layer_00.text_enhancer.residual_1
interaction.layer_00.text_enhancer.norm_2
interaction.layer_00.text_enhancer.ffn.linear_1
interaction.layer_00.text_enhancer.ffn.activation
interaction.layer_00.text_enhancer.ffn.linear_2
interaction.layer_00.text_enhancer.residual_2
interaction.layer_00.text_enhancer.output
```

### Layer output

```text
interaction.layer_00.visual.output
interaction.layer_00.text.output
```

Lặp cho toàn bộ interaction layers.

### Condition concat

```text
condition.visual_tokens
condition.text_tokens
condition.concatenated
```

Ghi rõ concat axis.

---

## 7.6. State projection

```text
state_projection.input_raw
state_projection.input_normalized
state_projection.norm
state_projection.linear_1
state_projection.activation
state_projection.linear_2
state_projection.reshape
state_projection.position_embedding
state_projection.position_add
state_projection.output_norm
state_projection.output
```

---

## 7.7. Action decoder

### Memory và queries

```text
action.memory.condition
action.memory.state_tokens
action.memory.concatenated
action.queries
```

### Mỗi decoder layer

```text
action.decoder.layer_00.input
```

#### Self attention

```text
action.decoder.layer_00.self_attn.q_linear
action.decoder.layer_00.self_attn.k_linear
action.decoder.layer_00.self_attn.v_linear
action.decoder.layer_00.self_attn.q_heads
action.decoder.layer_00.self_attn.k_heads
action.decoder.layer_00.self_attn.v_heads
action.decoder.layer_00.self_attn.logits
action.decoder.layer_00.self_attn.mask
action.decoder.layer_00.self_attn.masked_logits
action.decoder.layer_00.self_attn.probs
action.decoder.layer_00.self_attn.context_heads
action.decoder.layer_00.self_attn.context_merged
action.decoder.layer_00.self_attn.output_projection
action.decoder.layer_00.self_attn.after_residual
action.decoder.layer_00.self_attn.after_norm
```

#### Cross attention

```text
action.decoder.layer_00.cross_attn.q_linear
action.decoder.layer_00.cross_attn.k_linear
action.decoder.layer_00.cross_attn.v_linear
action.decoder.layer_00.cross_attn.q_heads
action.decoder.layer_00.cross_attn.k_heads
action.decoder.layer_00.cross_attn.v_heads
action.decoder.layer_00.cross_attn.logits
action.decoder.layer_00.cross_attn.mask
action.decoder.layer_00.cross_attn.masked_logits
action.decoder.layer_00.cross_attn.probs
action.decoder.layer_00.cross_attn.context_heads
action.decoder.layer_00.cross_attn.context_merged
action.decoder.layer_00.cross_attn.output_projection
action.decoder.layer_00.cross_attn.after_residual
action.decoder.layer_00.cross_attn.after_norm
```

#### FFN

```text
action.decoder.layer_00.ffn.linear_1
action.decoder.layer_00.ffn.activation
action.decoder.layer_00.ffn.linear_2
action.decoder.layer_00.ffn.after_residual
action.decoder.layer_00.output
```

Lặp cho mọi decoder layers.

---

## 7.8. Action MLP và output

```text
action.mlp.input
action.mlp.linear_0
action.mlp.activation_0
action.mlp.linear_1
action.mlp.activation_1
action.mlp.linear_2
action.before_tanh
action.normalized
action.denormalized
```

Ngoài full action chunk, lưu thêm:

```text
action.first_step
action.executed_steps
```

nếu policy chỉ thực thi một phần action chunk.

---

# 8. Manifest JSONL

Mỗi dòng tương ứng một tensor.

Ví dụ:

```json
{
  "trace_id": 1913,
  "semantic_name": "interaction.layer_02.cross.visual_to_text.probs",
  "module_path": "vision_language_interaction.fusion_layers.2.attention",
  "operation": "softmax",
  "call_index": 0,
  "io": "output",
  "stage": "interaction",
  "shape": [1, 4, 512, 21],
  "layout": "B,H,V,L",
  "strides": [43008, 10752, 21, 1],
  "source_dtype": "bfloat16",
  "storage_dtype": "float32",
  "endianness": "little",
  "contiguous": true,
  "numel": 43008,
  "file": "tensors/40_interaction/layer_02/0001913__interaction.layer_02.cross.visual_to_text.probs.f32le.bin",
  "pt_file": "tensors_pt/0001913.pt",
  "raw_bf16_file": "tensors_bf16/0001913.bf16le.bin",
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
  "sha256_f32": "<hash>",
  "timestamp_ns": 0
}
```

## 8.1. Trường bắt buộc

```text
trace_id
semantic_name
operation
shape
layout
source_dtype
storage_dtype
endianness
contiguous
numel
file
min
max
mean
std
abs_max
nan_count
inf_count
sha256_f32
```

## 8.2. Trường nên có

```text
module_path
call_index
io
stage
strides
pt_file
raw_bf16_file
l1_norm
l2_norm
zero_count
zero_ratio
first_values
last_values
```

---

# 9. Binary format

## 9.1. Float activation canonical

```python
array = (
    tensor.detach()
    .to(device="cpu", dtype=torch.float32)
    .contiguous()
    .numpy()
    .astype("<f4", copy=False)
)

array.tofile(path)
```

Quy ước:

```text
header: none
dtype: float32
endianness: little
order: C row-major
shape: đọc từ manifest
layout: đọc từ manifest
```

## 9.2. Integer tensors

```python
input_ids = tensor.detach().cpu().contiguous().numpy().astype("<i8")
input_ids.tofile(path)
```

Masks nên canonicalize thành `uint8`:

```python
mask = tensor.detach().cpu().to(torch.uint8).contiguous().numpy()
mask.tofile(path)
```

## 9.3. Raw BF16

```python
raw = (
    tensor.detach()
    .cpu()
    .contiguous()
    .view(torch.uint16)
    .numpy()
    .astype("<u2", copy=False)
)
raw.tofile(path)
```

Chỉ lưu raw BF16 khi:

```text
TURBOVLA_TRACE_SAVE_RAW_BF16=1
```

---

# 10. Kiến trúc code phía Python

Đề xuất thêm package:

```text
turbovla/debug/
├── __init__.py
├── trace_config.py
├── tensor_record.py
├── tensor_writer.py
├── trace_context.py
├── hook_manager.py
├── semantic_registry.py
├── fixture_writer.py
└── trace_utils.py
```

---

## 10.1. `TraceConfig`

```python
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TraceConfig:
    enabled: bool
    root_dir: Path
    level: str
    max_forwards: int
    include: tuple[str, ...]
    exclude: tuple[str, ...]
    save_pt: bool
    save_f32: bool
    save_raw_bf16: bool
    save_stats: bool
    save_first_last_values: int
    fail_on_nan: bool
    overwrite: bool
```

Supported level:

```text
boundary
layer
op
```

---

## 10.2. `TraceContext`

Nhiệm vụ:

- giữ forward index;
- rank;
- trace counter;
- semantic scope stack;
- call index;
- filter;
- writer;
- disable sau `max_forwards`.

Ví dụ API:

```python
with tracer.scope("interaction.layer_02"):
    tracer.tensor("visual.input", visual, layout="B,N,D")
    ...
```

Semantic name đầy đủ:

```text
interaction.layer_02.visual.input
```

---

## 10.3. `TensorWriter`

Interface:

```python
class TensorWriter:
    def write(
        self,
        *,
        semantic_name: str,
        tensor: torch.Tensor,
        layout: str,
        operation: str,
        module_path: str | None = None,
        io: str = "intermediate",
        metadata: dict | None = None,
    ) -> TensorRecord:
        ...
```

Responsibilities:

1. `detach`.
2. Move CPU.
3. Make contiguous.
4. Save canonical float32.
5. Save `.pt` nếu bật.
6. Save raw BF16 nếu bật.
7. Tính stats.
8. Tính hash.
9. Append manifest JSONL.
10. Append summary CSV.
11. Print một dòng ngắn ra console.

---

## 10.4. Generic leaf hooks

Dùng hook cho các leaf module:

```python
TRACE_LEAF_TYPES = (
    torch.nn.Linear,
    torch.nn.LayerNorm,
    torch.nn.Conv2d,
    torch.nn.Embedding,
    torch.nn.GELU,
    torch.nn.ReLU,
)
```

Pseudo-code:

```python
def install_leaf_hooks(model, tracer):
    handles = []

    for module_name, module in model.named_modules():
        if list(module.children()):
            continue

        if not isinstance(module, TRACE_LEAF_TYPES):
            continue

        handles.append(
            module.register_forward_pre_hook(
                make_pre_hook(module_name, tracer)
            )
        )

        handles.append(
            module.register_forward_hook(
                make_post_hook(module_name, tracer)
            )
        )

    return handles
```

Lưu ý:

- Module có thể nhận tuple/dict.
- Chỉ dump tensor arguments.
- Có `call_index`.
- Không dump parameter ở mỗi forward.
- Parameter manifest nên tách riêng.

---

## 10.5. Manual trace

Bắt buộc chèn trực tiếp trong code cho:

```text
reshape
transpose
permute
flatten
split heads
merge heads
QK matmul
mask addition
softmax
V matmul
residual add
concat
position embedding add
gamma scaling
tanh
denormalization
```

Ví dụ:

```python
q_linear = self.q_proj(query)
trace.tensor(
    "cross.q_visual_linear",
    q_linear,
    layout="B,N,HxDh",
    operation="linear",
)

q_heads = (
    q_linear.view(batch, tokens, heads, head_dim)
    .transpose(1, 2)
    .contiguous()
)

trace.tensor(
    "cross.q_visual_heads",
    q_heads,
    layout="B,H,N,Dh",
    operation="reshape_transpose",
)

logits = torch.matmul(q_heads, k_heads.transpose(-2, -1)) * scale

trace.tensor(
    "cross.visual_to_text.logits",
    logits,
    layout="B,H,V,L",
    operation="matmul_scale",
)

masked_logits = logits + additive_mask

trace.tensor(
    "cross.visual_to_text.masked_logits",
    masked_logits,
    layout="B,H,V,L",
    operation="mask_add",
)

probs = torch.softmax(masked_logits, dim=-1)

trace.tensor(
    "cross.visual_to_text.probs",
    probs,
    layout="B,H,V,L",
    operation="softmax",
)
```

---

# 11. Files cần sửa trong TurboVLA

Tên file chính xác cần đối chiếu với repo hiện tại, nhưng phạm vi dự kiến:

```text
turbovla/debug/                              tạo mới

turbovla/evaluation/policy.py
turbovla/evaluation/suite_policy.py

turbovla/models/turbovla.py
turbovla/models/text_encoder.py
turbovla/models/vision_encoder.py
turbovla/models/action_head.py

turbovla/models/components/fusion.py
turbovla/models/components/transformer.py

third_party/vla_adapter/vla_adapter/rollout.py

scripts/export_parity_fixture.py
scripts/compare_tensor_traces.py
scripts/inspect_trace.py
scripts/inspect_legacy_trace.py
scripts/migrate_legacy_trace.py              optional
scripts/validate_trace_manifest.py
```

---

# 12. CLI đề xuất

## 12.1. Evaluation CLI

Thêm flags:

```text
--trace-enabled
--trace-root
--trace-level
--trace-max-forwards
--trace-include
--trace-exclude
--trace-save-pt
--trace-save-f32
--trace-save-raw-bf16
--trace-fail-on-nan
--trace-overwrite
--fixture-id
--fixture-output-dir
--fixture-only
--load-fixture
```

Ví dụ:

```bash
python experiments/libero/evaluate.py \
  ... \
  --task_ids 0 \
  --num_trials_per_task 1 \
  --trace-enabled true \
  --trace-root outputs/parity_traces/python \
  --trace-level boundary \
  --trace-max-forwards 1 \
  --trace-save-pt true \
  --trace-save-f32 true \
  --trace-save-raw-bf16 true \
  --fixture-id object_task0_episode0_step0
```

## 12.2. Environment variables

Hỗ trợ tương đương:

```bash
export TURBOVLA_TRACE_ENABLED=1
export TURBOVLA_TRACE_ROOT=outputs/parity_traces/python
export TURBOVLA_TRACE_LEVEL=boundary
export TURBOVLA_TRACE_MAX_FORWARDS=1
export TURBOVLA_TRACE_INCLUDE=""
export TURBOVLA_TRACE_EXCLUDE="*.dropout*"
export TURBOVLA_TRACE_SAVE_PT=1
export TURBOVLA_TRACE_SAVE_F32=1
export TURBOVLA_TRACE_SAVE_RAW_BF16=1
```

CLI ưu tiên hơn environment variables.

---

# 13. Deterministic parity mode

Tạo một mode riêng:

```bash
--parity-mode true
```

Mode này thực hiện:

```python
model.eval()

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

torch.use_deterministic_algorithms(True, warn_only=True)
```

Ngoài ra:

- disable dropout;
- không dùng stochastic augmentation;
- dùng eager/manual attention nếu backend hỗ trợ;
- không dùng Flash Attention;
- không dùng fused SDPA khi cần dump logits/probs;
- cùng LayerNorm epsilon;
- cùng GELU variant;
- cùng attention scale;
- cùng cast position.

## 13.1. Precision strategy

Thứ tự kiểm tra:

```text
1. PyTorch FP32 ↔ C++ FP32
2. PyTorch BF16 checkpoint cast ↔ C++ BF16
3. GGUF F16/BF16
4. Q8
5. Q4
```

Không bắt đầu bằng quantized model.

---

# 14. Parameter manifest

Ngoài activation trace, xuất một `weights_manifest.jsonl`.

Mỗi parameter:

```json
{
  "name": "vision_projection.mlp.0.weight",
  "shape": [1024, 768],
  "layout": "OUT,IN",
  "dtype": "bfloat16",
  "numel": 786432,
  "sha256_raw": "<hash>",
  "sha256_f32": "<hash>",
  "min": -0.1,
  "max": 0.1,
  "mean": 0.0,
  "std": 0.02
}
```

Mục tiêu:

- phát hiện mapping sai tên weight;
- transpose sai Linear weight;
- QKV packed order sai;
- missing bias;
- wrong LayerNorm weight/bias;
- wrong checkpoint conversion.

Phía C++ nên dump parameter metadata tương ứng sau khi load GGUF.

---

# 15. Compare script

Tạo:

```text
scripts/compare_tensor_traces.py
```

CLI:

```bash
python scripts/compare_tensor_traces.py \
  --reference outputs/parity_traces/python/object_task0_episode0_step0/forward_000000_rank_0 \
  --candidate outputs/parity_traces/cpp/object_task0_episode0_step0/forward_000000_rank_0 \
  --output outputs/parity_traces/compare/object_task0_episode0_step0 \
  --atol 0.02 \
  --rtol 0.02 \
  --stop-after-first-failure false
```

## 15.1. Matching

Match theo:

```text
semantic_name
```

Không match theo numeric trace ID vì thứ tự implementation Python và C++ có thể khác.

Numeric ID chỉ dùng để hiển thị execution order.

## 15.2. Metrics

Mỗi tensor tính:

```text
shape_match
layout_match
numel_match
exact_match
max_abs_error
mean_abs_error
median_abs_error
rmse
max_relative_error
mean_relative_error
cosine_similarity
pearson_correlation
first_bad_flat_index
first_bad_coordinate
reference_value
candidate_value
nan_count_reference
nan_count_candidate
inf_count_reference
inf_count_candidate
```

## 15.3. Status

```text
PASS_EXACT
PASS_TOLERANCE
FAIL_SHAPE
FAIL_LAYOUT
FAIL_NAN
FAIL_TOLERANCE
MISSING_REFERENCE
MISSING_CANDIDATE
```

## 15.4. `compare.csv`

Columns:

```text
reference_trace_id
candidate_trace_id
semantic_name
reference_shape
candidate_shape
reference_layout
candidate_layout
status
max_abs_error
mean_abs_error
rmse
max_relative_error
cosine_similarity
first_bad_flat_index
first_bad_coordinate
reference_value
candidate_value
```

## 15.5. `first_divergence.txt`

Ví dụ:

```text
First failing tensor:
interaction.layer_02.cross.visual_to_text.logits

Previous matched tensor:
interaction.layer_02.cross.k_text_heads
status=PASS_TOLERANCE
max_abs_error=0.0031
cosine_similarity=0.99998

Current tensor:
reference_shape=[1,4,512,21]
candidate_shape=[1,4,512,21]
reference_layout=B,H,V,L
candidate_layout=B,H,V,L
max_abs_error=0.421
mean_abs_error=0.088
rmse=0.116
cosine_similarity=0.973
first_bad_coordinate=[0,2,140,7]
reference_value=-1.942
candidate_value=-1.521

Likely suspects:
- K transpose order
- attention scale
- head dimension
- mask application
- BF16 cast position
```

## 15.6. Tạo diff tensor

Khi fail:

```text
diff = candidate - reference
abs_diff = abs(diff)
relative_diff = abs(diff) / max(abs(reference), epsilon)
```

Lưu:

```text
<semantic_name>.diff.f32le.bin
<semantic_name>.abs_diff.f32le.bin
<semantic_name>.relative_diff.f32le.bin
```

---

# 16. Summary và inspect tools

## 16.1. `summary.csv`

```text
trace_id
semantic_name
shape
layout
source_dtype
storage_dtype
min
max
mean
std
abs_max
l2_norm
nan_count
inf_count
file
```

## 16.2. `trace_tree.txt`

Ví dụ:

```text
00 input
  input.pixel_values
  input.input_ids
  input.state_normalized

10 text_encoder
  bert.embeddings
  bert.layer_00
    attn
    ffn
  ...
  projection

20 vision_encoder
  view_0
    patch_embedding
    block_00
    ...
  view_1
    ...

30 vision_projection

40 interaction
  layer_00
    cross
    text_enhancer
  ...

50 state_projection

60 action_decoder
  layer_00
  layer_01
  layer_02

70 output
  before_tanh
  normalized
  denormalized
```

## 16.3. `inspect_trace.py`

Ví dụ:

```bash
python scripts/inspect_trace.py \
  outputs/parity_traces/python/... \
  --name "interaction.layer_02.cross.visual_to_text.probs"
```

Output:

```text
name: interaction.layer_02.cross.visual_to_text.probs
shape: [1, 4, 512, 21]
layout: B,H,V,L
dtype: bfloat16 -> float32
min: 0
max: 0.71
mean: 0.047619
std: 0.062
nan: 0
inf: 0
first values: ...
```

---

# 17. Storage safeguards

Op-level trace có thể rất lớn.

Cần hỗ trợ:

```text
--trace-max-forwards
--trace-max-bytes
--trace-max-tensor-numel
--trace-include
--trace-exclude
--trace-save-pt
--trace-save-raw-bf16
```

## 17.1. Chính sách đề xuất

### Boundary

- Lưu full tensor.
- Lưu `.pt`.
- Lưu `.f32le.bin`.
- Có thể lưu raw BF16.

### Layer

- Lưu full tensor.
- `.pt` optional.
- `.f32le.bin` bắt buộc.

### Op

- Chỉ bật cho một block.
- Không lưu `.pt` mặc định.
- Không dump dropout.
- Không dump lặp lại tensor alias giống hệt nếu hash trùng và muốn tiết kiệm.

## 17.2. Không overwrite im lặng

Nếu output tồn tại:

```text
raise FileExistsError
```

trừ khi:

```text
--trace-overwrite true
```

---

# 18. Validation của trace

Tạo:

```text
scripts/validate_trace_manifest.py
```

Kiểm tra:

1. Mọi file trong manifest tồn tại.
2. `numel × dtype_size == file_size`.
3. Shape product đúng `numel`.
4. Hash đúng.
5. Không semantic name trùng ngoài trường hợp có `call_index`.
6. Layout rank khớp số chiều.
7. Không NaN/Inf ngoài tensor được cho phép.
8. Trace ID tăng đơn điệu.
9. Fixture hash đúng.
10. Model checkpoint hash đúng.

CLI:

```bash
python scripts/validate_trace_manifest.py \
  outputs/parity_traces/python/.../forward_000000_rank_0
```

---

# 19. Plan triển khai theo phase

## Phase 1 — Bảo toàn legacy output và chuẩn hóa collector

Mục tiêu:

- đổi tên output cũ thành `embedding_dumps_legacy_<timestamp>`;
- tuyệt đối không sửa hoặc overwrite legacy output;
- giữ logic dump hiện tại làm baseline;
- thêm semantic name;
- thêm layout;
- thêm dtype suffix;
- chuyển manifest sang JSONL;
- thêm summary CSV;
- thêm canonical `.f32le.bin`;
- thêm validation và migration report cho format cũ.

Deliverables:

```text
TensorWriter
TensorRecord
manifest.jsonl
summary.csv
scripts/inspect_legacy_trace.py
optional: scripts/migrate_legacy_trace.py
```

Acceptance:

- Legacy output vẫn tồn tại và đọc được.
- Collector mới ghi sang `outputs/parity_traces`.
- Mỗi tensor mới có semantic name.
- File size khớp shape.
- Python load lại binary đúng.
- Không phá evaluation khi trace disabled.
- Không có code path tự động xoá legacy output.

---

## Phase 2 — Fixture export

Mục tiêu:

- lưu input preprocessing đầy đủ;
- cho phép replay không cần simulator.

Deliverables:

```text
FixtureWriter
scripts/export_parity_fixture.py
--fixture-only
--load-fixture
```

Acceptance:

- Chạy model từ simulator và từ fixture cho output action giống nhau trong tolerance.
- Fixture có image, token, mask, state.
- Hash ổn định.

---

## Phase 3 — Boundary trace

Mục tiêu:

- dump block lớn;
- tìm stage đầu tiên lệch với C++.

Deliverables:

```text
boundary traces cho text, vision, interaction, action
trace tree
```

Acceptance:

- Có tối thiểu output từng transformer block.
- Có final normalized và denormalized action.
- Trace một forward duy nhất.

---

## Phase 4 — Leaf module hooks

Mục tiêu:

- tự động trace Linear, LayerNorm, Conv, activation.

Deliverables:

```text
HookManager
module path mapping
call index
```

Acceptance:

- Không duplicate hook.
- Module gọi nhiều lần có call index.
- Trace disabled không gây overhead đáng kể.

---

## Phase 5 — Manual op traces

Mục tiêu:

- trace Q/K/V;
- logits;
- masks;
- softmax;
- residual;
- reshape/transpose;
- concat;
- tanh;
- denormalization.

Ưu tiên:

```text
1. interaction fusion
2. action decoder
3. vision projection
4. text encoder
5. DINO
```

Acceptance:

- Tìm được tensor trước và sau mọi attention matmul.
- Có layout cho từng Q/K/V/logits/probs.
- Có pre/post residual.

---

## Phase 6 — C++ TensorWriter

Mục tiêu:

- `vla.cpp` ghi cùng semantic name và format.

Yêu cầu:

- same semantic name;
- same canonical float32 binary;
- same manifest schema;
- same layout notation;
- same fixture.

Acceptance:

- Compare script match được tensor hai bên.
- Không cần custom mapping thủ công cho từng run.

---

## Phase 7 — Compare tool

Mục tiêu:

- tự động tìm first divergence.

Deliverables:

```text
compare.csv
compare.json
first_divergence.txt
missing files
diff tensors
```

Acceptance:

- Unit test với tensor giống nhau.
- Unit test shape mismatch.
- Unit test transpose mismatch.
- Unit test BF16 tolerance.
- Unit test NaN.

---

## Phase 8 — BF16 parity

Thứ tự:

```text
input exact
preprocessing exact
text tokens exact
DINO block-by-block
BERT block-by-block
vision projection
interaction layers
state projection
action decoder
action MLP
tanh
denormalization
```

Không chuyển sang quantized model trước khi hoàn thành phase này.

---

## Phase 9 — Quantization analysis

Sau khi BF16 parity đạt:

```text
BF16 reference
↔ F16 GGUF
↔ Q8
↔ Q4
```

Báo cáo thêm:

```text
error growth by layer
cosine degradation
final action error
success-rate impact
```

---

# 20. Acceptance criteria tổng thể

Hệ thống hoàn thành khi đáp ứng:

## Functional

- Trace disabled: model chạy như cũ.
- Trace enabled: chỉ dump số forward yêu cầu.
- Replay fixture cho output giống rollout origin.
- Python và C++ đọc cùng fixture.
- Compare theo semantic name.
- Tìm được first divergence.

## Data integrity

- Mọi tensor có shape/layout/dtype.
- Canonical binary là float32 little-endian contiguous.
- Hash hợp lệ.
- Manifest không thiếu file.
- Không overwrite ngoài ý muốn.
- Legacy output không bị thay đổi hoặc xoá tự động.
- Output mới và legacy output nằm ở hai namespace riêng.
- Chỉ xoá legacy sau khi fixture replay, validator và compare workflow đều đạt.

## Debug quality

- Có boundary trace.
- Có layer trace.
- Có op trace.
- Có Q/K/V/logits/probs.
- Có residual.
- Có preprocessing.
- Có final action trước/sau denormalization.

## Performance

- Trace disabled overhead gần 0.
- Trace boundary không làm evaluation OOM.
- Có filter và byte limit.
- Có thể trace một block riêng.

---

# 21. Các lỗi parity phổ biến cần ghi trong report

Khi compare fail, tool nên gợi ý các nguyên nhân dựa trên tensor đầu tiên lệch.

## Input image lệch

Kiểm tra:

```text
RGB/BGR
rotation
vertical flip
resize interpolation
0..255 vs 0..1
mean/std
NCHW/NHWC
camera order
```

## Text lệch

Kiểm tra:

```text
tokenizer version
special tokens
padding side
truncation
position IDs
mask type
custom self-attention mask
```

## Linear lệch

Kiểm tra:

```text
weight transpose
bias missing
QKV packing order
GGUF tensor mapping
BF16 decode
```

## Attention logits lệch

Kiểm tra:

```text
head split order
K transpose
scale
mask before/after scale
mask polarity
softmax axis
cast position
```

## Softmax lệch nhưng logits đúng

Kiểm tra:

```text
numerical stabilization
max subtraction
precision
softmax dimension
masked values
```

## Residual lệch nhưng delta đúng

Kiểm tra:

```text
wrong residual source
pre-norm/post-norm
in-place alias
gamma scaling
```

## LayerNorm lệch

Kiểm tra:

```text
epsilon
variance definition
reduction precision
weight/bias mapping
axis
```

## Final action lệch

Kiểm tra:

```text
tanh
action dimension order
normalization min/max
gripper handling
binary gripper conversion
chunk slicing
```

---

# 22. Lệnh workflow đề xuất

## 22.1. Tạo fixture

```bash
python experiments/libero/evaluate.py \
  ... \
  --task_ids 0 \
  --num_trials_per_task 1 \
  --fixture-only true \
  --fixture-id object_task0_episode0_step0 \
  --fixture-output-dir outputs/parity_traces/fixtures
```

## 22.2. Boundary trace từ fixture

```bash
python scripts/run_fixture_inference.py \
  --fixture outputs/parity_traces/fixtures/object_task0_episode0_step0 \
  --checkpoint pretrained/TurboVLA/checkpoints/libero/object.pth \
  --trace-root outputs/parity_traces/python \
  --trace-level boundary \
  --trace-max-forwards 1
```

## 22.3. Trace một interaction layer

```bash
python scripts/run_fixture_inference.py \
  --fixture outputs/parity_traces/fixtures/object_task0_episode0_step0 \
  --checkpoint pretrained/TurboVLA/checkpoints/libero/object.pth \
  --trace-root outputs/parity_traces/python \
  --trace-level op \
  --trace-include "interaction.layer_02" \
  --trace-max-forwards 1
```

## 22.4. Validate trace

```bash
python scripts/validate_trace_manifest.py \
  outputs/parity_traces/python/object_task0_episode0_step0/forward_000000_rank_0
```

## 22.5. Compare

```bash
python scripts/compare_tensor_traces.py \
  --reference outputs/parity_traces/python/object_task0_episode0_step0/forward_000000_rank_0 \
  --candidate outputs/parity_traces/cpp/object_task0_episode0_step0/forward_000000_rank_0 \
  --output outputs/parity_traces/compare/object_task0_episode0_step0
```

---

# 23. Unit tests cần viết

```text
tests/debug/test_tensor_writer.py
tests/debug/test_manifest.py
tests/debug/test_fixture_roundtrip.py
tests/debug/test_trace_filter.py
tests/debug/test_compare_exact.py
tests/debug/test_compare_tolerance.py
tests/debug/test_compare_shape_mismatch.py
tests/debug/test_compare_layout_mismatch.py
tests/debug/test_compare_nan.py
tests/debug/test_bf16_raw_bits.py
```

## 23.1. Tensor writer roundtrip

- tạo tensor;
- write;
- read binary;
- reshape;
- compare exact float32.

## 23.2. BF16 raw roundtrip

- tạo BF16;
- save raw uint16;
- reconstruct;
- compare exact bits.

## 23.3. Fixture replay

- inference trực tiếp;
- inference từ fixture;
- compare final action.

## 23.4. Semantic duplicate

- hai tensor cùng name và call index phải báo lỗi.

## 23.5. Compare transpose

- cùng shape/numel nhưng transpose layout;
- report phải phát hiện layout mismatch hoặc numerical mismatch.

---

# 24. Yêu cầu cho coding agent

Agent cần:

1. Đọc toàn bộ forward path trước khi sửa.
2. Trước khi chạy collector mới, đổi tên output cũ thành `embedding_dumps_legacy_<timestamp>`.
3. Không xoá, overwrite hoặc mutate legacy output.
4. Không hard-code shape.
5. Không thay đổi numerical behavior khi trace disabled.
6. Không đổi public API ngoài các flag mới.
7. Viết code theo từng phase.
8. Chạy test sau mỗi phase.
9. Không bật op-level trace toàn model mặc định.
10. Giữ backward compatibility với dump hiện tại nếu khả thi.
11. Tạo migration note và legacy inspection report.
12. Chỉ đề xuất xoá legacy output khi toàn bộ deletion gates trong mục 0.1 đã đạt.
13. Cập nhật README với ví dụ lệnh.

---

# 25. Prompt ngắn để giao cho coding agent

```text
Implement the TurboVLA tensor tracing and PyTorch-vla.cpp parity debugging system described in TURBOVLA_VLACPP_TENSOR_DUMP_PLAN.md.

Work phase by phase. First inspect the actual TurboVLA forward graph and map the real module/file names before editing. Do not hard-code tensor shapes. Preserve existing inference behavior when tracing is disabled.

Core requirements:
- Preserve the existing dump before any changes by moving it to `outputs/embedding_dumps_legacy_<timestamp>`. Never delete, overwrite, or mutate legacy dumps automatically.
- Write all new traces under `outputs/parity_traces`.
- Add a read-only legacy inspection/migration report and keep unresolved legacy tensor names explicitly marked rather than guessed.
- Export a deterministic input fixture containing raw/preprocessed images, tokens, all masks, position IDs, raw/normalized state, metadata, and hashes.
- Add semantic tensor names, explicit layouts, dtype/endianness metadata, JSONL manifest, CSV summary, and canonical float32 little-endian binaries.
- Support boundary, layer, and op trace levels with include/exclude filters and max-forward/max-byte limits.
- Use hooks for leaf nn.Modules and manual trace points for reshape, transpose, Q/K/V split, logits, masks, softmax, residual additions, concat, positional additions, tanh, and action denormalization.
- Add a replay-from-fixture path that does not require LIBERO.
- Add trace validation and Python-vla.cpp comparison scripts that report missing tensors, shape/layout mismatches, numerical metrics, and the first divergent tensor.
- Add unit tests for binary roundtrip, BF16 raw bits, fixture replay, trace filters, duplicate semantic names, exact/tolerance comparison, shape/layout mismatch, and NaN handling.
- Keep tracing disabled by default.
- Document all CLI flags and example workflows.

Implement and validate each phase separately. After every phase, summarize modified files, tests run, and remaining work.
```

---


# 26. Checklist bắt đầu triển khai

Chạy các bước này trước khi giao agent sửa code:

```bash
cd ~/Desktop/TurboVLA

timestamp=$(date +%Y%m%d_%H%M%S)

if [ -d outputs/embedding_dumps ]; then
  mv outputs/embedding_dumps \
     "outputs/embedding_dumps_legacy_${timestamp}"
fi

mkdir -p outputs/parity_traces
```

Kiểm tra:

```bash
find outputs -maxdepth 2 -type d | sort
du -sh outputs/embedding_dumps_legacy_* 2>/dev/null || true
```

Thêm `.gitignore`:

```bash
cat >> .gitignore <<'EOF'

# Tensor parity traces
outputs/embedding_dumps/
outputs/embedding_dumps_legacy_*/
outputs/parity_traces/
outputs/*.tar.zst
EOF
```

Kiểm tra working tree:

```bash
git status --short
```

Sau đó mới giao coding agent thực hiện theo phase.

Prompt khởi động đề xuất:

```text
Read TURBOVLA_VLACPP_TENSOR_DUMP_PLAN.md completely before modifying code.

Start with the pre-implementation legacy-output preservation step and Phase 1 only. Inspect the real repository structure and current dump implementation first. Do not delete, overwrite, or mutate any legacy output. Write new traces only under outputs/parity_traces.

After Phase 1:
1. list every modified file;
2. explain the actual forward-path/module mapping discovered;
3. show the new manifest schema;
4. run focused unit tests;
5. validate one small trace;
6. report disk usage;
7. stop and wait before implementing Phase 2.
```

---

# 27. Kết luận


Dump hiện tại là nền tốt nhưng chưa đủ để debug parity ở mức implementation.

Các thay đổi quan trọng nhất:

```text
numeric ID + semantic name
explicit layout
full deterministic fixture
canonical f32 little-endian binary
JSONL manifest
boundary/layer/op levels
manual functional-op trace
first-divergence compare tool
BF16 parity before quantization
```

Mục tiêu cuối cùng không phải dump càng nhiều càng tốt. Mục tiêu là:

```text
tìm tensor đầu tiên lệch
→ xác định operation gây lệch
→ sửa đúng block
→ chạy lại từ fixture
→ xác nhận sai số không lan sang layer sau
```
