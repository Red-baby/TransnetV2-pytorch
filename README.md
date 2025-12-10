# TransNetV2 的 PyTorch 训练与推理

本目录提供基于 `transnetv2_pytorch.py` 的最小 PyTorch 训练 / 评估 / 推理流程，权重可用 `inference-pytorch/convert_weights.py` 从 TensorFlow 转换得到。
（已内置 `training_pytorch/transnetv2_pytorch.py`，无需额外路径配置；若需转换权重，可继续使用仓库根下的转换脚本生成 `.pth`。）

## 数据清单（manifest）格式
逐行 JSON（`.jsonl`），每行代表一个样本：
```json
{
  "video_path": "/path/to/video.mp4",   // 或者 "frames_dir": "/path/to/frames",
  "labels": [0, 0, 1, ...],             // 每帧二分类标签，长度等于帧数
  "many_hot_labels": [0, 0, 1, ...]     // 可选；若缺省则复用 labels
}
```
推荐使用已解码的 `frames_dir`（RGB 帧目录，文件名排序即可），否则用 `video_path` 通过 OpenCV 解码。

## 命令行用法（argparse，无 gin）
训练（可带验证集）：
```bash
python training_pytorch/train.py \
  --train-manifest /path/train.jsonl \
  --val-manifest /path/val.jsonl \
  --weights transnetv2-pytorch-weights.pth \
  --output-dir runs/pytorch \
  --epochs 10 --batch-size 8 --lr 1e-4 --window 100 --stride 25
```

评估一个 manifest：
```bash
python training_pytorch/eval.py \
  --manifest /path/val.jsonl \
  --weights runs/pytorch/last_state_dict.pt
```

单视频推理 / 验证：
```bash
python training_pytorch/infer.py \
  --video-path /path/video.mp4 \
  --weights runs/pytorch/last_state_dict.pt \
  --window 100 --stride 25 --threshold 0.5 --save-npy preds.npy
```

## 说明
- 模型输入为 27x48 RGB uint8 帧，数据加载/推理工具会自动 resize 和排列维度。
- 需要安装 OpenCV 解码：`pip install opencv-python`。
- `train.py` 会保存 `checkpoint_epoch_*.pt`（含 optimizer）以及 `last_state_dict.pt`（仅模型）。
