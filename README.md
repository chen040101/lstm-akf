# newlstm-akf

这是一个用于装甲板中心点预测的工程目录，包含数据集构建、主模型训练、模型验证和固定基线对比。

## 当前流程

- 检测层：YOLO 输出当前最好打击目标的中心点。
- 状态层：AKF 负责平滑观测、短时补位和未来基线外推。
- 主模型：`LSTM-AKF` 采用门控融合，不再是固定相加。
- 主训练：支持早停，避免长期无提升时继续训练。
- 模型对比：`LSTM-AKF` 只读取外部 checkpoint，没有就自动跳过该项。

## 目录说明

- `docs/`：项目背景、数据集方案、模型结构、操作记录等文档。
- `scripts/`：数据集构建、训练、验证、模型对比的入口脚本。
- `lstm_akf/`：核心源码。
- `data/video/`：输入视频。
- `data/dataset/`：生成后的 `train/`、`val/` 和 `split_manifest.json`。
- `weights/yolo/`：YOLO 权重。
- `outputs/`：主模型训练输出目录。
- `Baseline Models/`：基线模型产物与对比结果。

## 数据说明

- 神经网络输入特征目前只有历史 `xy`。
- 速度、加速度由 AKF 内部估计与外推。
- 数据集支持 `sample`、`segment`、`video` 三种划分方式。
- 当前默认划分方式是 `segment`。
- 预测目标为未来 `1~15` 帧归一化中心点。

## 快速开始

```bash
conda activate yolo
python scripts/build_dataset.py --val-ratio 0.2 --split-mode segment
python scripts/train.py --device cuda --epochs 200 --batch-size 64 --early-stopping-patience 20
python scripts/validate.py --device cuda --checkpoint outputs/exp001/checkpoints/best.pt
python scripts/compare_models.py --device cuda --point-frames 8 15 --sequence-horizons 8 15
python scripts/visualize_predictions.py --device cuda --sample-index 0 --num-samples 5 --horizons 1 8 15
```

## 说明

- 数据集构建后会输出训练集、验证集和划分清单。
- 新构建的数据集 JSON 会额外保留 `meta`，用于把样本映射回原视频帧。
- 主模型默认一次输出未来 `1~15` 帧预测结果。
- 对比脚本期望 `Baseline Models/lstm_akf/checkpoints/best.pt` 已存在。
- 可视化脚本默认每张图只画一个来源，不生成局部放大图；用 `--num-samples` 控制输出多少组样本。
- 更详细的说明请查看 `docs/` 目录。
