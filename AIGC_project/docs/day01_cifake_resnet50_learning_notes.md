# Day 01 教学引导：CIFAKE + ResNet50 训练闭环

日期：2026-06-24

## 今日目标

用 CIFAKE 跑通一个最小但完整的 AIGC 图像二分类流程：数据读取、预处理、模型构建、训练、验证、保存 checkpoint、测试并输出指标。

## 已完成内容

1. 使用 `BinaryImageDataset` 读取 `REAL/FAKE` 目录结构。
2. 统一标签约定：`REAL/nature -> 0`，`FAKE/ai -> 1`。
3. 使用 ResNet50 作为第一个 baseline。
4. 训练脚本保存 `best_f1.pth` 和 `last.pth`。
5. 评估脚本输出 Accuracy、Precision、Recall、F1、AUC、Loss、Time/Image。
6. 修复小样本抽样时类别不均衡导致 AUC 为 `nan` 的问题。

## 关键文件

```text
configs/resnet50_cifake_quick_gpu.yaml
src/datasets.py
src/transforms.py
src/models/rgb_classifier.py
src/train.py
src/eval.py
src/metrics.py
src/utils.py
```

## 运行命令

```powershell
conda run -n aigc python scripts\check_dataset.py --root data\cifake
conda run -n aigc python src\train.py --config configs\resnet50_cifake_quick_gpu.yaml
conda run -n aigc python src\eval.py --config configs\resnet50_cifake_quick_gpu.yaml --ckpt results\checkpoints\resnet50_cifake_quick_gpu\best_f1.pth --out results\tables\resnet50_cifake_quick_gpu_eval.csv
```

如果使用外部已有 checkpoint：

```powershell
conda run -n aigc python src\eval.py --config configs\resnet50_cifake_quick_gpu.yaml --ckpt results\checkpoints\resnet50_cifake_quick_gpu\best_f1.pth
```

## 今天要理解的知识

Baseline 是后续所有改进的参照物。ResNet50 不一定是最终最强模型，但它结构稳定、复现成本低，适合作为第一条基线。

Smoke test 的目的不是追求高分，而是确认数据、模型、loss、optimizer、checkpoint、eval 全链路没有断点。

F1 比 Accuracy 更适合二分类检测场景，因为它同时考虑误报和漏报。AUC 衡量模型把真实图和 AI 图整体区分开的能力。

## 验收标准

```text
1. CIFAKE 数据集能被脚本识别。
2. ResNet50 能至少完成 1 个 epoch 训练。
3. results/checkpoints/.../best_f1.pth 存在。
4. results/tables/...csv 中包含 acc、precision、recall、f1、auc。
5. 你能解释为什么测试集抽样要保持 REAL/FAKE 均衡。
```

## 面试表达

我先用 CIFAKE 建立了 AIGC 图像检测的完整训练闭环，并用 ResNet50 作为可复现 baseline。这个阶段的重点不是刷最高分，而是把数据读取、训练、验证、测试、指标保存和 checkpoint 管理模块化，为后续切换 GenImage、ConvNeXt、FFT 融合和外部泛化测试打基础。
