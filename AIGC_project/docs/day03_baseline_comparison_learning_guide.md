# Day 03 教学引导：三组 RGB Baseline 对比

日期：2026-06-24

## 今日目标

建立正式对照组：ResNet50、EfficientNet-B0、ConvNeXt-Tiny。后续 FFT-only 和 RGB+FFT Fusion 都要和这些 baseline 比较。

## 已完成内容

1. `src/models/rgb_classifier.py` 支持 `resnet50`、`efficientnet_b0`、`convnext_tiny`。
2. 新增 CIFAKE quick 配置：
   - `configs/efficientnet_b0_cifake_quick_gpu.yaml`
   - `configs/convnext_tiny_cifake_quick_gpu.yaml`
3. 新增 GenImage 正式配置：
   - `configs/resnet50_genimage.yaml`
   - `configs/efficientnet_b0_genimage.yaml`
   - `configs/convnext_tiny_genimage.yaml`
4. `src/eval.py` 支持按 source 分组评估。

## 训练命令

```powershell
conda run -n aigc python src\train.py --config configs\resnet50_genimage.yaml
conda run -n aigc python src\train.py --config configs\efficientnet_b0_genimage.yaml
conda run -n aigc python src\train.py --config configs\convnext_tiny_genimage.yaml
```

## 评估命令

```powershell
conda run -n aigc python src\eval.py --config configs\convnext_tiny_genimage.yaml --ckpt results\checkpoints\convnext_tiny_genimage\best_f1.pth --out results\tables\baseline_results.csv --group-by-source
```

## 如何读 baseline 表

```text
model,input_size,acc,precision,recall,f1,auc,time_per_image_ms
```

Accuracy 看整体正确率，Precision 看误报控制，Recall 看漏检控制，F1 看综合检测能力，AUC 看模型排序区分能力。

## 验收标准

```text
1. 至少 ResNet50 和 ConvNeXt-Tiny 能完成训练。
2. 每个模型都有 best_f1.pth。
3. results/tables/baseline_results.csv 存在。
4. 能说清楚哪个模型更适合作为主力模型，以及原因。
```

## 面试表达

我没有直接上复杂方法，而是先建立了三组 RGB baseline。这样后续 RGB+FFT 是否有效，不是凭感觉判断，而是能和 ResNet、EfficientNet、ConvNeXt 这些标准视觉骨干进行量化比较。
