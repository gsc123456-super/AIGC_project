# Day 06 教学引导：鲁棒性测试与可解释性分析

日期：2026-06-24

## 今日目标

模拟真实传播场景中的图像退化，并用 Grad-CAM 与 FFT 频谱解释模型判断依据。

## 已完成内容

1. 新增 `src/robustness.py`。
2. 支持 `clean`、JPEG 压缩、模糊、缩放、中心裁剪、高斯噪声。
3. 新增 `src/visualize_gradcam.py`。
4. 新增 `src/visualize_fft.py`。
5. 输出目录统一放在 `results/tables` 和 `results/figures`。

## 鲁棒性命令

```powershell
conda run -n aigc python src\robustness.py ^
  --config configs\convnext_tiny_genimage.yaml ^
  --ckpt results\checkpoints\convnext_tiny_genimage\best_f1.pth ^
  --out results\tables\robustness_results.csv
```

指定外部测试集：

```powershell
conda run -n aigc python src\robustness.py ^
  --config configs\rgb_fft_fusion_genimage.yaml ^
  --ckpt results\checkpoints\rgb_fft_fusion_genimage\best_f1.pth ^
  --test-csv data\synthbuster_eval\meta\synthbuster_eval.csv
```

## 可解释性命令

```powershell
conda run -n aigc python src\visualize_gradcam.py --config configs\convnext_tiny_genimage.yaml --ckpt results\checkpoints\convnext_tiny_genimage\best_f1.pth --num-samples 12
conda run -n aigc python src\visualize_fft.py --config configs\convnext_tiny_genimage.yaml --num-samples 12
```

## 验收标准

```text
1. results/tables/robustness_results.csv 存在。
2. CSV 中包含 clean、jpeg_q75、jpeg_q50、blur_k3 等退化项。
3. results/figures/gradcam 下有热力图。
4. results/figures/fft 下有原图 + 频谱对照图。
5. 能结合失败案例解释模型可能依赖了哪些线索。
```

## 你要能讲清楚

真实平台上的图片通常会被压缩、缩放、二次传播。只在 clean test 上表现好不够，鲁棒性测试能说明模型在真实内容安全场景中是否稳定。
