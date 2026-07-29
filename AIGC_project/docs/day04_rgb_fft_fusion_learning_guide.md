# Day 04 教学引导：FFT-only 与 RGB+FFT Fusion

日期：2026-06-24

## 今日目标

完成项目创新点：把 RGB 空间域特征和 FFT 频域特征融合，用来检测 AI 生成图像可能留下的频域伪影。

## 已完成内容

1. 新增 `src/models/fft_classifier.py`。
2. 新增 `src/models/fusion_classifier.py`。
3. 新增 `src/models/model_factory.py`，统一构建 RGB、FFT-only、RGB+FFT 模型。
4. 新增配置：
   - `configs/fft_only_genimage.yaml`
   - `configs/rgb_fft_fusion_genimage.yaml`
5. `src/train.py` 和 `src/eval.py` 已支持新模型。

## 方法直觉

RGB 分支学习纹理、语义和局部视觉模式。FFT 分支把图像转换到频域，关注周期性纹理、异常高频分布和生成器残留伪影。融合后模型既看“图像长什么样”，也看“频率分布像不像真实相机图像”。

## 训练命令

```powershell
conda run -n aigc python src\train.py --config configs\fft_only_genimage.yaml
conda run -n aigc python src\train.py --config configs\rgb_fft_fusion_genimage.yaml
```

## 可视化频谱

```powershell
conda run -n aigc python src\visualize_fft.py --config configs\convnext_tiny_genimage.yaml --num-samples 12 --out-dir results\figures\fft
```

## 验收标准

```text
1. FFT-only 能正常前向、训练和评估。
2. RGB+FFT Fusion 能正常前向、训练和评估。
3. results/figures/fft 下生成真实图和 AI 图的频谱示例。
4. results/tables/fusion_results.csv 或 eval 输出中能比较 convnext_tiny、fft_only、rgb_fft_fusion。
```

## 你要能讲清楚

FFT 不是为了替代 RGB，而是补充 RGB 不容易直接捕捉的频域线索。如果 Fusion 没有在 clean test 上超过 ConvNeXt，也可以重点观察它在 JPEG、Blur 或外部数据集上的稳定性。
