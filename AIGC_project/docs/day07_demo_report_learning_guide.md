# Day 07 教学引导：Gradio Demo、README 与项目报告

日期：2026-06-24

## 今日目标

把项目整理成可以展示、可以复现、可以写进简历并能在面试中讲清楚的版本。

## 已完成内容

1. 新增 `app/gradio_app.py`。
2. 新增 `src/predict.py`，支持单图预测。
3. 更新 `README.md`。
4. 新增 `report.md`。
5. 每日教学引导文档补齐到 Day07。

## Demo 启动命令

默认使用外部已有 CIFAKE ResNet50 checkpoint：

```powershell
conda run -n aigc python app\gradio_app.py
```

指定自己的模型：

```powershell
conda run -n aigc python app\gradio_app.py ^
  --config configs\rgb_fft_fusion_genimage.yaml ^
  --ckpt results\checkpoints\rgb_fft_fusion_genimage\best_f1.pth
```

打开：

```text
http://127.0.0.1:7860
```

## 单图预测命令

```powershell
conda run -n aigc python src\predict.py --config configs\resnet50_cifake_quick_gpu.yaml --ckpt results\checkpoints\resnet50_cifake_quick_gpu\best_f1.pth --image path\to\image.jpg
```

## 最终交付物

```text
README.md
report.md
configs/*.yaml
src/*.py
src/models/*.py
scripts/*.py
app/gradio_app.py
docs/day01...day07...md
results/tables/*.csv
results/figures/fft/*.png
results/figures/gradcam/*.png
```

## 验收标准

```text
1. README 能让别人安装、准备数据、训练、评估、启动 Demo。
2. report 能解释问题定义、数据、方法、实验、鲁棒性、可解释性和局限。
3. Demo 能上传图片并输出 Real/AI-generated 与 AI 概率。
4. 你能用 1 分钟讲清楚项目主线。
```

## 1 分钟表达

我做的是 AIGC 图像检测系统，目标是判断输入图像是真实图还是 AI 生成图。项目先用 CIFAKE 跑通训练闭环，再用 GenImage 子集作为正式训练和内部测试数据，并用 Synthbuster 做外部生成器泛化测试。方法上，我实现了 ResNet50、EfficientNet-B0、ConvNeXt-Tiny baseline，并加入 RGB+FFT 双分支模型，让模型同时学习空间域纹理和频域伪影。实验部分包含 baseline 对比、外部泛化、JPEG/Blur 等鲁棒性测试，以及 Grad-CAM 和 FFT 可解释性分析，最后用 Gradio 做成可上传图片预测的 Demo。
