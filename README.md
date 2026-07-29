# AIGC-Guard

一个用于区分真实图像与 AI 生成图像的 PyTorch 项目。项目包含数据 manifest 生成、训练、评估、鲁棒性测试、FFT 可视化、Grad-CAM 可解释性分析和 Gradio 单图预测 Demo。

## 项目结构

```text
AIGC_project/
  app/                 # Gradio Demo
  configs/             # 训练与评估配置
  data/*/meta/         # 可提交的 CSV manifest
  docs/                # 实验记录与总结
  results/tables/      # 实验结果表
  scripts/             # 数据准备脚本
  src/                 # 训练、评估、模型与工具代码
```

大数据、训练权重、缓存和日志不建议上传 GitHub，已通过 `.gitignore` 排除。

## 快速开始

```powershell
cd AIGC_project
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

如果需要使用 CUDA，请按你的显卡和 CUDA 版本安装对应的 PyTorch wheel。

## 数据准备

把原始数据放到项目内的相对目录：

```text
AIGC_project/data/raw/genimage/
AIGC_project/data/raw/synthbuster/
AIGC_project/data/cifake/
```

生成 GenImage 子集 manifest：

```powershell
python scripts\prepare_genimage_subset.py --raw-root data\raw\genimage --out-root data\genimage_subset
```

生成 Synthbuster 外部测试 manifest：

```powershell
python scripts\prepare_synthbuster_eval.py --fake-root data\raw\synthbuster --real-csv data\genimage_subset\meta\test.csv --out-root data\synthbuster_eval
```

## 训练与评估

```powershell
python src\train.py --config configs\convnext_tiny_genimage_quick_final.yaml
python src\eval.py --config configs\convnext_tiny_genimage_quick_final.yaml --ckpt results\checkpoints\convnext_tiny_genimage_quick_final\best_f1.pth --group-by-source
```

启动 Demo：

```powershell
python app\gradio_app.py --config configs\convnext_tiny_genimage_quick_final.yaml --ckpt results\checkpoints\convnext_tiny_genimage_quick_final\best_f1.pth
```

> 注意：仓库不包含训练好的 `.pth` 权重文件。请先训练生成 checkpoint，或手动提供自己的 checkpoint。
