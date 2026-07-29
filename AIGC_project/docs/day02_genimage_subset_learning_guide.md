# Day 02 教学引导：整理 GenImage 子集

日期：2026-06-24

## 今日目标

把正式训练数据从 CIFAKE 切换到 GenImage 子集。GenImage 更适合作为主训练集，因为它包含不同生成器来源，能支撑“跨生成器泛化”的项目叙事。

## 已完成内容

1. 新增 `scripts/prepare_genimage_subset.py`。
2. 脚本支持从解压后的 GenImage 目录中扫描 `ai/fake/nature/real` 图像。
3. 默认生成 CSV manifest，不强制复制图片，节省磁盘空间。
4. 支持按 `source + label` 分组后划分 `train/val/test`。
5. 支持可选 `copy` 或 `hardlink` 导出 ImageFolder 结构。
6. 新增 `train_csv/val_csv/test_csv` 数据配置能力。

## 运行命令

先确认 BigGAN/ADM 已经解压。你当前外部目录里有分卷压缩包，但还没有可扫描图片时，脚本会提示先解压。

```powershell
conda run -n aigc python scripts\prepare_genimage_subset.py ^
  --raw-root data\raw\genimage ^
  --out-root data\genimage_subset ^
  --max-per-source-class 3000
```

检查 manifest：

```powershell
conda run -n aigc python scripts\check_dataset.py --csv data\genimage_subset\meta\train.csv
conda run -n aigc python scripts\check_dataset.py --csv data\genimage_subset\meta\val.csv
conda run -n aigc python scripts\check_dataset.py --csv data\genimage_subset\meta\test.csv
```

如果想导出目录结构：

```powershell
conda run -n aigc python scripts\prepare_genimage_subset.py --raw-root data\raw\genimage --out-root data\genimage_subset --export hardlink
```

## 为什么用 CSV manifest

GenImage 图片很多，复制一份会浪费大量磁盘。CSV 只记录图片路径、标签、来源和 split，可以让训练代码直接读取原图，同时保持实验划分可复现。

## 验收标准

```text
1. data/genimage_subset/meta/train.csv 存在。
2. val.csv 和 test.csv 存在。
3. 每个 CSV 都有 path,label,source,split 四列。
4. real/fake 数量基本均衡。
5. source 字段能区分 BigGAN、ADM 等生成器。
```

## 你要能讲清楚

为什么不能只在 CIFAKE 上做最终实验？因为 CIFAKE 更像流程验证集，无法证明模型面对不同生成器时仍然有效。GenImage 子集承担正式训练和内部测试角色。
