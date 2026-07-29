# Day 05 教学引导：Synthbuster 外部泛化测试

日期：2026-06-24

## 今日目标

用 Synthbuster 测试模型面对未知生成器时是否仍然有效。这个阶段证明项目不是只在同源测试集上“刷分”。

## 已完成内容

1. 新增 `scripts/prepare_synthbuster_eval.py`。
2. 支持从解压后的 Synthbuster 目录收集 fake 图像。
3. 支持从 GenImage test real 或 CSV manifest 抽取真实图补齐二分类评估。
4. 默认生成 `data/synthbuster_eval/meta/synthbuster_eval.csv`。
5. `src/eval.py` 支持 `--group-by-source`，可以按生成器来源输出指标。

## 准备命令

先解压：

```powershell
Expand-Archive -Path data\raw\synthbuster\synthbuster.zip -DestinationPath data\raw\synthbuster
```

生成评估 manifest：

```powershell
conda run -n aigc python scripts\prepare_synthbuster_eval.py ^
  --fake-root data\raw\synthbuster ^
  --real-csv data\genimage_subset\meta\test.csv ^
  --out-root data\synthbuster_eval ^
  --max-per-source 500
```

## 评估命令

```powershell
conda run -n aigc python src\eval.py ^
  --config configs\convnext_tiny_genimage.yaml ^
  --ckpt results\checkpoints\convnext_tiny_genimage\best_f1.pth ^
  --test-csv data\synthbuster_eval\meta\synthbuster_eval.csv ^
  --out results\tables\synthbuster_results.csv ^
  --group-by-source
```

## 验收标准

```text
1. synthbuster_eval.csv 存在。
2. CSV 中同时有 label=0 和 label=1。
3. source 字段能区分 Synthbuster 中不同生成器。
4. results/tables/synthbuster_results.csv 存在。
5. 能说明外部泛化测试和普通随机划分测试的区别。
```

## 面试表达

我把 Synthbuster 作为外部测试集，而不是训练集。这样可以评估模型是否真正学到了 AI 生成图像的泛化特征，而不是只记住 GenImage 训练源的统计偏差。
