# 最终总结文档：AIGC 项目终跑结果与复盘

日期：2026-06-24

## 总结目标

把用户提供的原始数据解压并跑完终版实验，回答一个核心问题：最后加入 RGB+FFT 频域分支的新模型，到底有没有比纯 RGB baseline 更好。

## 最终完成内容

1. 解压 `data\raw` 下的 GenImage BigGAN、ADM 与 Synthbuster 数据。
2. 生成 GenImage 训练、验证、测试 manifest。
3. 生成 Synthbuster 外部泛化测试 manifest。
4. 训练 `convnext_tiny` baseline 与 `rgb_fft_fusion` 新模型。
5. 完成 GenImage 内部测试、Synthbuster 外部测试、JPEG/Blur 鲁棒性测试。
6. 修复 `src/eval.py` 中 source 分组结果 CSV 字段顺序不稳定的问题。
7. 输出最终结果表到 `results/tables`。

## 数据状态

原始数据目录：

```text
data\raw
```

本次 GenImage 使用 BigGAN 与 ADM 两个 source，每个 source、每个类别抽样 800 张，形成均衡子集：

| split | total | real label=0 | fake label=1 |
|---|---:|---:|---:|
| train | 3200 | 1600 | 1600 |
| val | 3200 | 1600 | 1600 |
| test | 3200 | 1600 | 1600 |

Synthbuster 外部测试 manifest：

| dataset | total | real label=0 | fake label=1 |
|---|---:|---:|---:|
| synthbuster_eval.csv | 3600 | 1800 | 1800 |

注意：GenImage BigGAN/ADM 的分卷压缩包解压时出现少量 CRC 报错，说明原始压缩包并非完全干净。但本次用于训练和评估的 manifest 已做图片可读性检查，抽样到的 train/val/test 图片均可正常读取。若后续做论文级或线上级结果，建议重新下载完整无损压缩包。

## 本次配置

终跑使用的是 quick final 配置，目的是在本机上快速得到可复现的完整闭环结果。

| 模型 | 配置文件 | 训练轮数 | 输入尺寸 | 训练样本上限 | 测试样本上限 |
|---|---|---:|---:|---:|---:|
| ConvNeXt-Tiny | `configs/convnext_tiny_genimage_quick_final.yaml` | 3 | 224 | 3200 | 1600 |
| RGB+FFT Fusion | `configs/rgb_fft_fusion_genimage_quick_final.yaml` | 3 | 224 | 3200 | 1600 |

最终 checkpoint：

```text
results/checkpoints/convnext_tiny_genimage_quick_final/best_f1.pth
results/checkpoints/rgb_fft_fusion_genimage_quick_final/best_f1.pth
```

## 验证集最好结果

| 模型 | 最佳 epoch | val F1 | val AUC |
|---|---:|---:|---:|
| ConvNeXt-Tiny | 2 | 0.9874 | 0.9988 |
| RGB+FFT Fusion | 3 | 0.9901 | 0.9994 |

验证集上，新模型略好于 ConvNeXt-Tiny，说明频域分支在同源数据上确实带来了增益。

## GenImage 内部测试结果

结果文件：

```text
results/tables/quick_final_genimage_test.csv
```

| 模型 | Acc | Precision | Recall | F1 | AUC | 单图耗时 |
|---|---:|---:|---:|---:|---:|---:|
| ConvNeXt-Tiny | 0.9856 | 0.9949 | 0.9762 | 0.9855 | 0.9994 | 7.086 ms |
| RGB+FFT Fusion | 0.9931 | 0.9888 | 0.9975 | 0.9932 | 0.9998 | 7.628 ms |

结论：在 GenImage 内部测试集上，新模型效果更好。F1 从 0.9855 提升到 0.9932，AUC 从 0.9994 提升到 0.9998，代价是单图推理时间略高。

## Synthbuster 外部泛化结果

结果文件：

```text
results/tables/quick_final_synthbuster_eval.csv
```

| 模型 | Acc | Precision | Recall | F1 | AUC |
|---|---:|---:|---:|---:|---:|
| ConvNeXt-Tiny | 0.2850 | 0.1126 | 0.0625 | 0.0804 | 0.3341 |
| RGB+FFT Fusion | 0.2994 | 0.1704 | 0.1037 | 0.1290 | 0.3466 |

结论：在没见过的 Synthbuster 生成器上，新模型仍然比 baseline 好，但绝对效果不好。F1 从 0.0804 提升到 0.1290，只能说明频域分支缓解了一点泛化问题，不能说明模型已经具备可靠跨生成器检测能力。

source 分组行中有些 AUC 为 `nan`，这是因为单个 source 分组可能只包含单一类别，无法计算真正的二分类 AUC。判断外部泛化时应优先看 `source=all` 的总体行。

## 鲁棒性结果

结果文件：

```text
results/tables/quick_final_robustness.csv
```

| 退化方式 | ConvNeXt F1 | Fusion F1 | 结论 |
|---|---:|---:|---|
| clean | 0.9855 | 0.9932 | Fusion 更好 |
| jpeg_q75 | 0.5436 | 0.6527 | Fusion 更抗压缩 |
| jpeg_q50 | 0.3122 | 0.6061 | Fusion 明显更抗压缩 |
| blur_k3 | 0.9490 | 0.9055 | ConvNeXt 更抗轻微模糊 |

这个结果很有教学价值：FFT 分支对 JPEG 压缩后的频域伪影更敏感，所以压缩场景下更稳；但模糊会抹掉一部分高频信息，融合模型反而可能被影响。

## 最终回答：新模型效果好吗？

简短结论：新模型是当前项目里更好的模型，但还不是可以放心部署的最终检测器。

具体解释：

1. 在同源 GenImage 测试集上，RGB+FFT Fusion 明显好，F1 达到 0.9932。
2. 在 JPEG 压缩鲁棒性上，RGB+FFT Fusion 明显好，尤其 `jpeg_q50` 下 F1 是 0.6061，而 ConvNeXt 只有 0.3122。
3. 在 Synthbuster 外部泛化上，RGB+FFT Fusion 虽然优于 ConvNeXt，但 F1 只有 0.1290，说明模型仍然强依赖训练分布。

所以项目汇报时可以这样说：

```text
我加入 RGB+FFT 双分支后，同源测试和 JPEG 鲁棒性都有提升，证明频域特征对 AIGC 检测有效。
但 Synthbuster 外部测试暴露了明显的跨生成器泛化不足，因此后续重点不是继续堆单一模型，
而是扩大生成器覆盖、加入跨域增强、做阈值校准和失败案例分析。
```

## 复现实验命令

训练：

```powershell
python src\train.py --config configs\convnext_tiny_genimage_quick_final.yaml
python src\train.py --config configs\rgb_fft_fusion_genimage_quick_final.yaml
```

内部测试：

```powershell
python src\eval.py --config configs\convnext_tiny_genimage_quick_final.yaml --ckpt results\checkpoints\convnext_tiny_genimage_quick_final\best_f1.pth --out results\tables\quick_final_genimage_test.csv --group-by-source
python src\eval.py --config configs\rgb_fft_fusion_genimage_quick_final.yaml --ckpt results\checkpoints\rgb_fft_fusion_genimage_quick_final\best_f1.pth --out results\tables\quick_final_genimage_test.csv --group-by-source
```

外部泛化：

```powershell
python src\eval.py --config configs\convnext_tiny_genimage_quick_final.yaml --ckpt results\checkpoints\convnext_tiny_genimage_quick_final\best_f1.pth --test-csv data\synthbuster_eval\meta\synthbuster_eval.csv --out results\tables\quick_final_synthbuster_eval.csv --group-by-source
python src\eval.py --config configs\rgb_fft_fusion_genimage_quick_final.yaml --ckpt results\checkpoints\rgb_fft_fusion_genimage_quick_final\best_f1.pth --test-csv data\synthbuster_eval\meta\synthbuster_eval.csv --out results\tables\quick_final_synthbuster_eval.csv --group-by-source
```

鲁棒性：

```powershell
python src\robustness.py --config configs\convnext_tiny_genimage_quick_final.yaml --ckpt results\checkpoints\convnext_tiny_genimage_quick_final\best_f1.pth --out results\tables\quick_final_robustness.csv --degradations clean jpeg_q75 jpeg_q50 blur_k3
python src\robustness.py --config configs\rgb_fft_fusion_genimage_quick_final.yaml --ckpt results\checkpoints\rgb_fft_fusion_genimage_quick_final\best_f1.pth --out results\tables\quick_final_robustness.csv --degradations clean jpeg_q75 jpeg_q50 blur_k3
```

## 下一步优化

1. 用更多 GenImage source 或 Synthbuster 风格图片参与训练，降低跨生成器分布偏移。
2. 在训练中加入 JPEG、resize、blur 等数据增强，让模型提前见到真实传播退化。
3. 对外部数据单独做阈值校准，不只使用默认 0.5 阈值。
4. 做失败案例可视化，观察模型是看内容语义、局部纹理，还是压缩伪影。
5. 若磁盘和时间允许，重新下载无 CRC 报错的完整 GenImage 压缩包，再扩大训练规模。

## 验收标准

```text
1. docs 目录只保留 7 个每日总结文档和 1 个最终总结文档。
2. results/checkpoints 下存在两个 best_f1.pth。
3. results/tables/quick_final_genimage_test.csv 有两个模型的 source=all 结果。
4. results/tables/quick_final_synthbuster_eval.csv 有两个模型的 source=all 结果。
5. results/tables/quick_final_robustness.csv 有 clean、jpeg_q75、jpeg_q50、blur_k3。
6. 能准确说出：Fusion 内部测试更好、JPEG 更稳、外部泛化仍弱。
```
