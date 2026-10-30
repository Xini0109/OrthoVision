# 实验设计与复现说明

本文件说明三个实验的协议、命令与产物位置，保证任何人可以在同一份数据与代码上复现论文中的数字。

## 0. 环境

| 项目 | 值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4060 Laptop（8 GB） |
| OS | Windows 11 |
| Python | 3.11 |
| PyTorch | 2.14.1+cu126 |
| Ultralytics | 8.x |

> 实际运行时的版本号会写入 `results/eval_test.json` 与训练日志，论文引用的是该文件中的值。

## 1. 实验一：基础检测效果

**目的**：回答"模型在独立测试集上到底能做到什么程度"。

**协议**

1. 使用 `scripts/prepare_data.py` 生成的固定划分（seed=42）。
2. 不使用测试集做任何调参；阈值与分析均基于验证集确定后固定。
3. 报告两组指标：
   - 框级：mAP@0.5、mAP@0.5:0.95、Precision、Recall、F1
   - 图级：敏感度、特异度、准确率、PPV、NPV
4. 同时记录推理耗时（mean / p50 / p95）。

**命令**

```bash
python scripts/evaluate.py --split test --conf 0.25 --iou 0.45 --match-iou 0.5
```

**产物**

- `results/eval_test.json`：全部指标 + 运行环境
- `results/per_image_test.csv`：逐图记录（GT 框数、预测框数、TP/FP/FN、耗时）
- `results/figures/test_BoxPR_curve.png`、`test_confusion_matrix.png`

## 2. 实验二：可解释性与错误案例

**目的**：回答"模型凭什么做出判断、在哪里失败"。

**协议**

1. 从逐图记录中按 `outcome` 字段抽取 TP / FP / FN 案例，**不按指标好坏挑选**。
2. 对每个案例计算 Eigen-CAM 与 Grad-CAM，并把热力图与检测框并排展示。
3. 记录"框内响应占比"：CAM 能量落在预测框（外扩 8 px）内的比例。
4. 人工阅读案例，归纳失败模式（视野不全、金属/石膏干扰、细微骨折线）。

**命令**

```bash
python scripts/make_figures.py          # 生成图 5：三类案例 + Eigen-CAM
```

界面中的「可解释性分析」页可交互复现同一计算。

**解读约束（写入论文）**

- CAM 反映的是特征响应，不是骨折分割；
- 一致性高 ≠ 模型正确，可能只是学到了数据集偏倚；
- 定性案例只能提出假设，不能得出总体结论。

## 3. 实验三：图像质量扰动

**目的**：回答"影像质量下降时性能退化多快"。

**扰动定义**

| 名称 | 操作 | 参数 |
| --- | --- | --- |
| blur_mild | 高斯模糊 | σ=1.0 |
| blur_moderate | 高斯模糊 | σ=2.0 |
| brightness_±30 | 灰度整体偏移 | ±30 / 255 |
| contrast_0.7 | 围绕均值的对比度缩放 | ×0.7 |
| noise_σ12 | 加性高斯噪声 | σ=12 |

**协议**

1. 只对**测试集**图像施加扰动，标注框保持不变（框仍对应原解剖结构）。
2. 每个条件重新跑一遍推理与匹配，评测脚本与实验一完全一致。
3. 基线（original）与各扰动条件在同一进程内连续执行，避免环境漂移。

**命令**

```bash
python scripts/perturbation_experiment.py            # 全部 612 张
python scripts/perturbation_experiment.py --max-images 150 --no-map   # 快速版本
```

**产物**：`results/perturbation.json`、`results/figures/perturbation.png`

**解读约束**：合成扰动只能说明对*该变换*的敏感性，不能等同于真实成像条件差异。

## 4. 结果一致性检查

论文中的每一个数字都来自 `results/` 下的文件；`paper/build_paper.py` 在生成文档时直接读取这些 JSON。
若重新训练模型，只需重新运行上述三条命令并重新生成论文，数字会自动同步。
