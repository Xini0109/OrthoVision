# 数据集说明：FracAtlas

## 1. 来源

FracAtlas 是 2023 年发表于 *Data in Brief* 的肌骨 X 光公开数据集，图像来自孟加拉国三家医院，覆盖上肢、下肢、骨盆等部位。

- 论文：Abedeen I., Rahman M. A., Prottyasha F. Z., Ahmed T., Chowdhury T. M., Shatabda S.
  *FracAtlas: A Dataset for Fracture Classification, Localization and Segmentation of
  Musculoskeletal Radiographs.* Data in Brief, 2023, 51:109684.
- 数据地址：figshare article **22363012**
- 许可：**CC BY 4.0**（允许再分发与商用，需署名）

## 2. 规模与结构

| 项目 | 数量 |
| --- | --- |
| 图像总数 | 4,083 |
| 含骨折图像 | 717 |
| 无骨折图像 | 3,366 |
| 骨折边界框 | 922 |

原始目录：

```
FractureAtlas/
├── images/
│   ├── Fractured/        # 717 张
│   └── Non_fractured/    # 3366 张
├── Annotations/
│   ├── YOLO/             # 单类 txt：0 cx cy w h（归一化）
│   ├── COCO JSON/
│   └── PASCAL VOC/
└── Utilities/            # 官方格式转换 notebook
```

## 3. 本项目如何使用

- **只用 YOLO 标注**（`0 fractured`），因为本项目做的是单类目标检测。
- 官方发布**没有**划分 train/val/test，本项目用 `scripts/prepare_data.py` 做**分层随机划分**：
  - 随机种子 `42`，比例 70% / 15% / 15%，分别对阳性、阴性两组独立抽样；
  - 划分结果写入磁盘，之后所有实验**固定复用**这同一份文件，不再重新随机。
- **不修改**任何标签内容与图像像素（除扰动实验中的临时副本）。

划分结果（`data/fracatlas_yolo/dataset_stats.json`）：

| 划分 | 图像 | 含骨折 | 框数 |
| --- | --- | --- | --- |
| train | 2,858 | 502 | 662 |
| val | 613 | 108 | 133 |
| test | 612 | 107 | 127 |

## 4. 已知局限（论文中需如实说明）

1. **样本不平衡**：阳性仅占 17.6%，模型容易偏向"无检出"，因此报告图级敏感度与特异度而不能只报准确率。
2. **无患者级分组信息**：发布数据没有患者 ID，随机划分可能在训练/测试间混入同一患者的多张影像，指标可能偏乐观。这是本数据集无法消除的局限。
3. **来源集中**：图像来自三家医院，设备与拍摄规范相对一致，跨中心泛化能力未被验证。
4. **标注粒度**：边界框是"大致区域"，不是像素级骨折线；不同标注者对骨折线的框选可能不一致，mAP 会受此影响。
5. **语言与场景**：骨折类型未细分（无 AO/OTA 分型），模型只能输出"疑似骨折区域"。
6. **少量源文件损坏**：发布包中有 59 张 JPEG 为截断文件（44 张在训练集、6 张在验证集、
   9 张在测试集），OpenCV 与 PIL 均无法完整解码。训练脚本与评测脚本均跳过这些文件并记录数量，
   因此实际参与训练 2814 张、测试集实际评测 603 张。这是数据集本身的质量问题，本项目未做修复或补全。

## 5. 引用格式

```bibtex
@article{abedeen2023fracatlas,
  title   = {FracAtlas: A Dataset for Fracture Classification, Localization and
             Segmentation of Musculoskeletal Radiographs},
  author  = {Abedeen, Iftekharul and Rahman, Md. Ashiqur and Prottyasha, Fatema Zohra
             and Ahmed, Tasnim and Chowdhury, Tareque Mohmud and Shatabda, Swakkhar},
  journal = {Data in Brief},
  volume  = {51},
  pages   = {109684},
  year    = {2023},
  doi     = {10.1016/j.dib.2023.109684}
}
```
