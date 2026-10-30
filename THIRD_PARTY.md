# 第三方资源与来源说明

本项目在课程要求下明确区分"自己写的代码"与"第三方资源"，逐项说明来源、许可与改动。

## 1. 数据集：FracAtlas

| 项目 | 内容 |
| --- | --- |
| 名称 | FracAtlas: A Dataset for Fracture Classification, Localization and Segmentation of Musculoskeletal Radiographs |
| 作者 | Iftekharul Abedeen, Md. Ashiqur Rahman, Fatema Zohra Prottyasha, Tasnim Ahmed, Tareque Mohmud Chowdhury, Swakkhar Shatabda |
| 发表 | *Data in Brief*, 2023, 51:109684 |
| 原始地址 | figshare, article **22363012**, `https://doi.org/10.6084/m9.figshare.22363012` |
| 许可 | **CC BY 4.0** |
| 规模 | 4,083 张肌骨 X 光片；717 张含骨折区域边界框；3366 张无骨折 |
| 本项目的使用方式 | 只使用其 `Annotations/YOLO/` 下的单类标注（`0 fractured`），按 70/15/15 分层随机划分 train/val/test（`seed=42`）。**未修改任何标签内容**，未做重标注。 |
| 本项目的改动 | 仅重新组织目录结构（`scripts/prepare_data.py`），图像本身未做任何处理。 |
| 署名要求 | 论文与 README 中均需引用原文，保留数据集名称与作者信息（CC BY 4.0 要求）。 |

> 数据集原文件体积约 338 MB，**未随仓库分发**。使用者需按 `scripts/prepare_data.py` 顶部注释中的地址自行下载。

## 2. 检测框架：Ultralytics YOLO11

| 项目 | 内容 |
| --- | --- |
| 仓库 | https://github.com/ultralytics/ultralytics |
| 版本 | `ultralytics>=8.3.0`（实际训练版本见 `results/eval_test.json` 或训练日志） |
| 许可 | **AGPL-3.0** |
| 本项目的使用方式 | 作为 Python 依赖调用；未修改其源码。 |
| 初始权重 | `yolo11s.pt`（COCO 预训练），由 ultralytics 在首次运行时自动下载 |
| 由此产生的约束 | 本项目训练得到的 `orthovision_yolo11s.pt` 属于 AGPL-3.0 的衍生作品，**按 AGPL-3.0 发布**；若需闭源商用，必须替换检测框架或取得 Ultralytics 商业授权。 |
| 引用 | Jocher G., Qiu J., Chaurasia A. *Ultralytics YOLO* (2023–), https://github.com/ultralytics/ultralytics |

## 3. 论文 / 方法引用

- YOLO 系列：Redmon J. et al. *You Only Look Once*, CVPR 2016；Ultralytics YOLO11, 2024.
- Grad-CAM：Selvaraju R. R. et al. *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, ICCV 2017.
- Eigen-CAM：Muhammet B. et al. *Eigen-CAM: Class Activation Map using Principal Components*, IJCNN 2020.
- COCO 评测协议：Lin T.-Y. et al. *Microsoft COCO*, ECCV 2014.

## 4. 本项目自研部分

以下模块为本课程作业自行编写，采用 MIT 许可（见 `LICENSE`）：

- `orthovision/detector.py`：推理封装、耗时测量
- `orthovision/explain.py`：面向检测头的 Eigen-CAM / Grad-CAM 实现与"框内响应占比"指标
- `orthovision/metrics.py`：框级贪心匹配、图级敏感度/特异度统计
- `orthovision/image_ops.py`：显示增强、可控扰动、可视化绘制
- `scripts/*.py`：数据转换、训练、评测、扰动实验
- `app.py`：Streamlit 交互界面

## 5. 合规声明

- 本项目为《医学人工智能导论》课程研究原型，**不用于临床诊断**。
- 所有指标均来自 `results/` 下的可复现实验文件，未使用任何虚构数据。
- 界面中的"热力图"为特征响应可视化，不是病灶分割，论文与界面均已注明。
