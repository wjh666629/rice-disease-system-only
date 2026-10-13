# Rice Disease System - 水稻病害智能识别系统

基于 **YOLO11n-cls** 的水稻叶片病害六分类识别与离线诊断系统，覆盖数据治理、模型训练、专项优化、评测与演示的完整流程。

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO11-00bff3)
![Accuracy](https://img.shields.io/badge/验证集准确率-98.96%25-brightgreen)

---

## 项目简介

水稻病害的人工识别效率低、主观性强。本项目使用 Kaggle 公开的 Rice Disease Dataset（3829 张图像），基于 Ultralytics YOLO11n-cls 构建轻量级分类模型，实现以下六类水稻叶片状态的自动识别：

| 序号 | 英文名称 | 中文名称 | 索引 |
|------|----------|----------|------|
| 1 | bacterial_leaf_blight | 细菌性叶枯病 | 0 |
| 2 | brown_spot | 褐斑病 | 1 |
| 3 | healthy | 健康叶片 | 2 |
| 4 | leaf_blast | 稻瘟病 | 3 |
| 5 | leaf_scald | 叶灼病 | 4 |
| 6 | sheath_blight | 纹枯病 | 5 |

系统支持离线批量推理、预测置信度与概率分布展示、随机/分层抽样评测，并自动导出 CSV / JSON 评测报告。

## 系统架构

### 数据流程

```
原始数据 → 数据治理 → 数据集划分 → 模型训练 → 模型评估 → 推理部署
   ↓           ↓            ↓            ↓           ↓          ↓
Rice_Leaf_AUG/  质量检查    8:1:1 划分   YOLO11n-cls  混淆矩阵   predict()
                重复检测    train/val/test  多阶段训练  分类报告   实时推理
                无效过滤    data_info.json  最优选模    评测报告   结果导出
```

### 目录结构

```
rice-disease-system/
├── configs/                        # 配置文件
├── data/                           # 数据目录（不入库，需自行准备）
│   └── processed/                  # 划分后的数据集 train/val/test
├── evaluation_results/             # 演示评测结果（CSV + JSON）
├── runs/                           # 训练输出（不入库）
├── scripts/                        # 核心脚本
│   ├── data_processing.py          # 数据治理、质量检查、分层划分
│   ├── model_training.py           # 基础模型训练
│   ├── model_training_optimized.py # 两阶段优化训练 + 最佳模型选择
│   ├── optimize_bacterial.py       # 细菌性叶枯病专项优化训练
│   ├── hard_example_mining.py      # 难例挖掘 + 多种数据增强
│   ├── eval_val.py                 # 验证集评估脚本
│   ├── generate_evaluation_report.py # 评估报告生成
│   └── demo_system.py              # 演示系统（随机/分层抽样评测）
├── src/
│   └── inference.py                # 推理接口 RiceDiseasePredictor
├── tests/                          # 测试
└── requirements.txt
```

## 数据处理

### 数据治理流程

- **完整性校验**：验证图像可读性，过滤非 RGB/L 模式与尺寸小于 32×32 的无效图像
- **重复检测**：对全部图像计算 MD5 哈希，剔除重复样本
- **分层随机划分**：按 8:1:1 划分 train/val/test，保证各类别比例一致（随机种子固定，可复现）
- **测试集封存**：测试集独立封存，不参与任何超参数调优，仅用于最终评估

### 数据分布（3829 张）

| 类别 | 总数 | 训练集 | 验证集 | 测试集 |
|------|------|--------|--------|--------|
| 细菌性叶枯病 | 636 | 508 | 64 | 64 |
| 褐斑病 | 646 | 516 | 65 | 65 |
| 健康叶片 | 653 | 521 | 66 | 66 |
| 稻瘟病 | 634 | 506 | 64 | 64 |
| 叶灼病 | 628 | 502 | 63 | 63 |
| 纹枯病 | 632 | 504 | 64 | 64 |
| **合计** | **3829** | **3057** | **386** | **386** |

## 模型训练

### 训练流程

```
阶段1 数据准备 → 阶段2 基础训练 → 阶段3 两阶段优化训练 → 阶段4 专项优化 → 阶段5 评估与选模
```

### 训练参数对比

| 参数 | 基础训练 | 两阶段（阶段1） | 两阶段（阶段2） | 专项优化 |
|------|----------|-----------------|-----------------|----------|
| 模型 | YOLO11n-cls | YOLO11n-cls | YOLO11n-cls | YOLO11n-cls |
| 训练轮数 | 50 | 30 | 50 | 80 |
| 批次大小 | 32 | 32 | 32 | 32 |
| 输入尺寸 | 224 | 224 | 224 | 224 |
| 初始学习率 | 0.001 | 0.001 | 0.0001 | 0.0005 |
| 优化器 | AdamW | AdamW | AdamW | AdamW |
| 早停耐心值 | 10 | 15 | 20 | 25 |
| 冻结层数 | 无 | 10 | 0 | 无 |
| Cosine LR | 否 | 是 | 是 | 是 |
| Warmup | 无 | 3 轮 | 3 轮 | 5 轮 |
| MixUp | 默认 | 0.1 | 0.15 | 0.2 |
| Copy-Paste | 默认 | 0.1 | 0.15 | 0.2 |

### 技术亮点

- **两阶段训练**：先冻结骨干网络（10 层）训练分类头，再全网络微调；配合综合评分（准确率 40% + Macro F1 30% + 目标类别 F1 30%）自动选择最佳模型
- **难例挖掘与增强**：旋转、翻转、颜色扰动、噪声等多种增强策略，结合难例挖掘定向补充训练样本
- **类别加权专项优化**：针对细菌性叶枯病实施 3 倍类别加权、颜色增强与 MixUp/Copy-Paste 增强
- **完整评测体系**：验证集全量评估输出混淆矩阵、精确率/召回率/F1、Macro F1；演示系统支持随机与分层抽样评测

## 实验结果

### 模型性能对比（验证集 386 张）

| 模型文件 | 训练方式 | 验证集准确率 | Macro F1 |
|----------|----------|--------------|----------|
| `best_model_final.pt` | 两阶段优化训练 | 98.71% | 98.43% |
| `best_model_bacterial_opt.pt` | 颜色增强 + 类别加权 | **98.96%** | **98.96%** |

### 各类别指标（优化模型）

| 类别 | 精确率 | 召回率 | F1分数 | 样本数 |
|------|--------|--------|--------|--------|
| 细菌性叶枯病 | 1.0000 | 1.0000 | 1.0000 | 64 |
| 褐斑病 | 0.9688 | 1.0000 | 0.9841 | 65 |
| 健康叶片 | 1.0000 | 1.0000 | 1.0000 | 66 |
| 稻瘟病 | 1.0000 | 0.9688 | 0.9841 | 64 |
| 叶灼病 | 1.0000 | 1.0000 | 1.0000 | 63 |
| 纹枯病 | 1.0000 | 1.0000 | 1.0000 | 64 |

### 推理性能

| 指标 | 数值 |
|------|------|
| 参数量 | ~1.5M |
| 模型大小 | ~5MB |
| CPU 推理耗时 | ~20-50ms |
| GPU 推理耗时 | ~5-10ms |

## 快速开始

### 1. 环境准备

```bash
pip install -r requirements.txt
```

### 2. 数据准备

数据集不随本仓库分发，请自行从 Kaggle 下载 Rice Disease Dataset 并放置到 `data/` 目录，然后执行数据治理与划分：

```bash
python scripts/data_processing.py
```

划分结果为 train/val/test = 3057/386/386，元信息写入 `data/processed/data_info.json`。

### 3. 模型训练

```bash
# 基础训练
python scripts/model_training.py

# 两阶段优化训练
python scripts/model_training_optimized.py --mode two_stage

# 长训练
python scripts/model_training_optimized.py --mode long

# 细菌性叶枯病专项优化
python scripts/optimize_bacterial.py --augment_factor 2 --epochs 80
```

训练输出位于 `runs/classify/rice_disease_exp/weights/`（best.pt / last.pt）。

### 4. 模型评估

```bash
# 验证集评估
python scripts/eval_val.py

# 生成评估报告
python scripts/generate_evaluation_report.py
```

### 5. 演示评测

```bash
# 分层随机抽样评测（推荐）
python scripts/demo_system.py --samples 20 --stratified

# 指定随机种子
python scripts/demo_system.py --samples 20 --seed 123

# 使用优化模型评测
python scripts/demo_system.py --samples 20 --stratified --model best_model_bacterial_opt.pt
```

评测结果自动导出到 `evaluation_results/` 目录（CSV 明细 + JSON 摘要）。

### 6. Python 推理接口

```python
from src.inference import init_model, predict

# 初始化模型（只需调用一次）
init_model('best_model_final.pt')

# 执行推理
result = predict('leaf.jpg')
```

返回格式：

```json
{
    "predicted_label": "稻瘟病",
    "confidence": 0.9856,
    "probabilities": {
        "细菌性叶枯病": 0.0012,
        "褐斑病": 0.0034,
        "健康叶片": 0.0021,
        "稻瘟病": 0.9856,
        "叶灼病": 0.0045,
        "纹枯病": 0.0032
    },
    "latency_ms": 23.45
}
```

## 优化策略总览

| 优化策略 | 实施文件 | 核心方法 | 预期提升 |
|----------|----------|----------|----------|
| 两阶段微调 | model_training_optimized.py | 冻结骨干网络训练 + 全网络微调 | 5-10% |
| 数据增强 | hard_example_mining.py | 旋转、翻转、颜色扰动、噪声等 | 3-8% |
| 颜色增强 | optimize_bacterial.py | 亮度/对比度/饱和度变化 | 针对色差场景 |
| 类别加权 | optimize_bacterial.py | 细菌性叶枯病权重 3 倍 | 提升目标类别 F1 |
| 增强强度 | optimize_bacterial.py | MixUp 0.2, Copy-Paste 0.2 | 提升鲁棒性 |
