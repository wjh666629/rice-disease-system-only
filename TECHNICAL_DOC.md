# 水稻病害智能识别系统 - 技术文档

## 目录
1. [项目概述](#1-项目概述)
2. [模块架构](#2-模块架构)
3. [数据集目录详解](#3-数据集目录详解)
4. [数据处理模块](#4-数据处理模块)
5. [模型训练模块](#5-模型训练模块)
6. [推理接口模块](#6-推理接口模块)
7. [演示系统模块](#7-演示系统模块)
8. [评估报告生成](#8-评估报告生成)
9. [数据流程](#9-数据流程)
10. [接口规范](#10-接口规范)

---

## 1. 项目概述

### 1.1 项目简介
本项目是一个基于深度学习的水稻叶片病害智能识别与离线诊断系统，使用 **YOLO11n-cls** 模型实现六类水稻病害的分类识别。

### 1.2 技术栈
| 组件 | 技术 | 版本 |
|------|------|------|
| 框架 | Ultralytics YOLO | 8.4+ |
| 深度学习 | PyTorch | 2.1+ |
| 数据处理 | NumPy / Pandas | 最新 |
| 评估指标 | scikit-learn | 1.3+ |

### 1.3 分类任务
| 序号 | 英文名称 | 中文名称 |
|------|----------|----------|
| 1 | bacterial_leaf_blight | 细菌性叶枯病 |
| 2 | brown_spot | 褐斑病 |
| 3 | healthy | 健康叶片 |
| 4 | leaf_blast | 稻瘟病 |
| 5 | leaf_scald | 叶灼病 |
| 6 | sheath_blight | 纹枯病 |

---

## 2. 模块架构

```
rice-disease-system/
├── src/
│   └── inference.py          # 推理接口（核心API）
├── scripts/
│   ├── data_processing.py    # 数据治理与划分
│   ├── model_training.py     # YOLO11n-cls训练
│   ├── generate_evaluation_report.py  # 评估报告生成
│   └── demo_system.py        # 演示系统
├── data/
│   └── processed/            # 划分后的数据集（核心目录）
├── runs/
│   └── classify/             # 训练输出目录
└── evaluation_results/       # 演示评测结果
```

---

## 3. 数据集目录详解

### 3.1 目录结构

```
data/processed/
├── train/                    # 训练集（80%）
│   ├── bacterial_leaf_blight/
│   ├── brown_spot/
│   ├── healthy/
│   ├── leaf_blast/
│   ├── leaf_scald/
│   └── sheath_blight/
├── val/                      # 验证集（10%）
│   ├── bacterial_leaf_blight/
│   ├── brown_spot/
│   ├── healthy/
│   ├── leaf_blast/
│   ├── leaf_scald/
│   └── sheath_blight/
├── test/                     # 测试集（10%，已封存）
│   ├── bacterial_leaf_blight/
│   ├── brown_spot/
│   ├── healthy/
│   ├── leaf_blast/
│   ├── leaf_scald/
│   └── sheath_blight/
└── data_info.json            # 数据集元信息文件
```

### 3.2 目录功能说明

| 目录 | 功能 | 作用 |
|------|------|------|
| `train/` | 训练集 | 用于模型训练，占总数据的 80% |
| `val/` | 验证集 | 用于训练过程中的模型评估和超参数调优，占 10% |
| `test/` | 测试集 | **已封存**，仅用于最终模型评估，不参与训练，占 10% |

### 3.3 数据集划分原则

**分层随机划分**：
- 采用 `sklearn.model_selection.train_test_split` 进行分层划分
- 固定随机种子 **42**，确保划分结果可复现
- 每个类别按照 8:1:1 比例分配到 train/val/test

### 3.4 各类别数据分布

| 类别 | 英文名称 | 总数 | 训练集 | 验证集 | 测试集 |
|------|----------|------|--------|--------|--------|
| 细菌性叶枯病 | bacterial_leaf_blight | 636 | 508 | 64 | 64 |
| 褐斑病 | brown_spot | 646 | 516 | 65 | 65 |
| 健康叶片 | healthy | 653 | 521 | 66 | 66 |
| 稻瘟病 | leaf_blast | 634 | 506 | 64 | 64 |
| 叶灼病 | leaf_scald | 628 | 502 | 63 | 63 |
| 纹枯病 | sheath_blight | 632 | 504 | 64 | 64 |
| **合计** | - | **3829** | **3057** | **386** | **386** |

### 3.5 data_info.json 文件详解

该文件记录了数据集的完整元信息：

```json
{
  "dataset_name": "Rice Disease Dataset",
  "data_version": "v1.0",
  "creation_date": "2026-06-16 22:22:00",
  "statistics": {
    "total_images": 3829,
    "train_images": 3057,
    "val_images": 386,
    "test_images": 386,
    "split_ratio": "8:1:1",
    "classes": [...],
    "class_mapping": {...}
  },
  "invalid_images": [],
  "duplicate_groups": 31,
  "split_method": "分层随机划分，确保各类别比例一致",
  "random_seed": 42,
  "notes": "测试集已封存，不参与模型训练和参数调优"
}
```

**关键字段说明**：

| 字段 | 说明 |
|------|------|
| `total_images` | 数据集中图像总数 |
| `train_images/val_images/test_images` | 各数据集图像数量 |
| `split_ratio` | 划分比例（训练:验证:测试） |
| `classes` | 各类别详细统计信息 |
| `class_mapping` | 英文名称到中文名称的映射 |
| `invalid_images` | 无效/损坏图像列表（已过滤） |
| `duplicate_groups` | 重复图像组数（已标记） |
| `random_seed` | 随机种子，确保可复现 |

### 3.6 测试集封存机制

**测试集的特殊地位**：
- ✅ **不参与训练**：测试集图像不会被用于模型训练
- ✅ **不参与调优**：验证集用于超参数调整，测试集保持独立
- ✅ **最终评估**：仅在模型训练完成后，用于最终性能评估
- ✅ **公平对比**：确保模型在未知数据上的真实泛化能力

### 3.7 图像文件命名规则

图像文件名格式：
```
{原始文件名}_{索引}.jpg
```

例如：
- `aug_1_1.jpg` - 增强后的第1类第1张图像
- `DSC_001.jpg` - 原始图像

### 3.8 数据质量检查结果

| 检查项 | 结果 | 说明 |
|--------|------|------|
| 无效图像 | 0 张 | 所有图像均可正常读取 |
| 重复图像组 | 31 组 | 已识别但未删除（数据增强产物） |
| 图像格式 | JPG | 统一格式 |
| 最小尺寸 | ≥ 32x32 | 已过滤过小图像 |

---

## 4. 数据处理模块

### 文件路径：`scripts/data_processing.py`

#### 4.1 核心类：`RiceDiseaseDataProcessor`

**功能**：负责数据质量检查、重复图像检测、数据集分层划分

#### 4.2 关键方法

| 方法 | 功能 | 参数 | 返回值 |
|------|------|------|--------|
| `__init__` | 初始化处理器 | `raw_data_dir`, `output_dir` | 无 |
| `process_raw_data()` | 数据质量检查 | 无 | `all_images`, `invalid_images`, `duplicates` |
| `split_dataset()` | 分层随机划分 | `train_ratio=0.8`, `val_ratio=0.1`, `test_ratio=0.1` | `train_images`, `val_images`, `test_images` |
| `save_dataset()` | 保存划分结果 | `train_images`, `val_images`, `test_images` | 无 |
| `run()` | 执行完整流程 | 无 | `data_info` |

#### 4.3 核心代码解析

**数据质量检查**：
```python
def _is_valid_image(self, image_path):
    try:
        img = Image.open(image_path)
        img.verify()
        img = Image.open(image_path)
        if img.mode not in ('RGB', 'L'):
            return False
        if img.size[0] < 32 or img.size[1] < 32:
            return False
        return True
    except Exception:
        return False
```
- 使用 PIL 验证图像完整性
- 检查图像模式（仅支持 RGB 和灰度）
- 过滤过小图像（小于 32x32）

**重复图像检测**：
```python
def _compute_image_hash(self, image_path):
    hasher = hashlib.md5()
    with open(image_path, 'rb') as f:
        hasher.update(f.read())
    return hasher.hexdigest()
```
- 使用 MD5 哈希检测重复图像
- 相同哈希值表示完全相同的图像

**分层随机划分**：
```python
train_val, test = train_test_split(class_images, test_size=test_ratio, random_state=42)
train, val = train_test_split(train_val, test_size=val_size, random_state=42)
```
- 使用 `sklearn.model_selection.train_test_split`
- 固定随机种子 42，确保可复现
- 按类别分层，保证各类别比例一致

#### 4.4 输入输出

**输入**：原始数据集目录 `Rice_Leaf_AUG/`
```
Rice_Leaf_AUG/
├── Bacterial Leaf Blight/
├── Brown Spot/
├── Healthy Rice Leaf/
├── Leaf Blast/
├── Leaf scald/
└── Sheath Blight/
```

**输出**：划分后的数据集目录 `data/processed/`
```
data/processed/
├── train/          # 80% 训练集
├── val/            # 10% 验证集
├── test/           # 10% 测试集（已封存）
└── data_info.json  # 数据集统计信息
```

---

## 5. 模型训练模块

### 文件路径：`scripts/model_training.py`

#### 5.1 核心类：`RiceDiseaseModelTrainer`

**功能**：YOLO11n-cls 模型训练、验证、评估

#### 5.2 关键方法

| 方法 | 功能 | 参数 | 返回值 |
|------|------|------|--------|
| `train()` | 训练模型 | `epochs=50`, `batch_size=32`, `img_size=224`, `patience=10` | `model`, `results` |
| `evaluate()` | 评估模型 | `model`, `data_dir` | `metrics`, `results` |
| `compute_confusion_matrix()` | 计算混淆矩阵 | `model`, `test_dir` | `cm`, `report`, `y_true`, `y_pred`, `all_probs` |
| `save_evaluation_report()` | 保存评估报告 | `metrics`, `cm`, `report`, ... | `eval_report` |

#### 5.3 训练配置

```python
model = YOLO('yolo11n-cls.pt')

results = model.train(
    data=str(self.data_dir),
    epochs=50,
    batch=32,
    imgsz=224,
    patience=10,
    seed=42,
    optimizer='AdamW',
    lr0=0.001,
    augment=True,
    val=True
)
```

**训练参数说明**：
- **epochs=50**：训练轮数
- **batch=32**：批次大小
- **imgsz=224**：输入图像尺寸
- **patience=10**：早停耐心值（10轮无提升则停止）
- **optimizer='AdamW'**：优化器选择
- **augment=True**：启用数据增强

#### 5.4 评估指标

```python
metrics = {
    'accuracy': float(results.top1),      # Top-1 准确率
    'top5_accuracy': float(results.top5)  # Top-5 准确率
}
```

**分类报告**（使用 scikit-learn）：
```python
report = classification_report(
    y_true, y_pred,
    target_names=self.classes_cn,
    output_dict=True
)
```

输出包含：Precision、Recall、F1-score、Support

#### 5.5 训练输出

```
runs/classify/rice_disease_exp/
├── weights/
│   ├── best.pt      # 最佳模型（验证集性能最优）
│   └── last.pt      # 最后一轮模型
├── results.csv      # 训练指标数据
├── results.png      # 训练曲线图表
├── confusion_matrix.png
└── evaluation_report.json  # 完整评估报告
```

---

## 6. 推理接口模块

### 文件路径：`src/inference.py`

#### 6.1 核心类：`RiceDiseasePredictor`

**功能**：提供统一的推理接口，支持单张图像预测

#### 6.2 关键方法

| 方法 | 功能 | 参数 | 返回值 |
|------|------|------|--------|
| `__init__` | 初始化预测器 | `model_path=None` | 无 |
| `load_model()` | 加载模型权重 | `model_path` | 无 |
| `predict()` | 执行推理 | `image_path` | 推理结果字典 |

#### 6.3 推理流程

```python
def predict(self, image_path):
    start_time = time.time()
    
    results = self.model.predict(image_path, verbose=False)
    
    latency_ms = (time.time() - start_time) * 1000
    
    pred_idx = results[0].probs.top1           # 预测类别索引
    confidence = float(results[0].probs.top1conf)  # 置信度
    probs = results[0].probs.data.tolist()     # 各类概率
    
    result = {
        "predicted_label": predicted_label,
        "confidence": confidence,
        "probabilities": probabilities,
        "latency_ms": latency_ms
    }
    
    return result
```

**推理流程说明**：
1. 记录开始时间
2. 调用 YOLO 模型进行预测
3. 计算推理耗时（毫秒）
4. 解析预测结果：
   - `top1`：最高置信度类别的索引
   - `top1conf`：最高置信度值
   - `probs.data`：所有类别的概率分布

#### 6.4 接口规范

**调用方式**：
```python
from src.inference import init_model, predict

init_model('runs/classify/rice_disease_exp/weights/best.pt')
result = predict('test_image.jpg')
```

**返回格式**：
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

**字段说明**：
| 字段 | 类型 | 说明 |
|------|------|------|
| `predicted_label` | str | 预测类别（中文） |
| `confidence` | float | 置信度（0-1） |
| `probabilities` | dict | 六类概率分布 |
| `latency_ms` | float | 推理耗时（毫秒） |

#### 6.5 全局单例模式

```python
_predictor = None

def init_model(model_path):
    global _predictor
    _predictor = RiceDiseasePredictor(model_path)

def predict(image_path):
    if _predictor is None:
        raise RuntimeError("模型尚未初始化，请先调用 init_model()")
    return _predictor.predict(image_path)
```

- 使用全局单例模式，避免重复加载模型
- 先调用 `init_model()` 初始化，再调用 `predict()` 进行推理

---

## 7. 演示系统模块

### 文件路径：`scripts/demo_system.py`

#### 7.1 核心类：`RiceDiseaseDemoSystem`

**功能**：提供完整的演示流程，支持随机抽样、批量预测、结果导出

#### 7.2 关键方法

| 方法 | 功能 | 参数 | 返回值 |
|------|------|------|--------|
| `random_sampling()` | 简单随机抽样 | `n_samples=20`, `seed=42` | `samples` |
| `stratified_sampling()` | 分层随机抽样 | `n_samples=20`, `seed=42` | `samples` |
| `batch_predict()` | 批量预测 | `samples`, `save_results=True` | `predictions` |
| `print_summary()` | 打印评测摘要 | 无 | 无 |

#### 7.3 抽样策略

**简单随机抽样**：
```python
sampled = random.sample(self.test_samples, n_samples)
```
- 从测试集中随机抽取指定数量样本

**分层随机抽样**：
```python
samples_per_class = n_samples // n_classes

for label, samples in samples_by_class.items():
    stratified_samples.extend(random.sample(samples, samples_per_class))
```
- 确保每种类别抽取数量均衡
- 适用于类别分布不均衡的场景

#### 7.4 结果导出

**CSV 格式**：
```python
results_df = pd.DataFrame([{
    '序号': p['index'],
    '文件名': p['filename'],
    '真实标签': p['true_label'],
    '预测标签': p['predicted_label'],
    '置信度': f"{p['confidence']:.4f}",
    '预测正确': '是' if p['correct'] else '否',
    '推理耗时(ms)': f"{p['latency_ms']:.2f}"
} for p in self.predictions])

results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
```

**JSON 格式（评测摘要）**：
```python
summary = {
    'evaluation_time': timestamp,
    'total_samples': len(self.predictions),
    'correct_predictions': correct,
    'accuracy': accuracy,
    'average_latency_ms': avg_latency,
    'class_metrics': {}
}
```

#### 7.5 命令行使用

```bash
# 随机抽样 20 张图像
python scripts/demo_system.py --samples 20

# 分层随机抽样
python scripts/demo_system.py --samples 20 --stratified

# 指定随机种子
python scripts/demo_system.py --samples 20 --seed 123
```

---

## 8. 评估报告生成

### 文件路径：`scripts/generate_evaluation_report.py`

#### 8.1 功能说明

自动生成完整的模型评估报告，包括：
- 模型指标（Accuracy、Top-5 Accuracy）
- 混淆矩阵
- 分类报告（Precision、Recall、F1）
- 各类别详细指标

#### 8.2 执行流程

```python
# 1. 加载模型
model = YOLO(model_path)

# 2. 评估验证集
metrics, val_results = trainer.evaluate(model)

# 3. 计算测试集混淆矩阵
cm, report, y_true, y_pred, all_probs = trainer.compute_confusion_matrix(model, test_dir)

# 4. 保存评估报告
eval_report = trainer.save_evaluation_report(metrics, cm, report, y_true, y_pred, all_probs)

# 5. 保存混淆矩阵CSV
trainer.save_confusion_matrix_csv(cm)
```

#### 8.3 评估报告结构

```json
{
    "evaluation_date": "2026-06-16 23:29:10",
    "metrics": {
        "accuracy": 0.9871,
        "top5_accuracy": 1.0
    },
    "confusion_matrix": [[64,0,0,0,0,0], ...],
    "classification_report": {
        "细菌性叶枯病": {"precision": 0.9846, "recall": 1.0, "f1-score": 0.9922},
        ...
    },
    "macro_f1": 0.9843,
    "macro_precision": 0.9844,
    "macro_recall": 0.9843
}
```

---

## 9. 优化训练模块

### 文件路径：`scripts/optimize_bacterial.py`

#### 9.1 核心类：`RiceDiseaseOptimizer`

**功能**：针对实际检测场景中的色差和精确度问题，对细菌性叶枯病进行专项优化训练

#### 9.2 问题背景

在实际检测中，由于光照、拍摄设备、图像预处理等因素，模型可能表现不佳。主要问题：
- **色差问题**：不同光照条件下拍摄的图片颜色差异大
- **精确度问题**：细菌性叶枯病与其他病害特征相似，容易混淆

#### 9.3 优化策略

**策略一：数据增强（针对色差问题）**
```python
# 亮度变化：模拟不同光照条件
brightness_variants = [0.7, 0.85, 1.15, 1.3]

# 对比度变化：模拟不同明暗程度
contrast_variants = [0.7, 0.85, 1.15, 1.3]

# 饱和度变化：模拟不同色彩鲜艳程度
saturation_variants = [0.7, 0.85, 1.15, 1.3]
```

**策略二：增强数据增强强度（针对精确度问题）**
```python
# 训练参数调整
mixup = 0.2        # 原值 0.1
copy_paste = 0.2   # 原值 0.1
shear = 10         # 原值 8
degrees = 20       # 旋转角度
```

#### 9.4 训练配置

```python
results = model.train(
    data=str(enhanced_dir),
    epochs=80,
    batch=32,
    imgsz=224,
    patience=25,
    seed=42,
    optimizer='AdamW',
    lr0=0.0005,
    augment=True,
    cos_lr=True,
    warmup_epochs=5,
    weight_decay=0.0005,
    mixup=0.2,
    copy_paste=0.2,
)
```

#### 9.5 优化效果对比

| 指标 | 优化前 | 优化后 |
|------|--------|--------|
| 总体准确率 | 98.71% | **98.96%** |
| Macro F1 | 98.43% | **98.96%** |
| 细菌性叶枯病准确率 | 100% | 100% |

#### 9.6 使用方法

```bash
# 运行优化训练
python scripts/optimize_bacterial.py --augment_factor 2 --epochs 80
```

#### 9.7 输出文件

| 文件 | 说明 |
|------|------|
| `best_model_bacterial_opt.pt` | 优化后的最佳模型 |
| `bacterial_optimization_result.json` | 优化结果评估报告 |
| `data/enhanced_for_bacterial/` | 增强后的数据集 |

---

## 10. 验证集评估脚本

### 文件路径：`scripts/eval_val.py`

#### 10.1 功能说明

在验证集上评估模型性能，输出各类别的准确率、精确率、召回率和F1分数。

#### 10.2 使用方法

```bash
# 评估 best_model_final.pt
python scripts/eval_val.py

# 评估优化后的模型
python scripts/eval_bacterial_opt.py
```

---

## 11. 模型对比与选择

### 11.1 现有模型

| 模型文件 | 训练方式 | 验证集准确率 | 适用场景 |
|----------|----------|--------------|----------|
| `best_model_final.pt` | 标准训练 | 98.71% | 通用场景 |
| `best_model_bacterial_opt.pt` | 增强训练 | 98.96% | 色差较大场景 |

### 11.2 选择建议

- **标准场景**：使用 `best_model_final.pt`
- **光照变化大、色差明显**：使用 `best_model_bacterial_opt.pt`
- **比赛演示**：根据实际检测图片情况选择

---

## 9. 数据流程

### 9.1 完整流程

```
原始数据 → 数据治理 → 数据集划分 → 模型训练 → 模型评估 → 推理部署
     ↓            ↓            ↓           ↓           ↓           ↓
 Rice_Leaf_AUG/ → 质量检查    8:1:1划分   YOLO11n-cls  混淆矩阵    predict()
              → 重复检测    train/val/test  50轮训练     分类报告   实时推理
              → 无效过滤    data_info.json  best.pt      evaluation_report.json
```

### 9.2 各阶段输出

| 阶段 | 输出文件 | 说明 |
|------|----------|------|
| 数据治理 | `data/processed/data_info.json` | 数据集统计信息 |
| 模型训练 | `runs/classify/rice_disease_exp/weights/best.pt` | 最佳模型权重 |
| 模型评估 | `runs/classify/rice_disease_exp/evaluation_report.json` | 完整评估报告 |
| 演示评测 | `evaluation_results/*.csv` | 批量预测结果 |
| 演示评测 | `evaluation_results/*.json` | 评测摘要 |

---

## 10. 接口规范

### 10.1 predict() 接口

**函数签名**：
```python
predict(image_path: str) -> dict
```

**参数**：
- `image_path`: 本地图片路径（str）

**返回值**：
```python
{
    "predicted_label": str,        # 预测类别（中文）
    "confidence": float,           # 置信度（0-1）
    "probabilities": dict,         # 六类概率分布
    "latency_ms": float            # 推理耗时（毫秒）
}
```

**使用示例**：
```python
from src.inference import init_model, predict

# 初始化模型（只需调用一次）
init_model('runs/classify/rice_disease_exp/weights/best.pt')

# 执行推理
result = predict('path/to/rice_leaf.jpg')

print(f"预测结果: {result['predicted_label']}")
print(f"置信度: {result['confidence']:.4f}")
print(f"推理耗时: {result['latency_ms']:.2f} ms")
```

### 10.2 类别名称映射

| 索引 | 英文名称 | 中文名称 |
|------|----------|----------|
| 0 | bacterial_leaf_blight | 细菌性叶枯病 |
| 1 | brown_spot | 褐斑病 |
| 2 | healthy | 健康叶片 |
| 3 | leaf_blast | 稻瘟病 |
| 4 | leaf_scald | 叶灼病 |
| 5 | sheath_blight | 纹枯病 |

---

## 附录：模型性能

### 原始模型

| 指标 | 数值 |
|------|------|
| Top-1 Accuracy | 98.71% |
| Top-5 Accuracy | 100% |
| Macro F1-score | 98.43% |
| 参数量 | 1.5M |
| 推理耗时 | ~20-50ms（CPU） |

### 优化模型

| 指标 | 数值 |
|------|------|
| Top-1 Accuracy | **98.96%** |
| Top-5 Accuracy | 100% |
| Macro F1-score | **98.96%** |
| 参数量 | 1.5M |
| 推理耗时 | ~20-50ms（CPU） |

---

**文档版本**: v1.2  
**生成日期**: 2026-06-26  
**作者**: 吴杰涵