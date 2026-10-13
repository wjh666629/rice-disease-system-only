# 水稻病害智能识别系统 - 项目总结

---

## 目录

1. [项目概述](#1-项目概述)
2. [系统架构](#2-系统架构)
3. [数据集详解](#3-数据集详解)
4. [核心代码模块](#4-核心代码模块)
5. [模型训练流程](#5-模型训练流程)
6. [推理接口设计](#6-推理接口设计)
7. [演示系统](#7-演示系统)
8. [模型优化策略](#8-模型优化策略)
9. [实验结果](#9-实验结果)
10. [YOLO-Master论文改进方案](#10-yolo-master论文改进方案)
11. [项目文件清单](#11-项目文件清单)

---

## 1. 项目概述

### 1.1 项目目标

本项目基于 **YOLO11n-cls** 图像分类模型，构建了一个完整的水稻叶片病害智能识别与离线诊断系统。系统实现了：

- ✅ 水稻叶片病害六分类识别
- ✅ 离线批量图像推理
- ✅ 预测类别、置信度及概率分布展示
- ✅ 测试集随机/分层抽样评测
- ✅ 准确率、耗时自动统计
- ✅ 推理历史记录与结果导出

### 1.2 分类任务

| 序号 | 英文名称 | 中文名称 | 索引 |
|------|----------|----------|------|
| 1 | bacterial_leaf_blight | 细菌性叶枯病 | 0 |
| 2 | brown_spot | 褐斑病 | 1 |
| 3 | healthy | 健康叶片 | 2 |
| 4 | leaf_blast | 稻瘟病 | 3 |
| 5 | leaf_scald | 叶灼病 | 4 |
| 6 | sheath_blight | 纹枯病 | 5 |

### 1.3 技术栈

| 组件 | 技术 | 版本 |
|------|------|------|
| 深度学习框架 | Ultralytics YOLO | 8.3+ |
| 神经网络后端 | PyTorch | 2.1+ |
| 数据处理 | NumPy / Pandas | 最新 |
| 评估指标 | scikit-learn | 1.3+ |
| 图像处理 | PIL / OpenCV | 最新 |

---

## 2. 系统架构

### 2.1 项目目录结构

```
rice-disease-system/
├── src/                        # 核心模块
│   └── inference.py            # 推理接口（核心API）
├── scripts/                    # 脚本模块
│   ├── data_processing.py      # 数据治理与划分
│   ├── model_training.py       # 基础训练
│   ├── model_training_optimized.py  # 两阶段优化训练
│   ├── optimize_bacterial.py   # 细菌性叶枯病专项优化
│   ├── hard_example_mining.py  # 难例挖掘与数据增强
│   ├── demo_system.py          # 演示系统
│   ├── eval_val.py             # 验证集评估
│   └── generate_evaluation_report.py  # 评估报告生成
├── data/                       # 数据集目录
│   └── processed/              # 划分后的数据集
│       ├── train/              # 训练集（80%）
│       ├── val/                # 验证集（10%）
│       ├── test/               # 测试集（10%，已封存）
│       └── data_info.json      # 数据集元信息
├── runs/                       # 训练输出目录
│   └── classify/               # YOLO分类训练结果
├── evaluation_results/         # 演示评测结果
├── 论文/                       # 论文资料
│   ├── 2512.23273v1.pdf        # YOLO-Master原文
│   └── YOLO-Master_中文版.md   # 中文翻译
├── PLAN.md                     # 项目实施方案
├── planning.md                 # YOLO-Master改进计划
├── TECHNICAL_DOC.md                      # 技术文档
├── OPTIMIZATION_GUIDE.md       # 优化指南
├── DATA_SPLIT说明.md           # 数据划分说明
└── requirements.txt            # 依赖清单
```

### 2.2 数据流程图

```
原始数据 → 数据治理 → 数据集划分 → 模型训练 → 模型评估 → 推理部署
     ↓            ↓            ↓           ↓           ↓           ↓
 Rice_Leaf_AUG/ → 质量检查    8:1:1划分   YOLO11n-cls  混淆矩阵    predict()
              → 重复检测    train/val/test  50轮训练     分类报告   实时推理
              → 无效过滤    data_info.json  best.pt      evaluation_report.json
```

---

## 3. 数据集详解

### 3.1 数据集来源与规模

采用 Kaggle 公开的 **Rice Disease Dataset**，包含约 3829 张 RGB 水稻叶片图像，已按六个类别划分。

### 3.2 数据分布

| 类别 | 英文名称 | 总数 | 训练集 | 验证集 | 测试集 |
|------|----------|------|--------|--------|--------|
| 细菌性叶枯病 | bacterial_leaf_blight | 636 | 508 | 64 | 64 |
| 褐斑病 | brown_spot | 646 | 516 | 65 | 65 |
| 健康叶片 | healthy | 653 | 521 | 66 | 66 |
| 稻瘟病 | leaf_blast | 634 | 506 | 64 | 64 |
| 叶灼病 | leaf_scald | 628 | 502 | 63 | 63 |
| 纹枯病 | sheath_blight | 632 | 504 | 64 | 64 |
| **合计** | - | **3829** | **3057** | **386** | **386** |

### 3.3 数据治理流程

**数据质量检查**（关键代码）：

```python
# scripts/data_processing.py - 第58-69行
def _is_valid_image(self, image_path):
    try:
        img = Image.open(image_path)
        img.verify()                  # 验证图像完整性
        img = Image.open(image_path)
        if img.mode not in ('RGB', 'L'):  # 检查图像模式
            return False
        if img.size[0] < 32 or img.size[1] < 32:  # 过滤过小图像
            return False
        return True
    except Exception:
        return False
```

**重复图像检测**（关键代码）：

```python
# scripts/data_processing.py - 第52-56行
def _compute_image_hash(self, image_path):
    hasher = hashlib.md5()
    with open(image_path, 'rb') as f:
        hasher.update(f.read())
    return hasher.hexdigest()  # 使用MD5哈希检测重复
```

**分层随机划分**（关键代码）：

```python
# scripts/data_processing.py - 第125-127行
train_val, test = train_test_split(class_images, test_size=test_ratio, random_state=42)
val_size = val_ratio / (train_ratio + val_ratio)
train, val = train_test_split(train_val, test_size=val_size, random_state=42)
```

### 3.4 测试集封存机制

测试集在划分后立即封存，遵循以下规则：
- ✅ 不参与模型训练
- ✅ 不参与超参数调优
- ✅ 仅用于最终性能评估（单次评估）
- ✅ 保存文件清单和哈希值，确保可复现

---

## 4. 核心代码模块

### 4.1 数据处理模块（data_processing.py）

**核心类：`RiceDiseaseDataProcessor`**

| 方法 | 功能 | 参数 | 返回值 |
|------|------|------|--------|
| `__init__` | 初始化处理器 | `raw_data_dir`, `output_dir` | 无 |
| `process_raw_data()` | 数据质量检查 | 无 | `all_images`, `invalid_images`, `duplicates` |
| `split_dataset()` | 分层随机划分 | `train_ratio=0.8`, `val_ratio=0.1`, `test_ratio=0.1` | `train_images`, `val_images`, `test_images` |
| `save_dataset()` | 保存划分结果 | 三组图像列表 | 无 |
| `run()` | 执行完整流程 | 无 | `data_info` |

**执行入口**：

```python
# scripts/data_processing.py - 第215-222行
if __name__ == '__main__':
    raw_data_dir = 'Rice_Leaf_AUG'
    output_dir = 'data/processed'
    processor = RiceDiseaseDataProcessor(raw_data_dir, output_dir)
    processor.run()
```

### 4.2 模型训练模块（model_training.py）

**核心类：`RiceDiseaseModelTrainer`**

| 方法 | 功能 | 参数 | 返回值 |
|------|------|------|--------|
| `train()` | 训练模型 | `epochs=50`, `batch_size=32`, `img_size=224` | `model`, `results` |
| `evaluate()` | 评估模型 | `model`, `data_dir` | `metrics`, `results` |
| `compute_confusion_matrix()` | 计算混淆矩阵 | `model`, `test_dir` | `cm`, `report`, `y_true`, `y_pred`, `all_probs` |
| `save_evaluation_report()` | 保存评估报告 | 多项指标 | `eval_report` |

**训练配置**（关键代码）：

```python
# scripts/model_training.py - 第38-58行
def train(self, epochs=50, batch_size=32, img_size=224, patience=10):
    model = YOLO('yolo11n-cls.pt')
    
    results = model.train(
        data=str(self.data_dir),
        epochs=epochs,           # 训练轮数
        batch=batch_size,       # 批次大小
        imgsz=img_size,         # 输入尺寸
        patience=patience,      # 早停耐心值
        seed=42,                # 随机种子
        optimizer='AdamW',      # 优化器
        lr0=0.001,              # 初始学习率
        augment=True,           # 启用数据增强
        val=True                # 验证集评估
    )
    return model, results
```

**评估报告结构**：

```python
# scripts/model_training.py - 第115-140行
eval_report = {
    'evaluation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'metrics': metrics,                    # Top-1/Top-5准确率
    'confusion_matrix': cm.tolist(),       # 混淆矩阵
    'classification_report': report,       # 分类报告
    'class_mapping': self.class_mapping,   # 中英文映射
    'total_samples': len(y_true),          # 样本总数
    'correct_predictions': sum(1 for t, p in zip(y_true, y_pred) if t == p),
    'macro_f1': report['macro avg']['f1-score'],
    'macro_precision': report['macro avg']['precision'],
    'macro_recall': report['macro avg']['recall']
}
```

### 4.3 优化训练模块（model_training_optimized.py）

**核心类：`RiceDiseaseModelTrainerOptimized`**

实现了**两阶段微调策略**：

**阶段1：冻结骨干网络训练**（关键代码）：

```python
# scripts/model_training_optimized.py - 第42-82行
def train_stage1(self, model_name='yolo11n-cls.pt', epochs=30, batch_size=32, img_size=224):
    model = YOLO(model_name)
    
    results = model.train(
        data=str(self.data_dir),
        epochs=epochs,
        freeze=10,              # 冻结前10层
        cos_lr=True,            # 余弦退火学习率
        warmup_epochs=3,        # 预热轮数
        weight_decay=0.0005,    # 权重衰减
        degrees=15,             # 旋转角度
        mixup=0.1,              # MixUp增强
        copy_paste=0.1          # Copy-Paste增强
    )
```

**阶段2：全网络微调**（关键代码）：

```python
# scripts/model_training_optimized.py - 第84-131行
def train_stage2(self, model_path, epochs=50, batch_size=32, img_size=224):
    model = YOLO(model_path)
    
    results = model.train(
        data=str(self.data_dir),
        epochs=epochs,
        lr0=0.0001,             # 更低的学习率
        freeze=0,               # 解冻全网络
        patience=20,            # 更大的早停耐心值
        degrees=20,             # 更强的数据增强
        mixup=0.15,
        copy_paste=0.15
    )
```

**最佳模型选择**（关键代码）：

```python
# scripts/model_training_optimized.py - 第309-350行
def find_best_model(results_list):
    best_result = None
    best_metric = 0
    
    for name, report, model_path in results_list:
        accuracy = report['metrics']['accuracy']
        macro_f1 = report['macro_f1']
        target_class_f1 = report['classification_report']['细菌性叶枯病']['f1-score']
        
        # 综合评分：准确率40% + Macro F1 30% + 细菌性叶枯病F1 30%
        combined_score = accuracy * 0.4 + macro_f1 * 0.3 + target_class_f1 * 0.3
        
        if combined_score > best_metric:
            best_metric = combined_score
            best_result = {
                'name': name, 'report': report, 'model_path': model_path, 'combined_score': combined_score
            }
    
    # 复制最佳模型到项目根目录
    shutil.copy2(best_result['model_path'], 'best_model_final.pt')
    return best_result
```

### 4.4 细菌性叶枯病专项优化（optimize_bacterial.py）

**核心类：`RiceDiseaseOptimizer`**

针对细菌性叶枯病识别的专项优化，主要策略：

**策略1：颜色增强（关键代码）**：

```python
# scripts/optimize_bacterial.py - 第32-95行
def create_enhanced_dataset(self, augment_factor=2):
    for img_path in images:
        img = cv2.imread(str(img_path))
        
        # 亮度变化：模拟不同光照条件
        brightness_variants = [0.7, 0.85, 1.15, 1.3]
        for i, brightness in enumerate(brightness_variants):
            adjusted = cv2.convertScaleAbs(img, alpha=brightness, beta=0)
            cv2.imwrite(str(train_target / f'{img_path.stem}_bright_{i}.jpg'), adjusted)
        
        # 对比度变化
        contrast_variants = [0.7, 0.85, 1.15, 1.3]
        # ... 类似处理
        
        # 饱和度变化（HSV空间）
        for i, saturation in enumerate(saturation_variants):
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hsv[:, :, 1] = hsv[:, :, 1] * saturation
            adjusted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
```

**策略2：增强数据增强强度（关键代码）**：

```python
# scripts/optimize_bacterial.py - 第97-139行
def train_with_class_weighting(self, data_dir, model_name='yolo11n-cls.pt', epochs=80):
    model = YOLO(model_name)
    
    class_weights = [3.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # 细菌性叶枯病权重3倍
    
    results = model.train(
        data=str(data_dir),
        epochs=epochs,
        lr0=0.0005,
        cos_lr=True,
        warmup_epochs=5,
        mixup=0.2,           # 增强MixUp
        copy_paste=0.2,      # 增强Copy-Paste
        shear=10,            # 剪切增强
        degrees=20,
    )
```

### 4.5 难例挖掘模块（hard_example_mining.py）

**核心类：`HardExampleMiner`**

实现了多种数据增强方式：

| 增强类型 | 代码实现 | 参数范围 |
|----------|----------|----------|
| 旋转 | `img.rotate(angle)` | ±5°, ±10°, ±15° |
| 翻转 | `img.transpose()` | 水平/垂直翻转 |
| 亮度调整 | `ImageEnhance.Brightness` | 0.7-1.3 |
| 对比度调整 | `ImageEnhance.Contrast` | 0.7-1.3 |
| 饱和度调整 | `ImageEnhance.Color` | 0.5-1.5 |
| 高斯噪声 | NumPy随机噪声 | σ=15 |
| 高斯模糊 | `ImageFilter.GaussianBlur` | radius=2 |
| 透视变换 | OpenCV `getPerspectiveTransform` | 偏移10像素 |
| 缩放裁剪 | `img.crop()` + `img.resize()` | 裁剪10%边界 |

**难例挖掘逻辑**（关键代码）：

```python
# scripts/hard_example_mining.py - 第91-117行
def mine_hard_examples(self, model, test_dir, num_hard=50):
    hard_examples = []
    
    for img_path in target_dir.glob('*.jpg'):
        results = yolo_model.predict(str(img_path), verbose=False)
        probs = results[0].probs.data.tolist()
        pred_idx = results[0].probs.top1
        confidence = probs[pred_idx]
        
        # 难例定义：预测错误 或 置信度低于0.9
        if pred_idx != self.target_class_idx or confidence < 0.9:
            hard_examples.append({
                'path': str(img_path),
                'confidence': confidence,
                'pred_class': pred_idx,
                'true_class': self.target_class_idx
            })
    
    # 按置信度排序，取前num_hard个
    hard_examples.sort(key=lambda x: x['confidence'])
    return hard_examples[:num_hard]
```

---

## 5. 模型训练流程

### 5.1 训练流程总览

```
阶段1：数据准备
    ↓
阶段2：基础训练（model_training.py）
    ↓
阶段3：优化训练（model_training_optimized.py）
    ↓
阶段4：专项优化（optimize_bacterial.py）
    ↓
阶段5：模型评估与选择
```

### 5.2 训练参数对比

| 参数 | 基础训练 | 两阶段训练（阶段1） | 两阶段训练（阶段2） | 专项优化 |
|------|----------|---------------------|---------------------|----------|
| 模型 | YOLO11n-cls | YOLO11n-cls | YOLO11n-cls | YOLO11n-cls |
| 训练轮数 | 50 | 30 | 50 | 80 |
| 批次大小 | 32 | 32 | 32 | 32 |
| 输入尺寸 | 224 | 224 | 224 | 224 |
| 初始学习率 | 0.001 | 0.001 | 0.0001 | 0.0005 |
| 优化器 | AdamW | AdamW | AdamW | AdamW |
| 早停耐心值 | 10 | 15 | 20 | 25 |
| 冻结层数 | 无 | 10 | 0 | 无 |
| Cosine LR | 否 | 是 | 是 | 是 |
| Warmup | 无 | 3轮 | 3轮 | 5轮 |
| MixUp | 默认 | 0.1 | 0.15 | 0.2 |
| Copy-Paste | 默认 | 0.1 | 0.15 | 0.2 |

### 5.3 训练输出结构

```
runs/classify/rice_disease_exp/
├── weights/
│   ├── best.pt          # 最佳模型（验证集最优）
│   └── last.pt          # 最后一轮模型
├── results.csv          # 训练指标数据
├── results.png          # 训练曲线图表
├── confusion_matrix.png
└── evaluation_report.json  # 完整评估报告
```

---

## 6. 推理接口设计

### 6.1 核心类：`RiceDiseasePredictor`（src/inference.py）

**推理流程**（关键代码）：

```python
# src/inference.py - 第39-70行
def predict(self, image_path):
    if self.model is None:
        raise RuntimeError("模型尚未加载，请先调用 load_model()")
    
    start_time = time.time()
    
    # 执行推理
    results = self.model.predict(image_path, verbose=False)
    
    # 计算推理耗时
    latency_ms = (time.time() - start_time) * 1000
    
    # 解析结果
    pred_idx = results[0].probs.top1           # 预测类别索引
    confidence = float(results[0].probs.top1conf)  # 置信度
    probs = results[0].probs.data.tolist()     # 各类概率
    
    # 构建返回结果
    predicted_label = self.class_idx_to_cn[pred_idx]
    probabilities = {self.classes_cn[idx]: float(probs[idx]) for idx in range(len(self.classes))}
    
    result = {
        "predicted_label": predicted_label,
        "confidence": confidence,
        "probabilities": probabilities,
        "latency_ms": latency_ms
    }
    return result
```

### 6.2 全局单例模式

```python
# src/inference.py - 第72-82行
_predictor = None

def init_model(model_path):
    global _predictor
    _predictor = RiceDiseasePredictor(model_path)

def predict(image_path):
    if _predictor is None:
        raise RuntimeError("模型尚未初始化，请先调用 init_model()")
    return _predictor.predict(image_path)
```

### 6.3 接口规范

**调用方式**：

```python
from src.inference import init_model, predict

# 初始化模型（只需调用一次）
init_model('runs/classify/rice_disease_exp/weights/best.pt')

# 执行推理
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

---

## 7. 演示系统

### 7.1 核心类：`RiceDiseaseDemoSystem`（scripts/demo_system.py）

**抽样策略**：

**简单随机抽样**：

```python
# scripts/demo_system.py - 第59-74行
def random_sampling(self, n_samples=20, seed=42):
    random.seed(seed)
    self.test_samples = self._get_test_samples()
    sampled = random.sample(self.test_samples, n_samples)
    return sampled
```

**分层随机抽样**：

```python
# scripts/demo_system.py - 第76-110行
def stratified_sampling(self, n_samples=20, seed=42):
    random.seed(seed)
    self.test_samples = self._get_test_samples()
    
    samples_by_class = {}
    for sample in self.test_samples:
        label = sample['true_label']
        samples_by_class.setdefault(label, []).append(sample)
    
    n_classes = len(samples_by_class)
    samples_per_class = n_samples // n_classes
    
    stratified_samples = []
    for label, samples in samples_by_class.items():
        if len(samples) <= samples_per_class:
            stratified_samples.extend(samples)
        else:
            stratified_samples.extend(random.sample(samples, samples_per_class))
    
    random.shuffle(stratified_samples)
    return stratified_samples
```

**批量预测**（关键代码）：

```python
# scripts/demo_system.py - 第112-157行
def batch_predict(self, samples=None, save_results=True):
    self.predictions = []
    
    for i, sample in enumerate(samples):
        result = predict(sample['path'])
        
        prediction = {
            'index': i + 1,
            'filename': sample['filename'],
            'true_label': sample['true_label'],
            'predicted_label': result['predicted_label'],
            'confidence': result['confidence'],
            'probabilities': result['probabilities'],
            'latency_ms': result['latency_ms'],
            'correct': sample['true_label'] == result['predicted_label']
        }
        self.predictions.append(prediction)
    
    if save_results:
        self._save_results()
    return self.predictions
```

**结果导出**：

- **CSV格式**：包含序号、文件名、真实标签、预测标签、置信度、是否正确、推理耗时
- **JSON格式**：包含评测摘要（总数、正确数、准确率、平均耗时、各类别指标）

### 7.2 命令行使用

```bash
# 随机抽样20张图像
python scripts/demo_system.py --samples 20

# 分层随机抽样（推荐）
python scripts/demo_system.py --samples 20 --stratified

# 指定随机种子
python scripts/demo_system.py --samples 20 --seed 123

# 指定模型路径
python scripts/demo_system.py --samples 20 --model best_model_bacterial_opt.pt
```

---

## 8. 模型优化策略

### 8.1 优化策略总览

| 优化策略 | 实施文件 | 核心方法 | 预期提升 |
|----------|----------|----------|----------|
| 两阶段微调 | model_training_optimized.py | 冻结骨干网络训练 + 全网络微调 | 5-10% |
| 数据增强 | hard_example_mining.py | 旋转、翻转、颜色扰动、噪声等 | 3-8% |
| 颜色增强 | optimize_bacterial.py | 亮度/对比度/饱和度变化 | 针对色差场景 |
| 类别加权 | optimize_bacterial.py | 细菌性叶枯病权重3倍 | 提升目标类别F1 |
| 增强增强强度 | optimize_bacterial.py | MixUp 0.2, Copy-Paste 0.2 | 提升鲁棒性 |

### 8.2 优化实施顺序

```
阶段1（基础）：运行两阶段优化训练
    ↓
阶段2（数据）：数据增强 + 难例挖掘
    ↓
阶段3（专项）：细菌性叶枯病颜色增强 + 类别加权
    ↓
阶段4（评估）：验证集评估 + 选择最佳模型
```

---

## 9. 实验结果

### 9.1 模型性能对比

| 模型文件 | 训练方式 | 验证集准确率 | Macro F1 | 适用场景 |
|----------|----------|--------------|----------|----------|
| `best_model_final.pt` | 两阶段优化训练 | 98.71% | 98.43% | 通用场景 |
| `best_model_bacterial_opt.pt` | 颜色增强 + 类别加权 | 98.96% | 98.96% | 色差较大场景 |

### 9.2 各类别详细指标（优化模型）

| 类别 | 精确率 | 召回率 | F1分数 | 样本数 |
|------|--------|--------|--------|--------|
| 细菌性叶枯病 | 1.0000 | 1.0000 | 1.0000 | 64 |
| 褐斑病 | 0.9688 | 1.0000 | 0.9841 | 65 |
| 健康叶片 | 1.0000 | 1.0000 | 1.0000 | 66 |
| 稻瘟病 | 1.0000 | 0.9688 | 0.9841 | 64 |
| 叶灼病 | 1.0000 | 1.0000 | 1.0000 | 63 |
| 纹枯病 | 1.0000 | 1.0000 | 1.0000 | 64 |

### 9.3 推理性能

| 指标 | 数值 |
|------|------|
| 参数量 | ~1.5M |
| 模型大小 | ~5MB |
| CPU推理耗时 | ~20-50ms |
| GPU推理耗时 | ~5-10ms |

---

## 10. YOLO-Master论文改进方案

### 10.1 论文核心思想

YOLO-Master引入**混合专家（MoE）**机制，让模型根据输入复杂度动态激活专家网络，打破传统YOLO的静态计算范式。

### 10.2 对本项目的改进方向

| 改进方案 | 风险 | 实现难度 | 预期收益 |
|----------|------|----------|----------|
| 多尺度专家模块 | 低 | 低 | 特征表达增强，提升1-3%准确率 |
| 自适应推理策略 | 中 | 中 | 简单场景延迟降低30%+ |
| 完整MoE架构改造 | 高 | 高 | 最大提升3-5%准确率 |

### 10.3 推荐实施路径

```
阶段1（1-2天）：多尺度专家模块 → 验证特征增强效果
    ↓
阶段2（2-3天）：自适应推理策略 → 验证效率提升
    ↓
阶段3（3-5天）：完整MoE架构 → 追求最大性能提升
```

详细改进方案请参考 [planning.md](planning.md)

---

## 11. 项目文件清单

### 11.1 核心源代码

| 文件路径 | 功能描述 |
|----------|----------|
| `src/inference.py` | 推理接口，提供统一的predict() API |
| `scripts/data_processing.py` | 数据治理、质量检查、分层划分 |
| `scripts/model_training.py` | 基础模型训练 |
| `scripts/model_training_optimized.py` | 两阶段优化训练 + 最佳模型选择 |
| `scripts/optimize_bacterial.py` | 细菌性叶枯病专项优化训练 |
| `scripts/hard_example_mining.py` | 难例挖掘 + 多种数据增强 |
| `scripts/demo_system.py` | 演示系统，支持随机/分层抽样评测 |
| `scripts/eval_val.py` | 验证集评估脚本 |
| `scripts/generate_evaluation_report.py` | 评估报告生成脚本 |

### 11.2 文档文件

| 文件路径 | 内容描述 |
|----------|----------|
| `PLAN.md` | 项目实施方案（v1.2） |
| `TECHNICAL_DOC.md` | 详细技术文档 |
| `OPTIMIZATION_GUIDE.md` | 模型优化指南 |
| `DATA_SPLIT说明.md` | 数据划分说明 |
| `planning.md` | YOLO-Master论文改进计划 |
| `PROJECT_SUMMARY.md` | 项目总结（本文档） |

### 11.3 论文资料

YOLO-Master 论文原文与中文翻译为本地学习资料，不随本仓库分发。

### 11.4 数据与模型文件

| 文件路径 | 内容描述 |
|----------|----------|
| `data/processed/` | 划分后的数据集（train/val/test） |
| `data/processed/data_info.json` | 数据集元信息 |
| `runs/classify/` | 训练输出目录 |
| `best_model_final.pt` | 最佳通用模型 |
| `best_model_bacterial_opt.pt` | 优化后的专项模型 |
| `bacterial_optimization_result.json` | 优化结果报告 |
| `evaluation_results/` | 演示评测结果 |

---

## 附录：项目关键命令

### 数据处理

```bash
python scripts/data_processing.py
```

### 模型训练

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

### 模型评估

```bash
# 验证集评估
python scripts/eval_val.py

# 生成评估报告
python scripts/generate_evaluation_report.py
```

### 演示系统

```bash
# 随机抽样评测
python scripts/demo_system.py --samples 20

# 分层随机抽样评测（推荐）
python scripts/demo_system.py --samples 20 --stratified

# 使用优化模型评测
python scripts/demo_system.py --samples 20 --stratified --model best_model_bacterial_opt.pt
```

### 数据增强

```bash
# 数据增强（针对细菌性叶枯病）
python scripts/hard_example_mining.py --mode augment --augment_factor 3

# 难例挖掘
python scripts/hard_example_mining.py --mode mine --model best_model_final.pt
```

---

**文档版本**: v1.0  
**生成日期**: 2026年7月23日  
**项目名称**: Rice Disease System（水稻叶片病害智能识别系统）