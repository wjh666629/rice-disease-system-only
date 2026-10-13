# 模型优化方案 - 细菌性叶枯病识别准确率提升

## 问题分析

当前模型在测试集上准确率约为 75%，其中**细菌性叶枯病**识别准确率显著低于其他类别。

### 可能的原因

1. **数据质量问题**：细菌性叶枯病的图像特征与其他病害相似，容易混淆
2. **数据量不足**：虽然数量均衡，但多样性可能不够
3. **训练不充分**：50轮训练可能不足以收敛
4. **学习率设置**：学习率可能过高或过低
5. **数据增强不足**：增强策略不够丰富

---

## 优化方案

### 方案一：两阶段微调训练（推荐）

**核心思路**：先冻结骨干网络训练分类头，再解冻全网络微调

**执行命令**：
```bash
python scripts/model_training_optimized.py --mode two_stage --model yolo11n-cls.pt
```

**优化点**：
- 阶段1：冻结前10层，训练30轮，学习率 0.001
- 阶段2：全网络训练50轮，学习率 0.0001（更精细的微调）
- 增加数据增强强度（旋转、颜色扰动、噪声等）
- 早停耐心值从10增加到20
- 使用余弦退火学习率调度

**预期提升**：准确率提升 5-10%

---

### 方案二：更大模型训练

**核心思路**：使用 YOLO11s-cls（small版本）替代 nano 版本

**执行命令**：
```bash
python scripts/model_training_optimized.py --mode large --model yolo11s-cls.pt
```

**优化点**：
- 模型参数量从 1.5M 增加到约 5.5M
- 输入尺寸从 224 增加到 256
- 训练80轮，充分收敛
- 更强的数据增强

**预期提升**：准确率提升 8-15%

**权衡**：模型文件更大（约 50MB），推理速度稍慢

---

### 方案三：数据增强 + 难例挖掘

**核心思路**：增加细菌性叶枯病的数据量和多样性

#### 步骤1：数据增强
```bash
python scripts/hard_example_mining.py --mode augment --augment_factor 3
```

**增强方式**：
- 旋转（±5°, ±10°, ±15°）
- 水平/垂直翻转
- 亮度/对比度/饱和度调整
- 高斯噪声
- 高斯模糊
- 透视变换
- 缩放裁剪

#### 步骤2：使用增强数据训练
```bash
python scripts/model_training.py --data_dir data/augmented
```

#### 步骤3：难例挖掘（可选）
```bash
python scripts/hard_example_mining.py --mode mine --model runs/classify/rice_disease_exp/weights/best.pt
```

**预期提升**：准确率提升 3-8%

---

### 方案四：综合优化（推荐）

**执行顺序**：
1. 先做数据增强（方案三）
2. 再用两阶段训练（方案一）
3. 最后用大模型验证（方案二）

**执行命令**：
```bash
# 步骤1：数据增强
python scripts/hard_example_mining.py --mode augment --augment_factor 2

# 步骤2：两阶段微调
python scripts/model_training_optimized.py --mode two_stage --data_dir data/augmented

# 步骤3：大模型训练（可选）
python scripts/model_training_optimized.py --mode large --data_dir data/augmented
```

**预期提升**：准确率提升 10-20%

---

## 训练参数对比

| 参数 | 原始配置 | 优化配置（两阶段） | 优化配置（大模型） |
|------|----------|---------------------|---------------------|
| 模型 | YOLO11n-cls | YOLO11n-cls | YOLO11s-cls |
| 输入尺寸 | 224 | 224 | 256 |
| 训练轮数 | 50 | 30+50=80 | 80 |
| 初始学习率 | 0.001 | 0.001+0.0001 | 0.001 |
| 优化器 | AdamW | AdamW | AdamW |
| 早停耐心值 | 10 | 15+20 | 20 |
| 数据增强 | 标准 | 增强 | 增强 |
| 冻结层数 | 无 | 阶段1冻结10层 | 无 |

---

## 错误分析指南

如果优化后准确率仍不理想，按以下步骤排查：

### 1. 检查数据质量
```bash
# 检查细菌性叶枯病图像
ls data/processed/train/bacterial_leaf_blight/ | head -20
```

**检查要点**：
- 图像是否清晰
- 叶片是否完整
- 是否有标注错误
- 是否混入其他类别的图像

### 2. 分析混淆情况
查看 `evaluation_report.json` 中的混淆矩阵：
- 细菌性叶枯病最容易和哪个类别混淆？
- 错误样本有什么共同特征？

### 3. 训练过程分析
查看 `results.csv`：
- 训练损失是否持续下降？
- 验证损失是否出现过拟合？
- 准确率曲线是否平稳？

---

## 快速调试技巧

### 技巧1：先小数据验证
```bash
# 用少量数据快速测试训练流程
python -c "
from ultralytics import YOLO
model = YOLO('yolo11n-cls.pt')
model.train(data='data/processed', epochs=5, imgsz=224, batch=8)
"
```

### 技巧2：检查类别顺序
确保训练时的类别顺序与数据目录一致：
```python
from ultralytics import YOLO
model = YOLO('runs/classify/rice_disease_exp/weights/best.pt')
print(model.names)  # 查看类别名称
```

### 技巧3：单张图像测试
```python
from src.inference import init_model, predict
init_model('runs/classify/rice_disease_exp/weights/best.pt')
result = predict('data/processed/test/bacterial_leaf_blight/some_image.jpg')
print(result)
```

---

## 推荐执行顺序

1. **第一步**：运行两阶段优化训练
   ```bash
   python scripts/model_training_optimized.py --mode two_stage
   ```

2. **第二步**：评估结果
   - 查看 `runs/classify/rice_disease_optimized_stage2/evaluation_report.json`
   - 对比细菌性叶枯病的准确率提升

3. **第三步**：如果提升不明显，尝试大模型
   ```bash
   python scripts/model_training_optimized.py --mode large
   ```

4. **第四步**：如果还需提升，做数据增强
   ```bash
   python scripts/hard_example_mining.py --mode augment --augment_factor 2
   python scripts/model_training_optimized.py --mode two_stage --data_dir data/augmented
   ```

---

## 常见问题

### Q: 训练时显存不足怎么办？
A: 减小 batch_size，例如从32改为16或8。

### Q: 训练速度太慢怎么办？
A: 可以减少训练轮数，或者先试用小数据集验证。

### Q: 如何查看训练过程？
A: 查看 `runs/classify/实验名/results.png` 或 `results.csv`。

### Q: 模型文件在哪里？
A: 最佳模型在 `runs/classify/实验名/weights/best.pt`。

---

**文档版本**: v1.0  
**生成日期**: 2026-06-25  
**针对问题**: 细菌性叶枯病识别准确率低