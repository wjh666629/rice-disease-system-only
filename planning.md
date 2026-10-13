# 水稻病害识别项目改进方案：权重区分专家 + 噪声扩充 + YOLOv11 Mask升级

---

## 一、项目现状分析

### 1.1 当前数据集状况

| 项目 | 现状 |
|------|------|
| 数据格式 | 纯分类格式（按类别文件夹组织） |
| 类别数量 | 6类（细菌性叶枯病、褐斑病、健康叶片、稻瘟病、叶灼病、纹枯病） |
| 图像总数 | ~3829张 |
| 分割标注 | **无**（当前只有分类标签，无叶片mask标注） |
| Ultralytics版本 | 8.4.68（支持YOLO11系列） |

### 1.2 当前模型架构

```
当前模型：YOLO11n-cls (分类模型)
输入：224x224 RGB图像
输出：6类概率分布
训练方式：两阶段训练（冻结骨干→全网络微调）
```

### 1.3 核心挑战

| 挑战 | 说明 |
|------|------|
| **做深不做广** | 当前6类中，细菌性叶枯病是重点关注对象，需要针对性优化 |
| **数据量有限** | 每类约600-700张，需通过噪声扩充增强泛化能力 |
| **模型能力受限** | 分类模型只能判断是否患病，无法定位病害区域 |

---

## 二、核心改进方案

### 2.1 方案总览

```
┌─────────────────────────────────────────────────────────────┐
│                    项目改进三驾马车                          │
├─────────────────────────┬───────────────────────────────────┤
│   1. 权重区分专家机制    │   2. 噪声数据扩充                 │   3. YOLOv11 Mask升级   │
│   (做深不做广)          │   (3-4种噪声)                    │   (分割+分类双任务)     │
├─────────────────────────┼───────────────────────────────────┤
│ • 专家差异化初始化       │ • 高斯噪声                       │ • SAM自动生成伪掩码     │
│ • 专家差异化学习率       │ • 椒盐噪声                       │ • 分类→分割格式转换     │
│ • 目标类别专家增强       │ • 散斑噪声                       │ • YOLO11n-seg训练       │
│                         │ • 泊松噪声                       │ • 分割+分类联合推理     │
└─────────────────────────┴───────────────────────────────────┘
```

---

## 三、方案一：权重区分专家机制（做深不做广）

### 3.1 设计理念

**核心思想**：在MoE架构中，让不同专家专注于不同类型的特征提取，通过**差异化权重初始化**和**差异化学习率**，使专家具备不同的"专长"。

```
传统MoE（权重相等）：
    专家1 → 权重1
    专家2 → 权重1
    专家3 → 权重1
    专家4 → 权重1

权重区分MoE（做深不做广）：
    专家1 → 权重3.0 (专注：细菌性叶枯病特征)
    专家2 → 权重1.5 (专注：微小斑点特征)
    专家3 → 权重1.0 (专注：宏观形态特征)
    专家4 → 权重1.0 (专注：纹理细节特征)
```

### 3.2 权重区分专家模块实现

```python
import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv

class WeightedExpertBlock(nn.Module):
    """权重区分专家模块——做深不做广策略"""
    
    def __init__(self, in_channels, out_channels, expert_weights=[3.0, 1.5, 1.0, 1.0]):
        super().__init__()
        self.num_experts = len(expert_weights)
        self.expert_weights = torch.tensor(expert_weights)
        
        # 专家1：细菌性叶枯病专家（大权重，专注长条状病变特征）
        self.expert1 = nn.Sequential(
            Conv(in_channels, in_channels, 3),
            Conv(in_channels, in_channels, 7, padding=3),  # 大感受野捕捉长条特征
            Conv(in_channels, out_channels, 1)
        )
        
        # 专家2：斑点特征专家（中等权重，专注微小斑点）
        self.expert2 = nn.Sequential(
            Conv(in_channels, in_channels, 3),
            Conv(in_channels, in_channels, 3),
            Conv(in_channels, out_channels, 1)
        )
        
        # 专家3：宏观形态专家（标准权重）
        self.expert3 = nn.Sequential(
            Conv(in_channels, in_channels, 5, padding=2),
            Conv(in_channels, out_channels, 1)
        )
        
        # 专家4：纹理细节专家（标准权重）
        self.expert4 = nn.Sequential(
            Conv(in_channels, in_channels, 3),
            Conv(in_channels, out_channels, 1)
        )
        
        # 门控网络
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, self.num_experts),
            nn.Softmax(dim=1)
        )
        
        # 应用专家权重初始化
        self._init_expert_weights()
    
    def _init_expert_weights(self):
        """根据专家权重对每个专家进行差异化初始化"""
        for i, (expert, weight) in enumerate([
            (self.expert1, self.expert_weights[0]),
            (self.expert2, self.expert_weights[1]),
            (self.expert3, self.expert_weights[2]),
            (self.expert4, self.expert_weights[3])
        ]):
            for param in expert.parameters():
                if param.dim() > 1:  # 仅对卷积层权重应用缩放
                    nn.init.kaiming_normal_(param)
                    param.data *= weight ** 0.5  # 权重平方根缩放
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # 计算门控权重
        gate_weights = self.gate(x)  # [batch, num_experts]
        
        # 获取专家输出
        out1 = self.expert1(x)
        out2 = self.expert2(x)
        out3 = self.expert3(x)
        out4 = self.expert4(x)
        
        # 加权融合（专家权重 × 门控权重）
        w1 = gate_weights[:, 0].view(batch_size, 1, 1, 1) * self.expert_weights[0]
        w2 = gate_weights[:, 1].view(batch_size, 1, 1, 1) * self.expert_weights[1]
        w3 = gate_weights[:, 2].view(batch_size, 1, 1, 1) * self.expert_weights[2]
        w4 = gate_weights[:, 3].view(batch_size, 1, 1, 1) * self.expert_weights[3]
        
        # 归一化权重
        total_weight = w1 + w2 + w3 + w4 + 1e-8
        w1, w2, w3, w4 = w1/total_weight, w2/total_weight, w3/total_weight, w4/total_weight
        
        return w1 * out1 + w2 * out2 + w3 * out3 + w4 * out4
```

### 3.3 专家差异化学习率策略

```python
def set_expert_lr_strategy(model, base_lr=0.001):
    """为不同专家设置差异化学习率"""
    param_groups = []
    
    for name, param in model.named_parameters():
        if 'expert1' in name:  # 细菌性叶枯病专家：较高学习率，快速适应
            param_groups.append({
                'params': param,
                'lr': base_lr * 2.0,
                'weight_decay': 0.0005
            })
        elif 'expert2' in name:  # 斑点特征专家：中等学习率
            param_groups.append({
                'params': param,
                'lr': base_lr * 1.5,
                'weight_decay': 0.0005
            })
        elif 'expert3' in name or 'expert4' in name:  # 通用专家：标准学习率
            param_groups.append({
                'params': param,
                'lr': base_lr,
                'weight_decay': 0.0005
            })
        else:  # 其他层：标准学习率
            param_groups.append({
                'params': param,
                'lr': base_lr,
                'weight_decay': 0.0005
            })
    
    return param_groups
```

---

## 四、方案二：噪声数据扩充（3-4种噪声）

### 4.1 噪声选择与参数设计

| 噪声类型 | 参数范围 | 适用场景 | 代码实现 |
|----------|----------|----------|----------|
| **高斯噪声** | σ ∈ [0.01, 0.05] | 模拟传感器噪声 | `np.random.normal(0, σ, img.shape)` |
| **椒盐噪声** | p ∈ [0.005, 0.02] | 模拟图像传输错误 | 随机像素置0或255 |
| **散斑噪声** | factor ∈ [0.02, 0.08] | 模拟激光成像噪声 | `img + factor * img * noise` |
| **泊松噪声** | 无参数（基于像素值） | 模拟低光成像噪声 | `np.random.poisson(img)` |

### 4.2 噪声增强模块完整实现

```python
import numpy as np
import cv2
from PIL import Image, ImageEnhance

class NoiseAugmentor:
    """噪声数据增强器——支持4种噪声类型"""
    
    def __init__(self, noise_prob=0.5):
        self.noise_prob = noise_prob
    
    def add_gaussian_noise(self, image, sigma_range=(0.01, 0.05)):
        """添加高斯噪声"""
        img = np.array(image, dtype=np.float32) / 255.0
        sigma = np.random.uniform(*sigma_range)
        noise = np.random.normal(0, sigma, img.shape)
        noisy_img = np.clip(img + noise, 0, 1)
        return Image.fromarray((noisy_img * 255).astype(np.uint8))
    
    def add_salt_pepper_noise(self, image, p_range=(0.005, 0.02)):
        """添加椒盐噪声"""
        img = np.array(image)
        p = np.random.uniform(*p_range)
        salt_mask = np.random.random(img.shape[:2]) < p/2
        pepper_mask = np.random.random(img.shape[:2]) < p/2
        
        noisy_img = img.copy()
        noisy_img[salt_mask] = 255
        noisy_img[pepper_mask] = 0
        return Image.fromarray(noisy_img)
    
    def add_speckle_noise(self, image, factor_range=(0.02, 0.08)):
        """添加散斑噪声（乘性噪声）"""
        img = np.array(image, dtype=np.float32) / 255.0
        factor = np.random.uniform(*factor_range)
        noise = np.random.normal(0, 1, img.shape)
        noisy_img = np.clip(img + factor * img * noise, 0, 1)
        return Image.fromarray((noisy_img * 255).astype(np.uint8))
    
    def add_poisson_noise(self, image):
        """添加泊松噪声（基于像素值的噪声）"""
        img = np.array(image, dtype=np.float32)
        noisy_img = np.random.poisson(img).astype(np.float32)
        # 归一化回原始范围
        max_val = np.max(img)
        if max_val > 0:
            noisy_img = noisy_img * (255.0 / max_val)
        return Image.fromarray(np.clip(noisy_img, 0, 255).astype(np.uint8))
    
    def augment(self, image):
        """随机应用一种噪声增强"""
        if np.random.random() > self.noise_prob:
            return image
        
        noise_choices = [
            self.add_gaussian_noise,
            self.add_salt_pepper_noise,
            self.add_speckle_noise,
            self.add_poisson_noise
        ]
        
        # 加权选择：高斯噪声概率更高（更贴近真实场景）
        weights = [0.35, 0.25, 0.20, 0.20]
        noise_func = np.random.choice(noise_choices, p=weights)
        
        return noise_func(image)

# 使用示例
if __name__ == '__main__':
    augmentor = NoiseAugmentor(noise_prob=0.6)  # 60%概率应用噪声
    
    # 对单张图像增强
    img = Image.open('test.jpg')
    augmented_img = augmentor.augment(img)
    
    # 批量增强并保存
    import os
    input_dir = 'data/processed/train/bacterial_leaf_blight'
    output_dir = 'data/noise_augmented/train/bacterial_leaf_blight'
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.jpg'):
            img = Image.open(os.path.join(input_dir, filename))
            # 每张图生成3种不同噪声版本
            for i in range(3):
                augmented = augmentor.augment(img)
                augmented.save(os.path.join(output_dir, f'{os.path.splitext(filename)[0]}_noise_{i}.jpg'))
```

### 4.3 噪声增强流程设计

```
原始数据集 (3829张)
    ↓
噪声增强（每张图生成3种版本，60%概率应用）
    ↓
增强后数据集 (~3829 × 3 = 11487张)
    ↓
合并原始数据 + 增强数据
    ↓
最终训练集 (~15316张)
```

---

## 五、方案三：YOLOv11 Mask升级（从分类到分割）

### 5.1 升级路径规划

```
阶段1：生成伪分割标注（SAM自动标注）
    ↓
阶段2：转换为YOLO Seg格式
    ↓
阶段3：训练YOLO11n-seg模型
    ↓
阶段4：联合分类+分割推理
```

### 5.2 阶段1：SAM自动生成伪掩码

```python
import os
import numpy as np
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

class SAMMaskGenerator:
    """使用SAM自动生成叶片区域掩码"""
    
    def __init__(self, model_type='vit_h', checkpoint='sam_vit_h_4b8939.pth', device='cuda'):
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint)
        self.sam.to(device=device)
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)
    
    def generate_mask(self, image_path):
        """为单张图像生成叶片掩码"""
        image = np.array(Image.open(image_path))
        masks = self.mask_generator.generate(image)
        
        if len(masks) == 0:
            return None
        
        # 选择面积最大的掩码（通常是叶片主体）
        best_mask = max(masks, key=lambda x: x['area'])
        return best_mask['segmentation']
    
    def generate_masks_for_dataset(self, input_dir, output_dir):
        """为整个数据集生成掩码"""
        os.makedirs(output_dir, exist_ok=True)
        
        for class_name in os.listdir(input_dir):
            class_dir = os.path.join(input_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            output_class_dir = os.path.join(output_dir, class_name)
            os.makedirs(output_class_dir, exist_ok=True)
            
            for filename in os.listdir(class_dir):
                if filename.endswith('.jpg'):
                    img_path = os.path.join(class_dir, filename)
                    mask = self.generate_mask(img_path)
                    
                    if mask is not None:
                        # 保存为PNG格式的掩码
                        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                        mask_filename = os.path.splitext(filename)[0] + '_mask.png'
                        mask_img.save(os.path.join(output_class_dir, mask_filename))
        
        print("掩码生成完成！")

# 使用示例
if __name__ == '__main__':
    generator = SAMMaskGenerator()
    
    # 为训练集生成掩码
    generator.generate_masks_for_dataset(
        input_dir='data/processed/train',
        output_dir='data/masks/train'
    )
    
    # 为验证集生成掩码
    generator.generate_masks_for_dataset(
        input_dir='data/processed/val',
        output_dir='data/masks/val'
    )
```

### 5.3 阶段2：转换为YOLO Seg格式

```python
import os
import numpy as np
import cv2
from PIL import Image

class SegFormatConverter:
    """将掩码转换为YOLO Seg格式"""
    
    def __init__(self, class_names):
        self.class_names = class_names
        self.class_to_id = {name: idx for idx, name in enumerate(class_names)}
    
    def mask_to_yolo_seg(self, mask_path, image_path):
        """将单张掩码转换为YOLO Seg标注格式"""
        mask = np.array(Image.open(mask_path))
        img = Image.open(image_path)
        img_w, img_h = img.size
        
        # 找到掩码的轮廓
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if len(contours) == 0:
            return None
        
        # 选择最大的轮廓
        contour = max(contours, key=cv2.contourArea)
        
        # 转换为归一化坐标
        coords = []
        for point in contour:
            x = point[0][0] / img_w
            y = point[0][1] / img_h
            coords.append(f"{x:.6f} {y:.6f}")
        
        return " ".join(coords)
    
    def convert_dataset(self, image_dir, mask_dir, output_dir):
        """转换整个数据集"""
        os.makedirs(output_dir, exist_ok=True)
        
        for class_name in os.listdir(image_dir):
            class_id = self.class_to_id.get(class_name, -1)
            if class_id == -1:
                continue
            
            img_class_dir = os.path.join(image_dir, class_name)
            mask_class_dir = os.path.join(mask_dir, class_name)
            output_class_dir = os.path.join(output_dir, class_name)
            os.makedirs(output_class_dir, exist_ok=True)
            
            for filename in os.listdir(img_class_dir):
                if filename.endswith('.jpg'):
                    img_path = os.path.join(img_class_dir, filename)
                    mask_filename = os.path.splitext(filename)[0] + '_mask.png'
                    mask_path = os.path.join(mask_class_dir, mask_filename)
                    
                    if os.path.exists(mask_path):
                        seg_coords = self.mask_to_yolo_seg(mask_path, img_path)
                        
                        if seg_coords is not None:
                            txt_filename = os.path.splitext(filename)[0] + '.txt'
                            with open(os.path.join(output_class_dir, txt_filename), 'w') as f:
                                f.write(f"{class_id} {seg_coords}")

# 使用示例
if __name__ == '__main__':
    class_names = [
        'bacterial_leaf_blight',
        'brown_spot',
        'healthy',
        'leaf_blast',
        'leaf_scald',
        'sheath_blight'
    ]
    
    converter = SegFormatConverter(class_names)
    
    # 转换训练集
    converter.convert_dataset(
        image_dir='data/processed/train',
        mask_dir='data/masks/train',
        output_dir='data/seg_format/train'
    )
    
    # 转换验证集
    converter.convert_dataset(
        image_dir='data/processed/val',
        mask_dir='data/masks/val',
        output_dir='data/seg_format/val'
    )
```

### 5.4 阶段3：YOLO11n-seg训练脚本

```python
import os
from pathlib import Path
from ultralytics import YOLO

os.environ['YOLO_OFFLINE'] = '1'

class RiceDiseaseSegTrainer:
    """YOLO11n-seg模型训练器"""
    
    def __init__(self, data_dir, output_dir='runs/segment'):
        self.data_dir = Path(data_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.classes = [
            'bacterial_leaf_blight',
            'brown_spot',
            'healthy',
            'leaf_blast',
            'leaf_scald',
            'sheath_blight'
        ]
        
    def train(self, epochs=100, batch_size=16, img_size=640):
        print("="*60)
        print("开始YOLO11n-seg模型训练")
        print("="*60)
        
        # 加载分割模型
        model = YOLO('yolo11n-seg.pt')
        
        # 训练配置
        results = model.train(
            data=str(self.data_dir),
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            patience=20,
            seed=42,
            verbose=True,
            project=str(self.output_dir.parent),
            name=self.output_dir.name,
            pretrained=True,
            optimizer='AdamW',
            lr0=0.001,
            augment=True,
            val=True,
            cos_lr=True,
            warmup_epochs=5,
            weight_decay=0.0005,
            degrees=15,
            translate=0.1,
            scale=0.5,
            shear=5,
            flipud=0.5,
            fliplr=0.5,
            mosaic=0.5,
            mixup=0.1
        )
        
        best_model_path = self.output_dir / 'weights' / 'best.pt'
        print(f"\n训练完成！最佳模型: {best_model_path}")
        
        return model, results, best_model_path
    
    def evaluate(self, model, data_dir=None):
        print("\n开始模型评估...")
        
        if data_dir is None:
            data_dir = self.data_dir / 'val'
        
        results = model.val(data=str(data_dir))
        
        metrics = {
            'mAP50': float(results.box.map50),
            'mAP50_95': float(results.box.map),
            'mAP50_seg': float(results.seg.map50),
            'mAP50_95_seg': float(results.seg.map)
        }
        
        print(f"评估结果:")
        print(f"Box mAP50: {metrics['mAP50']:.4f}")
        print(f"Box mAP50-95: {metrics['mAP50_95']:.4f}")
        print(f"Seg mAP50: {metrics['mAP50_seg']:.4f}")
        print(f"Seg mAP50-95: {metrics['mAP50_95_seg']:.4f}")
        
        return metrics, results

# 使用示例
if __name__ == '__main__':
    trainer = RiceDiseaseSegTrainer('data/seg_format')
    model, results, best_path = trainer.train(epochs=100, batch_size=16, img_size=640)
    
    # 评估
    metrics, val_results = trainer.evaluate(model)
```

### 5.5 阶段4：联合分类+分割推理

```python
class RiceDiseaseSegPredictor:
    """联合分类+分割推理器"""
    
    def __init__(self, model_path):
        self.model = YOLO(model_path)
    
    def predict(self, image_path):
        """执行联合推理"""
        results = self.model.predict(image_path, verbose=False)
        
        result = results[0]
        
        # 获取分割结果
        masks = result.masks
        boxes = result.boxes
        
        # 获取分类信息（从分割结果中推导）
        predictions = []
        if boxes is not None:
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                confidence = float(boxes.conf[i])
                
                # 获取掩码信息
                mask_area = None
                if masks is not None:
                    mask_area = float(masks.xy[i].shape[0])  # 掩码面积
                
                predictions.append({
                    'class_id': cls_id,
                    'confidence': confidence,
                    'mask_area': mask_area,
                    'bbox': boxes.xyxy[i].tolist()
                })
        
        return {
            'predictions': predictions,
            'image_shape': result.orig_shape
        }
```

---

## 六、完整实施路线图

### 6.1 阶段划分与时间规划

| 阶段 | 任务 | 预计时间 | 产出 |
|------|------|----------|------|
| **阶段0** | 数据预处理：噪声增强 | 1-2天 | 扩充后数据集（~15316张） |
| **阶段1** | SAM自动生成伪掩码 | 2-3天 | 叶片区域掩码标注 |
| **阶段2** | 转换为YOLO Seg格式 | 1天 | Seg格式数据集 |
| **阶段3** | 训练YOLO11n-seg基础模型 | 3-5天 | 基础分割模型 |
| **阶段4** | 集成权重区分专家模块 | 2-3天 | 专家增强模型 |
| **阶段5** | 联合推理接口开发 | 1天 | 分类+分割联合推理 |
| **阶段6** | 模型评估与优化 | 2-3天 | 最终模型+评估报告 |

### 6.2 关键配置清单

```yaml
# 数据集配置
dataset:
  original_size: 3829
  augmented_size: 15316
  noise_types: [gaussian, salt_pepper, speckle, poisson]
  noise_prob: 0.6

# 模型配置
model:
  type: yolo11n-seg
  img_size: 640
  batch_size: 16
  epochs: 100
  
# 专家权重配置
expert_weights:
  bacterial_expert: 3.0      # 细菌性叶枯病专家
  spot_expert: 1.5           # 斑点特征专家
  morphology_expert: 1.0     # 宏观形态专家
  texture_expert: 1.0        # 纹理细节专家

# 学习率配置
learning_rates:
  base_lr: 0.001
  bacterial_expert_lr: 0.002  # 2x base
  spot_expert_lr: 0.0015     # 1.5x base
  other_experts_lr: 0.001    # 1x base
```

---

## 七、预期效果与评估指标

### 7.1 预期性能提升

| 指标 | 当前模型 | 改进后模型 | 预期提升 |
|------|----------|------------|----------|
| Top-1 Accuracy | 98.71% | 99.2%+ | +0.5%+ |
| 细菌性叶枯病F1 | ~97% | 99%+ | +2%+ |
| 推理延迟 | ~20ms | ~25ms | 可接受增加 |
| 病害定位能力 | 无 | 有（分割掩码） | 新增功能 |

### 7.2 评估方法

```python
def evaluate_combined_model(model, test_dir):
    """评估分类+分割联合模型"""
    y_true = []
    y_pred = []
    mask_accuracies = []
    
    for class_idx, class_name in enumerate(classes):
        class_dir = test_dir / class_name
        for img_path in class_dir.glob('*.jpg'):
            results = model.predict(str(img_path), verbose=False)
            pred_idx = results[0].probs.top1
            
            y_true.append(class_idx)
            y_pred.append(pred_idx)
            
            # 评估分割质量（如果有掩码）
            if hasattr(results[0], 'masks') and results[0].masks is not None:
                mask_acc = evaluate_mask_accuracy(results[0].masks)
                mask_accuracies.append(mask_acc)
    
    # 分类指标
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
    report = classification_report(y_true, y_pred, target_names=classes_cn)
    
    # 分割指标
    avg_mask_acc = np.mean(mask_accuracies) if mask_accuracies else None
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'avg_mask_accuracy': avg_mask_acc
    }
```

---

## 八、风险与应对策略

| 风险 | 概率 | 应对策略 |
|------|------|----------|
| SAM生成掩码质量不佳 | 中 | 人工抽检+修正，或使用更精确的标注工具 |
| 噪声增强过度导致过拟合 | 中 | 控制噪声概率和强度，使用验证集监控 |
| 专家权重过高导致模型偏置 | 低 | 交叉验证调整权重，使用负载平衡损失 |
| YOLO11-seg训练资源需求高 | 中 | 使用较小batch_size，缩短训练轮数 |

---

**创建日期**：2026年7月24日  
**适用项目**：Rice Disease System（水稻叶片病害识别系统）  
**文档版本**：v2.0（权重区分专家 + 噪声扩充 + YOLOv11 Mask升级）