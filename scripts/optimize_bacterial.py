import os
os.environ['YOLO_OFFLINE'] = '1'

import json
from pathlib import Path
from ultralytics import YOLO
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import shutil


class RiceDiseaseOptimizer:
    def __init__(self, data_dir='data/processed', output_dir='runs/classify'):
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
        
        self.classes_cn = ['细菌性叶枯病', '褐斑病', '健康叶片', '稻瘟病', '叶灼病', '纹枯病']
        self.target_class_idx = 0
        self.target_class_name = self.classes[self.target_class_idx]
        self.target_class_cn = self.classes_cn[self.target_class_idx]
        
    def create_enhanced_dataset(self, augment_factor=2):
        enhanced_dir = Path('data/enhanced_for_bacterial')
        enhanced_dir.mkdir(parents=True, exist_ok=True)
        
        for split in ['train', 'val']:
            src_split = self.data_dir / split
            dst_split = enhanced_dir / split
            dst_split.mkdir(parents=True, exist_ok=True)
            
            for cls in self.classes:
                src_cls = src_split / cls
                dst_cls = dst_split / cls
                if src_cls.exists():
                    if dst_cls.exists():
                        shutil.rmtree(dst_cls)
                    shutil.copytree(src_cls, dst_cls)
        
        train_target = enhanced_dir / 'train' / self.target_class_name
        val_target = enhanced_dir / 'val' / self.target_class_name
        
        images = list(train_target.glob('*.jpg'))
        print(f'原始细菌性叶枯病训练样本: {len(images)}')
        
        import cv2
        aug_count = 0
        for img_path in images:
            img = cv2.imread(str(img_path))
            
            brightness_variants = [0.7, 0.85, 1.15, 1.3]
            contrast_variants = [0.7, 0.85, 1.15, 1.3]
            saturation_variants = [0.7, 0.85, 1.15, 1.3]
            
            for i, brightness in enumerate(brightness_variants):
                adjusted = cv2.convertScaleAbs(img, alpha=brightness, beta=0)
                aug_name = f'{img_path.stem}_bright_{i}.jpg'
                cv2.imwrite(str(train_target / aug_name), adjusted)
                aug_count += 1
                if aug_count >= len(images) * (augment_factor - 1):
                    break
            
            for i, contrast in enumerate(contrast_variants):
                adjusted = cv2.convertScaleAbs(img, alpha=contrast, beta=0)
                aug_name = f'{img_path.stem}_contrast_{i}.jpg'
                cv2.imwrite(str(train_target / aug_name), adjusted)
                aug_count += 1
                if aug_count >= len(images) * (augment_factor - 1):
                    break
            
            for i, saturation in enumerate(saturation_variants):
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                hsv[:, :, 1] = hsv[:, :, 1] * saturation
                adjusted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                aug_name = f'{img_path.stem}_saturation_{i}.jpg'
                cv2.imwrite(str(train_target / aug_name), adjusted)
                aug_count += 1
                if aug_count >= len(images) * (augment_factor - 1):
                    break
            
            if aug_count >= len(images) * (augment_factor - 1):
                break
        
        print(f'增强后细菌性叶枯病训练样本: {len(list(train_target.glob("*.jpg")))}')
        return enhanced_dir
    
    def train_with_class_weighting(self, data_dir, model_name='yolo11n-cls.pt', epochs=80):
        print(f'开始训练（针对{self.target_class_cn}优化）')
        print(f'数据集: {data_dir}')
        print(f'模型: {model_name}')
        
        model = YOLO(model_name)
        
        class_weights = [3.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        
        results = model.train(
            data=str(data_dir),
            epochs=epochs,
            batch=32,
            imgsz=224,
            patience=25,
            seed=42,
            verbose=True,
            project=str(self.output_dir.parent),
            name=f'{self.output_dir.name}_bacterial_opt',
            pretrained=True,
            optimizer='AdamW',
            lr0=0.0005,
            augment=True,
            val=True,
            cos_lr=True,
            warmup_epochs=5,
            weight_decay=0.0005,
            degrees=20,
            translate=0.15,
            scale=0.6,
            shear=10,
            perspective=0.001,
            flipud=0.5,
            fliplr=0.5,
            mosaic=0.0,
            mixup=0.2,
            copy_paste=0.2,
            close_mosaic=0,
            overlap_mask=True,
        )
        
        best_model_path = self.output_dir.parent / f'{self.output_dir.name}_bacterial_opt' / 'weights' / 'best.pt'
        return model, results, best_model_path
    
    def evaluate_on_val(self, model_path):
        print(f'\n评估模型: {model_path}')
        model = YOLO(model_path)
        
        y_true = []
        y_pred = []
        
        val_dir = self.data_dir / 'val'
        for class_idx, class_name in enumerate(self.classes):
            class_dir = val_dir / class_name
            if not class_dir.exists():
                continue
            
            for img_path in class_dir.glob('*.jpg'):
                results = model.predict(str(img_path), verbose=False)
                pred_idx = results[0].probs.top1
                y_true.append(class_idx)
                y_pred.append(pred_idx)
        
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=self.classes_cn, output_dict=True)
        
        print(f'\n总体准确率: {report["accuracy"]:.4f}')
        print(f'Macro F1: {report["macro avg"]["f1-score"]:.4f}')
        
        print(f'\n{self.target_class_cn}指标:')
        print(f'  准确率: {cm[self.target_class_idx][self.target_class_idx] / report[self.target_class_cn]["support"]:.4f}')
        print(f'  精确率: {report[self.target_class_cn]["precision"]:.4f}')
        print(f'  召回率: {report[self.target_class_cn]["recall"]:.4f}')
        print(f'  F1分数: {report[self.target_class_cn]["f1-score"]:.4f}')
        
        return report, cm
    
    def run_optimization_pipeline(self, augment_factor=2, epochs=80):
        print('=' * 60)
        print(f'针对{self.target_class_cn}的优化训练')
        print('=' * 60)
        
        enhanced_dir = self.create_enhanced_dataset(augment_factor)
        
        print('\n步骤1: 创建增强数据集完成')
        print(f'增强数据集位置: {enhanced_dir}')
        
        print('\n步骤2: 开始训练...')
        model, results, best_model_path = self.train_with_class_weighting(enhanced_dir, epochs=epochs)
        
        print(f'\n步骤3: 评估验证集...')
        report, cm = self.evaluate_on_val(best_model_path)
        
        shutil.copy2(best_model_path, 'best_model_bacterial_opt.pt')
        print(f'\n优化模型已保存至: best_model_bacterial_opt.pt')
        
        result_data = {
            'target_class': self.target_class_cn,
            'enhanced_dataset': str(enhanced_dir),
            'augment_factor': augment_factor,
            'epochs': epochs,
            'model_path': str(best_model_path),
            'evaluation': {
                'total_accuracy': report['accuracy'],
                'macro_f1': report['macro avg']['f1-score'],
                'target_class_metrics': {
                    'accuracy': float(cm[self.target_class_idx][self.target_class_idx] / report[self.target_class_cn]['support']),
                    'precision': float(report[self.target_class_cn]['precision']),
                    'recall': float(report[self.target_class_cn]['recall']),
                    'f1_score': float(report[self.target_class_cn]['f1-score'])
                }
            }
        }
        
        with open('bacterial_optimization_result.json', 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        print(f'优化结果已保存至: bacterial_optimization_result.json')
        
        return result_data


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='针对细菌性叶枯病的优化训练')
    parser.add_argument('--augment_factor', type=int, default=2, help='增强倍数')
    parser.add_argument('--epochs', type=int, default=80, help='训练轮数')
    args = parser.parse_args()
    
    optimizer = RiceDiseaseOptimizer()
    result = optimizer.run_optimization_pipeline(augment_factor=args.augment_factor, epochs=args.epochs)
    
    print(f'\n优化完成！')
    print(f'{optimizer.target_class_cn} F1分数: {result["evaluation"]["target_class_metrics"]["f1_score"]:.4f}')