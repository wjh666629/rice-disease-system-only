import os
import json
import time
from pathlib import Path
from ultralytics import YOLO
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


os.environ['YOLO_OFFLINE'] = '1'


class RiceDiseaseModelTrainerOptimized:
    def __init__(self, data_dir, output_dir='runs/classify'):
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
        
        self.classes_cn = [
            '细菌性叶枯病',
            '褐斑病',
            '健康叶片',
            '稻瘟病',
            '叶灼病',
            '纹枯病'
        ]
        
        self.class_mapping = dict(zip(self.classes, self.classes_cn))
        self.class_idx_to_cn = dict(zip(range(len(self.classes)), self.classes_cn))
        
    def train_stage1(self, model_name='yolo11n-cls.pt', epochs=30, batch_size=32, img_size=224):
        print("="*60)
        print("阶段1：冻结骨干网络训练")
        print("="*60)
        
        model = YOLO(model_name)
        
        results = model.train(
            data=str(self.data_dir),
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            patience=15,
            seed=42,
            verbose=True,
            project=str(self.output_dir.parent),
            name=self.output_dir.name,
            pretrained=True,
            optimizer='AdamW',
            lr0=0.001,
            augment=True,
            val=True,
            freeze=10,
            cos_lr=True,
            warmup_epochs=3,
            weight_decay=0.0005,
            degrees=15,
            translate=0.1,
            scale=0.5,
            shear=5,
            perspective=0.0005,
            flipud=0.5,
            fliplr=0.5,
            mosaic=0.0,
            mixup=0.1,
            copy_paste=0.1
        )
        
        stage1_best = self.output_dir / 'weights' / 'best.pt'
        print(f"阶段1训练完成！最佳模型: {stage1_best}")
        return model, results, stage1_best
    
    def train_stage2(self, model_path, epochs=50, batch_size=32, img_size=224):
        print("\n" + "="*60)
        print("阶段2：全网络微调")
        print(f"加载模型: {model_path}")
        print("="*60)
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        model_path = str(Path(model_path).resolve())
        
        model = YOLO(model_path)
        
        results = model.train(
            data=str(self.data_dir),
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            patience=20,
            seed=42,
            verbose=True,
            project=str(self.output_dir.parent),
            name=f"{self.output_dir.name}_stage2",
            pretrained=False,
            optimizer='AdamW',
            lr0=0.0001,
            lrf=0.01,
            augment=True,
            val=True,
            freeze=0,
            cos_lr=True,
            warmup_epochs=3,
            weight_decay=0.0005,
            degrees=20,
            translate=0.15,
            scale=0.6,
            shear=8,
            perspective=0.001,
            flipud=0.5,
            fliplr=0.5,
            mosaic=0.0,
            mixup=0.15,
            copy_paste=0.15
        )
        
        stage2_best = self.output_dir.parent / f"{self.output_dir.name}_stage2" / 'weights' / 'best.pt'
        print(f"阶段2训练完成！最佳模型: {stage2_best}")
        return model, results, stage2_best
    
    def evaluate(self, model, data_dir=None):
        print("\n开始模型评估...")
        
        if data_dir is None:
            data_dir = self.data_dir / 'test'
        
        results = model.val(data=str(data_dir))
        
        metrics = {
            'accuracy': float(results.top1),
            'top5_accuracy': float(results.top5)
        }
        
        print(f"评估结果:")
        print(f"Top-1 Accuracy: {metrics['accuracy']:.4f}")
        print(f"Top-5 Accuracy: {metrics['top5_accuracy']:.4f}")
        
        return metrics, results
    
    def compute_confusion_matrix(self, model, test_dir):
        print("\n计算混淆矩阵...")
        
        y_true = []
        y_pred = []
        all_probs = []
        error_samples = []
        
        for class_idx, class_name in enumerate(self.classes):
            class_dir = test_dir / class_name
            if not class_dir.exists():
                continue
            
            for img_path in class_dir.glob('*.jpg'):
                try:
                    results = model.predict(str(img_path), verbose=False)
                    pred_idx = results[0].probs.top1
                    probs = results[0].probs.data.tolist()
                    
                    y_true.append(class_idx)
                    y_pred.append(pred_idx)
                    all_probs.append(probs)
                    
                    if pred_idx != class_idx:
                        error_samples.append({
                            'image': str(img_path),
                            'true_label': self.classes_cn[class_idx],
                            'pred_label': self.classes_cn[pred_idx],
                            'confidence': float(probs[pred_idx])
                        })
                except Exception as e:
                    print(f"处理图像 {img_path} 时出错: {e}")
        
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(
            y_true, y_pred,
            target_names=self.classes_cn,
            output_dict=True
        )
        
        return cm, report, y_true, y_pred, all_probs, error_samples
    
    def save_evaluation_report(self, metrics, cm, report, y_true, y_pred, all_probs, error_samples, save_path):
        report_dir = Path(save_path).parent
        report_dir.mkdir(parents=True, exist_ok=True)
        
        eval_report = {
            'evaluation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': metrics,
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'class_mapping': self.class_mapping,
            'class_list_cn': self.classes_cn,
            'total_samples': len(y_true),
            'correct_predictions': sum(1 for t, p in zip(y_true, y_pred) if t == p),
            'macro_f1': report['macro avg']['f1-score'],
            'macro_precision': report['macro avg']['precision'],
            'macro_recall': report['macro avg']['recall'],
            'error_samples': error_samples[:20]
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(eval_report, f, ensure_ascii=False, indent=2)
        
        print(f"\n评估报告已保存至: {save_path}")
        
        return eval_report
    
    def save_confusion_matrix_csv(self, cm, save_path):
        cm_df = pd.DataFrame(cm, index=self.classes_cn, columns=self.classes_cn)
        cm_df.to_csv(save_path, encoding='utf-8-sig')
        print(f"混淆矩阵已保存至: {save_path}")
    
    def run_two_stage(self, model_name='yolo11n-cls.pt'):
        print("开始两阶段优化训练...")
        print(f"使用模型: {model_name}")
        
        stage1_model, stage1_results, stage1_best = self.train_stage1(model_name=model_name)
        
        print(f"\n阶段1最佳模型路径: {stage1_best}")
        if not stage1_best.exists():
            print(f"警告：阶段1模型文件不存在！")
            return None, None, None
        
        stage2_model, stage2_results, stage2_best = self.train_stage2(str(stage1_best))
        
        test_dir = self.data_dir / 'test'
        metrics, val_results = self.evaluate(stage2_model, test_dir)
        cm, report, y_true, y_pred, all_probs, error_samples = self.compute_confusion_matrix(stage2_model, test_dir)
        
        report_path = stage2_best.parent.parent / 'evaluation_report.json'
        eval_report = self.save_evaluation_report(
            metrics, cm, report, y_true, y_pred, all_probs, error_samples, str(report_path)
        )
        
        cm_path = stage2_best.parent.parent / 'confusion_matrix.csv'
        self.save_confusion_matrix_csv(cm, str(cm_path))
        
        print(f"\n最佳模型已保存至: {stage2_best}")
        
        return stage2_model, eval_report, stage2_best
    
    def run_single_stage_long(self, model_name='yolo11n-cls.pt', epochs=100, batch_size=32, img_size=224):
        print("开始长训练优化...")
        print(f"使用模型: {model_name}, 训练轮数: {epochs}")
        
        model = YOLO(model_name)
        
        results = model.train(
            data=str(self.data_dir),
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            patience=25,
            seed=42,
            verbose=True,
            project=str(self.output_dir.parent),
            name=f"{self.output_dir.name}_long",
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
            shear=8,
            perspective=0.001,
            flipud=0.5,
            fliplr=0.5,
            mosaic=0.0,
            mixup=0.15,
            copy_paste=0.15
        )
        
        best_model_path = self.output_dir.parent / f"{self.output_dir.name}_long" / 'weights' / 'best.pt'
        
        test_dir = self.data_dir / 'test'
        metrics, val_results = self.evaluate(model, test_dir)
        cm, report, y_true, y_pred, all_probs, error_samples = self.compute_confusion_matrix(model, test_dir)
        
        report_path = best_model_path.parent.parent / 'evaluation_report.json'
        eval_report = self.save_evaluation_report(
            metrics, cm, report, y_true, y_pred, all_probs, error_samples, str(report_path)
        )
        
        cm_path = best_model_path.parent.parent / 'confusion_matrix.csv'
        self.save_confusion_matrix_csv(cm, str(cm_path))
        
        print(f"\n最佳模型已保存至: {best_model_path}")
        
        return model, eval_report, best_model_path


def find_best_model(results_list):
    print("\n" + "="*60)
    print("选择最佳模型")
    print("="*60)
    
    best_result = None
    best_metric = 0
    
    for name, report, model_path in results_list:
        accuracy = report['metrics']['accuracy']
        macro_f1 = report['macro_f1']
        target_class_f1 = report['classification_report']['细菌性叶枯病']['f1-score']
        
        combined_score = accuracy * 0.4 + macro_f1 * 0.3 + target_class_f1 * 0.3
        
        print(f"\n{name}:")
        print(f"  Top-1 Accuracy: {accuracy:.4f}")
        print(f"  Macro F1: {macro_f1:.4f}")
        print(f"  细菌性叶枯病 F1: {target_class_f1:.4f}")
        print(f"  综合评分: {combined_score:.4f}")
        print(f"  模型路径: {model_path}")
        
        if combined_score > best_metric:
            best_metric = combined_score
            best_result = {
                'name': name,
                'report': report,
                'model_path': model_path,
                'combined_score': combined_score
            }
    
    if best_result:
        print(f"\n" + "="*60)
        print(f"最佳模型: {best_result['name']}")
        print(f"综合评分: {best_result['combined_score']:.4f}")
        print(f"模型路径: {best_result['model_path']}")
        print("="*60)
        
        shutil.copy2(best_result['model_path'], 'best_model_final.pt')
        print(f"\n最佳模型已复制到项目根目录: best_model_final.pt")
    
    return best_result


if __name__ == '__main__':
    import argparse
    import shutil
    
    parser = argparse.ArgumentParser(description='水稻病害模型优化训练（YOLO11n-cls）')
    parser.add_argument('--mode', type=str, default='both', 
                        choices=['two_stage', 'long', 'both'],
                        help='训练模式: two_stage(两阶段训练), long(长训练), both(都训练)')
    parser.add_argument('--model', type=str, default='yolo11n-cls.pt',
                        help='基础模型名称（保持yolo11n-cls不变）')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='数据集目录')
    
    args = parser.parse_args()
    
    print("="*60)
    print("YOLO11n-cls 模型优化训练")
    print("="*60)
    
    trainer = RiceDiseaseModelTrainerOptimized(args.data_dir)
    
    results_list = []
    
    if args.mode in ['two_stage', 'both']:
        print("\n" + "="*60)
        print("方案1：两阶段微调训练")
        print("="*60)
        try:
            model1, report1, path1 = trainer.run_two_stage(args.model)
            if report1:
                results_list.append(('两阶段训练', report1, path1))
        except Exception as e:
            print(f"两阶段训练失败: {e}")
    
    if args.mode in ['long', 'both']:
        print("\n" + "="*60)
        print("方案2：长训练（100轮）")
        print("="*60)
        try:
            model2, report2, path2 = trainer.run_single_stage_long(args.model, epochs=100)
            if report2:
                results_list.append(('长训练', report2, path2))
        except Exception as e:
            print(f"长训练失败: {e}")
    
    if results_list:
        best = find_best_model(results_list)
    else:
        print("\n所有训练方案均失败！")
    
    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)