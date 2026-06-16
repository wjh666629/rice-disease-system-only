import os
import json
import time
from pathlib import Path
from ultralytics import YOLO
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

class RiceDiseaseModelTrainer:
    def __init__(self, data_dir, output_dir='runs/classify'):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
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
        
    def train(self, epochs=50, batch_size=32, img_size=224, patience=10):
        print("开始模型训练...")
        
        model = YOLO('yolo11n-cls.pt')
        
        results = model.train(
            data=str(self.data_dir),
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            patience=patience,
            seed=42,
            verbose=True,
            project=str(self.output_dir),
            name='rice_disease_exp',
            pretrained=True,
            optimizer='AdamW',
            lr0=0.001,
            augment=True,
            val=True
        )
        
        print("训练完成！")
        return model, results
    
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
                except Exception as e:
                    print(f"处理图像 {img_path} 时出错: {e}")
        
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(
            y_true, y_pred,
            target_names=self.classes_cn,
            output_dict=True
        )
        
        return cm, report, y_true, y_pred, all_probs
    
    def save_evaluation_report(self, metrics, cm, report, y_true, y_pred, all_probs):
        report_dir = self.output_dir / 'rice_disease_exp'
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = report_dir / 'evaluation_report.json'
        
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
            'macro_recall': report['macro avg']['recall']
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(eval_report, f, ensure_ascii=False, indent=2)
        
        print(f"\n评估报告已保存至: {report_path}")
        
        return eval_report
    
    def save_confusion_matrix_csv(self, cm):
        cm_df = pd.DataFrame(cm, index=self.classes_cn, columns=self.classes_cn)
        cm_path = self.output_dir / 'rice_disease_exp' / 'confusion_matrix.csv'
        cm_df.to_csv(cm_path, encoding='utf-8-sig')
        print(f"混淆矩阵已保存至: {cm_path}")
    
    def run(self):
        model, train_results = self.train()
        
        metrics, val_results = self.evaluate(model)
        
        test_dir = self.data_dir / 'test'
        cm, report, y_true, y_pred, all_probs = self.compute_confusion_matrix(model, test_dir)
        
        eval_report = self.save_evaluation_report(metrics, cm, report, y_true, y_pred, all_probs)
        self.save_confusion_matrix_csv(cm)
        
        best_model_path = self.output_dir / 'rice_disease_exp' / 'weights' / 'best.pt'
        print(f"\n最佳模型已保存至: {best_model_path}")
        
        return model, eval_report, best_model_path

if __name__ == '__main__':
    # 训练数据目录（已处理的数据集）
    data_dir = 'data/processed'
    
    trainer = RiceDiseaseModelTrainer(data_dir)
    model, eval_report, best_model_path = trainer.run()
    
    print("\n训练完成！评估报告摘要:")
    print(f"Top-1 Accuracy: {eval_report['metrics']['accuracy']:.4f}")
    print(f"Macro F1-score: {eval_report['macro_f1']:.4f}")
    print(f"模型路径: {best_model_path}")