import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.model_training import RiceDiseaseModelTrainer
from ultralytics import YOLO

if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    trainer = RiceDiseaseModelTrainer(
        os.path.join(project_root, 'data/processed'), 
        os.path.join(project_root, 'runs/classify')
    )
    
    model_path = os.path.join(project_root, 'runs/classify/runs/classify/rice_disease_exp/weights/best.pt')
    model = YOLO(model_path)
    
    print("开始评估...")
    metrics, val_results = trainer.evaluate(model)
    
    test_dir = trainer.data_dir / 'test'
    cm, report, y_true, y_pred, all_probs = trainer.compute_confusion_matrix(model, test_dir)
    
    eval_report = trainer.save_evaluation_report(metrics, cm, report, y_true, y_pred, all_probs)
    trainer.save_confusion_matrix_csv(cm)
    
    print('评估报告已生成！')
    print(f"Top-1 Accuracy: {eval_report['metrics']['accuracy']:.4f}")
    print(f"Macro F1-score: {eval_report['macro_f1']:.4f}")