import os
os.environ['YOLO_OFFLINE'] = '1'

from pathlib import Path
from ultralytics import YOLO
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import json

classes = [
    'bacterial_leaf_blight',
    'brown_spot', 
    'healthy',
    'leaf_blast',
    'leaf_scald',
    'sheath_blight'
]
classes_cn = ['细菌性叶枯病', '褐斑病', '健康叶片', '稻瘟病', '叶灼病', '纹枯病']

model_path = 'best_model_final.pt'
val_dir = Path('data/processed/val')

print('加载模型:', model_path)
model = YOLO(model_path)

y_true = []
y_pred = []

print('开始验证集评估...')
for class_idx, class_name in enumerate(classes):
    class_dir = val_dir / class_name
    if not class_dir.exists():
        print(f'  {class_name} 目录不存在，跳过')
        continue
    
    images = list(class_dir.glob('*.jpg'))
    print(f'  处理 {classes_cn[class_idx]} ({class_name}): {len(images)} 张')
    
    for img_path in images:
        results = model.predict(str(img_path), verbose=False)
        pred_idx = results[0].probs.top1
        
        y_true.append(class_idx)
        y_pred.append(pred_idx)

print()
print('=' * 60)
print('验证集评估结果')
print('=' * 60)

cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=classes_cn, output_dict=True)

total_acc = report['accuracy']
macro_f1 = report['macro avg']['f1-score']

print(f'\n总体准确率: {total_acc:.4f} ({total_acc * 100:.2f}%)')
print(f'Macro F1: {macro_f1:.4f}')

print(f'\n各类别准确率:')
header = f'{"类别":<12} {"准确率":>8} {"精确率":>8} {"召回率":>8} {"F1分数":>8} {"样本数":>6}'
print(header)
print('-' * len(header))

for i, cn_name in enumerate(classes_cn):
    precision = report[cn_name]['precision']
    recall = report[cn_name]['recall']
    f1 = report[cn_name]['f1-score']
    support = int(report[cn_name]['support'])
    per_class_acc = cm[i][i] / support if support > 0 else 0
    print(f'{cn_name:<12} {per_class_acc:>8.4f} {precision:>8.4f} {recall:>8.4f} {f1:>8.4f} {support:>6}')

print(f'\n混淆矩阵:')
cm_df = pd.DataFrame(cm, index=classes_cn, columns=classes_cn)
print(cm_df)

result_data = {
    'model': 'best_model_final.pt',
    'dataset': 'val',
    'total_accuracy': total_acc,
    'macro_f1': macro_f1,
    'per_class_accuracy': {},
    'confusion_matrix': cm.tolist()
}
for i, cn_name in enumerate(classes_cn):
    result_data['per_class_accuracy'][cn_name] = {
        'accuracy': float(cm[i][i] / report[cn_name]['support']) if report[cn_name]['support'] > 0 else 0,
        'precision': float(report[cn_name]['precision']),
        'recall': float(report[cn_name]['recall']),
        'f1_score': float(report[cn_name]['f1-score']),
        'support': int(report[cn_name]['support'])
    }

output_path = 'val_evaluation_result.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(result_data, f, ensure_ascii=False, indent=2)
print(f'\n结果已保存至: {output_path}')
