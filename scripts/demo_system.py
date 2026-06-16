import os
import sys
import random
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import init_model, predict

class RiceDiseaseDemoSystem:
    def __init__(self, project_root=None):
        if project_root is None:
            self.project_root = Path(__file__).parent.parent
        else:
            self.project_root = Path(project_root)
        
        self.model_path = self.project_root / 'runs/classify/runs/classify/rice_disease_exp/weights/best.pt'
        self.test_dir = self.project_root / 'data/processed/test'
        self.output_dir = self.project_root / 'evaluation_results'
        self.output_dir.mkdir(exist_ok=True)
        
        # 测试集目录使用英文名称
        self.classes_en = [
            'bacterial_leaf_blight', 'brown_spot', 'healthy',
            'leaf_blast', 'leaf_scald', 'sheath_blight'
        ]
        
        # 中文名称映射
        self.classes_cn = [
            '细菌性叶枯病', '褐斑病', '健康叶片',
            '稻瘟病', '叶灼病', '纹枯病'
        ]
        
        self.en_to_cn = dict(zip(self.classes_en, self.classes_cn))
        
        self.test_samples = []
        self.predictions = []
        
    def _get_test_samples(self):
        samples = []
        
        for class_dir in self.test_dir.iterdir():
            if class_dir.is_dir() and class_dir.name in self.classes_en:
                class_name_en = class_dir.name
                class_name_cn = self.en_to_cn[class_name_en]
                for img_path in class_dir.glob('*.jpg'):
                    samples.append({
                        'path': str(img_path),
                        'true_label': class_name_cn,  # 使用中文标签
                        'true_label_en': class_name_en,  # 保存英文标签
                        'filename': img_path.name
                    })
        
        return samples
    
    def random_sampling(self, n_samples=20, seed=42):
        random.seed(seed)
        
        self.test_samples = self._get_test_samples()
        
        if len(self.test_samples) < n_samples:
            print(f"警告：测试集只有 {len(self.test_samples)} 张图像，将全部使用")
            return self.test_samples
        
        sampled = random.sample(self.test_samples, n_samples)
        
        print(f"\n随机抽样完成！")
        print(f"抽样数量: {len(sampled)} 张")
        print(f"随机种子: {seed}")
        
        return sampled
    
    def stratified_sampling(self, n_samples=20, seed=42):
        random.seed(seed)
        
        self.test_samples = self._get_test_samples()
        
        stratified_samples = []
        samples_by_class = {}
        
        for sample in self.test_samples:
            label = sample['true_label']
            if label not in samples_by_class:
                samples_by_class[label] = []
            samples_by_class[label].append(sample)
        
        n_classes = len(samples_by_class)
        samples_per_class = n_samples // n_classes
        
        for label, samples in samples_by_class.items():
            if len(samples) <= samples_per_class:
                stratified_samples.extend(samples)
            else:
                stratified_samples.extend(random.sample(samples, samples_per_class))
        
        remaining = n_samples - len(stratified_samples)
        if remaining > 0:
            extra_samples = [s for s in stratified_samples if s not in stratified_samples[:remaining]]
            stratified_samples.extend(random.sample(extra_samples, remaining))
        
        random.shuffle(stratified_samples)
        
        print(f"\n分层随机抽样完成！")
        print(f"抽样数量: {len(stratified_samples)} 张")
        print(f"每类样本数: {samples_per_class} 张")
        
        return stratified_samples
    
    def batch_predict(self, samples=None, save_results=True):
        if samples is None:
            samples = self.test_samples
        
        if not samples:
            raise ValueError("没有样本可预测，请先执行随机抽样")
        
        print(f"\n开始批量预测，共 {len(samples)} 张图像...")
        
        self.predictions = []
        
        for i, sample in enumerate(samples):
            try:
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
                
                status = "✓" if prediction['correct'] else "✗"
                print(f"  [{status}] {prediction['filename']}: "
                      f"真实={prediction['true_label']}, "
                      f"预测={prediction['predicted_label']} "
                      f"(置信度: {prediction['confidence']:.2%})")
                
            except Exception as e:
                print(f"  [错误] {sample['filename']}: {str(e)}")
        
        print(f"\n批量预测完成！")
        print(f"总样本数: {len(self.predictions)}")
        print(f"正确预测: {sum(1 for p in self.predictions if p['correct'])}")
        print(f"准确率: {sum(1 for p in self.predictions if p['correct']) / len(self.predictions):.2%}")
        
        if save_results:
            self._save_results()
        
        return self.predictions
    
    def _save_results(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        results_df = pd.DataFrame([{
            '序号': p['index'],
            '文件名': p['filename'],
            '真实标签': p['true_label'],
            '预测标签': p['predicted_label'],
            '置信度': f"{p['confidence']:.4f}",
            '预测正确': '是' if p['correct'] else '否',
            '推理耗时(ms)': f"{p['latency_ms']:.2f}"
        } for p in self.predictions])
        
        csv_path = self.output_dir / f'prediction_results_{timestamp}.csv'
        results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"预测结果已保存至: {csv_path}")
        
        summary = {
            'evaluation_time': timestamp,
            'total_samples': len(self.predictions),
            'correct_predictions': sum(1 for p in self.predictions if p['correct']),
            'accuracy': sum(1 for p in self.predictions if p['correct']) / len(self.predictions),
            'average_latency_ms': sum(p['latency_ms'] for p in self.predictions) / len(self.predictions),
            'class_metrics': {}
        }
        
        for class_name in self.classes_cn:
            class_preds = [p for p in self.predictions if p['true_label'] == class_name]
            if class_preds:
                summary['class_metrics'][class_name] = {
                    'count': len(class_preds),
                    'correct': sum(1 for p in class_preds if p['correct']),
                    'accuracy': sum(1 for p in class_preds if p['correct']) / len(class_preds)
                }
        
        json_path = self.output_dir / f'evaluation_summary_{timestamp}.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"评测摘要已保存至: {json_path}")
        
        return csv_path, json_path
    
    def print_summary(self):
        if not self.predictions:
            print("没有预测结果可显示")
            return
        
        total = len(self.predictions)
        correct = sum(1 for p in self.predictions if p['correct'])
        accuracy = correct / total
        avg_latency = sum(p['latency_ms'] for p in self.predictions) / total
        
        print("\n" + "="*60)
        print("评测结果摘要")
        print("="*60)
        print(f"总样本数: {total}")
        print(f"正确预测: {correct}")
        print(f"准确率: {accuracy:.2%}")
        print(f"平均推理耗时: {avg_latency:.2f} ms")
        print("\n各类别表现:")
        
        for class_name in self.classes_cn:
            class_preds = [p for p in self.predictions if p['true_label'] == class_name]
            if class_preds:
                class_correct = sum(1 for p in class_preds if p['correct'])
                class_acc = class_correct / len(class_preds)
                print(f"  {class_name}: {class_correct}/{len(class_preds)} ({class_acc:.2%})")
        
        print("="*60)
        
        if accuracy >= 0.90:
            print("✓ 验收标准达成：准确率 ≥ 90%")
        if avg_latency <= 1000:
            print("✓ 推理速度达标：单张 ≤ 1000 ms")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='水稻病害识别演示系统')
    parser.add_argument('--samples', type=int, default=20, help='随机抽样数量（默认20）')
    parser.add_argument('--stratified', action='store_true', help='使用分层随机抽样')
    parser.add_argument('--seed', type=int, default=42, help='随机种子（默认42）')
    parser.add_argument('--model', type=str, default=None, help='模型路径（可选）')
    
    args = parser.parse_args()
    
    demo = RiceDiseaseDemoSystem()
    
    if args.model:
        demo.model_path = Path(args.model)
    
    print("="*60)
    print("水稻病害智能识别系统 - 演示模式")
    print("="*60)
    print(f"模型路径: {demo.model_path}")
    print(f"测试集路径: {demo.test_dir}")
    
    init_model(str(demo.model_path))
    
    if args.stratified:
        samples = demo.stratified_sampling(n_samples=args.samples, seed=args.seed)
    else:
        samples = demo.random_sampling(n_samples=args.samples, seed=args.seed)
    
    demo.batch_predict(samples)
    demo.print_summary()

if __name__ == '__main__':
    main()
