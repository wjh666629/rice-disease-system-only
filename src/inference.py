import os
import time
from pathlib import Path
from ultralytics import YOLO

class RiceDiseasePredictor:
    def __init__(self, model_path=None):
        self.model = None
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
        
        self.class_idx_to_cn = dict(zip(range(len(self.classes)), self.classes_cn))
        
        if model_path is not None:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        self.model = YOLO(model_path)
        print(f"模型加载成功: {model_path}")
    
    def predict(self, image_path):
        if self.model is None:
            raise RuntimeError("模型尚未加载，请先调用 load_model()")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        
        start_time = time.time()
        
        results = self.model.predict(image_path, verbose=False)
        
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        pred_idx = results[0].probs.top1
        confidence = float(results[0].probs.top1conf)
        probs = results[0].probs.data.tolist()
        
        predicted_label = self.class_idx_to_cn[pred_idx]
        
        probabilities = {}
        for idx, class_name_cn in enumerate(self.classes_cn):
            probabilities[class_name_cn] = float(probs[idx])
        
        result = {
            "predicted_label": predicted_label,
            "confidence": confidence,
            "probabilities": probabilities,
            "latency_ms": latency_ms
        }
        
        return result

_predictor = None

def init_model(model_path):
    global _predictor
    _predictor = RiceDiseasePredictor(model_path)

def predict(image_path):
    if _predictor is None:
        raise RuntimeError("模型尚未初始化，请先调用 init_model()")
    
    return _predictor.predict(image_path)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='水稻病害识别推理')
    parser.add_argument('--model', type=str, required=True, help='模型文件路径')
    parser.add_argument('--image', type=str, required=True, help='测试图像路径')
    
    args = parser.parse_args()
    
    init_model(args.model)
    result = predict(args.image)
    
    print("推理结果:")
    print(f"预测类别: {result['predicted_label']}")
    print(f"置信度: {result['confidence']:.4f}")
    print("各类概率:")
    for cls_name, prob in result['probabilities'].items():
        print(f"  {cls_name}: {prob:.4f}")
    print(f"推理耗时: {result['latency_ms']:.2f} ms")