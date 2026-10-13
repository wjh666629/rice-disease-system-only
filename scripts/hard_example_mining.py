import os
import shutil
from pathlib import Path
import random
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import cv2


class HardExampleMiner:
    def __init__(self, data_dir, output_dir='data/augmented'):
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
        
        self.target_class = 'bacterial_leaf_blight'
        self.target_class_idx = 0
        
    def augment_image(self, image_path, augment_type='all'):
        img = Image.open(image_path).convert('RGB')
        augmented_images = []
        
        if augment_type in ['all', 'rotate']:
            for angle in [-15, -10, -5, 5, 10, 15]:
                rotated = img.rotate(angle, resample=Image.BICUBIC, expand=True)
                augmented_images.append(('rotate', rotated))
        
        if augment_type in ['all', 'flip']:
            flipped_h = img.transpose(Image.FLIP_LEFT_RIGHT)
            augmented_images.append(('flip_h', flipped_h))
            flipped_v = img.transpose(Image.FLIP_TOP_BOTTOM)
            augmented_images.append(('flip_v', flipped_v))
        
        if augment_type in ['all', 'color']:
            enhancer = ImageEnhance.Brightness(img)
            bright = enhancer.enhance(1.3)
            augmented_images.append(('bright', bright))
            dark = enhancer.enhance(0.7)
            augmented_images.append(('dark', dark))
            
            enhancer = ImageEnhance.Contrast(img)
            high_contrast = enhancer.enhance(1.3)
            augmented_images.append(('high_contrast', high_contrast))
            low_contrast = enhancer.enhance(0.7)
            augmented_images.append(('low_contrast', low_contrast))
            
            enhancer = ImageEnhance.Color(img)
            saturated = enhancer.enhance(1.5)
            augmented_images.append(('saturated', saturated))
            desaturated = enhancer.enhance(0.5)
            augmented_images.append(('desaturated', desaturated))
        
        if augment_type in ['all', 'noise']:
            img_array = np.array(img)
            noise = np.random.normal(0, 15, img_array.shape).astype(np.uint8)
            noisy_img = Image.fromarray(np.clip(img_array + noise, 0, 255).astype(np.uint8))
            augmented_images.append(('noise', noisy_img))
        
        if augment_type in ['all', 'blur']:
            blurred = img.filter(ImageFilter.GaussianBlur(radius=2))
            augmented_images.append(('blur', blurred))
        
        if augment_type in ['all', 'perspective']:
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            h, w = img_cv.shape[:2]
            
            src_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
            dst_pts = np.float32([[10, 0], [w-10, 0], [0, h], [w, h]])
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(img_cv, M, (w, h))
            warped_rgb = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
            augmented_images.append(('perspective', warped_rgb))
        
        if augment_type in ['all', 'zoom']:
            w, h = img.size
            zoomed = img.crop((int(w*0.1), int(h*0.1), int(w*0.9), int(h*0.9)))
            zoomed = zoomed.resize((w, h), Image.BICUBIC)
            augmented_images.append(('zoom_in', zoomed))
        
        return augmented_images
    
    def mine_hard_examples(self, model, test_dir, num_hard=50):
        print("挖掘难例样本...")
        
        from ultralytics import YOLO
        yolo_model = YOLO(model)
        
        hard_examples = []
        
        target_dir = Path(test_dir) / self.target_class
        if target_dir.exists():
            for img_path in target_dir.glob('*.jpg'):
                results = yolo_model.predict(str(img_path), verbose=False)
                probs = results[0].probs.data.tolist()
                pred_idx = results[0].probs.top1
                confidence = probs[pred_idx]
                
                if pred_idx != self.target_class_idx or confidence < 0.9:
                    hard_examples.append({
                        'path': str(img_path),
                        'confidence': confidence,
                        'pred_class': pred_idx,
                        'true_class': self.target_class_idx
                    })
        
        hard_examples.sort(key=lambda x: x['confidence'])
        print(f"找到 {len(hard_examples)} 个难例样本")
        return hard_examples[:num_hard]
    
    def augment_target_class(self, augment_factor=3, augment_type='all'):
        print(f"对 {self.target_class} 进行数据增强（增强倍数: {augment_factor}）...")
        
        for split in ['train', 'val']:
            source_dir = self.data_dir / split / self.target_class
            target_dir = self.output_dir / split / self.target_class
            target_dir.mkdir(parents=True, exist_ok=True)
            
            other_dir = self.output_dir / split
            for cls in self.classes:
                if cls != self.target_class:
                    src = self.data_dir / split / cls
                    dst = other_dir / cls
                    if src.exists():
                        if dst.exists():
                            shutil.rmtree(dst)
                        shutil.copytree(src, dst)
            
            if not source_dir.exists():
                continue
            
            images = list(source_dir.glob('*.jpg'))
            print(f"{split} 集原始 {self.target_class} 数量: {len(images)}")
            
            for img_path in images:
                shutil.copy2(img_path, target_dir / img_path.name)
            
            if augment_factor > 1:
                num_to_augment = int(len(images) * (augment_factor - 1))
                images_to_augment = random.sample(images, min(num_to_augment, len(images)))
                
                aug_count = 0
                for img_path in images_to_augment:
                    augmented = self.augment_image(img_path, augment_type)
                    random.shuffle(augmented)
                    
                    for aug_name, aug_img in augmented[:augment_factor-1]:
                        if aug_count >= num_to_augment:
                            break
                        aug_filename = f"aug_{aug_name}_{img_path.stem}_{aug_count}.jpg"
                        aug_img.save(target_dir / aug_filename)
                        aug_count += 1
                    
                    if aug_count >= num_to_augment:
                        break
                
                print(f"{split} 集增强后 {self.target_class} 数量: {len(images) + aug_count}")
    
    def create_balanced_dataset(self, augment_factor=3):
        print("创建类别均衡数据集...")
        self.augment_target_class(augment_factor)
        print(f"增强数据集已保存至: {self.output_dir}")
        return self.output_dir


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='难例挖掘与数据增强')
    parser.add_argument('--mode', type=str, default='augment', 
                        choices=['augment', 'mine', 'both'],
                        help='模式: augment(数据增强), mine(难例挖掘), both(都执行)')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='原始数据集目录')
    parser.add_argument('--output_dir', type=str, default='data/augmented',
                        help='增强后数据集目录')
    parser.add_argument('--augment_factor', type=int, default=3,
                        help='增强倍数')
    parser.add_argument('--model', type=str, default=None,
                        help='模型路径（用于难例挖掘）')
    
    args = parser.parse_args()
    
    miner = HardExampleMiner(args.data_dir, args.output_dir)
    
    if args.mode in ['augment', 'both']:
        miner.create_balanced_dataset(args.augment_factor)
    
    if args.mode in ['mine', 'both']:
        if args.model is None:
            print("请指定模型路径用于难例挖掘")
        else:
            test_dir = os.path.join(args.data_dir, 'test')
            hard_examples = miner.mine_hard_examples(args.model, test_dir)
            for i, ex in enumerate(hard_examples[:10]):
                print(f"难例 {i+1}: {ex['path']}, 置信度: {ex['confidence']:.4f}")


if __name__ == '__main__':
    main()