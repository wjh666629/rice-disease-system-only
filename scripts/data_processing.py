import os
import shutil
import hashlib
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
import pandas as pd

class RiceDiseaseDataProcessor:
    def __init__(self, raw_data_dir, output_dir):
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.train_dir = self.output_dir / 'train'
        self.val_dir = self.output_dir / 'val'
        self.test_dir = self.output_dir / 'test'
        
        # 原始数据目录中的类别名称（英文）
        self.raw_classes = [
            'Bacterial Leaf Blight',
            'Brown Spot',
            'Healthy Rice Leaf',
            'Leaf Blast',
            'Leaf scald',
            'Sheath Blight'
        ]
        
        # 标准化类别名称（英文+中文）
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
        
        # 原始名称到标准名称的映射
        self.raw_to_standard = dict(zip(self.raw_classes, self.classes))
        self.class_mapping = dict(zip(self.classes, self.classes_cn))
        
    def _compute_image_hash(self, image_path):
        hasher = hashlib.md5()
        with open(image_path, 'rb') as f:
            hasher.update(f.read())
        return hasher.hexdigest()
    
    def _is_valid_image(self, image_path):
        try:
            img = Image.open(image_path)
            img.verify()
            img = Image.open(image_path)
            if img.mode not in ('RGB', 'L'):
                return False
            if img.size[0] < 32 or img.size[1] < 32:
                return False
            return True
        except Exception:
            return False
    
    def process_raw_data(self):
        print("开始数据质量检查...")
        
        all_images = []
        invalid_images = []
        duplicate_hashes = {}
        
        for raw_class_name in self.raw_classes:
            class_dir = self.raw_data_dir / raw_class_name
            if not class_dir.exists():
                print(f"警告: 类别目录不存在 {class_dir}")
                continue
            
            standard_class_name = self.raw_to_standard[raw_class_name]
            
            for img_path in tqdm(list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.png')), desc=f"检查 {raw_class_name}"):
                if not self._is_valid_image(img_path):
                    invalid_images.append(str(img_path))
                    continue
                
                img_hash = self._compute_image_hash(img_path)
                if img_hash in duplicate_hashes:
                    duplicate_hashes[img_hash].append(str(img_path))
                else:
                    duplicate_hashes[img_hash] = [str(img_path)]
                
                all_images.append({
                    'path': str(img_path),
                    'class': standard_class_name,
                    'hash': img_hash
                })
        
        duplicates = {k: v for k, v in duplicate_hashes.items() if len(v) > 1}
        print(f"\n发现无效图像: {len(invalid_images)} 张")
        print(f"发现重复图像组: {len(duplicates)} 组")
        print(f"有效图像总数: {len(all_images)} 张")
        
        return all_images, invalid_images, duplicates
    
    def split_dataset(self, all_images, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        print(f"\n开始数据划分: train={train_ratio}, val={val_ratio}, test={test_ratio}")
        
        train_images = []
        val_images = []
        test_images = []
        
        for standard_class_name in self.classes:
            class_images = [img for img in all_images if img['class'] == standard_class_name]
            class_cn = self.class_mapping[standard_class_name]
            print(f"类别 {standard_class_name} ({class_cn}): {len(class_images)} 张")
            
            if len(class_images) == 0:
                continue
            
            train_val, test = train_test_split(class_images, test_size=test_ratio, random_state=42)
            val_size = val_ratio / (train_ratio + val_ratio)
            train, val = train_test_split(train_val, test_size=val_size, random_state=42)
            
            train_images.extend(train)
            val_images.extend(val)
            test_images.extend(test)
        
        print(f"\n划分结果:")
        print(f"训练集: {len(train_images)} 张")
        print(f"验证集: {len(val_images)} 张")
        print(f"测试集: {len(test_images)} 张")
        
        return train_images, val_images, test_images
    
    def save_dataset(self, train_images, val_images, test_images):
        print("\n开始保存数据集...")
        
        for split_dir in [self.train_dir, self.val_dir, self.test_dir]:
            split_dir.mkdir(parents=True, exist_ok=True)
            for class_name in self.classes:
                (split_dir / class_name).mkdir(exist_ok=True)
        
        for images, split_dir in [(train_images, self.train_dir), (val_images, self.val_dir), (test_images, self.test_dir)]:
            for img_info in tqdm(images):
                src_path = Path(img_info['path'])
                dst_path = split_dir / img_info['class'] / src_path.name
                shutil.copy(src_path, dst_path)
        
        print("数据集保存完成")
    
    def generate_statistics(self, all_images, train_images, val_images, test_images):
        stats = {
            'total_images': len(all_images),
            'train_images': len(train_images),
            'val_images': len(val_images),
            'test_images': len(test_images),
            'split_ratio': '8:1:1',
            'classes': [],
            'class_mapping': self.class_mapping
        }
        
        for class_name in self.classes:
            class_total = len([img for img in all_images if img['class'] == class_name])
            class_train = len([img for img in train_images if img['class'] == class_name])
            class_val = len([img for img in val_images if img['class'] == class_name])
            class_test = len([img for img in test_images if img['class'] == class_name])
            
            stats['classes'].append({
                'name': class_name,
                'name_cn': self.class_mapping[class_name],
                'total': class_total,
                'train': class_train,
                'val': class_val,
                'test': class_test
            })
        
        return stats
    
    def generate_data_info(self, stats, invalid_images, duplicates):
        info = {
            'dataset_name': 'Rice Disease Dataset',
            'data_version': 'v1.0',
            'creation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'statistics': stats,
            'invalid_images': invalid_images,
            'duplicate_groups': len(duplicates),
            'split_method': '分层随机划分，确保各类别比例一致',
            'random_seed': 42,
            'notes': '测试集已封存，不参与模型训练和参数调优'
        }
        
        return info
    
    def run(self):
        all_images, invalid_images, duplicates = self.process_raw_data()
        train_images, val_images, test_images = self.split_dataset(all_images)
        self.save_dataset(train_images, val_images, test_images)
        
        stats = self.generate_statistics(all_images, train_images, val_images, test_images)
        data_info = self.generate_data_info(stats, invalid_images, duplicates)
        
        with open(self.output_dir / 'data_info.json', 'w', encoding='utf-8') as f:
            json.dump(data_info, f, ensure_ascii=False, indent=2)
        
        print(f"\n数据处理完成！")
        print(f"数据信息已保存至: {self.output_dir / 'data_info.json'}")
        
        return data_info

if __name__ == '__main__':
    # 原始数据目录（Rice_Leaf_AUG文件夹）
    raw_data_dir = 'Rice_Leaf_AUG'
    # 处理后的数据目录
    output_dir = 'data/processed'
    
    processor = RiceDiseaseDataProcessor(raw_data_dir, output_dir)
    processor.run()