import os
from PIL import Image

def convert_images(source_dir, target_format='jpeg'):
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                img_path = os.path.join(root, file)
                img = Image.open(img_path)
                # 修改文件扩展名为目标格式
                new_file = os.path.splitext(file)[0] + '.' + target_format
                new_img_path = os.path.join(root, new_file)
                img = img.convert('RGB')  # 确保图像为RGB格式
                img.save(new_img_path, target_format.upper())
                print(f"Converted {img_path} to {new_img_path}")

                # 删除原来的文件
                if file.lower().endswith('png'):
                    os.remove(img_path)

# 转换训练集和验证集的图像格式
convert_images('D:/micosoft downlodes/fire/fire/train/images')
convert_images('D:/micosoft downlodes/fire/fire/val/images')
