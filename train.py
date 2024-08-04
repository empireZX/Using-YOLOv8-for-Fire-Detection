import matplotlib.pyplot as plt
from ultralytics import YOLO

def main():
    # 配置文件路径
    data_config = 'D:/micosoft downlodes/fire/dataset.yaml'

    # 加载预训练的YOLOv8s模型
    model = YOLO('yolov8s.pt')

    # 开始训练，启用数据增强，并使用早停法
    results = model.train(
        data=data_config,
        epochs=4,
        batch=16,
        imgsz=640,
        patience=10,  # 早停法的容忍度，若验证集损失在10个epoch内未改善，则停止训练
        hsv_h=0.015,  # 色调变换
        hsv_s=0.7,    # 饱和度变换
        hsv_v=0.4,    # 亮度变换
        degrees=0.0,  # 旋转
        translate=0.1,# 平移
        scale=0.5,    # 缩放
        shear=0.0,    # 剪切
        perspective=0.0, # 透视变换
        flipud=0.0,   # 垂直翻转
        fliplr=0.5,   # 水平翻转
        mosaic=1.0,   # 拼接
        mixup=0.0,    # 混合
        copy_paste=0.0 # 复制粘贴
    )

    # 提取训练和验证损失
    train_metrics = results.metrics
    train_loss = train_metrics.box + train_metrics.cls + train_metrics.dfl
    val_metrics = results.metrics
    val_loss = val_metrics.box + val_metrics.cls + val_metrics.dfl

    # 绘制训练和验证损失图像
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot.png')
    plt.show()

    # 加载训练好的模型
    trained_model_path = 'runs/train/exp/weights/best.pt'
    model = YOLO(trained_model_path)

    # 验证模型
    results = model.val(data=data_config)
    print("Validation Results: ", results)

    # 预测新图像
    new_image_path = 'path/to/new_image.jpg'
    prediction_results = model.predict(source=new_image_path, save=True)
    print("Prediction Results: ", prediction_results)

if __name__ == "__main__":
    main()
