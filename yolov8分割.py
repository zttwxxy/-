import cv2
import torch
from ultralytics import YOLO

# 加载预训练的YOLOv8模型
model = YOLO('yolov8s.pt')  # 可以使用'yolov8s.pt'或其他更大的模型来提高精度

# 读取图像
image_path = 'output_image2.jpg'
image = cv2.imread(image_path)
height, width, channels = image.shape

# 使用YOLOv8模型进行检测
results = model(image)

# 解析检测结果
# YOLOv8 返回的结果是一个列表，我们需要从列表中提取边界框信息
boxes = results[0].boxes.xyxy.cpu().numpy()  # 提取检测到的边界框
scores = results[0].boxes.conf.cpu().numpy()  # 提取置信度
class_ids = results[0].boxes.cls.cpu().numpy()  # 提取类别ID

# 保存检测到的汽车
car_count = 0
for i in range(len(boxes)):
    x1, y1, x2, y2 = boxes[i]
    confidence = scores[i]
    class_id = class_ids[i]
    if confidence > 0.05 and int(class_id) == 2:  # 置信度阈值设置为0.1，class_id 2 对应于汽车
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        crop_img = image[y1:y2, x1:x2]
        car_image_path = f"/mnt/data/car_{car_count}.jpg"
        cv2.imwrite(car_image_path, crop_img)
        car_count += 1

        # 绘制边界框和类别标签
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f'Car {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 显示检测结果
cv2.imshow("Detected Cars", image)
cv2.waitKey(0)
cv2.destroyAllWindows()



