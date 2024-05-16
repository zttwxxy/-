import cv2
import torch
from ultralytics import YOLO

# 加载预训练的YOLOv8模型
model = YOLO('yolov8n.pt')  # 可以使用'yolov8s.pt'或其他更大的模型来提高精度

# 视频路径
video_path = '6.mp4'
cap = cv2.VideoCapture(video_path)

# 获取视频帧率
fps = cap.get(cv2.CAP_PROP_FPS)
# 获取视频宽度和高度
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 输出视频设置
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 使用YOLOv8模型进行检测
    results = model(frame)

    # 解析检测结果
    boxes = results[0].boxes.xyxy.cpu().numpy()  # 提取检测到的边界框
    scores = results[0].boxes.conf.cpu().numpy()  # 提取置信度
    class_ids = results[0].boxes.cls.cpu().numpy()  # 提取类别ID

    car_count = 0
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        confidence = scores[i]
        class_id = class_ids[i]
        if confidence > 0.5 and int(class_id) == 2:  # 置信度阈值设置为0.5，class_id 2 对应于汽车
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'Car {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            car_count += 1

    # 在帧上显示车的数量
    cv2.putText(frame, f'Car Count: {car_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # 写入视频文件
    out.write(frame)

    # 显示当前帧
    cv2.imshow('Detected Cars', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()

