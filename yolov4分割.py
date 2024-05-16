import cv2
import numpy as np

# 读取类别名称
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# 读取图像
image_path = 'output_image2.jpg'
image = cv2.imread(image_path)

# 加载YOLOv4模型
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# 准备输入图像
height, width, channels = image.shape
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# 分析检测结果
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.3 and class_id == 2:  # class_id 2 对应于汽车，置信度阈值设置为0.1
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# 使用非极大值抑制来提高检测精度
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.2)  # NMS阈值设置为0.2

# 检查是否有检测到的物体
if len(indices) > 0:
    indices = indices.flatten()

    # 保存检测到的汽车
    car_count = 0
    for i in indices:
        x, y, w, h = boxes[i]
        crop_img = image[y:y+h, x:x+w]
        car_image_path = f"/mnt/data/car_{car_count}.jpg"
        cv2.imwrite(car_image_path, crop_img)
        car_count += 1

    # 显示检测结果
    for i in indices:
        x, y, w, h = boxes[i]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Detected Cars", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No cars detected.")


