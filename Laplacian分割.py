import cv2
import numpy as np

# 读取图像  Laplacian算子
image_path = 'output_image2.jpg'
image = cv2.imread(image_path)

# 灰度化
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 高斯模糊
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 拉普拉斯算子
edges = cv2.Laplacian(blurred, cv2.CV_16S)
edges = cv2.convertScaleAbs(edges)

# 轮廓检测
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 保存检测到的汽车
car_count = 0
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if w < 20 and h < 15:
        crop_img = image[y:y + h, x:x + w]
        car_image_path = f"/mnt/data/car_{car_count}.jpg"
        cv2.imwrite(car_image_path, crop_img)
        car_count += 1

# 显示检测结果
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if w > 40 and h > 40:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("Detected Cars with Laplacian", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
