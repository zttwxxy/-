import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读取图片
image_path = '1.jpg'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Robert 算子
roberts_cross_v = np.array([[1, 0], [0, -1]])
roberts_cross_h = np.array([[0, 1], [-1, 0]])
roberts_v = cv2.filter2D(img, -1, roberts_cross_v)
roberts_h = cv2.filter2D(img, -1, roberts_cross_h)
edges_roberts = roberts_v + roberts_h

# Sobel 算子
edges_sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
edges_sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
edges_sobel = cv2.magnitude(edges_sobel_x, edges_sobel_y)

# Prewitt 算子
kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
edges_prewitt_x = cv2.filter2D(img, -1, kernelx)
edges_prewitt_y = cv2.filter2D(img, -1, kernely)
edges_prewitt = edges_prewitt_x + edges_prewitt_y

# 拉普拉斯（Laplacian）算子
edges_laplacian = cv2.Laplacian(img, cv2.CV_64F)

# Canny 算子
edges_canny = cv2.Canny(img, 100, 200)

# 显示和保存分割后的图片
titles = ['Original Image', 'Roberts', 'Sobel', 'Prewitt', 'Laplacian', 'Canny']
images = [img, edges_roberts, edges_sobel, edges_prewitt, edges_laplacian, edges_canny]
save_paths = ['original.jpg', 'roberts.jpg', 'sobel.jpg', 'prewitt.jpg', 'laplacian.jpg', 'canny.jpg']

for i in range(6):
    plt.subplot(2, 3, i + 1), plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
    cv2.imwrite(save_paths[i], images[i])

plt.show()
