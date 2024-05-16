import cv2
import numpy as np
import math
from matplotlib import pyplot as plt


def denoise_image(image):
    # 使用高斯模糊去噪
    denoised = cv2.GaussianBlur(image, (5, 5), 0)
    return denoised


def dehaze_image(image):
    # 去雾算法 (Dark Channel Prior)
    def get_dark_channel(image, size=15):
        b, g, r = cv2.split(image)
        min_img = cv2.min(cv2.min(r, g), b)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
        dark_channel = cv2.erode(min_img, kernel)
        return dark_channel

    def get_atmosphere(image, dark_channel):
        image = image.astype(np.float64)
        dark_channel = dark_channel.astype(np.float64)
        [h, w] = image.shape[:2]
        num_pixel = h * w
        num_brightest_pixel = int(max(math.floor(num_pixel / 1000), 1))
        dark_vec = dark_channel.reshape(num_pixel)
        image_vec = image.reshape(num_pixel, 3)

        indices = np.argsort(dark_vec)
        indices = indices[num_pixel - num_brightest_pixel::]

        brightest_sum = np.zeros([1, 3])
        for idx in indices:
            brightest_sum += image_vec[idx]

        atmosphere = brightest_sum / num_brightest_pixel
        return atmosphere

    def get_transmission(image, atmosphere, omega=0.95, size=15):
        image = image.astype(np.float64)
        transmission = 1 - omega * get_dark_channel(image / atmosphere, size)
        return transmission

    def guided_filter(image, p, r, eps):
        image = image.astype(np.float64)
        p = p.astype(np.float64)
        mean_I = cv2.boxFilter(image, cv2.CV_64F, (r, r))
        mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
        mean_Ip = cv2.boxFilter(image * p, cv2.CV_64F, (r, r))
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = cv2.boxFilter(image * image, cv2.CV_64F, (r, r))
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
        mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

        q = mean_a * image + mean_b
        return q

    def recover_image(image, atmosphere, transmission, t0=0.1):
        image = image.astype(np.float64)
        transmission = np.maximum(transmission, t0)

        recovered = np.empty(image.shape, image.dtype)
        for i in range(3):
            recovered[:, :, i] = (image[:, :, i] - atmosphere[0, i]) / transmission + atmosphere[0, i]

        return recovered

    dark_channel = get_dark_channel(image)
    atmosphere = get_atmosphere(image, dark_channel)
    transmission = get_transmission(image, atmosphere)

    refined_transmission = guided_filter(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255, transmission, 60, 1e-3)
    dehazed = recover_image(image, atmosphere, refined_transmission)

    # 将去雾后的图像转换为 uint8 类型
    dehazed = cv2.normalize(dehazed, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return dehazed


# 导入图像
input_path = '1.jpg'  # 输入图像路径
output_path = 'output_image2.jpg'  # 输出图像保存路径
image = cv2.imread(input_path)

# 去噪处理
denoised_image = denoise_image(image)

# 去雾处理
dehazed_image = dehaze_image(denoised_image)

# 显示处理结果
plt.figure(figsize=(10, 10))
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.subplot(1, 3, 2)
plt.title('Denoised Image')
plt.imshow(cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB))

plt.subplot(1, 3, 3)
plt.title('Dehazed Image')
plt.imshow(cv2.cvtColor(dehazed_image, cv2.COLOR_BGR2RGB))
plt.show()

# 保存处理后的图像
cv2.imwrite(output_path, dehazed_image)

