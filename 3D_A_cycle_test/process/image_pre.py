import cv2
import numpy as np
import matplotlib.pyplot as plt

# 加载图片，把图像元素的数据类型转换成浮点数，处于255，把元素数值控制在0-1之间
img = cv2.imread('../process/images/z3d_rgb_1.png').astype(np.float32) / 255
# 色彩空间从BGR转换成RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gamma = 0.5
# np.power函数是进行n次方的运算，
# 在这里是图像的mei
corrected_img = np.power(img, gamma)
save_path = '../process/image_light/z3d_rgb_1.png'
cv2.imwrite(save_path, corrected_img)