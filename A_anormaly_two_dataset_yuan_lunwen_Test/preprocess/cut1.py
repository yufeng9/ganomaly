from cv2 import cv2
import numpy as np
from PIL import Image


# 将图片填充为正方形
def fill_image(pic_path, cut_size, result_path):
    image = Image.open(pic_path)
    width, height = image.size
    new_width = 0
    new_height = 0

    if width % cut_size != 0:
        new_width = cut_size * (width // cut_size + 1)
        new_image = Image.new(image.mode, (new_width, height), color='black')
        new_image.paste(image, (int((new_width - width) / 2), 0))
    if height % cut_size != 0:
        new_height = cut_size * (height // cut_size + 1)
        new_he_image = Image.new(new_image.mode, (new_width, new_height), color='black')
        new_he_image.paste(new_image, (0, int((new_height - height) / 2)))
        new_he_image.save(result_path)
    return new_image


def cut_1(pic_path, pic_target):
    # 要分割后的尺寸
    cut_width = 1024
    cut_length = 1024
    # 读取要分割的图片，以及其尺寸等数据
    # image = Image.open(pic_path)
    fill_image(pic_path, cut_width, 'T_11new.png')
    picture = cv2.imread('T_11new.png')
    (width, length, depth) = picture.shape
    # 预处理生成0矩阵
    pic = np.zeros((cut_width, cut_length, depth))
    # 计算可以划分的横纵的个数
    num_width = int(width / cut_width)
    num_length = int(length / cut_length)
    # for循环迭代生成
    for i in range(0, num_width):
        for j in range(0, num_length):
            pic = picture[i * cut_width: (i + 1) * cut_width, j * cut_length: (j + 1) * cut_length, :]
            result_path = pic_target + '{}_{}.png'.format(i + 1, j + 1)
            cv2.imwrite(result_path, pic)


