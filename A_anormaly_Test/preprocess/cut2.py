from PIL import Image
import cv2
import numpy as np


def cut_2(pic_path, pic_target):
    # 要分割后的尺寸
    cut_width = 1024
    cut_length = 1024
    # 读取要分割的图片，以及其尺寸等数据

    picture = cv2.imread(pic_path)
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

