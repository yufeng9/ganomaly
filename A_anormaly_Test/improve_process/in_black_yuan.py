import cv2
import numpy as np
from PIL import Image


def in_black_image(pic_path, save_path):
    src = cv2.imread(pic_path)
    cv2.namedWindow("input", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("input", src)
    """
    提取图中的红色部分
    """
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    low_hsv = np.array([0, 43, 46])
    high_hsv = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lowerb=low_hsv, upperb=high_hsv)
    cv2.imwrite(save_path, mask)


def in_show(pic, image_path, save_path, label):
    # 打开图片
    Pic = Image.open(pic)
    img1 = Image.open(image_path).convert("RGB")
    # 获取图片的尺寸
    width = Pic.size[0]
    height = Pic.size[1]
    xy = []
    # 遍历整个图片，标注图片中的颜色异常区域
    for i in range(1, width):
        for j in range(1, height):
            # 获取图片所有的像素点（机构为data[0]:R data[1]:G data[2]:B data[3]:A
            data = (Pic.getpixel((i, j)))
            # data_hou = (img1.getpixel((i, j)))
            # 灰色的RGB取169 169 169
            if data[0] > 0 and data[1] > 0 and data[2] > 0:
                # 则颜色异常区域改成纯蓝色
                img1.putpixel((i, j), (label, label, label))
    # 把图片转成RGB
    Pic = img1.convert("RGB")
    # 保存修改后的图片
    # img.save(save_path)
    width = Pic.size[0]
    height = Pic.size[1]
    xy = []
    # 遍历整个图片，标注图片中的颜色异常区域
    for i in range(1, width):
        for j in range(1, height):
            # 获取图片所有的像素点（机构为data[0]:R data[1]:G data[2]:B data[3]:A
            data = (Pic.getpixel((i, j)))
            # data_hou = (img1.getpixel((i, j)))
            # 灰色的RGB取169 169 169
            if data[0] < 100:
                Pic.putpixel((i, j), (4, 61, 150))
            elif data[0] > 100:
                Pic.putpixel((i, j), (104, 18, 22))

    Pic.save(save_path)
# in_black_image('T_11new.png', 'T_11_black.png')
in_show('result_merge_1_.png', 'T_11_black.png', 'result1.png', 255)
