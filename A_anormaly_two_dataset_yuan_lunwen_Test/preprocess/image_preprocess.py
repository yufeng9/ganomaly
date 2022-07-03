import os
import cv2
import numpy as np
from PIL import Image
from preprocess.cut1 import cut_1
from code1.demo_part1 import demo_part1
from preprocess.merge import merge_temp
from preprocess.cut2 import cut_2
from preprocess.contours_test import color_test, black_to_red


def change_black(image_path, image_new_path):
    threshold = 25
    img = cv2.imread(image_path, 0)  # load grayscale version
    # the indeces where the useful region starts and ends
    hStrart = 0
    hEnd = img.shape[0]
    vStart = 0
    vEnd = img.shape[1]

    # get row and column maxes for each row and column
    hMax = img.max(1)
    vMax = img.max(0)

    hDone_flag = False
    vDone_flag = False

    # go through the list of max and begin where the pixel value is greater
    # than the threshold
    for i in range(hMax.size):
        if not hDone_flag:
            if hMax[i] > threshold:
                hStart = i
                hDone_flag = True

        if hDone_flag:
            if hMax[i] < threshold:
                hEnd = i
                break

    for i in range(vMax.size):
        if not vDone_flag:
            if vMax[i] > threshold:
                vStart = i
                vDone_flag = True

        if vDone_flag:
            if vMax[i] < threshold:
                vEnd = i
                break

    # load the color image and choose only the useful area from it
    img2 = (cv2.imread(image_path))[hStart:hEnd, vStart:vEnd, :]

    # write the cropped image
    cv2.imwrite(image_new_path, img2)


def in_black_image(pic_path, save_path):
    src = cv2.imread(pic_path)
    # cv2.namedWindow("input", cv2.WINDOW_AUTOSIZE)
    # cv2.imshow("input", src)
    """
    提取图中的红色部分
    """
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    low_hsv = np.array([156, 43, 46])
    high_hsv = np.array([180, 255, 255])
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
    Pic.save(image_path)
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


def image_cut_white(pic_path, pic_path_yuan):
    img = Image.open(pic_path)  # 读取图片
    image_yuan = Image.open(pic_path_yuan)
    width, height = image_yuan.size
    new_width, new_height = img.size
    a = int((new_width - width) / 2)
    b = int((new_height - height) / 2)
    box = (a, b, a + width, b + height)  # 设定要剪切的位置
    img = img.crop(box)  # 剪切图片
    img.save("T_11_black_final.png")  # 存储图片


if __name__ == '__main__':
    pic_path = '../temp_data/'

    change_black('T11.png', 'T11_remove_black.png')
    color_test('T11_remove_black.png', 'red.png', 'result_contours.png')
    black_to_red('result_contours.png', 'red_contours.png')

    cut_1('red_contours.png', '../raw_data/round_test/part1/cut/')
    demo_part1()
    merge_temp(pic_path, '1')
    # yuan_image-----> black
    in_black_image('T_11new.png', 'T_11_black.png')
    in_show('result_merge_1_.png', 'T_11_black.png', 'result1.png', 255)

    # epoch 2 :
    cut_2('result1.png', '../raw_data/round_test/part1/cut/')
    demo_part1()
    merge_temp(pic_path, '2')
    in_show('result_merge_2_.png', 'T_11_black.png', 'result2.png', 255)

    # epoch 3 :
    cut_2('result2.png', '../raw_data/round_test/part1/cut/')
    demo_part1()
    merge_temp(pic_path, '3')
    in_show('result_merge_3_.png', 'T_11_black.png', 'result3.png', 0)

    # epoch 4 :
    cut_2('result3.png', '../raw_data/round_test/part1/cut/')
    demo_part1()
    merge_temp(pic_path, '4')
    in_show('result_merge_4_.png', 'T_11_black.png', 'result4.png', 255)

    # epoch 5 :
    cut_2('result4.png', '../raw_data/round_test/part1/cut/')
    demo_part1()
    merge_temp(pic_path, '5')
    in_show('result_merge_5_.png', 'T_11_black.png', 'result5.png', 255)

    image_cut_white('T_11_black.png', 'T11_remove_black.png')
    image_cut_white('result5.png', 'T11_remove_black.png')
    #
    #
    #
    #
    #
    #
    #
