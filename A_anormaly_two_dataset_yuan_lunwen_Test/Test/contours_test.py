import cv2
import numpy as np
from PIL import Image


def red_to_black(pic_path, con_path):
    src = cv2.imread(pic_path)
    # cv2.namedWindow("input", cv2.WINDOW_AUTOSIZE)
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    low_hsv = np.array([0, 43, 46])
    high_hsv = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lowerb=low_hsv, upperb=high_hsv)
    cv2.imwrite(con_path, mask)


def color_test(pic_path, con_path, result_path):
    red_to_black(pic_path, con_path)
    img = cv2.imread(con_path)
    h, w = img.shape[:2]  # h :  647 w:  918
    # 获取图像的高和宽 #显示原始图像
    blured = cv2.GaussianBlur(img, (5, 5), 0)
    # cv2.imwrite("blured_yuan.png", blured)
    # 进行滤波去掉噪声
    # 显示低通滤波后的图像
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # 掩码长和宽都比输入图像多两个像素点，满水填充不会超出掩码的非零边缘
    # 进行泛洪填充
    cv2.floodFill(blured, mask, (150, 150), (255, 255, 255), (2, 2, 2), (3, 3, 3), 8)
    cv2.imwrite("blured.png", blured)
    # 得到灰度图
    gray = cv2.cvtColor(blured, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('gray.png', gray)

    # 定义结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    # 开闭运算，先开运算去除背景噪声，再继续闭运算填充目标内的孔洞
    opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite('close.png', closed)
    # 求二值图
    ret, binary = cv2.threshold(closed, 250, 255, cv2.THRESH_BINARY)
    # 找到轮廓
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 绘制轮廓
    c = []
    ares = []
    for i in contours:
        ares.append(cv2.contourArea(i))
        if cv2.contourArea(i) > 2000:
            c.append(i)

    img = cv2.drawContours(img, c, -1, (0, 0, 0), -1)
    cv2.imwrite(result_path, img)
    # cv2.imshow("result_lunkuo", img)
    # 绘制结果
    # cv2.imshow("result", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def black_to_red(pic_path, result_path):
    # 打开图片
    Pic = Image.open(pic_path)

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
            if data[0] < 100:
                Pic.putpixel((i, j), (4, 61, 150))
            elif data[0] > 100:
                Pic.putpixel((i, j), (104, 18, 22))
    Pic.save(result_path)


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
