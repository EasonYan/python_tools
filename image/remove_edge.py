# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os


# 删除图片边缘黑色像素

def remove_the_blackborder(image, save_path):

    image = cv2.imread(image, 1)  # 读取图片
    img = cv2.medianBlur(image, 5)  # 中值滤波，去除黑色边际中可能含有的噪声干扰
    b = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY)  # 调整裁剪效果
    binary_image = b[1]  # 二值图--具有三通道
    binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
    # print(binary_image.shape)     #改为单通道

    # cut left
    bias = 6
    cut_right = 10
    edges_y, edges_x = np.where(binary_image == 255)  # h, w
    print("x,y", edges_x, edges_y)
    bottom = min(edges_y)
    top = max(edges_y)
    height = top - bottom

    left = min(edges_x)
    right = max(edges_x)
    height = top - bottom
    width = right - left - cut_right

    res_image = image[bottom:bottom+height, left+bias:left+width]
    cv2.imwrite(save_path, res_image)
    # plt.savefig(os.path.join("res_combine.jpg"))
    # plt.show()


def remove_black_edges(image_path, output_path, tolerance=1):
    # 读取图片
    image = cv2.imread(image_path)

    # 将图片转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算灰度图中非黑色区域的外接矩形
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        x, y, w, h = cv2.boundingRect(np.concatenate(contours))

        # 扩展矩形框以保留一些边缘
        x -= tolerance
        y -= tolerance
        w += 2 * tolerance
        h += 2 * tolerance

        # 裁剪图片
        cropped_image = image[y:y+h, x:x+w]
        # 保存处理后的图片
        print("w,h", w, h)
        cv2.imwrite(output_path, cropped_image)
    image = cv2.imread(image_path)

    # 将图片转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用Canny边缘检测
    edges = cv2.Canny(gray, 50, 150)

    # 寻找轮廓
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 寻找最大的轮廓
    max_contour = max(contours, key=cv2.contourArea)

    # 获取边界框
    x, y, w, h = cv2.boundingRect(max_contour)

    # 裁剪并保存去除黑色边缘的图片
    cropped_image = image[y:y+h, x:x+w]
    cv2.imwrite(output_path, cropped_image)

    print("黑色边缘已去除并保存为", output_path)


def list_files_in_directory(directory, dest_dir):
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if "jpg" not in file_path:
                continue
            dest_path = os.path.join(dest_dir, file)
            print(f"文件名: {file}，绝对路径: {file_path},dest:{dest_path}")
            remove_the_blackborder(file_path, dest_path)


dir = "/Users/yicheng.yan/project/resources/txvideo/"
dest_dir = "/Users/yicheng.yan/project/resources/processed/cut_edge"
source_path = "/Users/yicheng.yan/Downloads/s1.jpg"
save_path = "/Users/yicheng.yan/Downloads/s2_out2.jpg"

if __name__ == "__main__":
    # remove_the_blackborder(source_path,save_path)
    list_files_in_directory(dir, dest_dir)
