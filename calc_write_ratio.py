import cv2  # 读取图像
import os
from tqdm import tqdm

def calc_percent(fname):
    img = cv2.imread(fname)

    # 获取图像高度和宽度
    height, width = img.shape[:2]

    # 计算白色像素数量
    white_pixels = 0
    black_pixels = 0
    for row in img:
        for pixel in row:
            if pixel[0] > 220 and pixel[1] > 200 and pixel[2] > 200:
                white_pixels += 1
            if pixel[0] < 100 and pixel[1] < 100 and pixel[2] < 100:
                black_pixels += 1

    # 计算白色所占百分比
    w_percentage = white_pixels / (height * width)
    b_percentage = black_pixels / (height * width)
    return w_percentage, b_percentage


if __name__ == '__main__':
    image_path = 'K:/GDC_Patch_Result/m/'


    images_path = os.listdir(image_path)

    for temp_image_sub_path in tqdm(images_path):
        doubt_sub_path = image_path + temp_image_sub_path + 'doubt/'
        if not os.path.exists(doubt_sub_path):
            os.mkdir(doubt_sub_path)
        images = os.listdir(image_path + temp_image_sub_path)

        for image in tqdm(images):
            temp_full_path = image_path + temp_image_sub_path + "/" + image
            percent1, percent2 = calc_percent(temp_full_path)
            if percent1 > 0.3 or percent2 > 0.8:
                os.renames(temp_full_path, doubt_sub_path + image)
