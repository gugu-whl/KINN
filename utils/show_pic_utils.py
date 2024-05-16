import cv2
import matplotlib.pyplot as plt
import glob
import os

# 一个窗口窗口显示多张图片：python3 + opencv3的版本。
# 传入的参数是：
# 1. 图片的集合（大小、通道数需要一致，否则黑屏）
# 2. 想显示到一张图片的大小
# 3. 图片间隔大小。

# 如果图片太多，会自动省略多的图片，不够严谨。
def show_in_one(images, title="merge"):
    for idx, image in enumerate(images):
        temp_title = title + str(idx + 1)
        # plt.figure(figsize=(4, 4))
        # 行，列，索引
        plt.subplot((len(images) // 5) + 1, 5, idx + 1)
        plt.imshow(image)
        plt.title(temp_title, fontsize=8)
        plt.xticks([])
        plt.yticks([])
    plt.show()


if __name__ == '__main__':
    test_dir = "C:/Users/pc/Desktop/test"
    path = test_dir

    images = []
    for infile in glob.glob(os.path.join(path, '*.*')):
        ext = os.path.splitext(infile)[1][1:]  # get the filename extenstion
        if ext == "png" or ext == "jpg" or ext == "bmp" or ext == "tiff" or ext == "pbm":
            print(infile)
            img = cv2.imread(infile)
            images.append(img)

    show_in_one(images)