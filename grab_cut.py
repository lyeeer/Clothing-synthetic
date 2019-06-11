import numpy as np
import cv2
from os import listdir
from PIL import Image


IMAGES_PATH_0 = r'C:\Users\lyeee\PycharmProjects\aiwork00\img\joint_image_0\\'
IMAGES_PATH_1 = r'C:\Users\lyeee\PycharmProjects\aiwork00\img\joint_image_1\\'
WIDTH = 1346
HEIGHT = 1475

class Grab_cut(object):
    suffix = '.png'

    def __init__(self, filename=None):
        self.filename = filename
        self.height = None
        self.width = None

    def image_matting(self, image_file, shape, iteration=20):
        points = shape['points']
        xmin, ymin, xmax, ymax = Grab_cut.convertPoints2BndBox(points)
        self.width = xmax - xmin
        self.height = ymax - ymin

        src_img = cv2.imread(image_file)

        mask = np.zeros(src_img.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        rect = (xmin, ymin, self.width, self.height)

        # Grabcut
        cv2.grabCut(src_img, mask, rect, bgdModel, fgdModel,
                    iteration, cv2.GC_INIT_WITH_RECT)

        r_channel, g_channel, b_channel = cv2.split(src_img)
        a_channel = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')

        # crop image space
        for row in range(ymin, ymax):
            if sum(r_channel[row, xmin:xmax + 1]) > 0:
                out_ymin = row
                break
        for row in range(ymin, ymax)[::-1]:
            if sum(r_channel[row, xmin:xmax + 1]) > 0:
                out_ymax = row + 1
                break
        for col in range(xmin, xmax):
            if sum(a_channel[ymin:ymax + 1, col]) > 0:
                out_xmin = col
                break
        for col in range(xmin, xmax)[::-1]:
            if sum(a_channel[ymin:ymax + 1, col]) > 0:
                out_xmax = col + 1
                break

        # output image
        # cv2.merge()单通道合并为多通道
        img_RGBA = cv2.merge((r_channel[out_ymin:out_ymax, out_xmin:out_xmax],
                              g_channel[out_ymin:out_ymax, out_xmin:out_xmax],
                              b_channel[out_ymin:out_ymax, out_xmin:out_xmax],
                              a_channel[out_ymin:out_ymax, out_xmin:out_xmax]))

        return img_RGBA

    @staticmethod
    def convertPoints2BndBox(points):
        xmin = float('inf')
        ymin = float('inf')
        xmax = float('-inf')
        ymax = float('-inf')
        for p in points:
            x = p[0]
            y = p[1]
            xmin = min(x, xmin)
            ymin = min(y, ymin)
            xmax = max(x, xmax)
            ymax = max(y, ymax)

        # 0-valued coordinates of BB caused an error while
        # training faster-rcnn object detector.
        if xmin < 1:
            xmin = 1

        if ymin < 1:
            ymin = 1

        return (int(xmin), int(ymin), int(xmax), int(ymax))

    @staticmethod
    def resultSave(save_fold, save_path, image_np):
        # cv2.imwrite(r'C:\Users\lyeee\PycharmProjects\aiwork01\joint_image_test\%s' % save_path, image_np)
        cv2.imwrite(save_fold+save_path, image_np)
        print('saved!')


# 以数组形式读取，先把上衣贴上去,重叠部分上面这张图的颜色为255，则使他等于下面这张图的颜色
# 1.使切图的面积更大，上衣和裤子扣出来的width相同，然后拼接的时候直接拼出效果
# 2.合成图之后，把result的红色边缘的像素值打印出来，如果在这个范围内，就替代为白色
# 3.对红色通道进行抠图,打印出各自点红色通道的值
def joint1():
    im_list = [Image.open(IMAGES_PATH_0+fn) for fn in listdir(IMAGES_PATH_0) if fn.endswith('.png')]
    img_cloth = cv2.cvtColor(np.array(im_list[1]), cv2.COLOR_RGB2BGR)
    width = 1346
    height = 1536
    result = Image.new("RGBA", (2768, 4160), (245, 76, 66, 0))
    # 先把上衣贴进去
    result.paste(im_list[1], box=(0, 0))
    result.paste(im_list[0], box=(-30, height-750))
    img_result = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
    img_result_np = result.convert('L')

    for i in range(0, width):
        for j in range(height-100):
            if 80 < img_result_np.getpixel((i, j)) <= 255:
                img_result[j][i] = img_cloth[j][i]
    replace2white(img_result)
    cv2.imwrite('res_joint_final_1.png', img_result)

def joint2():
    im_list = [Image.open(IMAGES_PATH_0+fn) for fn in listdir(IMAGES_PATH_0) if fn.endswith('.png')]
    img_pant = cv2.cvtColor(np.array(im_list[0]), cv2.COLOR_RGB2BGR)
    result = Image.new("RGBA", (2768, 4160), (245, 76, 66, 0))
    # 先把上衣贴进去
    result.paste(im_list[0], box=(0, HEIGHT-700))
    result.paste(im_list[1], box=(50, 0))
    img_result = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

    for i in range(0, WIDTH):
        for j in range(HEIGHT):
            if 120 <= img_result[j, i, 2] <= 255 and 120 <= img_result[j, i, 1] <= 255 and 120 <= img_result[j, i, 0] <= 255:
                img_result[j][i] = img_pant[j-836][i]
    replace2white(img_result)
    cv2.imwrite('res_joint_final_2.png', img_result)

def joint3():
    im_list = [Image.open(IMAGES_PATH_1+fn) for fn in listdir(IMAGES_PATH_1) if fn.endswith('.png')]
    img_pant = cv2.cvtColor(np.array(im_list[0]), cv2.COLOR_RGB2BGR)

    result = Image.new("RGBA", (2768, 4160), (245, 76, 66, 0))
    # 先把上衣贴进去
    result.paste(im_list[0], box=(-150, HEIGHT-750))
    result.paste(im_list[1], box=(0, 0))
    img_result = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

    for i in range(200, WIDTH-301):
        for j in range(100, HEIGHT):
            if 120 <= img_result[j, i, 2] <= 255 and 120 <= img_result[j, i, 1] <= 255 and 120 <= img_result[j, i, 0] <= 255:
                img_result[j][i] = img_pant[j-HEIGHT+750][i+150]

    replace2white(img_result)
    cv2.imwrite('res_joint_final_3.png', img_result)


def joint4():
    im_list = [Image.open(IMAGES_PATH_1+fn) for fn in listdir(IMAGES_PATH_1) if fn.endswith('.png')]
    img_cloth = cv2.cvtColor(np.array(im_list[1]), cv2.COLOR_RGB2BGR)
    result = Image.new("RGBA", (2768, 4160), (245, 76, 66, 0))
    result.paste(im_list[1], box=(0, 0))
    result.paste(im_list[0], box=(-150, HEIGHT-700))
    img_result = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
    for i in range(0, WIDTH):
        for j in range(100, HEIGHT):
            if 252 <= img_result[j, i, 2] <= 255:
                img_result[j][i] = img_cloth[j][i]
    replace2white(img_result)
    cv2.imwrite('res_joint_final_4.png', img_result)

def replace2white(img_result):
    # 对红色通道进行操作, 若红色通道为红色范围且别的通道颜色小（保证不为白色）
    for i in range(2768):
        for j in range(4160):
            if 170 <= img_result[j, i, 2] <= 255 and img_result[j, i, 1] < 150:
                img_result[j][i] = 255

def matFile(image_fold, save_fold):
    image_out_np_pant = mattingFile.image_matting(image_fold+'input2.JPG', shape_pant, iteration=20)
    mattingFile.resultSave(save_fold, 'result_kuzi.png', image_out_np_pant)
    image_out_np_cloth = mattingFile.image_matting(image_fold+'input1.JPG', shape_cloth, iteration=10)
    mattingFile.resultSave(save_fold, 'result_shangyi.png', image_out_np_cloth)


if __name__ == '__main__':
    mattingFile = Grab_cut()
    # 裤子
    shape_pant = dict(line_color=(0, 255, 0, 128),
                      fill_color=(255, 0, 0, 128),
                      points=[(734, 1175.5), (2316, 1175.5), (2316, 4157.0), (734, 4157.0)])
    # 上衣
    shape_cloth = dict(line_color=(0, 255, 0, 128),
                       fill_color=(255, 0, 0, 128),
                       points=[(734, 334), (2316, 334), (2316, 1809), (734, 1809)])
    '''while True:
        choice = input(
            "Please select the group you want to work with:")
        if choice == "1":
            # matFile('./img/1/', IMAGES_PATH_0)
            joint1()

        if choice == "2":
            # matFile('./img/2/', IMAGES_PATH_0)
            joint2()

        if choice == "3":
            matFile('./img/3/', IMAGES_PATH_1)
            joint3()

        if choice == "4":
            matFile('./img/4/', IMAGES_PATH_1)
            joint4()
    '''
    matFile('./img/1/', IMAGES_PATH_0)
    matFile('./img/3/', IMAGES_PATH_1)
    joint1()
    joint2()
    joint3()
    joint4()