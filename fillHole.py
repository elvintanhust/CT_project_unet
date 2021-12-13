import cv2
import numpy as np

# 求连通域的中心点坐标
# 通过连通域的几何矩来求中心坐标
# m00表示目标图像的面积，一阶矩反映了目标图像的质心位置
def centroid(contour):
    moment = cv2.moments(contour)
    if moment['m00'] != 0:
        cx = int(moment['m10'] / moment['m00'])
        cy = int(moment['m01'] / moment['m00'])
        return cx, cy
    else:
        return None

def opening(mask):
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    return opening

def FillHole(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)    # （输入图像，轮廓检索模式，轮廓近似方式）
    len_contour = len(contours)     # 轮廓的个数

    if(len_contour == 0):   # 如果没有轮廓（连通域），则返回原图
        return mask, []


    contour_list = []   # 连通域表
    center_list = []    # 连通域中心坐标表
    for i in range(len_contour):
        drawing = np.zeros_like(mask, np.uint8)  # create a black image
        # 得到实心连通域
        img_contour = cv2.drawContours(drawing, contours, i, (255, 255, 255), -1)   # （底层图，轮廓，轮廓编号，填充颜色，轮廓线的宽度（-1为轮廓填充））
        contour_list.append(img_contour)

        center_list.append(centroid(img_contour))

    out = sum(contour_list)
    return out, center_list


if __name__ == '__main__':
    mask_in = cv2.imread('./dataset/test/result/8_predict.png', 0)
    mask_out, center_list = FillHole(mask_in)
    cv2.imwrite('./dataset/test/result/8_fillhole.png', mask_out)
    for center_coordinate in center_list:
        print(center_coordinate)