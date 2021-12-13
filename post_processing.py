# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# img_color = cv2.imread("sources/coins.jpg",1)
# img_gray = cv2.cvtColor(img_color,cv2.COLOR_BGR2GRAY)
# img_gray = cv2.Canny(img_gray, 100,180)
#
# img_contours,contours, hierarchy = cv2.findContours(img_gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(img_color,contours,-1,(0,0,255),2)  # 画出轮廓
#
# retval, markers = cv2.connectedComponents(img_contours)
# # cv2.watershed(img_contours, markers)
# print(markers[])
# cv2.imshow("1",markers)
# cv2.waitKey(0)

"""
完成分水岭算法步骤：
1、加载原始图像
2、阈值分割，将图像分割为黑白两个部分
3、对图像进行开运算，即先腐蚀在膨胀
4、对开运算的结果再进行 膨胀，得到大部分是背景的区域
5、通过距离变换 Distance Transform 获取前景区域
6、背景区域sure_bg 和前景区域sure_fg相减，得到即有前景又有背景的重合区域
7、连通区域处理
8、最后使用分水岭算法
"""

import cv2
import numpy as np
import os.path as osp
import os
import matplotlib.pyplot as plt


# Step1. 加载灰度图像
input_folder = "./dataset/val/val_result202-45"
output_folder = "./dataset/val/temp_val_result202-45"

input_list = os.listdir(input_folder)

for i in range(0,len(input_list)):

    img = cv2.imread(osp.join(input_folder,str(i) + '_predict45.jpg'),0) # 图像为灰度图像
    cv2.imwrite(osp.join(output_folder,str(i) + '_predict45.jpg'), img)
    open = 1
    if (open):
        # Step3. 对图像进行“开运算”，先腐蚀再膨胀
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=2)
        cv2.imwrite(osp.join(output_folder,str(i) + '_predict45_open.jpg'), opening)

        # Step4. 对“开运算”的结果进行膨胀
        open_dilate = cv2.dilate(opening, kernel, iterations=3)
        cv2.imwrite(osp.join(output_folder, str(i) + '_predict45_open+dilate.jpg'), open_dilate)
        open_erode = cv2.erode(opening, kernel, iterations=3)
        cv2.imwrite(osp.join(output_folder, str(i) + '_predict45_open+erode.jpg'), open_erode)
    else:
        # Step3. 对图像进行“闭运算”，先膨胀再腐蚀
        kernel = np.ones((3, 3), np.uint8)
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=2)
        cv2.imwrite(osp.join(output_folder, str(i) + '_predict45_close.jpg'), closing)

        # Step4. 对“闭运算”的结果进行膨胀
        close_dilate = cv2.dilate(closing, kernel, iterations=3)
        cv2.imwrite(osp.join(output_folder, str(i) + '_predict45_close+dilate.jpg'), close_dilate)
        close_erode = cv2.erode(closing, kernel, iterations=3)
        cv2.imwrite(osp.join(output_folder, str(i) + '_predict45_close+erode.jpg'), close_erode)


    # # Step5.通过distanceTransform获取前景区域
    # dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)  # DIST_L1 DIST_C只能 对应掩膜为3    DIST_L2 可以为3或者5
    # ret, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)
    #
    # cv2.imshow("sure_fg", sure_fg)
    #
    # # Step6. sure_bg与sure_fg相减,得到既有前景又有背景的重合区域   #此区域和轮廓区域的关系未知
    # sure_fg = np.uint8(sure_fg)
    # unknow = cv2.subtract(sure_bg, sure_fg)
    #
    # # Step7. 连通区域处理
    # ret, markers = cv2.connectedComponents(sure_fg,connectivity=8) #对连通区域进行标号  序号为 0 - N-1
    # markers = markers + 1           #OpenCV 分水岭算法对物体做的标注必须都 大于1 ，背景为标号 为0  因此对所有markers 加1  变成了  1  -  N
    # #去掉属于背景区域的部分（即让其变为0，成为背景）
    # # 此语句的Python语法 类似于if ，“unknow==255” 返回的是图像矩阵的真值表。
    # markers[unknow==255] = 0
    #
    # # Step8.分水岭算法
    # # preimg = img
    # markers = cv2.watershed(img, markers)  #分水岭算法后，所有轮廓的像素点被标注为  -1
    # print(markers)
    #
    #
    # for i in range(80,460):
    #     for j in range(80,460):
    #         if markers[i][j] == -1:
    #             img[i][j] = [0, 0, 255]
    #
    # # img[markers == -1] = [0, 0, 255]   # 标注为-1 的像素点标 红
    #
    # cv2.imshow("dst", img)
    # cv2.imwrite(data_root + "dst.jpg", img)
    # cv2.waitKey(0)
