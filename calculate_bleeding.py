import skimage.io as io
import os
import os.path as osp
import numpy as np
import cmath


# （文件路径列表，，，）
def calculate_bleeding(file_path, thk = 5, pixel_area = 0.238, need_print=True, with_sort=True):

    V1 = 0
    V2 = 0
    S = [0]
    # file_list = os.listdir(folder_root)
    if with_sort:
        file_path.sort()
    file_num = len(file_path)

    for i in range(file_num):
        # file_path = osp.join(folder_root, file_list[i])
        file_i = io.imread(file_path[i])
        file_i = np.array(file_i)
        file_i_P = np.int64(file_i > 127)

        S.append(sum(sum(file_i_P)) * pixel_area)
    S.append(0)

    for i in range(len(S) - 1):
        # 去掉头部和尾部的圆锥
        if (S[i] != 0 and S[i + 1] != 0):
            V1 += (S[i] + S[i + 1] + cmath.sqrt(S[i] * S[i + 1]).real) * thk / 3

        # 加上头部和尾部的圆锥
        if (S[i] != 0 or S[i + 1] != 0):
            V2 += (S[i] + S[i + 1] + cmath.sqrt(S[i] * S[i + 1]).real) * thk / 3
    if need_print:
        print('V1 = %fcm3' % (V1 / 1000))
        print('V2 = %fcm3' % (V2 / 1000))

    bleeding = True if (V1 / 1000 > 0) else False

    if bleeding:
        return bleeding, V2 / 1000
    else:
        return bleeding, 0



if __name__ == '__main__':
    # 计算单个子文件夹内的出血量
    # subfolder_root = './dataset/train/result/patient0001'
    # file_path = []
    # for file_name in os.listdir(subfolder_root):
    #     file_path.append(osp.join(subfolder_root, file_name))
    #
    # length_of_ruler = 205
    # pixel_area = (100 / length_of_ruler) * (100 / length_of_ruler)
    # calculate_bleeding(file_path, thk=5, pixel_area=pixel_area*4, need_print=True)


    # 计算文件夹内所有子文件夹的出血量
    folder_root = './dataset/test/256resultUnderlight'
    subfoldernames = os.listdir(folder_root)
    subfolder_roots = []
    for subfoldername in subfoldernames:
        subfolder_roots.append(osp.join(folder_root, subfoldername))

    for subfolder_root in subfolder_roots:
        file_path = []
        for file_name in os.listdir(subfolder_root):
            file_path.append(osp.join(subfolder_root, file_name))
        bleeding, vml = calculate_bleeding(file_path, thk=5, pixel_area=1, need_print=False)
        fold, name = osp.split(subfolder_root)  # 将目录和文件名分开
        print(str(vml) + 'ml   ' + name)