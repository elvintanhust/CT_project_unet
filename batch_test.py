import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
from unet import Unet
from DataHelper import *
from tqdm import tqdm
import numpy as np
import skimage.io as io
from skimage import transform
import os
import os.path as osp
import time
import argparse
import matplotlib
from fillHole import FillHole, opening
from calculate_bleeding import calculate_bleeding

# 定位部分的头文件
from location_predict.predict import *

# 是否使用cuda
device = torch.device("cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 对数据集的操作
x_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    # transforms.Grayscale(),             # 先将三通道转化为单通道
    # transforms.ColorJitter(brightness=(0.7,1), contrast=(0.5,1), saturation=0, hue=0),
    transforms.ToTensor(),              # ToTensor()能够把灰度范围从0-255变换到0-1之间
    transforms.Normalize([0.5], [0.5])  # transform.Normalize()则把0-1变换到(-1,1)。三通道则为([ , , ], [ , , ])
])

# 对标签mask的操作，mask只需要转换为tensor
y_transforms = transforms.ToTensor()

# 将Image的灰度图根据阈值转化为二值图像
def gray2bin(img255, threshold):
    table = []
    for i in range(256):
        if i<threshold:
            table.append(1)
        else:
            table.append(0)
    dst = img255.point(table,'1')
    return dst

# 输入ndarray，输出ndarray[0,1]
def gray2binary(img255, threshold):
    dst = (img255 >= threshold)  # 根据阈值进行分割
    dst = dst.astype(np.uint8)
    return dst


def test(check_point,
         input_img_paths,
         output_img_paths,
         need_print = True):
    model = Unet(3, 1)  # 加载模型框架
    model.load_state_dict(torch.load(check_point, map_location=device)) # 加载模型权重
    test_dataset = CLITestDataset(input_img_paths, transform=x_transforms,target_transform=y_transforms)
    dataloaders = DataLoader(test_dataset, batch_size=1)
    model.eval()

    location_list = []

    with torch.no_grad():
        for index, x in enumerate(dataloaders):     # x是当前图片的tensor形式

            # str_index = '%06d' % index
            str_index = str(index)
            start = time.time()
            y = model(x)                            # y是x经过网络的输出，也是tensor形式

            img_y = torch.squeeze(y).numpy()

            img_bin_predict = gray2binary(img_y, 0) * 255
            img_bin_predict = opening(img_bin_predict)    # 开运算（先腐蚀后膨胀），消除小的连通域
            img_bin_predict, locations = FillHole(img_bin_predict)   # 填充孔洞
            location_list.append(locations)

            # io.imsave(output_img_paths[index], img_bin_predict)
            img_bin_predict = Image.fromarray(img_bin_predict).resize((512, 512))
            img_bin_predict.save(output_img_paths[index])   # 按照指定文件保存

            end = time.time()
            if need_print:
                print('第%d/%d张，共处理图片的时间：%f' % (index + 1, dataloaders.__len__(), (end - start)))

    bleeding, vml = calculate_bleeding(output_img_paths, thk=5, pixel_area=0.25, need_print=False, with_sort=False)
    # print(str(vml) + 'ml')

    print('发生脑出血，出血量为： %.3f ml' % vml) if bleeding else print('未发生脑出血')
    return location_list


if __name__ == '__main__':


    parser = argparse.ArgumentParser()  # 创建ArgumentParser()对象
    # parser.add_argument('--model', '-m', default='./model/jitter1_unet_model_108_epoch_sd.pt')
    parser.add_argument('--seg_model', '-m1', default='./model/jitter1forall256size/jitter1_unet_model_96_epoch_sd.pt')
    parser.add_argument('--loc_model', '-m2', default='./location_predict/data/weights/resnext101_32x32d/epoch_190.pth')

    parser.add_argument('--input_img_paths', '-i', nargs='+')
    parser.add_argument('--output_img_paths', '-o', nargs='+')
    # parser.add_argument('--output_img_paths', '-o', default='./dataset/train/train_result_part')

    args = parser.parse_args()  # 使用parse_arg()解析添加的参数

    seg_check_point = args.seg_model
    loc_check_point = args.loc_model
    input_img_paths = args.input_img_paths
    input_img_paths = ['./dataset/val/image/patient1001/000010.jpg',
                       './dataset/val/image/patient1001/000011.jpg',
                       './dataset/val/image/patient1001/000012.jpg']
    # print(input_img_paths)
    output_img_paths = args.output_img_paths
    # output_img_paths = ['./dataset/test/result/(13)jitter.png']
    output_img_paths = ['./dataset/val/temp/010.jpg',
                        './dataset/val/temp/011.jpg',
                        './dataset/val/temp/012.jpg']


    start = time.time()
    location_list = test(seg_check_point, input_img_paths, output_img_paths, need_print = True)
    end = time.time()

    print('分割部分运行时间：%f'% (end - start))


    # 以上是分割部分
    ##################################################################
    # 以下是定位部分

    start = time.time()

    imgs_paths = output_img_paths

    _id, pred_list, pred_scores_list = predict(loc_check_point, imgs_paths)

    # print("原始分类序列：", end=' ')
    # print(pred_list)
    # print("分类分数   ：", end=' ')
    # print(pred_scores_list)

    optimized_seq = seq_optimization(pred_list, pred_scores_list, isprint=False)
    # print("优化序列   ：", end=' ')
    # print(optimized_seq)
    #
    # print("出血坐标序列：", end=' ')
    # print(location_list)

    regions = seq_location(optimized_seq, location_list=location_list, isprint=False)
    print("出血区域   ：", end=' ')
    print(regions)

    end = time.time()
    print('定位部分运行时间：%f'% (end - start))