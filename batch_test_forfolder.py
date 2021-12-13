import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
from unet import Unet
from DataHelper import *
from tqdm import tqdm
import numpy as np
import skimage.io as io
import os
import os.path as osp
import time
import argparse
import matplotlib
from fillHole import FillHole, opening

# 定位部分的头文件
from location_predict.predict import *


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 是否使用cuda
device = torch.device("cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 对数据集的操作
x_transforms = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(512),
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

# 输入ndarray，输出ndarray
def gray2binary(img255, threshold):
    dst = (img255 >= threshold)  # 根据阈值进行分割
    dst = dst.astype(np.uint8)
    return dst


def test(check_point,
         input_folder_path,
         output_img_paths,
         optimized_seq,
         need_print = True):
    model = Unet(3, 1)  # 加载模型框架
    # model.half()
    # model.cuda()



    test_dataset = multiFrameTestDataset(input_folder_path, frame=1, transform=x_transforms,target_transform=y_transforms)
    dataloaders = DataLoader(test_dataset, batch_size=1)


    filenames = os.listdir(input_folder_path)
    filenames.sort()

    location_list = []

    with torch.no_grad():
        for index, x in enumerate(dataloaders):     # x是当前图片的tensor形式

            # if optimized_seq[index] < 10:
            #     model.load_state_dict(torch.load(check_point[0], map_location=device))  # 加载模型权重
            #     print("选择模型0   ", end=' ')
            # elif optimized_seq[index] >= 10 and optimized_seq[index] < 20:
            #     model.load_state_dict(torch.load(check_point[1], map_location=device))  # 加载模型权重
            #     print("选择模型1   ", end=' ')
            # elif optimized_seq[index] >= 20:
            #     model.load_state_dict(torch.load(check_point[2], map_location=device))  # 加载模型权重
            #     print("选择模型2   ", end=' ')
            model.load_state_dict(torch.load(check_point[0], map_location=device))  # 加载模型权重
            model.eval()

            str_index = filenames[index][0:-4]

            start = time.time()

            # x = x.float()
            # x = x.cuda()
            # x = x.half()

            y = model(x)                            # y是x经过网络的输出，也是tensor形式
            # img_y = torch.squeeze(y).cpu().numpy()
            img_y = torch.squeeze(y).numpy()


            img_bin_predict = gray2binary(img_y, 0) * 255   # 二值分割
            img_bin_predict = opening(img_bin_predict)      # 开运算（先腐蚀后膨胀），消除小的连通域
            img_bin_predict, locations = FillHole(img_bin_predict)     # 填充孔洞，并得到连通域的中心坐标
            location_list.append(locations)


            # io.imsave(osp.join(output_img_paths, str_index + '_predict.png'), img_bin_predict)
            img_bin_predict = Image.fromarray(img_bin_predict).resize((512,512))
            img_bin_predict.save(osp.join(output_img_paths, str_index + '_predict.png'))

            end = time.time()
            if need_print:
                print('第%d/%d张，共处理图片的时间：%f' % (index + 1, dataloaders.__len__(), (end - start)))

    return location_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # 创建ArgumentParser()对象
    # parser.add_argument('--model', '-m', default='./model/jitter1forall256size/jitter1_unet_model_96_epoch_sd.pt')
    # parser.add_argument('--model', '-m', default='./model/multiframe1forall/multiframe1_unet_model_100_epoch_sd.pt')
    parser.add_argument('--model', '-m', default='./model/jitter1_unet_model_108_epoch_sd.pt')
    parser.add_argument('--input_img_paths', '-i',
                        default='./dataset/train/image/patient0001')
    parser.add_argument('--output_img_paths', '-o',
                        default='./dataset/train/result/patient0001')

    args = parser.parse_args()  # 使用parse_arg()解析添加的参数

    check_point = [args.model, args.model, args.model]
    input_img_paths = args.input_img_paths
    output_img_paths = args.output_img_paths
    if not osp.exists(input_img_paths):
        print(input_img_paths + ' not exist!!')
    if not osp.exists(output_img_paths):
        os.mkdir(output_img_paths)

###################################################

    trained_model = "/home/ubuntu/zipeng/ct_detect/U-net-master/location_predict/data/weights/resnext101_32x32d/epoch_191.pth"

    # test_path = "/home/ubuntu/zipeng/ct_detect/U-net-master/location_predict/data/test"
    test_path = "./dataset/train/image/patient0001"
    imgs_names = os.listdir(test_path)
    # 先对文件名排序，正式测试的时候不需要重新排序
    imgs_names.sort()
    imgs_paths = []
    for i in range(len(imgs_names)):
        imgs_paths.append(osp.join(test_path, imgs_names[i]))

    # _id, pred_list = tta_predict(trained_model)
    _id, pred_list, pred_scores_list = predict(trained_model, imgs_paths)

    # print("原始分类序列：", end=' ')
    # print(pred_list)
    # print("分类分数   ：", end=' ')
    # print(pred_scores_list)

    optimized_seq = seq_optimization(pred_list, pred_scores_list, isprint=False)
    print("优化序列   ：", end=' ')
    print(optimized_seq)

##############################################################

    start = time.time()
    location_list = test(check_point, input_img_paths, output_img_paths, optimized_seq)
    end = time.time()

    print('location_list:')
    print(location_list)
    print('程序总运行时间：%f'% (end - start))


    # 以上是分割部分
    ##################################################################
    # 以下是定位部分



    #

    regions = seq_location(optimized_seq, location_list=location_list, isprint=False)
    print("出血区域   ：", end=' ')
    print(regions)