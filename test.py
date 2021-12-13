import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
from unet import Unet
from DataHelper import *
from tqdm import tqdm
import numpy as np
import skimage.io as io
# from main import test
import time
import argparse

# 是否使用cuda
device = torch.device("cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 对数据集的操作
x_transforms = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(512),
    transforms.Grayscale(),             # 先将三通道转化为单通道
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
            table.append(0)
        else:
            table.append(1)
    out = img255.point(table,'1')
    return out

def test(check_point,
         input_img_paths,
         output_img_paths):
    model = Unet(1, 1)
    model.load_state_dict(torch.load(check_point, map_location=device))
    test_dataset = TestDataset(input_img_paths, transform=x_transforms,target_transform=y_transforms)
    dataloaders = DataLoader(test_dataset, batch_size=1)
    model.eval()
    # import matplotlib.pyplot as plt
    # plt.ion()

    start = time.time()
    with torch.no_grad():
        for index, x in enumerate(dataloaders):     # x是当前图片的tensor形式
            y = model(x)                            # y是x经过网络的输出，也是tensor形式
            img_y = torch.squeeze(y).numpy()
            img_y = img_y[:, :, np.newaxis]
            img = labelVisualize(2, COLOR_DICT, img_y) if False else img_y[:, :, 0]
            img255 = img.astype(np.uint8)     # 将img的值同时增加某个值，使得灰度扩展到0-255中的某个区间
            img255 = Image.fromarray(img255)
            img_bin_predict = gray2bin(img255, 245)     # 将灰度图通过阈值分割为二值图

            # io.imsave(output_img_paths[index], img)

            img_bin_predict.save(output_img_paths)
            # io.imsave(output_img_paths, img_bin_predict)

            # plt.pause(0.01)
        # plt.show()
    end = time.time()
    print('纯处理图片的时间：%f'% (end - start))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # 创建ArgumentParser()对象
    parser.add_argument('--model', '-m', default='./model/unet_model_400epoch.pt')
    parser.add_argument('--input_img_paths', '-i',
                        default='./dataset/val/image/42.jpg')
    parser.add_argument('--output_img_paths', '-o',
                        default='./dataset/val/val_result400/42_predict.jpg')

    args = parser.parse_args()  # 使用parse_arg()解析添加的参数

    check_point = args.model
    input_img_paths = args.input_img_paths
    output_img_paths = args.output_img_paths


    start = time.time()
    test(check_point, input_img_paths, output_img_paths)
    end = time.time()

    print('程序总运行时间：%f'% (end - start))