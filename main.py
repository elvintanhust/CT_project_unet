import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
from unet import Unet
from DataHelper import *
from tqdm import tqdm
import numpy as np
import skimage.io as io
# import PIL.Image
import os
import os.path as osp
# import cv2
from batch_test import gray2bin,test
from eval import evaluate_for_batch
import imgaug as ia
from imgaug import augmenters as iaa


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# PATH = './model/unet_model_204epoch.pt'  # check_point读取和加载路径

# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu_device = torch.device("cpu")
batch_size = 2
frame_num = 1
num_epochs = 61

# 对数据集的操作
light_change = 50
seq = iaa.Sequential([
    # iaa.Grayscale((0.2, 0.3)),
    iaa.Add((0, light_change)),
    iaa.Multiply((0.8, 0.85)),

    # 椒盐噪声
    iaa.Sometimes(0.5, iaa.OneOf([
        iaa.Alpha((0.2, 0.6), iaa.SaltAndPepper((0.01, 0.03)))
    ])),

    # 对比度调整
    iaa.Sometimes(0.5, iaa.OneOf([
        iaa.ContrastNormalization((0.5, 0.9)),
    ])),

    iaa.Sometimes(0.5, iaa.OneOf([
        iaa.AdditiveGaussianNoise(0, (3, 6)),
        iaa.AdditivePoissonNoise((3, 6)),
        iaa.JpegCompression((30, 60)),
        iaa.GaussianBlur(sigma=1),
        iaa.AverageBlur((1, 3)),
        iaa.MedianBlur((1, 3)),
    ])),
])

x_transforms = transforms.Compose([
    transforms.Resize(256),
    # transforms.Grayscale(),             # 先将三通道转化为单通道
    # transforms.ColorJitter(brightness=(0.7,0.8), contrast=(0.6,0.7), saturation=0, hue=0),
    # transforms.RandomVerticalFlip(p=0.5),   # 依概率随机垂直翻转
    # transforms.RandomRotation(degrees, resample=False, expand=False, center=None),

    transforms.ToTensor(),              # ToTensor()能够把灰度范围从0-255变换到0-1之间
    transforms.Normalize([0.5], [0.5])  # transform.Normalize()则把0-1变换到(-1,1)。三通道则为([ , , ], [ , , ])
])

# 对标签mask的操作，mask只需要转换为tensor
y_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor()
])

# 在训练过程中评估，计算f1和IOU
def evaluate_for_train(labels,outputs):
    outputs_temp = outputs  # 暂存输出
    labels_temp = labels  # 暂存mask
    # 将模型输出的tensor转化为二值图ndarray
    img_y = torch.squeeze(outputs_temp)
    img_y = img_y.cpu()
    img_y = img_y.detach().numpy()
    img_y = img_y[0]  # 将列表的维度(batch_size,512,512)变为(512,512)，即只选用第一层的图
    img_y = img_y[:, :, np.newaxis]
    img = img_y[:, :, 0]
    img255 = img.astype(np.uint8)  # 将img的值同时增加某个值，使得灰度扩展到0-255中的某个区间
    img255 = Image.fromarray(img255)
    img_bin_predict = gray2bin(img255, 45)  # 将灰度图通过阈值分割为二值图，利用Image库
    # 将mask的tensor转化为二值图ndarray
    labels_array = torch.squeeze(labels_temp)
    labels_array = labels_array.cpu().numpy()
    labels_array = labels_array[0]  # 将列表的维度(batch_size,512,512)变为(512,512)，即只选用第一层的图

    gt = np.array(labels_array)
    gt = np.int64(gt > 0)
    predict = np.array(img_bin_predict)
    predict = np.int64(predict > 0)
    intersection = gt * predict
    union = gt | predict

    sum_gt = sum(sum(gt))
    sum_predict = sum(sum(predict))
    sum_inter = sum(sum(intersection))
    sum_union = sum(sum(union))

    precision_rate = (sum_inter + 1) / (sum_predict + 1)
    recall_rate = (sum_inter + 1) / (sum_gt + 1)
    # dice_coefficient = (2 * sum_inter + 1) / (sum_predict + sum_gt + 1)
    IOU = (sum_inter + 1) / (sum_union + 1)
    F1_Score = 2 * precision_rate * recall_rate / (precision_rate + recall_rate)

    return F1_Score,IOU


# （模型网络，损失函数，优化器，数据加载器，轮）
def train_model(model, criterion, optimizer, dataload, num_epochs):
    best_model = model  # 最好的模型网络，best_model.state_dict()是模型权重
    min_loss = 1000
    best_epoch = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        dt_size = len(dataload.dataset)
        dt_size_batch = dt_size/batch_size
        epoch_loss = 0
        step = 0
        sum_F1 = 0
        sum_IOU = 0

        for x, y in tqdm(dataload):

            img_y = torch.squeeze(x).numpy()[1]
            # io.imsave(osp.join("/home/ubuntu/zipeng/ct_detect/U-net-master/dataset/test/result_temp", "_predict.jpg"),
            #           img_y)

            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)


            ################训练过程中，每一轮，对于训练集的评估
            # F1_Score,IOU = evaluate_for_train(labels,outputs)
            # sum_F1 = sum_F1 + F1_Score
            # sum_IOU = sum_IOU + IOU
            ################

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            # print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item()))

        print("epoch %d   loss:%0.3f" % (epoch, epoch_loss/step), end='   ')

        print("F1: %f   IOU: %f\n" % (sum_F1/dt_size_batch, sum_IOU/dt_size_batch))
        with open("./logging/log.txt", "a") as f:
            f.write("epoch: %d   loss: %0.3f   " % (epoch, epoch_loss/step))  # 自带文件关闭功能，不需要再写f.close()
            f.write("F1: %f   IOU: %f\n" % (sum_F1/dt_size_batch, sum_IOU/dt_size_batch))

        # 保存最小loss的模型
        if (epoch_loss/step) < min_loss:
            min_loss = (epoch_loss/step)
            best_model = model
            best_epoch = epoch
            print('best epoch is %d' % best_epoch)

        # 按照指定epoch保存模型权重
        if epoch != 0 and epoch % 10 == 0:
            torch.save(model.state_dict(),
                       './model/3layer_model/2layer/256jitter%d_unet_model_%d_epoch_sd.pt' % (frame_num, epoch+238))
            print("保存模型%depoch" % epoch)

        # 在指定epoch上利用验证集和部分训练集来评估模型
        # if epoch % 20 == 0:
        #     val_check_point = './model/val_model/val_model_sd.pt'
        #     torch.save(model.state_dict(), val_check_point)
        #     # 验证集上
        #     print('验证集上评估...')
        #     val_input_img_paths = './dataset/val/image'
        #     val_output_img_paths = './dataset/val/val_result'
        #     test(val_check_point, val_input_img_paths, val_output_img_paths, need_print=False)
        #     val_gt_folder = './dataset/val/label'
        #     val_F1, val_IOU = evaluate_for_batch(val_gt_folder, val_output_img_paths, need_print=False)
        #     with open("./logging/log_val.txt", "a") as f:
        #         f.write("epoch: %d   " % epoch)  # 自带文件关闭功能，不需要再写f.close()
        #         f.write("F1: %f   IOU: %f\n" % (val_F1, val_IOU))
        #     # 部分训练集上
        #     print('部分训练集上评估...')
        #     train_input_img_paths = './dataset/train/image_part'
        #     train_output_img_paths = './dataset/train/train_result_part'
        #     test(val_check_point, train_input_img_paths, train_output_img_paths, need_print=False)
        #     train_gt_folder = './dataset/train/label_part'
        #     train_F1, train_IOU = evaluate_for_batch(train_gt_folder, train_output_img_paths, need_print=False)
        #     with open("./logging/log_train.txt", "a") as f:
        #         f.write("epoch: %d   " % epoch)  # 自带文件关闭功能，不需要再写f.close()
        #         f.write("F1: %f   IOU: %f\n" % (train_F1, train_IOU))


    # torch.save(best_model.state_dict(), PATH)
    # torch.save(best_model.state_dict(), './model/jitter1forall256size/best_jitter%d_unet_model_%d_epoch_sd.pt' % (frame_num, best_epoch+108))
    torch.save(best_model.state_dict(),
               './model/3layer_model/2layer/best_256jitter%d_unet_model_%d_epoch_sd.pt' % (frame_num, best_epoch + 238))

    # torch.save(best_model, './model/unet_model_%d_epoch.pt' % num_epochs)     # 权重和模型一同保存

    return best_model


# 训练模型
def train():
    model = Unet(3, 1).to(device)

    # 加载预训练模型权重
    # model.load_state_dict(torch.load('./model/jitter1forall256size/jitter1_unet_model_108_epoch_sd.pt', map_location=device))
    model.load_state_dict(
        torch.load('./model/3layer_model/2layer/256jitter1_unet_model_238_epoch_sd.pt', map_location=device))

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())
    train_dataset = TrainDataset("dataset/train/image", "dataset/train/label", frame=frame_num,
                                 seq = seq, transform=x_transforms, target_transform=y_transforms)
    dataloaders = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    train_model(model, criterion, optimizer, dataloaders, num_epochs)


# 保存模型的输出结果
# def test():
#     model = Unet(1, 1)
#     model.load_state_dict(torch.load(PATH))
#     test_dataset = TestDataset(["dataset/train/train_img/0.jpg"], transform=x_transforms,target_transform=y_transforms)
#     dataloaders = DataLoader(test_dataset, batch_size=1)
#     model.eval()
#     import matplotlib.pyplot as plt
#     plt.ion()
#     with torch.no_grad():
#         for index, x in enumerate(dataloaders):     # x是当前图片的tensor形式
#             y = model(x)                            # y是x经过网络的输出，也是tensor形式
#             img_y = torch.squeeze(y).numpy()
#             img_y = img_y[:, :, np.newaxis]
#             img = labelVisualize(2, COLOR_DICT, img_y) if False else img_y[:, :, 0]
#
#             io.imsave("./dataset/train/train_result/" + str(index) + "_predict.jpg", img)
#             plt.pause(0.01)
#         plt.show()

# 将三通道的图片转化为单通道
def prepare_dataset():
    input_path = 'dataset/test/test_3channel'
    output_path = 'dataset/test/test_1channel'
    file_names =os.listdir(input_path)

    for file_num in range(0, len(file_names)):
        img = Image.open(osp.join(input_path, file_names[file_num])).convert('L')
        # img = PIL.Image.fromarray(img)
        img.save(osp.join(output_path, file_names[file_num]))

if __name__ == '__main__':

    print("开始训练")
    train()
    print("训练完成，保存模型")
    print("-"*20)
    # print("测试数据集准备")
    ## prepare_dataset()
    # print("开始预测")

    # test()
