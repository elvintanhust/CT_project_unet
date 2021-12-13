from torch.utils.data import Dataset
import PIL.Image as Image
import torch
import os
import numpy as np
import os.path as osp
from skimage import io, color

def is_jpgpng_file(filename):
    return any(filename.endswith(extension) for extension in ['.jpg', '.JPG', '.png', '.PNG'])

def get_index(index, size):
    if index < 0:
        return 0
    elif index >= 0 and index <= size - 1:
        return index
    else:
        return size - 1

# 输入图片和标签的文件夹路径，以列表的形式返回[(图片路径，标签路径),(图片路径，标签路径)]
def train_dataset(img_root, label_root):
    images = []
    n = len(os.listdir(img_root))
    for i in range(n):
        image = os.path.join(img_root, "%d.jpg" % i)
        label = os.path.join(label_root, "%d_mask.jpg" % i)
        images.append((image, label))
    return images
# def train_dataset(img_root, label_root):
#     image_filenames = [osp.join(img_root, x) for x in os.listdir(img_root) if is_jpgpng_file(x)]
#     image_filenames.sort()
#     label_filenames = [osp.join(label_root, x) for x in os.listdir(label_root) if is_jpgpng_file(x)]
#     label_filenames.sort()
#     images = list(zip(image_filenames, label_filenames))
#     return images


def my_train_dataset(image_root, label_root, frame=3):
    # 输入图片和标签的文件夹路径，以列表的形式返回[(图片路径，标签路径),(图片路径，标签路径)，...]
    if (frame == 1):
        # images_labels = []
        # n = len(os.listdir(image_root))
        # for i in range(n):
        #     image = os.path.join(image_root, "%d.jpg" % i)
        #     label = os.path.join(label_root, "%d_mask.jpg" % i)
        #     images_labels.append((image, label))
        # return images_labels
        images = []
        labels = []
        for sub_image_foldername in os.listdir(image_root):
            sub_folder_root = osp.join(image_root, sub_image_foldername)
            sub_images_list = os.listdir(sub_folder_root)
            sub_images_list.sort()
            sub_folder_size = len(sub_images_list)
            for i in range(sub_folder_size):
                # if i >= 0 and i < 15:   # 判断在上中下哪一大层
                if i >= 8 and i < 22:  # 判断在上中下哪一大层
                # if i >= 15:  # 判断在上中下哪一大层
                    images.append((osp.join(sub_folder_root, sub_images_list[i])))
                    labels.append(osp.join(label_root, sub_image_foldername, (sub_images_list[i][0:-4] + '_mask.jpg')))
        images_labels = list(zip(images, labels))
        return images_labels
    # 输入图片和标签的文件夹路径，以列表的形式返回[((上一帧路径，图片路径，下一帧路径)，标签路径),((上一帧路径，图片路径，下一帧路径)，标签路径)，...]
    elif (frame == 3):
        images = []
        labels = []
        for sub_image_foldername in os.listdir(image_root):
            sub_folder_root = osp.join(image_root, sub_image_foldername)
            sub_images_list = os.listdir(sub_folder_root)
            sub_images_list.sort()
            sub_folder_size = len(sub_images_list)
            for i in range(sub_folder_size):
                images.append((osp.join(sub_folder_root, sub_images_list[i - 1 if i - 1 >= 0 else 0]),
                               osp.join(sub_folder_root, sub_images_list[i]),
                               osp.join(sub_folder_root, sub_images_list[i + 1 if i + 1 <= sub_folder_size - 1 else sub_folder_size - 1])))
                labels.append(osp.join(label_root, sub_image_foldername, (sub_images_list[i][0:-4] + '_mask.jpg')))
        images_labels = list(zip(images, labels))
        return images_labels
    elif (frame == 5):
        images = []
        labels = []
        for sub_image_foldername in os.listdir(image_root):
            sub_folder_root = osp.join(image_root, sub_image_foldername)
            sub_images_list = os.listdir(sub_folder_root)
            sub_images_list.sort()
            sub_folder_size = len(sub_images_list)
            for i in range(sub_folder_size):
                images.append((osp.join(sub_folder_root, sub_images_list[i - 2 if i - 2 >= 0 else 0]),
                               osp.join(sub_folder_root, sub_images_list[i - 1 if i - 1 >= 0 else 0]),
                               osp.join(sub_folder_root, sub_images_list[i]),
                               osp.join(sub_folder_root, sub_images_list[
                                   i + 1 if i + 1 <= sub_folder_size - 1 else sub_folder_size - 1]),
                               osp.join(sub_folder_root, sub_images_list[
                                   i + 2 if i + 2 <= sub_folder_size - 1 else sub_folder_size - 1])))
                labels.append(osp.join(label_root, sub_image_foldername, (sub_images_list[i][0:-4] + '_mask.jpg')))
        images_labels = list(zip(images, labels))
        return images_labels

    else:
        print('error')


class TrainDataset(Dataset):
    def __init__(self, img_root, label_root, frame, seq=None, transform=None, target_transform=None):
        imgs = my_train_dataset(img_root, label_root, frame)
        self.imgs = imgs
        self.frame = frame
        self.seq = seq
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        if (self.frame == 1):
            img_x0 = io.imread(x_path)
            img_x = img_x0
        elif (self.frame == 3):
            img_x0 = (io.imread(x_path[0]))[:, :, 0]  # 读取上一帧的灰度图
            img_x1 = (io.imread(x_path[1]))[:, :, 0]  # 读取本图的灰度图
            img_x2 = (io.imread(x_path[2]))[:, :, 0]  # 读取下一帧
            img_x = np.dstack((img_x0,img_x1,img_x2))   # 将三张灰度图融合为一张通道为3的图
            # img_x = Image.fromarray(img_x)
        elif (self.frame == 5):
            img_x0 = (io.imread(x_path[0]))[:, :, 0]
            img_x1 = (io.imread(x_path[1]))[:, :, 0]
            img_x2 = (io.imread(x_path[2]))[:, :, 0]  # 读取本图
            img_x3 = (io.imread(x_path[3]))[:, :, 0]
            img_x4 = (io.imread(x_path[4]))[:, :, 0]
            w = 0.6
            img_x = np.dstack((img_x0*(1-w)+img_x1*w,img_x2,img_x3*w+img_x4*(1-w)))   # 将5张灰度图融合为一张通道为3的图
            img_x = img_x.astype(np.uint8)
            # img_x = Image.fromarray(img_x)
        # img_x = Image.open(x_path)
        if osp.exists(y_path):
            # 如果存在标签，则使用标签
            img_y = Image.open(y_path)
        else:
            # 如果不存在标签，则用全黑图作为标签
            img_y = Image.open('./dataset/train_fullname/zero_map/zero_map.jpg')

        if self.seq is not None:
            img_x = self.seq.augment_images(img_x)
        img_x = Image.fromarray(img_x)
        if self.transform is not None:
            img_x = self.transform(img_x)

        # img_y = torch.squeeze(img_x).numpy()[1]
        # io.imsave(osp.join("/home/ubuntu/zipeng/ct_detect/U-net-master/dataset/test/result_temp", "_predict.jpg"),
        #           img_y)

        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x, img_y

    def __len__(self):
        return len(self.imgs)

#########################################################################################

# 输入图片的文件夹路径，以列表的形式输出路径下的文件名
# def test_dataset(img_root):
#     imgs = []
#     n = len(os.listdir(img_root))
#     # 遍历路径下的文件名
#     for i in range(n):
#         img = os.path.join(img_root, "%d.jpg" % i)
#         imgs.append(img)
#     return imgs


# 输入图片的文件夹路径，以列表的形式输出路径下的文件名，文件名是按顺序的
# def my_test_dataset(img_root):
#     image_filenames = [osp.join(img_root, x) for x in os.listdir(img_root) if is_jpgpng_file(x)]
#     image_filenames.sort()
#     return image_filenames

# 输入图片的文件夹路径，以列表的形式输出[(上一帧路径，图片路径，下一帧路径),(上一帧路径，图片路径，下一帧路径)，...]
def multiframe_test_dataset(sub_folder_root, frame=3):
    if (frame == 1):
        image_filenames = [osp.join(sub_folder_root, x) for x in os.listdir(sub_folder_root) if is_jpgpng_file(x)]
        image_filenames.sort()
        return image_filenames
    elif (frame == 3):
        images = []
        sub_images_list = os.listdir(sub_folder_root)
        sub_images_list.sort()
        sub_folder_size = len(sub_images_list)
        for i in range(sub_folder_size):
            images.append((osp.join(sub_folder_root, sub_images_list[i - 1 if i - 1 >= 0 else 0]),
                           osp.join(sub_folder_root, sub_images_list[i]),
                           osp.join(sub_folder_root, sub_images_list[i + 1 if i + 1 <= sub_folder_size - 1 else sub_folder_size - 1])))
        return images
    elif (frame == 5):
        images = []
        sub_images_list = os.listdir(sub_folder_root)
        sub_images_list.sort()
        sub_folder_size = len(sub_images_list)
        for i in range(sub_folder_size):
            images.append((osp.join(sub_folder_root, sub_images_list[i - 2 if i - 2 >= 0 else 0]),
                           osp.join(sub_folder_root, sub_images_list[i - 1 if i - 1 >= 0 else 0]),
                           osp.join(sub_folder_root, sub_images_list[i]),
                           osp.join(sub_folder_root, sub_images_list[i + 1 if i + 1 <= sub_folder_size - 1 else sub_folder_size - 1]),
                           osp.join(sub_folder_root, sub_images_list[i + 2 if i + 2 <= sub_folder_size - 1 else sub_folder_size - 1])))
        return images


# class TestDataset(Dataset):
#     def __init__(self, img_root, transform=None, target_transform=None):
#         imgs = my_test_dataset(img_root)   # 输入图片的文件夹路径，以列表的形式输出路径下的文件名，列表为imgs
#         # imgs = [img_root]
#         self.imgs = imgs
#         self.transform = transform
#         self.target_transform = target_transform
#
#     def __getitem__(self, index):
#         x_path = self.imgs[index]
#         img_x = Image.open(x_path)
#         img_x = img_x.convert("L")
#         # img_x = img_x[0:256,0:256]
#         if self.transform is not None:
#             img_x = self.transform(img_x)
#         return img_x
#
#     def __len__(self):
#         return len(self.imgs)


class multiFrameTestDataset(Dataset):
    def __init__(self, img_root, frame, transform=None, target_transform=None):
        imgs = multiframe_test_dataset(img_root, frame)   # 输入图片的文件夹路径，以列表的形式输出路径下的文件名，列表为imgs
        self.imgs = imgs
        self.frame = frame
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path = self.imgs[index]
        if (self.frame == 1):
            x_path = self.imgs[index]

            # img_x = (io.imread(x_path))[:, :, 2]

            img_x = color.rgb2gray(io.imread(x_path)) * 255  # 彩色转灰度
            img_x = img_x.astype(np.uint8)

            img_x = np.dstack((img_x, img_x, img_x))
            img_x = Image.fromarray(img_x)
            # img_x = img_x.convert("L")
        elif (self.frame == 3):
            img_x0 = (io.imread(x_path[0]))[:, :, 0]  # 读取上一帧的灰度图
            img_x1 = (io.imread(x_path[1]))[:, :, 0]  # 读取本图的灰度图
            img_x2 = (io.imread(x_path[2]))[:, :, 0]  # 读取下一帧
            img_x = np.dstack((img_x0, img_x1, img_x2))  # 将三张灰度图融合为一张通道为3的图
            img_x = Image.fromarray(img_x)
        elif (self.frame == 5):
            img_x0 = (io.imread(x_path[0]))[:, :, 0]
            img_x1 = (io.imread(x_path[1]))[:, :, 0]
            img_x2 = (io.imread(x_path[2]))[:, :, 0]  # 读取本图
            img_x3 = (io.imread(x_path[3]))[:, :, 0]
            img_x4 = (io.imread(x_path[4]))[:, :, 0]
            w = 0.6
            img_x = np.dstack(
                (img_x0 * (1 - w) + img_x1 * w, img_x2, img_x3 * w + img_x4 * (1 - w)))  # 将5张灰度图融合为一张通道为3的图
            img_x = img_x.astype(np.uint8)
            img_x = Image.fromarray(img_x)

        if self.transform is not None:
            img_x = self.transform(img_x)
        return img_x

    def __len__(self):
        return len(self.imgs)


# 输入路径img_root为图像路径的列表
class CLITestDataset(Dataset):
    def __init__(self, img_root, transform=None, target_transform=None):
        imgs = img_root   # 输入图片的文件夹路径，以列表的形式输出路径下的文件名，列表为imgs
        # imgs = [img_root]
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        # x_path = self.imgs[index]
        # img_x = Image.open(x_path)

        x_path = self.imgs[index]
        img_x = io.imread(x_path)

        img_x = color.rgb2gray(io.imread(x_path)) * 255  # 彩色转灰度
        img_x = img_x.astype(np.uint8)

        img_x = np.dstack((img_x, img_x, img_x))
        img_x = Image.fromarray(img_x)

        if self.transform is not None:
            img_x = self.transform(img_x)
        return img_x

    def __len__(self):
        return len(self.imgs)