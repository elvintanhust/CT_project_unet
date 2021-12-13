import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import imgaug as ia
import os
from imgaug import augmenters as iaa

ia.seed(random.randint(0, 100000))


class Augment_class():
    def __init__(self, random_num=10):
        self.random_num = random_num
        self.shadow_num = round(random_num*0.5)
        self.shadow_min_ratio=0.5
        self.ill_seq = self.get_ill_seq()

    def aug_image(self, image_ori):
        image_ori = np.tile(image_ori, [self.random_num, 1, 1, 1])
        image_new = self.random_ill_change(image_ori)

        return image_new

    def random_ill_change(self, image_ori):
        image_new = self.ill_seq.augment_images(image_ori)

        # image_num = image_new.shape[0]
        # for image_id in tuple(np.random.choice(np.arange(image_num),
        #                                        self.shadow_num, replace=False)):
        #     # continue
        #     image_new[image_id] = self.insert_shadow(image_new[image_id])
        return image_new

    def insert_shadow(self, image_ori):
        img_new = image_ori.copy()
        image_row, image_col = image_ori.shape[:2]
        shadow_num = random.randint(5, 50)
        min_size, max_size = 10, round(min(image_ori.shape[:2]) / 4)
        mask = np.zeros(image_ori.shape[:2], np.uint8)
        rect_shrink_ratio = 1 / 3
        ellipse_prob = 0.3
        transparency_range = [self.shadow_min_ratio, 1]

        for i in range(shadow_num):
            # 阴影尺寸
            ax = random.randint(min_size, max_size)
            ay = random.randint(min_size, max_size)
            max_rad = max(ax, ay)
            # 阴影中心
            x = np.random.randint(max_rad, image_col - max_rad)
            y = np.random.randint(max_rad, image_row - max_rad)
            # 选取阴影形状
            if random.random() < ellipse_prob:
                angle = np.random.rand() * 90
                cv2.ellipse(mask, (x, y), (ax, ay), angle, 0, 360, 255, -1)
            else:
                shr_x_range = round(rect_shrink_ratio * ax)
                shr_y_range = round(rect_shrink_ratio * ay)
                rad_x, rad_y = round(ax / 2), round(ay / 2)
                rect_point = np.array([[x - rad_x, y - rad_y], [x + rad_x, y - rad_y],
                                       [x + rad_x, y + rad_y], [x - rad_x, y + rad_y]], dtype='int32')
                rect_point += np.c_[np.random.randint(-shr_x_range, shr_x_range, 4),
                                    np.random.randint(-shr_y_range, shr_y_range, 4)]
                cv2.fillConvexPoly(mask, rect_point, 255)

        mask = mask > 1
        mask = np.tile(np.expand_dims(mask, axis=2), (1, 1, 3))
        transparency = np.random.uniform(*transparency_range)
        shadow_value = 255
        img_new[mask] = img_new[mask] * transparency + shadow_value * (1 - transparency)

        return img_new

    def get_ill_seq(self):
        light_change = 50
        seq = iaa.Sequential([
            # 全局调整，含有颜色空间调整
            # iaa.Sometimes(0.5, iaa.OneOf([
            #     iaa.WithColorspace(
            #         to_colorspace="HSV",
            #         from_colorspace="RGB",
            #         children=iaa.OneOf([
            #             iaa.WithChannels(0, iaa.Add((-5, 5))),
            #             iaa.WithChannels(1, iaa.Add((-20, 20))),
            #             iaa.WithChannels(2, iaa.Add((-light_change, light_change))),
            #         ])
            #     ),
            #     iaa.Grayscale((0.2, 0.6)),
            #     iaa.ChannelShuffle(1),
            #     iaa.Add((-light_change, light_change)),
            #     iaa.Multiply((0.5, 1.5)),
            # ])),

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
        return seq


if __name__ == '__main__':
    path = './dataset/test/image/patient1001'
    img = cv2.imread(os.path.join(path, '16.0.jpg'))

    aug_obj = Augment_class()
    new_img = aug_obj.aug_image(img)

    for k in range(new_img.shape[0]):
        # plt.imshow(new_img[k])
        plt.imsave('./dataset/test/image/temp/' + str(k) + '.jpg', new_img[k])
        # plt.show()
        temp = 1
