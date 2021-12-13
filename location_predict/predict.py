import torch
import os
import os.path as osp
from PIL import Image
import pandas as pd
from tqdm import tqdm
import numpy as np
from collections import Counter

# import location_predict.cfg as cfg
from location_predict.data.transform import get_test_transform
from location_predict.data.adjust_order import seq_optimization
from location_predict.data.divisional_design import seq_location


os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def load_checkpoint(filepath):
    # '/home/ubuntu/zipeng/ct_detect/U-net-master/location_predict/data/weights/resnext101_32x32d/epoch_190.pth'
    # '/home/ubuntu/zipeng/ct_detect/U-net-master/location_predict/data/weights/resnext101_32x32d/epoch_190.pth'
    checkpoint = torch.load(filepath)
    model = checkpoint['model']  # 提取网络结构
    model.load_state_dict(checkpoint['model_state_dict'])  # 加载网络权重参数
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model

# 输入参数（模型pth文件路径，图片路径列表）；输出参数（图片id列表，预测类别列表，预测分数列表）
def predict(model, imgs_paths):
    # 读入模型
    model = load_checkpoint(model)
    # print('..... Finished loading model! ......')

    ##将模型放置在gpu上运行
    # if torch.cuda.is_available():
    #     model.cuda()

    pred_list, _id = [], []
    pred_scores_list = []  # 预测最高分的前后3个得分
    for i in tqdm(range(len(imgs_paths))):
        img_path = imgs_paths[i].strip()
        # print(img_path)
        _id.append(os.path.basename(img_path).split('.')[0])
        img = Image.open(img_path).convert('RGB')
        # print(type(img))
        # img = get_test_transform(size=cfg.INPUT_SIZE)(img).unsqueeze(0)
        img = get_test_transform(size=448)(img).unsqueeze(0)

        if torch.cuda.is_available():
            img = img.cuda()
        with torch.no_grad():
            out = model(img)
        prediction = torch.argmax(out, dim=1).cpu().item()
        pred_list.append(prediction * 2)

        # prediction表示下标
        if prediction == 0:
            pred_scores_list.append([0,
                                     round(out[0][prediction].cpu().item(), 2),
                                     round(out[0][prediction + 1].cpu().item(), 2)])
        elif prediction == len(out[0]) - 1:
            pred_scores_list.append([round(out[0][prediction - 1].cpu().item(), 2),
                                     round(out[0][prediction].cpu().item(), 2),
                                     0])
        else:
            pred_scores_list.append([round(out[0][prediction - 1].cpu().item(), 2),
                                     round(out[0][prediction].cpu().item(), 2),
                                     round(out[0][prediction + 1].cpu().item(), 2)])

    return _id, pred_list, pred_scores_list


if __name__ == "__main__":

    trained_model = "../location_predict/data/weights/resnext101_32x32d/epoch_190.pth"
    # trained_model = "/home/ubuntu/zipeng/ct_detect/U-net-master/location_predict/data/weights/resnext101_32x32d/epoch_190.pth"

    # model_name = cfg.model_name
    # with open(cfg.TEST_LABEL_DIR,  'r')as f:
    #     imgs = f.readlines()

    test_path = r"../dataset/train/image/patient0001"
    imgs_names = os.listdir(test_path)
    # 先对文件名排序，正式测试的时候不需要重新排序
    imgs_names.sort()
    imgs_paths = []
    for i in range(len(imgs_names)):
        imgs_paths.append(osp.join(test_path, imgs_names[i]))

    # _id, pred_list = tta_predict(trained_model)
    _id, pred_list, pred_scores_list = predict(trained_model, imgs_paths)

    print("原始分类序列：", end=' ')
    print(pred_list)
    print("分类分数   ：", end=' ')
    print(pred_scores_list)

    optimized_seq = seq_optimization(pred_list, pred_scores_list, isprint=False)
    print("优化序列   ：", end=' ')
    print(optimized_seq)

    # location_list来自分割部分的输出
    location_list = [[], [], [], [], [], [], [], [], [], [], [], [(176, 247)], [(190, 232)], [(192, 226)], [(193, 217)], [(193, 221)], [], [], [], [], [], [], [], [], [], [], []]

    print("出血坐标序列：", end=' ')
    print(location_list)

    regions = seq_location(optimized_seq, location_list=location_list, isprint=False)
    print("出血区域   ：", end=' ')
    print(regions)




