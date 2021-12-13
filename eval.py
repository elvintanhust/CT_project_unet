import skimage.io as io
from skimage import transform
import os
import os.path as osp
import numpy as np


def evaluate_for_each(gt_path,predict_path):
    print('对于单张预测图的评估：')

    gt = io.imread(gt_path)

    gt = np.array(gt)
    gt = np.int64(gt > 0)
    predict = io.imread(predict_path)
    predict = np.array(predict)
    predict = np.int64(predict > 0)

    intersection = gt * predict
    union = gt | predict

    sum_gt = sum(sum(gt))
    sum_predict = sum(sum(predict))
    sum_inter = sum(sum(intersection))
    sum_union = sum(sum(union))

    precision_rate = (sum_inter + 1) / (sum_predict + 1)
    recall_rate = (sum_inter + 1) / (sum_gt + 1)
    dice_coefficient = (2 * sum_inter + 1) / (sum_predict + sum_gt + 1)
    IOU = (sum_inter + 1) / (sum_union + 1)
    F1_Score = 2 * precision_rate * recall_rate / (precision_rate + recall_rate)

    return F1_Score,IOU
    # print('精确率：%f' % precision_rate, end='  ')
    # print('召回率：%f' % recall_rate, end='  ')
    # print('Dice系数：%f' % dice_coefficient, end='  ')
    # print('IOU：%f' % IOU, end='  ')
    # print('F1 Score：%f' % (2 * precision_rate * recall_rate / (precision_rate + recall_rate)))

def evaluate_for_batch(gt_folder, predict_folder, mask_ext='_mask.jpg', predict_ext='_predict.jpg', need_print=True):

    sum_pre, sum_rec, sum_acc, sum_dice, sum_iou, sum_fa, sum_ma = 0, 0, 0, 0, 0, 0, 0
    max_pre, min_pre, max_rec, min_rec, max_dice, min_dice, max_iou, min_iou = 0, 1, 0, 1, 0, 1, 0, 1

    gt_list = os.listdir(gt_folder)
    gt_list.sort()

    predict_list = os.listdir(predict_folder)
    predict_list.sort()

    num_for_gt = 1  # 1：以gt的数量作为总数；0：以predict的数量作为总数
    # num_for_gt ? num = len(gt_list) : num = len(predict_list)
    # num = len(gt_list) if num_for_gt else len(predict_list)     # python中的三目运算符：num = "变量1" if a>b else "变量2"
    sequence_list = [x[0:-(len(mask_ext))] for x in gt_list] if num_for_gt else [x[0:-(len(predict_ext))] for x in predict_list]
    num = len(sequence_list)
    if(num == 0):
        print(predict_folder)

    for i in range(0, num):
        gt_path = osp.join(gt_folder,sequence_list[i] + mask_ext)
        predict_path = osp.join(predict_folder,sequence_list[i] + predict_ext)

        gt = io.imread(gt_path)
        # gt = gt[504:3527,0:3023]
        gt = transform.resize(gt, (512,512))

        gt = np.array(gt)
        gt_P = np.int64(gt > 0)
        gt_N = np.int64(gt == 0)
        predict = io.imread(predict_path)
        predict = np.array(predict)
        pred_P = np.int64(predict > 0)
        pred_N = np.int64(predict == 0)

        tp_map = gt_P * pred_P
        tn_map = gt_N * pred_N
        fp_map = gt_N * pred_P
        fn_map = gt_P * pred_N
        intersection = tp_map
        union = gt_P | pred_P

        tp, tn, fp, fn = sum(sum(tp_map)), sum(sum(tn_map)), sum(sum(fp_map)), sum(sum(fn_map))
        sum_inter = sum(sum(intersection))
        sum_union = sum(sum(union))

        precision_rate = (tp + 1) / (tp + fp + 1)
        recall_rate = (tp + 1) / (tp + fn + 1)
        accuracy_rate = (tp + tn + 1) / (tp + fn + fp + tn + 1)
        dice_coefficient = (2 * sum_inter + 1) / (sum_inter + sum_union + 1)
        IOU = (sum_inter + 1) / (sum_union + 1)
        fa = (fp + 1)/(fp + tn + 1)
        ma = (fn + 1)/(tp + fn + 1)

        # 找出指标中最好和最差的图
        if (max_pre < precision_rate and sum_inter != 0 ): max_pre, max_pre_i = precision_rate, i
        if (max_rec < recall_rate and sum_inter != 0 ): max_rec, max_rec_i = recall_rate, i
        if (max_dice < dice_coefficient): max_dice, max_dice_i = dice_coefficient, i
        if (max_iou < IOU): max_iou, max_iou_i = IOU, i
        if (min_pre > precision_rate): min_pre, min_pre_i = precision_rate, i
        if (min_rec > recall_rate): min_rec, min_rec_i = recall_rate, i
        if (min_dice > dice_coefficient): min_dice, min_dice_i = dice_coefficient, i
        if (min_iou > IOU): min_iou, min_iou_i = IOU, i

        sum_pre += precision_rate
        sum_rec += recall_rate
        sum_acc += accuracy_rate
        sum_dice += dice_coefficient
        sum_iou += IOU
        sum_fa += fa
        sum_ma += ma

    precision_rate = sum_pre/num
    recall_rate = sum_rec/num
    accuracy_rate = sum_acc/num
    dice_coefficient = sum_dice/num
    IOU = sum_iou/num
    F1_Score = 2 * precision_rate * recall_rate / (precision_rate + recall_rate)
    fa_rate = sum_fa/num
    ma_rate = sum_ma / num

    if need_print:
        print('对于批量预测图的评估：')
        print('测试数量： %d' % num)
        print('精确率：%f' % precision_rate, end='  ')
        print('召回率：%f' % recall_rate, end='  ')
        print('正确率：%f' % accuracy_rate, end='  ')
        # print('Dice系数：%f' % dice_coefficient, end='  ')
        print('IOU：%f' % IOU, end='  ')
        print('F1 Score：%f' % F1_Score, end='  ')
        print('FA：%f' % fa_rate, end='  ')
        print('MA：%f' % ma_rate)
        print('最好精确率：%f，图%d，最差精确率：%f，图%d' % (max_pre,max_pre_i,min_pre,min_pre_i))
        print('最好召回率：%f，图%d，最差召回率：%f，图%d' % (max_rec,max_rec_i,min_rec,min_rec_i))
        print('最好Dice系数：%f，图%d，最差Dice系数：%f，图%d' % (max_dice,max_dice_i,min_dice,min_dice_i))
        print('最好IOU：%f，图%d，最差IOU：%f，图%d' % (max_iou,max_iou_i,min_iou,min_iou_i))

    return precision_rate, recall_rate, accuracy_rate, IOU, F1_Score, fa_rate, ma_rate


if __name__ == '__main__':

    ################################################
    # 单图评估
    # gt_path = './dataset/val/label/0_mask.jpg'
    # predict_path = './dataset/val/temp_val_result202-45/0_predict45_open+dilate.jpg'
    # evaluate_for_each(gt_path,predict_path)

    ################################################
    # 文件夹内批量图评估
    # mask_ext = '_mask.jpg'  # mask名的后缀
    # predict_ext = '_predict.png'  # 预测图像名的后缀
    # gt_folder = './dataset/train/label/patient0001'
    # predict_folder = './dataset/train/result/patient0001'
    # evaluate_for_batch(gt_folder, predict_folder, mask_ext, predict_ext, need_print=True)

    ################################################
    # 数据集中文件夹内批量图评估
    mask_ext = '_mask.jpg'          # mask名的后缀
    predict_ext = '_predict.png'  # 预测图像名的后缀
    gt_set_folder = './dataset/valpng/tmplabel'
    predict_set_folder = './dataset/valpng/2layer_result138e'# 0.6575
    # predict_set_folder = './dataset/valpng/1allresult100e'  # iou = 0.6918
    # predict_set_folder = './dataset/valpng/96e256size'      # iou = 0.6325

    sub_folder_index = 0
    sub_folder_num = len(os.listdir(predict_set_folder))
    precision_rate, recall_rate, accuracy_rate, IOU, F1_Score, fa_rate, ma_rate = 0, 0, 0, 0, 0, 0, 0

    for sub_foldername in os.listdir(predict_set_folder):
        gt_folder = osp.join(gt_set_folder, sub_foldername)
        predict_folder = osp.join(predict_set_folder, sub_foldername)

        print('Index: ' + str(sub_folder_index), end='  ')
        print(sub_foldername)
        eval_list = evaluate_for_batch(gt_folder, predict_folder, mask_ext, predict_ext, need_print=False)
        # print(eval_list)
        # print('Index: ' + str(sub_folder_index), end='  ')
        # print(sub_foldername)
        print('Precision：%0.4f' % eval_list[0], end='  ')
        print('Recall：%0.4f' % eval_list[1], end='  ')
        print('ACC：%0.4f' % eval_list[2], end='  ')
        print('IOU：%0.4f' % eval_list[3], end='  ')
        print('F1：%0.4f' % eval_list[4])
        # print('FA：%0.4f' % eval_list[5], end='  ')
        # print('MA：%0.4f' % eval_list[6])
        sub_folder_index += 1
        precision_rate += eval_list[0]
        recall_rate += eval_list[1]
        accuracy_rate += eval_list[2]
        IOU += eval_list[3]
        F1_Score += eval_list[4]
        fa_rate += eval_list[5]
        ma_rate += eval_list[6]
    print('For all folders:')
    print('Precision：%0.4f' % (precision_rate / sub_folder_num), end='  ')
    print('Recall：%0.4f' % (recall_rate / sub_folder_num), end='  ')
    print('ACC：%0.4f' % (accuracy_rate / sub_folder_num), end='  ')
    print('IOU：%0.4f' % (IOU / sub_folder_num), end='  ')
    print('F1：%0.4f' % (F1_Score / sub_folder_num), end='  ')
    print('FA：%0.4f' % (fa_rate / sub_folder_num), end='  ')
    print('MA：%0.4f' % (ma_rate / sub_folder_num))