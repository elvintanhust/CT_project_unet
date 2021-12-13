from batch_test_forfolder import *

# weight_path = './model/jitter1forall/jitter1_unet_model_108_epoch_sd.pt'    # 512size分割权重
# weight_path = './model/jitter1forall256size/jitter1_unet_model_96_epoch_sd.pt'    # 256size分割权重
weight_path = './model/3layer_model/2layer/256jitter1_unet_model_138_epoch_sd.pt'
input_set_path = './dataset/valpng/image'          # 输入图像集的路径
output_set_path = './dataset/valpng/2layer_result138e'   # 输出图像集的路径
frame = 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # 创建ArgumentParser()对象
    parser.add_argument('--model', '-m', default=weight_path)
    parser.add_argument('--input_set_path', '-i', default=input_set_path)
    parser.add_argument('--output_set_path', '-o', default=output_set_path)
    args = parser.parse_args()  # 使用parse_arg()解析添加的参数
    check_point = args.model
    input_set_path = args.input_set_path
    output_set_path = args.output_set_path

    if not osp.exists(input_set_path):
        print(input_set_path + ' not exist!!')
    if not osp.exists(output_set_path):
        os.mkdir(output_set_path)

    input_folders_name = os.listdir(input_set_path)

    for input_folder_name in input_folders_name:
        input_folder_path = osp.join(input_set_path, input_folder_name)
        output_folder_path = osp.join(output_set_path, input_folder_name)
        if not osp.exists(output_folder_path):
            os.mkdir(output_folder_path)

        print(input_folder_name)
        start = time.time()

        optimized_seq = []
        location_list = test(check_point, input_folder_path, output_folder_path, optimized_seq, need_print=False)

        end = time.time()
        print('当前文件夹下总运行时间：%f\n' % (end - start))

        # 以上是分割部分
        ##################################################################
        # 以下是定位部分

        trained_model = "./location_predict/data/weights/resnext101_32x32d/epoch_191.pth"
"
        imgs_names = os.listdir(input_folder_path)
        # 先对文件名排序，正式测试的时候不需要重新排序
        imgs_names.sort()
        imgs_paths = []
        for i in range(len(imgs_names)):
            imgs_paths.append(osp.join(input_folder_path, imgs_names[i]))

        _id, pred_list, pred_scores_list = predict(trained_model, imgs_paths)

        print("原始分类序列：", end=' ')
        print(pred_list)
        # print("分类分数   ：", end=' ')
        # print(pred_scores_list)

        optimized_seq = seq_optimization(pred_list, pred_scores_list, isprint=False)
        print("优化序列   ：", end=' ')
        print(optimized_seq)

        # print("出血坐标序列：", end=' ')
        # print(location_list)

        regions = seq_location(optimized_seq, location_list=location_list, isprint=False)
        print("出血区域   ：", end=' ')
        print(regions)

