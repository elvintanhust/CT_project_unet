import torch.onnx
import onnxruntime as ort
import numpy as np
import cv2
import PIL.Image as Image
import time

# 输入ndarray，输出ndarray
def gray2binary(img255, threshold):
    dst = (img255 >= threshold)  # 根据阈值进行分割
    dst = dst.astype(np.uint8)
    return dst

def preprocess(img_data):
    # mean_vec = np.array([0.485, 0.456, 0.406])
    # stddev_vec = np.array([0.229, 0.224, 0.225])
    mean_vec = np.array([0.5,0.5,0.5])
    stddev_vec = np.array([0.5,0.5,0.5])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[0]):
        # for each pixel in each channel, divide the value by 255 to get value between [0, 1] and then normalize
        norm_img_data[i, :, :] = (img_data[i, :, :] / 255 - mean_vec[i]) / stddev_vec[i]
    return norm_img_data


img = cv2.imread("./dataset/train/image/詹子文 男 28岁 右侧顶叶血肿破入脑室 鼻窦炎/000015.jpg")
img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
input_data = np.transpose(img, (2, 0, 1))

# normalize
input_data = preprocess(input_data)
input_data = input_data.reshape([1, 3, 512, 512])
sess = ort.InferenceSession("./model/onnx/512size_unet.onnx")
input_name = sess.get_inputs()[0].name

start = time.time()
result = sess.run([], {input_name: input_data})
end = time.time()
print('运行时间：%f'% (end - start))

result = result[0][0][0]


img_bin_predict = gray2binary(result, 0) * 255   # 二值分割

# img_y = Image.fromarray(img_y).resize((640, 512))
# img_y.save(osp.join(output_img_paths, str_index + '_gray.png'))

# io.imsave(osp.join(output_img_paths, str_index + '_predict.png'), img_bin_predict)
img_bin_predict = Image.fromarray(img_bin_predict).resize((512,512))
img_bin_predict.save('./dataset/train/result_temp/_predict.png')

# result = np.reshape(result, [1, -1])
index = np.argmax(result)
print("max index:", index)