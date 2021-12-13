# CT_project_unet
基于U-net的医学脑CT影像分割和定位，分割出脑出血区域并定位出出血脑区等功能 / pytorch实现，CT图像可来源于二次拍摄的图像

## 实现功能

- 脑CT图像/二次拍摄的脑CT图像脑出血区域的分割
- 出血量的计算
- 出血脑区的定位
- pytorch模型转onnx和转tensorFlow

## 运行环境

 * Ubuntu 16.04
 * python 3.6
 * cuda 9.0
 * pytorch 1.6.0
 * torchvision 0.7.0
 * Cython 0.29.21
 * imgaug 0.4.0
 * opencv-python 4.5.2.52
 * scikit-image 0.17.2
 * Pillow 8.0.0
 * onnxruntime1.10.0

## 代码仓库的使用

### 数据集形式

```
|-- dataset
	|-- train
		|-- image
			|-- patient0001
				|-- 000001.jpg
				|-- 000002.jpg
			|-- patient0002
				|-- 000001.jpg
				|-- 000002.jpg
			...
		|-- label
			|-- patient0001
				|-- 000001_mask.jpg
				|-- 000002_mask.jpg
			|-- patient0002
				|-- 000001_mask.jpg
				|-- 000002_mask.jpg
			...
	|-- test
		|-- image
			|-- patient1001
				|-- 000001.jpg
				|-- 000002.jpg
			|-- patient1002
				|-- 000001.jpg
				|-- 000002.jpg
			...
```

### 模型介绍

分割模型在unet.py中，脑区定位中的分类模型resnext来自于torchvision

### 训练

- 在main.py中设置合适的参数和路径

```
python main.py
```

### 预测

- 命令行测试脚本

```
python batch_test.py 
-i ./dataset/test/image/patient0001/000010.jpg ./dataset/test/image/patient0001/000011.jpg ./dataset/test/image/patient0001/000012.jpg 
-o ./dataset/test/result/patient0001/010.png ./dataset/test/result/patient0001/011.png ./dataset/test/result/patient0001/012.png
```

- 文件夹内的批量图像测试

```
python batch_test_forfolder.py
```

- pytorch模型转onnx

``` 
python pth2onnx.py
```

- 测试onnx模型

``` 
python test_onnx.py
```

- 评价指标

```
python eval.py
```



