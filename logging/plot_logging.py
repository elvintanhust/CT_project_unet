import matplotlib.pyplot as plt
import numpy as np

filename = './log_val.txt'  # 文档路径
str0 = 'epoch'
num0_len = 3    # 字符串str0后面数字的长度
str1 = 'IOU'
num1_len = 8    # 字符串str1后面数字的长度
str1 = 'F1'
num2_len = 8

# 打开文档，将内容放入contents中
try:
    with open(filename) as file_object:
        contents = file_object.read()   # 把文本内容以字符串保存至contents中
        # print(len(contents))
except FileNotFoundError:
    msg = 'Sorry, the file ' + filename + 'does not exist.'
    print(msg)
else:

    str0_len = len(str0) + 2
    str0_index = contents.find(str0, 0)
    str0_list = []

    for i in range(contents.count(str0, 0, len(contents))):  # 默认搜索整个字符串，循环次数为字符串中出现子字符串的次数。count()用于统计字符串中子字符串出现的个数
        str0_list.append(float(contents[str0_index + str0_len: str0_index + str0_len + num0_len]))
        str0_index = contents.find(str0, str0_index + 1)

    str1_len = len(str1) + 2
    str1_index = contents.find(str1, 0)
    str1_list = []

    for i in range(contents.count(str1,0,len(contents))): #默认搜索整个字符串，循环次数为字符串中出现子字符串的次数。count()用于统计字符串中子字符串出现的个数
        str1_list.append(float(contents[str1_index + str1_len : str1_index + str1_len + num1_len]))
        str1_index = contents.find(str1, str1_index + 1)

    # plt.scatter(range(contents.count(str1)), str1_list)  # 散点图
    # plt.plot(range(contents.count(str1)), str1_list)  # 折线图
    plt.plot(str0_list, str1_list)  # 折线图
    plt.xlabel('epoch')
    plt.ylim(0,1)
    plt.ylabel(str1)
    plt.grid()
    plt.show()

    print('Finished!')

