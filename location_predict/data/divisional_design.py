
seq = [0, 0, 2, 2, 4, 6, 8, 10, 10, 12, 12, 12, 14, 16, 18, 20, 20, 20, 22, 22, 22, 22, 22]
location_list = [[], [], [], [], [], [], [(234, 197), (250, 189)], [(232, 199), (248, 188)], [(245, 184), (234, 179), (355, 127)], [(348, 128)], [(350, 133)], [(350, 134)], [(352, 141)], [(353, 138), (362, 143)], [(356, 118), (347, 143)], [(347, 151), (366, 125)], [], [], [], [], [], [], []]

# seq = [0, 0, 2, 2, 2, 2, 2, 4, 6, 6, 8, 10, 12, 14, 16, 18, 18, 18, 18, 18, 20, 20, 20, 22, 22, 22, 22, 22]
# location_list = [[], [], [], [], [], [], [], [], [], [(289, 407)], [(293, 398)], [], [(263, 282)], [(193, 341), (269, 280)], [(274, 273)], [(280, 270)], [(244, 262)], [(246, 258)], [(245, 248)], [], [], [], [], [], [], [], [], []]

# 输入：（层号，坐标值）；输出：脑区名称
def location(layer_id, coordinate, size=256):
    x = coordinate[0]
    y = coordinate[1]

    if (layer_id == 22):
        if y <= 0.5 * size: region = '额叶'
        else: region = '顶叶'
    elif (layer_id == 20):
        if y <= 0.4 * size: region = '额叶'
        else: region = '顶叶'
    elif (layer_id == 18):
        if y <= 0.45 * size: region = '额叶'
        elif 0.45 * size <= y < 0.7 * size: region = '顶叶'
        else: region = '枕叶'
    elif (layer_id == 16):
        if y <= 0.4 * size: region = '额叶'
        elif 0.4 * size <= y < 0.5 * size: region = '颞叶'
        elif 0.5 * size <= y < 0.6 * size: region = '顶叶'
        else: region = '枕叶'
    elif (layer_id == 14):
        if 0.45 * size <= x < 0.55 * size and 0.5 * size <= y < 0.6 * size: region = '小脑'
        elif y <= 0.35 * size: region = '额叶'
        elif 0.35 * size <= y < 0.45 * size: region = '颞叶'
        elif 0.45 * size <= y < 0.6 * size: region = '顶叶'
        else: region = '枕叶'
    elif (layer_id == 12):
        if 0.45 * size <= x < 0.55 * size and 0.6 * size <= y < 0.7 * size: region = '小脑'
        elif 0.45 * size <= x < 0.55 * size and 0.5 * size <= y < 0.6 * size: region = '中脑'
        elif y <= 0.4 * size: region = '额叶'
        elif 0.4 * size <= y < 0.7 * size: region = '颞叶'
        else: region = '枕叶'
    elif (layer_id == 10):
        if 0.45 * size <= x < 0.55 * size and 0.6 * size <= y < 0.7 * size: region = '小脑'
        elif 0.45 * size <= x < 0.55 * size and 0.45 * size <= y < 0.6 * size: region = '中脑'
        elif y <= 0.35 * size: region = '额叶'
        elif 0.35 * size <= y < 0.7 * size: region = '颞叶'
        else: region = '枕叶'
    elif (layer_id == 8):
        if 0.45 * size <= x < 0.55 * size and 0.5 * size <= y < 0.6 * size: region = '脑桥'
        elif y <= 0.35 * size: region = '额叶'
        elif 0.35 * size <= y < 0.55 * size: region = '颞叶'
        else: region = '小脑'
    elif (layer_id == 6):
        if 0.45 * size <= x < 0.55 * size and 0.5 * size <= y < 0.6 * size: region = '脑桥'
        elif y <= 0.35 * size: region = '额叶'
        elif 0.35 * size <= y < 0.55 * size: region = '颞叶'
        else: region = '小脑'
    elif (layer_id == 4):
        if 0.45 * size <= x < 0.55 * size and 0.5 * size <= y < 0.65 * size: region = '脑桥'
        elif y <= 0.3 * size: region = '额叶'
        elif 0.3 * size <= y < 0.5 * size: region = '颞叶'
        else: region = '小脑'
    elif (layer_id == 2):
        if 0.45 * size <= x < 0.55 * size and 0.5 * size <= y < 0.65 * size: region = '脑桥'
        elif y <= 0.3 * size: region = '额叶'
        elif 0.3 * size <= y < 0.5 * size: region = '颞叶'
        else: region = '小脑'
    else:
        if 0.45 * size <= x < 0.55 * size and 0.5 * size <= y < 0.6 * size: region = '延髓'
        else: region = '小脑'

    return region

def seq_location(seq, location_list, isprint=False):
    # 脑区字典，用来记录各个脑区出现脑出血的次数
    brain_region = {'额叶':0, '顶叶':0, '枕叶':0, '颞叶':0, '小脑':0, '中脑':0, '脑桥':0, '延髓':0}
    # 脑出血区域列表，用于存放脑出血区域的名称
    celebral_hemorrhage_region = []

    # 通过location函数计算各个脑区出现脑出血的次数，存放在brain_region中
    # 遍历所有层
    for num in range(len(seq)):
        # 遍历每一层中的一个或多个中心点坐标
        for coordinate in location_list[num]:
            brain_region[location(seq[num], coordinate, size=256)] += 1

    # 将脑区字典中的值大于0的脑区键，存入脑出血区域列表celebral_hemorrhage_region
    for brain_region_name, count in brain_region.items():
        if(count > 0):
            celebral_hemorrhage_region.append(brain_region_name)

    if(isprint):
        print(celebral_hemorrhage_region)
        print("end")

    return celebral_hemorrhage_region