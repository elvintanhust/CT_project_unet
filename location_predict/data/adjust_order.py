# seq = [0, 0, 2, 2, 2, 2, 2, 6, 6, 6, 8, 12, 20, 12, 20, 20, 20, 20, 20, 20, 22, 22, 22, 22, 22, 22, 22]
seq = [0, 0, 2, 22, 2, 6, 8, 10, 18, 12, 18, 12, 20, 18, 18, 20, 20, 20, 22, 22, 22, 22, 22]

# scores_seq = [[0, 9.033186912536621, 5.613186359405518],[0, 9.031064987182617, 5.685033321380615],[0, 8.920525550842285, 5.9734907150268555],
#               [0, 7.515252113342285, 5.583392143249512],[0, 7.4545416831970215, 6.010153770446777],[6.267703056335449, 7.229067802429199, 6.47604513168335],
#               [4.356229782104492, 5.900481700897217, 5.405774116516113],[1.6753277778625488, 3.7602200508117676, 3.6594276428222656],
#               [2.2680373191833496, 2.6598496437072754, 1.5639475584030151],[0.4878985285758972, 1.9974606037139893, 1.9195481538772583],
#               [1.4769552946090698, 2.126319408416748, 1.1450762748718262],[1.1467162370681763, 2.0412049293518066, 1.0257841348648071],
#               [1.471242070198059, 2.631680727005005, 2.179579496383667],[2.7016589641571045, 3.2657904624938965, 1.8410272598266602],
#               [2.8827261924743652, 3.644536256790161, 3.5805838108062744],[3.276369571685791, 3.757749319076538, 3.4257137775421143],
#               [3.3702995777130127, 6.311985492706299, 6.167485237121582],[3.1150062084198, 7.354478359222412, 6.579658508300781],
#               [6.547810077667236, 7.020653247833252, 0],[6.831855773925781, 8.250619888305664, 0],[6.271189212799072, 9.322376251220703, 0],
#               [5.531214237213135, 8.949485778808594, 0],[5.379454135894775, 10.939595222473145, 0],[5.447601795196533, 11.950372695922852, 0],
#               [5.915984153747559, 14.684892654418945, 0],[6.148496150970459, 14.337355613708496, 0],[6.459311485290527, 14.24008846282959, 0]]
scores_seq = [[0, 15.962529182434082, 14.450174331665039], [0, 14.99944019317627, 13.717550277709961], [8.830686569213867, 9.148002624511719, -1.5343999862670898], [-0.962857723236084, 8.719061851501465, 0], [0.8026000261306763, 5.389798641204834, -3.6356422901153564], [-3.852513551712036, 8.096139907836914, 5.265137672424316], [2.626528739929199, 8.801519393920898, 6.572805404663086], [6.350579738616943, 10.784282684326172, 0.0651635080575943], [2.0554885864257812, 8.157853126525879, 2.2430672645568848], [7.649357795715332, 8.642839431762695, 3.4235332012176514], [2.583448886871338, 8.559179306030273, 4.100086212158203], [6.509147644042969, 10.293290138244629, 6.8081769943237305], [7.341824531555176, 8.282012939453125, -2.2673635482788086], [6.313859939575195, 9.605566024780273, 5.865984916687012], [6.474619388580322, 8.664670944213867, 4.813777923583984], [6.1913580894470215, 9.672931671142578, 5.298215389251709], [2.9559390544891357, 11.503613471984863, 8.88846206665039], [0.6157232522964478, 11.194825172424316, 8.29102897644043], [10.6860933303833, 13.355537414550781, 0], [8.976114273071289, 15.016359329223633, 0], [6.569857597351074, 18.80513572692871, 0], [7.33546257019043, 21.111621856689453, 0], [8.376696586608887, 24.228479385375977, 0]]


# 判断一个序列是否连续+2
def is_continue(seq):
    length = len(seq)
    for i in range(length - 1):
        if(seq[i + 1] - seq[i] > 2):
            return False
    return True


# 序列优化：去噪、递增、连续化
def seq_optimization(seq, scores_seq, isprint=False):

    # print('原始序列为：', end=' ')
    # print(seq)

    seq[0] = 0  # 序列首项为0
    seq[-1] = 22  # 序列尾项为22
    diff_seq = []  # seq的差分列表

    # 去噪
    for idx in range(1, len(seq) - 1):
        if(seq[idx] > seq[idx-1] and seq[idx] > seq[idx+1]
        or seq[idx] < seq[idx-1] and seq[idx] < seq[idx+1]):
            seq[idx] = seq[idx - 1]
    if(isprint):
        print('去噪序列为：',end=' ')
        print(seq)

    # 保证非严格递增
    for idx in range(len(seq) - 1):
        if(seq[idx + 1] < seq[idx]):
            seq[idx + 1] = seq[idx]
    if (isprint):
        print('递增序列为：', end=' ')
        print(seq)

    # 序列连续化
    count1,count2 = 0, 0
    while(not is_continue(seq)):
        for idx in range(1, len(seq) - 2):
            diff = seq[idx + 1] - seq[idx]
            if (diff == 0 or diff == 2):
                continue
            elif (diff == 4):
                score_diff0 = scores_seq[idx][1] - scores_seq[idx][2]
                score_diff1 = scores_seq[idx][1] - scores_seq[idx][0]
                if(score_diff0 < score_diff1):
                    seq[idx] = seq[idx] + 2
                else:
                    seq[idx + 1] = seq[idx + 1] - 2
            elif (diff == 6):
                seq[idx] = seq[idx] + 2
                seq[idx + 1] = seq[idx + 1] - 2
            elif (diff == 8):
                seq[idx - 1] = seq[idx - 1] + 2
                seq[idx] = seq[idx] + 4
                seq[idx + 1] = seq[idx + 1] - 2
            else:
                seq[idx - 1] = seq[idx - 1] + 2
                seq[idx] = seq[idx] + 4
                seq[idx + 1] = seq[idx + 1] - 4
                seq[idx + 2] = seq[idx + 2] - 2
        count1 += 1
        if (count1 > 10):
            break
    if (isprint):
        print('经过连续化处理的次数：' + str(count1))

    while(not is_continue(seq)):
        for idx in range(1, len(seq) - 2):
            diff = seq[idx + 1] - seq[idx]
            if (diff == 0 or diff == 2):
                continue
            elif (diff == 4):
                seq[idx] = seq[idx] + 2
        count2 += 1
        if (count2 > 6):
            break
    if (isprint):
        print('最终序列为：', end=' ')
        print(seq)

    return seq