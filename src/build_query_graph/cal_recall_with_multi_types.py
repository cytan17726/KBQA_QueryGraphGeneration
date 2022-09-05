from collections import defaultdict
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.utils.data_processing import readTrainFile, readCands, \
    addQue2AnswerOtherTypes, getQuestionsAndTypes
from src.utils.cal_f1 import f1Item, processAnswerEntity, processAnswerValue

if __name__ == "__main__":
    '''评价 F1_gen, 平均候选数量，最大候选数量'''
    
    fileName = BASE_DIR + '/data/dataset/CCKS2019/test.txt'
    que2answer = readTrainFile(fileName)
    
    fileName = BASE_DIR + '/data/dataset/CCKS2019_CompType/new_ex.json' # XXX
    que2answer = addQue2AnswerOtherTypes(fileName, que2answer)
    questions, que2type = getQuestionsAndTypes(fileName)

    quetype2true = {}
    quetype2all = {}
    
    '''CCKS2019'''
    # candsFile = BASE_DIR + '/data/candidates/CCKS2019_Yhi_test_top300000_0429DNS_100_1000_10000_100000_1000.txt'    # 85.13
    # candsFile = BASE_DIR + '/data/candidates/CCKS2019_Luo_test_top300000_0429DNS_100_1000_10000_100000_1000.txt'    # 86.23
    # candsFile = BASE_DIR + '/data/candidates/CCKS2019_Ours_test_2000_0429DNS_100_1000_10000_100000_1000.txt'    # 89.47
    '''CCKS2019-Comp'''
    # candsFile = BASE_DIR + '/data/candidates/CCKS2019_Comp_Yhi_test_top300000_0429DNS_100_1000_10000_100000_1000.txt'   # 71.07
    # candsFile = BASE_DIR + '/data/candidates/CCKS2019_Comp_Luo_test_top300000_0429DNS_100_1000_10000_100000_1000.txt'   # 71.93
    candsFile = BASE_DIR + '/data/candidates/CCKS2019_Comp_Ours_test_2000_0429DNS_100_1000_10000_100000_1000.txt'   # 86.91
    
    que2cands = readCands(candsFile)
    macroF1 = 0.0
    num = 0
    trueNum = 0
    scoreThreshold = 0.6
    Max_can_num = 0
    Sum_can_num = 0
    type2can = defaultdict(list)    # 按类型统计候选数量
    # type2F1 = defaultdict(list) # 按类型统计F1值
    que_num = 0     # 问句总数
    for que in que2answer:
        # import pdb; pdb.set_trace()
        goldAnswer = que2answer[que]
        # goldAnswer = processAnswer(goldAnswer)
        maxF1 = 0.0
        best_pred_ans = ''
        best_pred_path = ''
        if(que in que2cands):
            cands = que2cands[que]
            que_num += 1
            can_num = len(cands)
            Sum_can_num += can_num
            if que in que2type:
                type2can[que2type[que]].append(can_num)
            else:
                type2can['else'].append(can_num)
            if can_num > Max_can_num:   # 更大？更新
                Max_can_num = can_num
            # import pdb;pdb.set_trace()
            for cand in cands:
                predictionAnswer = cand[1]
                newGoldAnswer = processAnswerEntity(goldAnswer)
                predictionAnswer = processAnswerEntity(predictionAnswer)
                f1 = f1Item(newGoldAnswer, predictionAnswer)
                if(f1 < 0.00001):
                    newGoldAnswer = processAnswerValue(goldAnswer)
                    f1 = f1Item(newGoldAnswer, predictionAnswer)
                if(f1 > maxF1):
                    maxF1 = f1
                    best_pred_ans = cand[1]
                    best_pred_path = cand[0]
                    # import pdb;pdb.set_trace()
                    # print(cand)
            if(maxF1 < scoreThreshold):
                num += 1
            elif(maxF1 >= scoreThreshold):
                trueNum += 1
            else:
                print('error', maxF1)
            macroF1 += maxF1
            if(que in que2type):
                if(que2type[que] not in quetype2all):
                    quetype2all[que2type[que]] = 0
                    quetype2true[que2type[que]] = 0
                # print(que, que2type[que])
                quetype2all[que2type[que]] += 1
                quetype2true[que2type[que]] += maxF1
            else:
                if('else' not in quetype2all):
                    quetype2all['else'] = 0
                    quetype2true['else'] = 0
                quetype2all['else'] += 1
                quetype2true['else'] += maxF1

    print('false:', num, 'true:', trueNum)
    for queType in quetype2true:
        print(queType, quetype2true[queType] * 1.0 / quetype2all[queType])
    print(quetype2true, quetype2all)
    print('que num: %d'%que_num)
    print('Max can num: %d'%Max_can_num)
    print('ave can num: {} = {} / {}'.format(Sum_can_num/len(que2cands), Sum_can_num, len(que2cands)))
    print('F1_gen: {} = {} / {}'.format(macroF1 / len(que2cands), macroF1, len(que2cands)))
    for quetype in type2can:
        data = type2can[quetype]
        ave_num = sum(data)/len(data)
        print('{}\tmax_can: {}, ave_num: {:.4f}'.format(quetype,max(data),ave_num))


