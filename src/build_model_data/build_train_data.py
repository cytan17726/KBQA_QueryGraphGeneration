import sys
import os
import random
import json
random.seed(100)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.utils.data_processing import readTrainFile, readCands, \
    readEntityLinking, readTrainFileForSparql, readQue2AnswerFromTestset
from src.utils.cal_f1 import f1Item, processAnswerEntity, processAnswerValue
from src.utils.data_processing import readCandsInfo, getQuestions

MAX_LENGTH = 150

def write2file(trainData, fileName):
    with open(fileName, 'w', encoding='utf-8') as fout:
        for trainGroup in trainData:
            for item in trainGroup:
                fout.write(str((item[0], json.dumps(item[1], ensure_ascii=False))) + '\n')


def buildTrainDataNoCopyPos():
    fileName = BASE_DIR + '/dataset/ccks2021/ccks2021_task13_train.txt'
    questions = getQuestions(fileName)
    random.shuffle(questions)
    devQuestions = questions[0: int(len(questions) * 0.2)]
    devQuestionsDic = {item: 1 for item in devQuestions}
    # fileFeature = '2_1031'
    # fileFeature = '5_1101'
    # fileFeature = 'y_6_4_1102'
    # outFileFeature = 'y_6_4_1102'
    # fileFeature = 'norm_element_y_6_4_1102'
    # outFileFeature = 'norm_element_y_8_1_1105'
    # fileFeature = 'noanswer_y_8_2_1105'
    # outFileFeature = 'noanswer_y_8_2_1105'

    fileFeature = '9727_pos_f109_neg03_0114'
    outFileFeature = 'pos_f109_neg03_sparql_drop_allpath'
    candsFile = BASE_DIR + '/data/candidates/seq/trainset_seq_' + fileFeature + '.txt'
    ### stagg 方法生成的结果
    # fileFeature = '1115'
    # outFileFeature = 'stagg_1115'
    # candsFile = BASE_DIR + '/data/candidates/seq/stagg_9342_trainset_seq_' + fileFeature + '.txt'
    #######################
    que2cands = readCandsInfo(candsFile)
    que2posCands = {}
    que2negCands = {}
    for que in que2cands:
        if(que not in que2posCands):
            que2posCands[que] = []
            que2negCands[que] = []
        cands = que2cands[que]
        for cand in cands:
            if(cand["label"] == 1):
                que2posCands[que].append(cand)
            else:
                que2negCands[que].append(cand)
        random.shuffle(que2posCands[que])
        random.shuffle(que2negCands[que])
    devData = []
    devNeg = 100
    for que in que2posCands:
        if(que in devQuestionsDic):
            temp = []
            for cand in que2posCands[que]:
                temp.append((que, cand))
            for cand in que2negCands[que][0: devNeg]:
                temp.append((que, cand))
            devData.append(temp)
    devFile = BASE_DIR + '/data/train_data/' + outFileFeature + '/dev_20%_neg' + str(devNeg) + '.txt'
    write2file(trainData=devData, fileName=devFile)

    # negNums = [5, 10, 15, 20]
    negNums = [120, 100, 80, 60, 40, 20, 15, 10, 5]
    for negNum in negNums:
        trainData = []
        outFile = BASE_DIR + '/data/train_data/' + outFileFeature + '/train_neg' + str(negNum) + '.txt'
        for index, que in enumerate(que2posCands):
            if(que in devQuestionsDic):
                continue
            if(index % 100 == 0):
                print(negNum, index)
            for posCand in que2posCands[que]:
                lenNegCands = len(que2negCands[que])
                if(lenNegCands == 0):
                    print('0:', que)
                    break
                temp = []
                temp.append((que, posCand))
                for i in range(negNum):
                    temp.append((que, que2negCands[que][i % lenNegCands]))
                trainData.append(temp)
        print('train group num:', len(trainData))
        write2file(trainData=trainData, fileName=outFile)



if __name__ == "__main__":
    buildTrainDataNoCopyPos()

    # fileName = BASE_DIR + '/dataset/ccks2021/ccks2021_task13_train.txt'
    # questions = getQuestions(fileName)
    # random.shuffle(questions)
    # devQuestions = questions[0: int(len(questions) * 0.2)]
    # devQuestionsDic = {item: 1 for item in devQuestions}
    # # fileFeature = '2_1031'
    # seqFileFeature = '3_1101'
    # fileFeature = '7_1101'
    # candsFile = BASE_DIR + '/data/candidates/seq/trainset_seq_' + seqFileFeature + '.txt'
    # que2cands = readCandsInfo(candsFile)
    # que2posCands = {}
    # que2negCands = {}
    # for que in que2cands:
    #     if(que not in que2posCands):
    #         que2posCands[que] = []
    #         que2negCands[que] = []
    #     cands = que2cands[que]
    #     for cand in cands:
    #         if(cand["label"] == 1):
    #             que2posCands[que].append(cand)
    #         else:
    #             que2negCands[que].append(cand)
    #     random.shuffle(que2posCands[que])
    #     random.shuffle(que2negCands[que])
    # # devData = []
    # # devNeg = 50
    # # for que in que2posCands:
    # #     if(que in devQuestionsDic):
    # #         temp = []
    # #         for cand in que2posCands[que]:
    # #             temp.append((que, cand))
    # #         for cand in que2negCands[que][0: devNeg]:
    # #             temp.append((que, cand))
    # #         devData.append(temp)
    # # devFile = BASE_DIR + '/data/train_data/' + fileFeature + '/dev_20%_neg' + str(devNeg) + '.txt'
    # # write2file(trainData=devData, fileName=devFile)

    # negNums = [5, 10, 15, 20]
    # # negNums = [5]
    # copyNums = [2, 4, 6]
    # # copyNums = [2]
    # for negNum in negNums:
    #     for copyNum in copyNums:
    #         trainData = []
    #         outFile = BASE_DIR + '/data/train_data/' + fileFeature + '/train_neg_' + str(negNum) + '_copy_' + str(copyNum) + '.txt'
    #         for index, que in enumerate(que2posCands):
    #             if(que in devQuestionsDic):
    #                 continue
    #             if(index % 100 == 0):
    #                 print(negNum, index)
    #             for posCand in que2posCands[que]:
    #                 lenNegCands = len(que2negCands[que])
    #                 if(lenNegCands == 0):
    #                     print('0:', que)
    #                     break
    #                 for thCopyNum in range(copyNum):
    #                     temp = []
    #                     temp.append((que, posCand))
    #                     for i in range(negNum):
    #                         temp.append((que, que2negCands[que][(thCopyNum * negNum + i) % lenNegCands]))
    #                     trainData.append(temp)
    #             # break
    #         print('train group num:', len(trainData))
    #         write2file(trainData=trainData, fileName=outFile)
    # import pdb; pdb.set_trace()