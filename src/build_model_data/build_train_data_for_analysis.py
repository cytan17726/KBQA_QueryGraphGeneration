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
    # fileName = BASE_DIR + '/data/data_from_pzhou/trainset/train_query_triple.txt'
    # questions = getQuestions(fileName)
    # devQuestionsDic = {}
    # fileFeature = '9734_nopad_multiTypes_0115'
    # outFileFeature = 'multi_types_0116_nopad'
    # fileFeature = '9734_multiTypes_0115'
    # outFileFeature = 'multi_types_0116'
    # fileFeature = 'nopad_multiTypes_0305'   # 复现使用
    outFileFeature = '0413_ccks2019_only_Ours'
    candsFile = BASE_DIR + '/data/candidates/seq/0413_ccks2019_only_Ours/trainset_seq_nopad_top300.txt'
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
    negNums = [30]
    # negNums = [120, 100, 80, 60, 40, 20, 15, 10, 5]
    for negNum in negNums:
        trainData = []
        outFile = BASE_DIR + '/data/train_data/' + outFileFeature + '/train_neg' + str(negNum) + '.txt'
        for index, que in enumerate(que2posCands):
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

def buildDevDataNoCopyPos():
    # fileName = BASE_DIR + '/data/data_from_pzhou/devset/dev_query_triple.txt'
    # devQuestions = getQuestions(fileName)
    # devQuestionsDic = {item: 1 for item in devQuestions}
    # fileFeature = '9708_nopad_multiTypes_0115'
    # outFileFeature = 'multi_types_0116_nopad'
    # fileFeature = '9708_multiTypes_0115'
    # outFileFeature = 'multi_types_0116'
    # fileFeature = 'new/devset_seq_nopad_multiTypes_0305'
    outFileFeature = '0414_ccks2019Only_Ours_19m2e_noML'
    candsFile = BASE_DIR + '/data/candidates/seq/0414_ccks2019Only_Ours_19m2e_noML/devset_seq_nopad_top300.txt'
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
        temp = []
        for cand in que2posCands[que]:
            temp.append((que, cand))
        for cand in que2negCands[que][0: devNeg]:
            temp.append((que, cand))
        devData.append(temp)
    devFile = BASE_DIR + '/data/train_data/' + outFileFeature + '/dev_20%_neg' + str(devNeg) + '.txt'
    write2file(trainData=devData, fileName=devFile)


if __name__ == "__main__":
    buildTrainDataNoCopyPos()
    # buildDevDataNoCopyPos()
    pass