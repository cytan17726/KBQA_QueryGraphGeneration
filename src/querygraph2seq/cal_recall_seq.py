import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.utils.data_processing import readQue2AnswerFromTestset, readSequences, \
    readEntityLinking, readTrainFileForSparql
from src.utils.cal_f1 import f1Item

if __name__ == "__main__":
    fileName = BASE_DIR + '/dataset/ccks2021/testset_zhengqiu.json'
    que2answer = readQue2AnswerFromTestset(fileName)
    candsFile = BASE_DIR + '/data/candidates/seq/testset_seq_y_6_5_1102.txt'
    # candsFile = BASE_DIR + '/data/candidates/seq/testset_seq_5_1101.txt'
    # candsFile = BASE_DIR + '/data/candidates/seq/testset_seq_4_1101.txt'
    # candsFile = BASE_DIR + '/data/candidates/seq/testset_seq_3_1101.txt'
    # candsFile = BASE_DIR + '/data/candidates/seq/testset_seq_2_1031.txt'
    que2cands = readSequences(candsFile)

    # entityLinkingFile = BASE_DIR + '/data/entitylinking/Entity_linking_PlusSubstrPair_1019.json'
    # que2entitylinkings = readEntityLinking(entityLinkingFile)

    macroF1 = 0.0
    num = 0
    trueNum = 0
    scoreThreshold = 0.6
    for que in que2answer:
    # for que in que2cands:
        # import pdb; pdb.set_trace()
        cands = que2cands[que]
        maxF1 = 0.0
        for cand in cands:
            f1 = cand[2]
            if(f1 > maxF1):
                maxF1 = f1
        # print(maxF1)
        if(maxF1 < scoreThreshold):
            num += 1
            print(que)
        elif(maxF1 >= scoreThreshold):
            trueNum += 1
        else:
            print('error', maxF1)
        macroF1 += maxF1
        # if('第六天魔王"有哪些成就' in que):
        #     import pdb; pdb.set_trace()
                # import pdb; pdb.set_trace()
    print('false:', num, 'true:', trueNum)
    print(macroF1, len(que2answer), macroF1 / len(que2answer))

