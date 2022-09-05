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
from src.utils.data_processing import readCandsInfo
from src.build_model_data.build_train_data import write2file

            

if __name__ == "__main__":
    
    # fileFeature = 'new/testset_seq_nopad_multiTypes_0305'
    # outFileFeature = '0414_ccks2019Only_Ours_19m2e_noML'
    outFileFeature = '0607_A_EL3'
    candsFile = BASE_DIR + '/data/candidates/0525_A/seq/seq_0602_CCKS2022Dev_pred_baseline_ccks2022_1018reberta300_allHard_122-22_0524.txt'
    candsFile = BASE_DIR + '/data/candidates/0524_self/seq/testset_seq_nopad_122-22_0607_3_300.txt'
    candsFile = BASE_DIR + '/data/candidates/0607_self_EL3/seq/0607_testset_seq_nopad_CCKS2022_self_testset_maxhop4_topk2000_allHard_122-22_EL3.txt'
    candsFile = BASE_DIR + '/data/candidates/0524_self/seq/Seq_0524_CCKS2022_self_test-gold_300_allHard.txt'
    

    que2cands = readCandsInfo(candsFile)
    que2posCands = {}
    que2negCands = {}
    trainData = []
    for que in que2cands:
        if(que not in que2posCands):
            que2posCands[que] = []
            que2negCands[que] = []
        cands = que2cands[que]
        temp = []
        for cand in cands:
            temp.append((que, cand))
        trainData.append(temp)
    outFile = BASE_DIR + '/data/train_data/' + outFileFeature + '/testset_model_data_A_noAns_122-22_0602.txt'
    outFile = BASE_DIR + '/data/train_data/' + outFileFeature + '/testset_model_data_A_noAns_122-22_0607_3_300.txt'
    outFile = BASE_DIR + '/data/train_data/' + outFileFeature + '/testset_model_data_self_3_2k_EL3_0607.txt'
    outFile = BASE_DIR + '/data/train_data/' + outFileFeature + '/testset_model_data_A_4_2000_EL3_0607.txt'
    outFile = BASE_DIR + '/data/train_data/' + outFileFeature + '/testset_model_data_self_4_300_ELgold_0610.txt'
    # outFile = BASE_DIR + '/data/train_data/' + outFileFeature + '/testset_model_data_pjzhangTop3_0429DNSTop300NoCheat.txt'
    write2file(trainData=trainData, fileName=outFile)
    # import pdb; pdb.set_trace()