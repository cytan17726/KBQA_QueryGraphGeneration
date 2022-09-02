import sys
import os
import time
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.utils.data_processing import getQuestions, readEntityLinking, getQuestionsFromTestset,\
                                addMentionAndDigitToLinking, readQuestionType, getQuestionsWithComplex
from src.SearchPath import SearchPath
from src.mysql.MysqlConnection import MAX_ANSWER_NUM, MAX_VARIABLE_NUM, MAX_REL_NUM,\
                            MAX_TRIPLES_NUM_INIT, MAX_TRIPLES_NUM_EXPAND
from src.run import buildConflictMatrix
import time
from multiprocessing import Process
import random
from src.build_query_graph.rel_sim.cal_que_rel_sim import SimOfQueRel, RelRanker
from build_query_graph.main_based_filter_rel_for_train import main



if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    device = torch.device('cuda',0)
    topk = 2000
    simOfQueRel = SimOfQueRel(topk)
    
    N = 10
    '''CCKS2019'''
    # fileName = BASE_DIR + '/data/dataset/CCKS2019/que_newType_test.txt'
    # fileNormal = '/data/candidates/temp/CCKS2019_Ours_test_' + str(topk) + '_0429DNS_'
    # entityLinkingFile = BASE_DIR + '/data/dataset/CCKS2019/EL_test_实体识别结果_加长串_baike_19Order_0624reberta_top3.json' # pj师兄 EL取 Top3
    '''CCKS2019-Comp'''
    fileName = BASE_DIR + '/data/dataset/CCKS2019_CompType/que_newType_test.txt'
    fileNormal = '/data/candidates/temp/CCKS2019_Comp_Ours_test_' + str(topk) + '_0429DNS_'
    entityLinkingFile = BASE_DIR + '/data/dataset/CCKS2019/EL_test_实体识别结果_加长串_baike_19Order_0624reberta_top3.json' # pj师兄 EL取 Top3

    questions = getQuestionsWithComplex(fileName)
    que2type = readQuestionType(fileName)
    numPerClass = len(questions) // N
    
    if(N == 1):
        outFile = BASE_DIR + fileNormal + str(0) + '_' + str(MAX_REL_NUM) + '_'\
                + str(MAX_VARIABLE_NUM) + '_' + str(MAX_ANSWER_NUM) + '_' + str(MAX_TRIPLES_NUM_INIT) + '_' + \
                    str(MAX_TRIPLES_NUM_EXPAND) + '.txt'
        main(entityLinkingFile, questions, outFile, simOfQueRel, que2type, mode='test')
    else:
        for i in range(N):
            outFile = BASE_DIR + fileNormal + str(i) + '_' + str(MAX_REL_NUM) + '_'\
                + str(MAX_VARIABLE_NUM) + '_' + str(MAX_ANSWER_NUM) + '_' + str(MAX_TRIPLES_NUM_INIT) + '_' + \
                    str(MAX_TRIPLES_NUM_EXPAND) + '.txt'
            if(i != N - 1):
                # print(len(questions[numPerClass * i: numPerClass * (i + 1)]))
                p1 = Process(target=main, args=(entityLinkingFile, questions[numPerClass * i: numPerClass * (i + 1)], outFile, simOfQueRel, que2type, ))
                p1.start()
            else:
                # print(len(questions[numPerClass * i:]))
                p1 = Process(target=main, args=(entityLinkingFile, questions[numPerClass * i: ], outFile, simOfQueRel, que2type, ))
                p1.start()
    while(True):
        currentFile = __file__
        # print(currentFile)
        shellF = os.popen("ps -au| grep 'python " + currentFile + "'|grep -v grep")
        lines = shellF.readlines()
        print('进程个数：', len(lines))
        if(len(lines) <= 1):
            break
        time.sleep(5)
    newFileNormal = fileNormal.replace('/temp', '')
    outFile = BASE_DIR + newFileNormal + str(MAX_REL_NUM) + '_'\
            + str(MAX_VARIABLE_NUM) + '_' + str(MAX_ANSWER_NUM) + '_' + str(MAX_TRIPLES_NUM_INIT) + '_' + \
                str(MAX_TRIPLES_NUM_EXPAND) + '.txt'
    fout = open(outFile, 'w', encoding='utf-8')
    for i in range(N):
        fileName = BASE_DIR + fileNormal + str(i) + '_' + str(MAX_REL_NUM) + '_'\
            + str(MAX_VARIABLE_NUM) + '_' + str(MAX_ANSWER_NUM) + '_' + str(MAX_TRIPLES_NUM_INIT) + '_' + \
                str(MAX_TRIPLES_NUM_EXPAND) + '.txt'
        fread = open(fileName, 'r', encoding='utf-8')
        for line in fread:
            fout.write(line)
    print(outFile)
    print('success')