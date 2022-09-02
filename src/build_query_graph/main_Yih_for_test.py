import sys
import os
import time

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
from src.build_query_graph.rel_sim.cal_que_rel_sim import SimOfQueRel
from main_Yih_for_train import main




if __name__ == "__main__":
    topk = 300000
    mode = 'test'
    simOfQueRel = SimOfQueRel(topk)
    N = 4

    '''CCKS2019'''
    que_file = BASE_DIR + '/data/dataset/CCKS2019/que_newType_test.txt'  
    out_file_name = '/data/candidates/temp/CCKS2019_Yhi_test_top' + str(topk) + '_0429DNS_'
    entityLinkingFile = BASE_DIR + '/data/dataset/CCKS2019/EL_test_实体识别结果_加长串_baike_19Order_0624reberta_top3.json'
    '''CCKS2019-Comp'''
    # que_file = BASE_DIR + '/data/dataset/CCKS2019_CompType/que_newType_test.txt'  
    # out_file_name = '/data/candidates/temp/CCKS2019_Comp_Yhi_test_top' + str(topk) + '_0429DNS_'
    # entityLinkingFile = BASE_DIR + '/data/dataset/CCKS2019/EL_test_实体识别结果_加长串_baike_19Order_0624reberta_top3.json'

    print('topk:{},mode:{},N:{}\nqueFile:{}\noutput:{}\nELFile:{}'.format(topk,mode,N,que_file,out_file_name,entityLinkingFile))
    questions = getQuestionsWithComplex(que_file)
    que2type = readQuestionType(que_file)
    numPerClass = len(questions) // N    
    
    # exit()
    if(N == 1):
        outFile = BASE_DIR + out_file_name + str(0) + '_' + str(MAX_REL_NUM) + '_'\
                + str(MAX_VARIABLE_NUM) + '_' + str(MAX_ANSWER_NUM) + '_' + str(MAX_TRIPLES_NUM_INIT) + '_' + \
                    str(MAX_TRIPLES_NUM_EXPAND) + '.txt'
        main(entityLinkingFile, questions, outFile, simOfQueRel, que2type,mode)
    else:
        for i in range(N):
            outFile = BASE_DIR + out_file_name + str(i) + '_' + str(MAX_REL_NUM) + '_'\
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
    newFileNormal = out_file_name.replace('/temp', '')
    outFile = BASE_DIR + newFileNormal + str(MAX_REL_NUM) + '_'\
            + str(MAX_VARIABLE_NUM) + '_' + str(MAX_ANSWER_NUM) + '_' + str(MAX_TRIPLES_NUM_INIT) + '_' + \
                str(MAX_TRIPLES_NUM_EXPAND) + '.txt'
    fout = open(outFile, 'w', encoding='utf-8')
    for i in range(N):
        fileName = BASE_DIR + out_file_name + str(i) + '_' + str(MAX_REL_NUM) + '_'\
            + str(MAX_VARIABLE_NUM) + '_' + str(MAX_ANSWER_NUM) + '_' + str(MAX_TRIPLES_NUM_INIT) + '_' + \
                str(MAX_TRIPLES_NUM_EXPAND) + '.txt'
        fread = open(fileName, 'r', encoding='utf-8')
        for line in fread:
            fout.write(line)
    print(outFile)
    print('success')