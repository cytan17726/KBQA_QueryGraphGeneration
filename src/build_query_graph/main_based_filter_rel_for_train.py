import sys
import os
import time
from tqdm import tqdm
from collections import defaultdict
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.utils.data_processing import getQuestions, readEntityLinking, \
                                    addMentionAndDigitToLinking, getQuestionsFromRecallFalse,\
                                    addEntityLinking, getQuestionsWithComplex, readQuestionType,\
                                    preprocessRel
from src.SearchPath import SearchPath
from src.mysql.MysqlConnection import MAX_ANSWER_NUM, MAX_VARIABLE_NUM, MAX_REL_NUM,\
                            MAX_TRIPLES_NUM_INIT, MAX_TRIPLES_NUM_EXPAND
from config.MysqlConfig import CCKSConfig, CCKSConfig2021
import time
from multiprocessing import Process
import random
from src.build_query_graph.rel_sim.cal_que_rel_sim import SimOfQueRel


def buildConflictMatrix(entitylinkings):
    entitys = []
    entityIndex = 0
    mention2entityIndex = {}
    mentions = []
    virtualEntitys = []
    higherOrder = []
    for mentionGroup in entitylinkings:
        mentionTuple = eval(mentionGroup)
        # print(mentionTuple)
        # import pdb; pdb.set_trace()
        if(len(mentionTuple) == 5):
            virtualEntitys.append((mentionTuple, entitylinkings[mentionGroup][0]))
        elif(len(mentionTuple) == 4):
            higherOrder.append((mentionTuple, entitylinkings[mentionGroup][0]))
        else:
            if(mentionGroup not in mention2entityIndex):
                mention2entityIndex[mentionGroup] = []
            mentions.append(mentionGroup)
            for entity in entitylinkings[mentionGroup]:
                # if(len(mentionTuple) > 3):
                #     entitys.append((entity, mentionTuple[3], mentionTuple[4]))
                    # import pdb; pdb.set_trace()
                entitys.append(entity)
                mention2entityIndex[mentionGroup].append(entityIndex)
                entityIndex += 1
    conflictMatrix = []
    for i in range(len(entitys)): # 自己和自己冲突
        conflictMatrix.append([0] * len(entitys))
        conflictMatrix[i][i] = 1
    for mention in mention2entityIndex: # 同一个mention对应的链接实体之间冲突
        entitiyIndex = mention2entityIndex[mention]
        for i in entitiyIndex:
            for j in entitiyIndex:
                conflictMatrix[i][j] = 1
                # print(i, j, entitiyIndex)
    lenMentions = len(mentions)
    for i in range(lenMentions):    # mention之间重叠的冲突
        mention1 = eval(mentions[i])
        # if(mention1[1] == -1):
        #     import pdb; pdb.set_trace()
        #     continue
        for j in range(i+1, lenMentions):
            mention2 = eval(mentions[j])
            if((mention1[1] >= mention2[1] and mention1[1] <= mention2[2]) or\
                mention1[2] >= mention2[1] and mention1[2] <= mention2[2]):
                links1 = mention2entityIndex[str(mention1)]
                links2 = mention2entityIndex[str(mention2)]
                for link1Index in links1:
                    for link2Index in links2:
                        conflictMatrix[link1Index][link2Index] = 1
                        conflictMatrix[link2Index][link1Index] = 1
    # print(virtualEntitys)
    # import pdb; pdb.set_trace()
    return conflictMatrix, entitys, virtualEntitys, higherOrder

def main(entityLinkingFile, questions, outFile, simOfQueRel, que2type,maxhop = 4,mode='train'):
    que2entitylinkings = readEntityLinking(entityLinkingFile)
    '''把 规则识别的约束与其他的EL结果区分开会比较好'''
    que2entitylinkings = addMentionAndDigitToLinking(que2entitylinkings)    # XXX 2022-03-12 1:52 之前没有注释  # 增加 其余约束词
    # import pdb;pdb.set_trace()
    varAndNoentityFile = BASE_DIR + '/data/dataset/CCKS2019_CompType/new_ex.json'  # 数据new 第二次构建的数据
    que2entitylinkings = addEntityLinking(que2entitylinkings, varAndNoentityFile)

    searchPath = SearchPath()
    
    fout = open(outFile, 'w', encoding='utf-8')
    startTime = time.time()
    queTimeDict = defaultdict(list)
    for i, que in enumerate(questions):
        print(i, que)
        # if que != '"兰陵王"是对谁的尊称？':
        #     continue
        # pdb.set_trace()
        qs_time = time.time()
        # import pdb; pdb.set_trace()
        # if('与高娜恩参演同一部作品且属于同一国家的人有哪些？' not in que):
        #     continue
        # if('中的扬州市又称为？' not in que):
        #     continue
        relsDict, topkRelsList = simOfQueRel.predictionRelBasedThreshold(que,mode)   # 是有<>的
        # print(topkRelsList)
        # import pdb; pdb.set_trace()
        if(que in que2type):
            queType = que2type[que]
        else:
            print("question type error")
            import pdb; pdb.set_trace()
        fout.write(que + '\n')
        # import pdb; pdb.set_trace()
        if(que in que2entitylinkings and len(que2entitylinkings[que]) > 0):
            entitylinkings = que2entitylinkings[que]    # 该问句中的实体
            # 冲突矩阵 ||| 实体 ||| 虚拟结点 ||| 更高级比较？
            conflictMatrix, entitys, virtualEntitys, higherOrder = buildConflictMatrix(entitylinkings=entitylinkings)
            # if virtualEntitys:  # XXX
            #     import pdb;pdb.set_trace()
            # else:
            #     continue
            # print(entitys, queType)
            # import pdb;pdb.set_trace()
            if(queType in ['compareJudge','比较推理(计算)']):
                queryGraphs = searchPath.generateQueryGraphComp(entitys=entitys, conflictMatrix=conflictMatrix,\
                                                                candRels = relsDict, queType=queType)
                # import pdb; pdb.set_trace()
            else:
                queryGraphs = searchPath.generateQueryGraph(entitys=entitys, virtualEntitys=virtualEntitys,
                                conflictMatrix=conflictMatrix, higherOrder=higherOrder, candRels = relsDict, candRelsList=topkRelsList, MaxHop=maxhop)
                # import pdb; pdb.set_trace()
        else:
            # continue    # TODO
            print('没有链接到的实体结果:', que) # 筛选过后还只取前10 这怎么办？
            topkRelsList = preprocessRel(topkRelsList[0:100], que)
            # topkRelsList = preprocessRel(topkRelsList[0:100])
            # import pdb; pdb.set_trace()
            # 
            #"<哥哥>" in topkRelsList
            # topkRelsList.index("<哥哥>")
            print(topkRelsList)
            # import pdb; pdb.set_trace()
            queryGraphs = searchPath.generateQueryGraphNoEntity(entitys = [], candRelsList=topkRelsList[0:10])  # 由 无实体逻辑来处理
            # queryGraphs = searchPath.generateQueryGraphNoEntity(entitys = [], candRelsList=topkRelsList[0:20])  # 由 无实体逻辑来处理
            # import pdb; pdb.set_trace()
            # print('error:', que)
        qe_time = time.time()
        queTimeDict[queType].append(qe_time-qs_time)
        for queryGraph in queryGraphs:
            fout.write(queryGraph.serialization() + '\n')
        fout.write('\n')
        # break
    endTime = time.time()
    for quetype in queTimeDict:
        quenum = len(queTimeDict[quetype])
        aveTime = sum(queTimeDict[quetype])/quenum
       
        # import pdb;pdb.set_trace()
        print('{} ave time: {:.6f} s'.format(quetype, aveTime))
    totalTime = endTime - startTime
    print(totalTime, totalTime / 60.0)
    print(outFile)

if __name__ == "__main__":
    topk = 300
    simOfQueRel = SimOfQueRel(topk)
    fileName = BASE_DIR + '/dataset/ccks2021/ccks2021_task13_train.txt'
    questions = getQuestions(fileName)
    # fileName = BASE_DIR + '/data/candidates/9727_analysis_recall_false.txt'
    # questions = getQuestionsFromRecallFalse(fileName)
    # fileName = BASE_DIR + '/src/question_classification/build_data/que_newType_train.txt'
    fileName = BASE_DIR + '/src/question_classification/build_data/que_newType_dev.txt'
    questions = getQuestionsWithComplex(fileName)
    que2type = readQuestionType(fileName)
    N = 30
    # questions = questions[0: 2]
    # N = 1
    numPerClass = len(questions) // N
    fileNormal = '/data/candidates/temp/4hop4_dev_with_multiType_filter_rel_top' + str(topk) + '_query_graph_cands_1230_expand10_add_'
    entityLinkingFile = BASE_DIR + '/data/entitylinking/1031_EL_train.json'
    if(N == 1):
        outFile = BASE_DIR + fileNormal + str(0) + '_' + str(MAX_REL_NUM) + '_'\
                + str(MAX_VARIABLE_NUM) + '_' + str(MAX_ANSWER_NUM) + '_' + str(MAX_TRIPLES_NUM_INIT) + '_' + \
                    str(MAX_TRIPLES_NUM_EXPAND) + '.txt'
        main(entityLinkingFile, questions, outFile, simOfQueRel, que2type)
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
        # import pdb; pdb.set_trace()
        print('进程个数：', len(lines))
        if(len(lines) <= 1):
            break
        time.sleep(20)
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