import sys
import os
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.utils.data_processing import getQuestions, readEntityLinking
from src.SearchPath import SearchPath
from src.mysql.MysqlConnection import MAX_ANSWER_NUM, MAX_VARIABLE_NUM, MAX_REL_NUM,\
                            MAX_TRIPLES_NUM_INIT, MAX_TRIPLES_NUM_EXPAND

import time
from multiprocessing import Process
import random


def buildConflictMatrix(entitylinkings):
    entitys = []
    entityIndex = 0
    mention2entityIndex = {}
    mentions = []
    for mentionGroup in entitylinkings:
        if(mentionGroup not in mention2entityIndex):
            mention2entityIndex[mentionGroup] = []
        mentions.append(mentionGroup)
        for entity in entitylinkings[mentionGroup]:
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
    # import pdb; pdb.set_trace()
    return conflictMatrix, entitys


def main(questions, outFile):
    # entityLinkingFile = BASE_DIR + '/data/entitylinking/Entity_linking_PlusSubstrPair_1019.json'
    entityLinkingFile = BASE_DIR + '/data/entitylinking/1031_EL_train.json'
    que2entitylinkings = readEntityLinking(entityLinkingFile)
    searchPath = SearchPath()
    
    fout = open(outFile, 'w', encoding='utf-8')
    startTime = time.time()
    for i, que in enumerate(questions[:]):
        print(i, que)
        # if('武汉大学出了哪些科学家' not in que):
        #     continue
        # if('中的扬州市又称为？' not in que):
        #     continue
        
        if(que in que2entitylinkings):
            fout.write(que + '\n')
            entitylinkings = que2entitylinkings[que]
            # entitys = []
            # for mentionGroup in entitylinkings:
            #     entitys.extend(entitylinkings[mentionGroup])
            # import pdb; pdb.set_trace()
            conflictMatrix, entitys = buildConflictMatrix(entitylinkings=entitylinkings)
            
            queryGraphs = searchPath.searchQueryGraph(entitys=entitys,
                            conflictMatrix=conflictMatrix)
            for queryGraph in queryGraphs:
                fout.write(queryGraph.serialization() + '\n')
            fout.write('\n')
        else:
            print('error:', que)
    endTime = time.time()
    totalTime = endTime - startTime
    print(totalTime, totalTime / 60.0)
    print(outFile)

if __name__ == "__main__":
    
    fileName = BASE_DIR + '/dataset/ccks2021/ccks2021_task13_train.txt'
    questions = getQuestions(fileName)
    N = 50
    questions = questions[784:785]
    N = 1
    numPerClass = len(questions) // N
    fileNormal = '/data/candidates/temp/query_graph_cands_1110_expand10_add_'
    if(N == 1):
        outFile = BASE_DIR + fileNormal + str(0) + '_' + str(MAX_REL_NUM) + '_'\
                + str(MAX_VARIABLE_NUM) + '_' + str(MAX_ANSWER_NUM) + '_' + str(MAX_TRIPLES_NUM_INIT) + '_' + \
                    str(MAX_TRIPLES_NUM_EXPAND) + '.txt'
        main(questions, outFile)
    else:
        for i in range(N):
            outFile = BASE_DIR + fileNormal + str(i) + '_' + str(MAX_REL_NUM) + '_'\
                + str(MAX_VARIABLE_NUM) + '_' + str(MAX_ANSWER_NUM) + '_' + str(MAX_TRIPLES_NUM_INIT) + '_' + \
                    str(MAX_TRIPLES_NUM_EXPAND) + '.txt'
            if(i != N - 1):
                # print(len(questions[numPerClass * i: numPerClass * (i + 1)]))
                p1 = Process(target=main, args=(questions[numPerClass * i: numPerClass * (i + 1)], outFile, ))
                p1.start()
            else:
                # print(len(questions[numPerClass * i:]))
                p1 = Process(target=main, args=(questions[numPerClass * i: ], outFile, ))
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