# 生成子路径
import sys
import os
import json
from typing import Dict
import re
from typing import List, Dict, Tuple
import copy

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
from src.utils.data_processing import readCandsInfo
from src.utils.cal_f1 import f1Item, processAnswerEntity, processAnswerValue
from src.utils.data_processing import readTrainFile, readQue2AnswerFromTestset,\
                            readTrainFileForSparql

index2Label = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']

def getPathTriplesSubPath(querygraphInfo):
    pathTriples = querygraphInfo["path"]
    variableNodeIds = querygraphInfo["variableNodeIds"]
    answerNodeId = querygraphInfo["answerNodeId"]
    answers = querygraphInfo["answer"].split('\t')
    pathKey = []
    triples = []
    for i in range(len(pathTriples)):
        if(i == answerNodeId):
            # tripleStr += '<answer>'
            triples.append("#" + answers[0] + "#")
            # import pdb; pdb.set_trace()
        elif(i not in variableNodeIds and i <= 2):
            # tripleStr += pathTriples[i]
            triples.append(pathTriples[i])
        elif(i % 3 == 1): # 表示关系位置
            # tripleStr += pathTriples[i]
            triples.append(pathTriples[i])
        else:
            if(i in variableNodeIds):
                index = variableNodeIds.index(i)
            else:
                thTriple = (i // 3)
                index = thTriple - 1
            triples.append('<' + index2Label[index] + '>')
    # if(len(pathKey) > 0):
    #     print(pathKey)
    #     import pdb; pdb.set_trace()
    return triples

def getEntityConstrainSubPath(querygraphInfo):
    pathKey = []
    triples = []
    entityConstrainTriples = querygraphInfo['entityPath']
    entityIDs = querygraphInfo['entityIDs']
    entityConstrainIDs = querygraphInfo["entityConstrainIDs"]
    variableNodeIds = querygraphInfo['variableNodeIds']
    for orderTriple, triple in enumerate(entityConstrainTriples):
        for orderIndex in range(len(triple)):
            i = orderTriple
            j = orderIndex
            if(j == querygraphInfo['entityIDs'][i] or j % 3 == 1):
                # tripleStr += triple[j]
                triples.append(triple[j])
            else:
                thMainpath = entityConstrainIDs[i]
                if(thMainpath in variableNodeIds):
                    index = variableNodeIds.index(thMainpath)
                else:
                    thTriple = (thMainpath // 3) # 确定变量是第几个三元组里的，取值为1,2,3
                    # index = thTriple // 2 - 1 # 修改bug
                    index = thTriple - 1
                triples.append('<' + index2Label[index] + '>')
    return triples

def getVirtualEntityConstrainSubPath(querygraphInfo):
    pathKey = []
    triples = []
    virtualConstrainTriples = querygraphInfo['virtualConstrainTriples']
    virtualConstrainIDs = querygraphInfo['virtualConstrainIDs']
    variableNodeIds = querygraphInfo['variableNodeIds']
    for orderTriple, triple in enumerate(virtualConstrainTriples):
        thMainPath = virtualConstrainIDs[orderTriple]
        # import pdb; pdb.set_trace()
        if(thMainPath in variableNodeIds):
            index = variableNodeIds.index(thMainPath)
        else:
            thTriple = (thMainPath // 3)
            index = thTriple - 1
        # index = thVariable // 2
        triples.append('<' + index2Label[index] + '>')
        triples.append(triple[1])
        triples.append(triple[2])
    return triples


def getHigherOrderConstrainSubPath(querygraphInfo):
    ''''
    功能：获取最高级路径对应的key
    '''
    pathKey = []
    triples = []
    higherOrderTriples = querygraphInfo['higherOrderTriples']
    higherOrderConstrainIDs = querygraphInfo['higherOrderConstrainIDs']
    variableNodeIds = querygraphInfo['variableNodeIds']
    for orderTriple, triple in enumerate(higherOrderTriples):
        thMainPath = higherOrderConstrainIDs[orderTriple]
        # import pdb; pdb.set_trace()
        if(thMainPath in variableNodeIds):
            index = variableNodeIds.index(thMainPath)
        else:
            thTriple = (thMainPath // 3)
            index = thTriple - 1
        # index = thVariable // 2
        triples.append('<' + index2Label[index] + '>')
        triples.append(triple[1])
        triples.append(triple[2])
    return triples


def getBasePathTriplesSubPath(querygraphInfo):
    ''''
    功能：获取基础路径对应的key
    '''
    basePathTriples = querygraphInfo["basePathTriples"]
    basePathVariableNodeIds = querygraphInfo["basePathVariableIds"]
    # answerNodeId = querygraphInfo["answerNodeId"]
    pathKey = []
    triples = []
    for order, pathTriples in enumerate(basePathTriples): # bug:缺少聚合信息
        variableNodeIds = basePathVariableNodeIds[order]
        if(len(pathTriples) == 1):
            pathTriples = pathTriples[0]
        for i in range(len(pathTriples)):
            if(i not in variableNodeIds and i <= 2):
                # tripleStr += pathTriples[i]
                triples.append(pathTriples[i])
            elif(i % 3 == 1): # 表示关系位置
                # tripleStr += pathTriples[i]
                triples.append(pathTriples[i])
            else:
                if(i in variableNodeIds):
                    index = variableNodeIds.index(i)
                else:
                    thTriple = (i // 3)
                    index = thTriple - 1
                triples.append('<' + index2Label[index] + '>')
    if(len(triples) == 3):
        import pdb; pdb.set_trace()
    return triples

def getRelConstrainSubPath(querygraphInfo):
    '''
    实现和baseKey一致
    '''
    relConstrainTriples = querygraphInfo["relConstrainTriples"]
    relConstrainIds = querygraphInfo["relConstrainIDs"]
    # answerNodeId = querygraphInfo["answerNodeId"]
    pathKey = []
    triples = []
    for order, pathTriples in enumerate(relConstrainTriples): # bug:缺少聚合信息
        variableNodeIds = relConstrainIds[order]
        for i in range(len(pathTriples)):
            if(i not in variableNodeIds and i <= 2):
                triples.append(pathTriples[i])
            elif(i % 3 == 1): # 表示关系位置
                triples.append(pathTriples[i])
            else:
                if(i in variableNodeIds):
                    index = variableNodeIds.index(i)
                else:
                    thTriple = (i // 3)
                    index = thTriple - 1
                triples.append('<' + index2Label[index] + '>')
    return triples