import sys
import os
import json
from typing import Dict
import re
from typing import List, Dict, Tuple
import copy
# from querygraph_seq.selection_cands import isTrueEntityPath

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
from src.utils.data_processing import readCandsInfo
from src.utils.cal_f1 import f1Item, processAnswerEntity, processAnswerValue
from src.utils.data_processing import readTrainFile, readQue2AnswerFromTestset,\
                            readTrainFileForSparql, addQue2AnswerOtherTypes,\
                            addQue2GoldTriples, addQue2GoldTriples2
from src.querygraph2seq.get_subPath import *
from src.querygraph2seq.selection_cands import *

index2Label = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']

def getPathTriplesKey(querygraphInfo):
    pathTriples = querygraphInfo["path"]
    variableNodeIds = querygraphInfo["variableNodeIds"]
    answerNodeId = querygraphInfo["answerNodeId"]
    answers = querygraphInfo["answer"].split('\t')
    pathKey = []
    triples = []
    for i in range(len(pathTriples)):
        if(i == answerNodeId):
            # tripleStr += '<answer>'
            triples.append(answers[0])
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
                thTriple = (i // 3) + 1
                index = thTriple - 1
            # tripleStr += '<' + index2Label[index] + '>'
            triples.append('<' + index2Label[index] + '>')
        if((i + 1) % 3 == 0):
            pathKey.append(str(triples))
            triples = []
    if(len(triples) > 0):
        pathKey.append(str(triples))
    # if(len(pathKey) > 0):
    #     print(pathKey)
    #     import pdb; pdb.set_trace()
    pathKey.sort()
    return pathKey

def getEntityConstrainKey(querygraphInfo):
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
                    thTriple = (thMainpath // 3) + 1 # 确定变量是第几个三元组里的，取值为1,2,3
                    # index = thTriple // 2 - 1 # 修改bug
                    index = thTriple - 1
                # tripleStr += '<' + index2Label[index] + '>'
                triples.append('<' + index2Label[index] + '>')
        pathKey.append(str(triples))
        triples = []
    pathKey.sort()
    return pathKey

def getVirtualEntityConstrainKey(querygraphInfo):
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
            thTriple = (thMainPath // 3) + 1
            index = thTriple - 1
        # index = thVariable // 2
        # tripleStr += '<' + index2Label[index] + '>'
        triples.append('<' + index2Label[index] + '>')
        triples.append(triple[1])
        triples.append(triple[2])
        # tripleStr += triple[1] + triple[2]
        pathKey.append(str(triples))
        triples = []
        # import pdb; pdb.set_trace()
    pathKey.sort()
    return pathKey


def getHigherOrderConstrainKey(querygraphInfo):
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
            thTriple = (thMainPath // 3) + 1
            index = thTriple - 1
        # index = thVariable // 2
        triples.append('<' + index2Label[index] + '>')
        triples.append(triple[1])
        triples.append(triple[2])
        # tripleStr += triple[1] + triple[2]
        pathKey.append(str(triples))
        triples = []
    pathKey.sort()
    return pathKey


def getBasePathTriplesKey(querygraphInfo):
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
                    thTriple = (i // 3) + 1
                    index = thTriple - 1
                # tripleStr += '<' + index2Label[index] + '>'
                triples.append('<' + index2Label[index] + '>')
            if((i + 1) % 3 == 0):
                pathKey.append(str(triples))
                triples = []
        # import pdb; pdb.set_trace()
    if(len(triples) > 0):
        pathKey.append(str(triples))
    pathKey.sort()
    # import pdb; pdb.set_trace()
    return pathKey

def getRelConstrainKey(querygraphInfo):
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
                    thTriple = (i // 3) + 1
                    index = thTriple - 1
                triples.append('<' + index2Label[index] + '>')
            if((i + 1) % 3 == 0):
                pathKey.append(str(triples))
                triples = []
        # import pdb; pdb.set_trace()
    if(len(triples) > 0):
        pathKey.append(str(triples))
    pathKey.sort()
    # import pdb; pdb.set_trace()
    return pathKey


def getKey(querygraphInfo) -> str:
    pathTriplesKey = getPathTriplesKey(querygraphInfo)
    entityConstrainKey = getEntityConstrainKey(querygraphInfo)
    virtualEntityConstrainKey = getVirtualEntityConstrainKey(querygraphInfo)
    higherOrderConstrainKey = getHigherOrderConstrainKey(querygraphInfo)
    basePathKey = getBasePathTriplesKey(querygraphInfo)
    relConstrainKey = getRelConstrainKey(querygraphInfo)
    key = pathTriplesKey + entityConstrainKey + virtualEntityConstrainKey \
        + higherOrderConstrainKey + basePathKey + relConstrainKey
    key.sort()
    return str(key)


def getMainPathSeq(graphInfo, normElement = False):
    mainPath = graphInfo['path']
    variableNodeIds = graphInfo['variableNodeIds']
    answerNodeId = graphInfo['answerNodeId']
    mainPathSeq = ''
    for i, item in enumerate(mainPath):
        if(i == answerNodeId):
            answer = graphInfo['answer'].split('\t')[0]
            if(normElement):
                answer = normalizationElement(answer)
            # answer = '<>'
            mainPathSeq += '#' + answer + '#'
            # import pdb; pdb.set_trace()
        elif(i not in variableNodeIds and i <= 2):
            if(normElement):
                item = normalizationElement(item)
            mainPathSeq += item
        elif(i % 3 == 1): # 表示关系位置
            if(normElement):
                item = normalizationElement(item)
            mainPathSeq += item
        else:
            if(i in variableNodeIds):
                index = variableNodeIds.index(i)
            else:
                thTriple = (i // 3) + 1
                index = thTriple // 2 - 1
            mainPathSeq += '<' + index2Label[index] + '>'
    return mainPathSeq


def getConstrainPathSeq(graphInfo, normElement):
    constrainPathSeq = ''
    variableNodeIds = graphInfo['variableNodeIds']
    constrainPath = graphInfo['entityPath']
    entityConstrainIDs = graphInfo['entityConstrainIDs']
    for i, triple in enumerate(constrainPath):
        for j, item in enumerate(triple):
            if(j == graphInfo['entityIDs'][i] or j % 3 == 1):
                if(normElement):
                    item = normalizationElement(item)
                constrainPathSeq += item
            else:
                thMainpath = entityConstrainIDs[i]
                if(thMainpath in variableNodeIds):
                    index = variableNodeIds.index(thMainpath)
                else:
                    thTriple = (thMainpath // 3) + 1
                    index = thTriple // 2 - 1
                constrainPathSeq += '<' + index2Label[index] + '>'
    return constrainPathSeq


def deleteExplanation(entity:str):
    pos = entity.find('_（')
    if(pos != -1):
        entity = entity[0: pos] + '>'
    return entity


def normalizationElement(element: str):
    if(len(element) < 2):
        return element
    element = deleteExplanation(element)
    # if('\"' == element[0] and '\"' == element[-1]):
    #     return '<' + element[1:-1] + '>'
    return element


def transQueryGraph2Seq(graphInfo, normElement):
    mainPathSeq = getMainPathSeq(graphInfo, normElement)
    constrainPathSeq = getConstrainPathSeq(graphInfo, normElement)
    return {"mainPath": mainPathSeq, "entityPath": constrainPathSeq}

def transQueryGraph2SeqItems(graphInfo, normElement):
    mainPathSeq = getPathTriplesSubPath(graphInfo)
    entityConstrainKey = getEntityConstrainSubPath(graphInfo)
    virtualEntityConstrainKey = getVirtualEntityConstrainSubPath(graphInfo)
    higherOrderConstrainKey = getHigherOrderConstrainSubPath(graphInfo)
    basePathKey = getBasePathTriplesSubPath(graphInfo)
    relConstrainKey = getRelConstrainSubPath(graphInfo)
    # import pdb; pdb.set_trace()
    if(len(basePathKey) > 0):
        # print(basePathKey)
        if(len(basePathKey) == 6 and basePathKey[2] == '<A>'):
        # import pdb; pdb.set_trace()
            # basePathKey = basePathKey[0: 2] + [basePathKey[4]] # 这种模式能达到0.5078
            basePathKey = basePathKey[0: 2] + basePathKey[3: 5] # 这种模式能达到0.5124
            # basePathKey = basePathKey[0: 3] + basePathKey[3: 5] + ['<B>'] 
    # import pdb;pdb.set_trace()
    return {"mainPath": mainPathSeq, "entityPath": entityConstrainKey,\
            "virtualConstrain": virtualEntityConstrainKey,\
            "higherOrderConstrain": higherOrderConstrainKey,\
            "basePath": basePathKey,\
            "relConstrain": relConstrainKey,\
            "answerType": graphInfo['answerType']}


def writeSeq(que2seqs, fileName):
    with open(fileName, 'w', encoding='utf-8') as fout:
        for que in que2seqs:
            fout.write(que + '\n')
            for seq in que2seqs[que]:
                fout.write(json.dumps(seq, ensure_ascii=False) + '\n')
            fout.write('\n')

def data2json(inputPath: str) -> Dict[str, str]:
    '抽取 转为'
    with open(inputPath, "r") as fin:    # 这样可以一起写
        ques, sql, line_idx = [], [], 1
        for line in fin:
            line = line.rstrip()
            if (line_idx - 1) % 4 == 0:
                ques.append(line)
            elif (line_idx - 2) % 4 == 0:
                sql.append(line)
            line_idx += 1
        print("共{}条问句".format(len(ques)))
        res = {}
        for idx, i in enumerate(ques):
            que = ques[idx]
            index = que.find(':')
            que = que[index+1:]
            # if que in res and sql[idx] != res[que]: # 问句存在且SPARQL不一致 打印
            #     print(que, res[que], sql[idx])
            res[que] = sql[idx]
    return res


def json2elements(inputPath: str, outputPath: str):
    data = data2json(inputPath)
    
    with open(outputPath, "a") as fout:
        # mention2enti = loadMention2Enti(mention2entiPath)
        print('que count: %d'%len(data))
        number, res, noneEnti ,mulMen, lackMen= 0, {}, [], [], []
        regex = [
            r"(<[^>]+>)\s*(<[^>]+>)\s*(\?)[a-z]+",  # <enti><rel>?x
            r"(\?)[a-z]+\s*(<[^>]+>)\s*(<[^>]+>)",  # ?x<rel><enti>
            r"(<[^>]+>)\s*(\?)[a-z]+\s*(<[^>]+>)",  # <enti>?x<enti>
            r'(\?)[a-z]+\s*(<[^>]+>)\s*("[^"]+")',  # ?x<rel>"rel_val"
            r"(<[^>]+>)\s*(\?)[a-z+]\s*(\?)[a-z]+",  # <enti>?x?y
            r"(\?)[a-z+]\s*(\?)[a-z]+\s*(<[^>]+>)",  # ?x?y<enti>
            r'regex\((\?)[a-z],\s*("[^"]+")\)',  # regex(?x "val")
        ]
        count = [0] * len(regex)    # [4710, 1792, 6, 544, 4, 8, 14]
        
        que2triples = {}
        for que, sql in data.items():   # 遍历问句
            # if(len(res)>10):
            #     print(res)
            #     break
            # 获取实体
            results = []
            entitys, regexFlag = [], False   # SPARQL中的实体
            for idx, item in enumerate(regex):
                group = re.findall(item, sql)   # re 匹配实体
                if idx == 2 and group:
                    # import pdb; pdb.set_trace()
                    matchRes = group
                    # group = list(set(group[0]))
                    
                else:
                    # group = list(set(group))
                    if(group):
                        matchRes = group
                    # import pdb; pdb.set_trace()
                # if('我们会在' in que):
                #     print(group)
                entitys.extend(group)
                if group:
                    count[idx] += 1   
                    results.extend(matchRes)
            # if('我们会在' in que):
            #     import pdb; pdb.set_trace()
            que2triples[que] = results
    return que2triples


def json2elements2(inputPath: str, outputPath: str = None):
    data = data2json(inputPath)
    # mention2enti = loadMention2Enti(mention2entiPath)
    print('que count: %d'%len(data))
    que2triples = {}
    for que, sql in data.items():   # 遍历问句
        pos = sql.find('{')
        triples = []
        if(pos != -1):
            sql = sql[pos + 1:].strip()
            sql = sql.replace('{', '')
            sql = sql.replace('}', '')
            i = 0
            while(sql != ''):
                pos = sql.find('.')
                if(pos != -1):
                    tripleStr = sql[0: pos].strip()
                    triples.append(tripleStr)
                    sql = sql[pos + 1:].strip()
                else:
                    # print('final triple:', sql)
                    tripleStr = sql
                    triples.append(tripleStr)
                    sql = ''
                i += 1
                if(i > 10):
                    print('死循环')
                    import pdb; pdb.set_trace()
        else:
            print('sparql中不包含{')
            import pdb; pdb.set_trace()
        if(len(triples) == 0):
            import pdb; pdb.set_trace()
        que2triples[que] = triples
    # import pdb; pdb.set_trace()
    return que2triples

def isSameTriple(goldTriple, triple):
    if(len(goldTriple) == 3):
        if('?' == goldTriple[0] and (goldTriple[1] == triple[1] or goldTriple[2] == triple[2])):
            # print(goldTriple, triple)
            return True
        elif('?' == goldTriple[1] and (goldTriple[0] == triple[0] or goldTriple[2] == triple[2])):
            # print(goldTriple, triple)
            return True
        elif('?' == goldTriple[2] and (goldTriple[0] == triple[0] or goldTriple[1] == triple[1])):
            # print(goldTriple, triple)
            return True
    return False
    # import pdb; pdb.set_trace()

def getLabel(goldCand: List[str], cand):
    '''
    功能：根据标准sparql三元组得到候选cand的标签
    '''
    goldTriples = copy.deepcopy(goldCand)
    mainPath = cand['path']
    entityPath = cand['entityPath']
    triples = []
    i = 0
    while(i < len(mainPath)):
        triples.append((mainPath[i + 0], mainPath[i + 1], mainPath[i + 2]))
        i += 3
    i = 0
    while(i < len(entityPath)):
        # print(i, entityPath)
        triples.append((entityPath[i][0], entityPath[i][1], entityPath[i][2]))
        i += 3
    if(len(goldTriples) < len(triples)):
        return 0
    else:   # 正确答案的三元组数需大于或等于预测答案
        trueNum = 0
        lenGoldTriples = len(goldTriples)
        for i in range(len(triples)):
            flag = True
            for j in range(lenGoldTriples):
                if(isSameTriple(goldTriples[j], triples[i])):
                    trueNum += 1
                    flag = False
                    break
            if(flag):
                break
        if(trueNum == len(triples)):
            # print(goldCand, cand)
            return 1
    return 0
    

def generateTrainSeq():
    fileName = BASE_DIR + '/dataset/ccks2021/ccks2021_task13_train.txt'
    que2sparql = readTrainFileForSparql(fileName)
    que2goldTriples = json2elements(fileName, BASE_DIR + '/src/querygraph2seq/elements.txt')
    # candsFile = BASE_DIR + '/data/candidates/885_query_graph_cands_1027_expand10_add_100_1000_10000_100000_1000.txt'
    candsFile = BASE_DIR + '/data/candidates/9375_query_graph_cands_1031_expand10_add_100_1000_10000_100000_1000.txt'
    que2cands = readCandsInfo(candsFile)
    que2seqs = {}
    fileName = BASE_DIR + '/dataset/ccks2021/ccks2021_task13_train.txt'
    que2answer = readTrainFile(fileName)
    notGoldPathNum = 0
    for i, que in enumerate(que2cands):
        tempDic = {}
        answerTrue = 0
        sparqlTrue = 0
        goldAnswer = que2answer[que]
        if(i % 100 == 0):
            print(i)
        if(que not in que2seqs):
            que2seqs[que] = []
        for cand in que2cands[que]:
            key = getKey(cand)
            if(key in tempDic):
                continue
            else:
                tempDic[key] = cand
            predictionAnswer = cand['answer']
            newGoldAnswer = processAnswerEntity(goldAnswer)
            predictionAnswer = processAnswerEntity(predictionAnswer)
            f1 = f1Item(newGoldAnswer, predictionAnswer)
            if(f1 < 0.00001):
                newGoldAnswer = processAnswerValue(goldAnswer)
                f1 = f1Item(newGoldAnswer, predictionAnswer)
            candSeq = transQueryGraph2Seq(cand)
            # print(candSeq)
            candSeq["f1"] = f1
            label = getLabel(que2goldTriples[que], cand)
            # if('我们会在' in que):
            #     print(label, cand)
            #     print(que2goldTriples[que])
            if(f1 >= 0.5 and label == 1):
                candSeq["label"] = 1
            else:
                candSeq["label"] = 0
            if(f1 >= 0.5):
                answerTrue = 1
            if(label == 1):
                sparqlTrue = 1
            if(not (f1 >= 0.5 and label == 0)): # f1值较高，而且路径不是标准路径的会剔除掉
                que2seqs[que].append(candSeq)
        if(answerTrue > sparqlTrue):
            que2seqs[que] = []
            tempDic = {}
            for cand in que2cands[que]:
                key = getKey(cand)
                if(key in tempDic):
                    continue
                else:
                    tempDic[key] = cand
                predictionAnswer = cand['answer']
                newGoldAnswer = processAnswerEntity(goldAnswer)
                predictionAnswer = processAnswerEntity(predictionAnswer)
                f1 = f1Item(newGoldAnswer, predictionAnswer)
                if(f1 < 0.00001):
                    newGoldAnswer = processAnswerValue(goldAnswer)
                    f1 = f1Item(newGoldAnswer, predictionAnswer)
                candSeq = transQueryGraph2Seq(cand)
                # print(candSeq)
                candSeq["f1"] = f1
                if(f1 >= 0.5):
                    candSeq["label"] = 1
                else:
                    candSeq["label"] = 0
                que2seqs[que].append(candSeq)
            print('answer true > spqral true:', que)
            notGoldPathNum += 1
    print('没有gold path，但有相对正确答案的问句个数为%d' %(notGoldPathNum))
    seqsFile = BASE_DIR + '/data/candidates/seq/trainset_seq_y_6_5_1102.txt'
    writeSeq(que2seqs=que2seqs, fileName = seqsFile)

def generateTestSeq():
    fileName = BASE_DIR + '/dataset/ccks2021/testset_zhengqiu.json'
    que2answer = readQue2AnswerFromTestset(fileName)
    # candsFile = BASE_DIR + '/data/candidates/8328_testset_query_graph_cands_1027_expand10_add_100_200_2000_20000_200.txt'
    candsFile = BASE_DIR + '/data/candidates/8386_testset_query_graph_cands_1031_expand10_add_100_1000_10000_100000_1000.txt'
    que2cands = readCandsInfo(candsFile)
    que2seqs = {}
    for i, que in enumerate(que2cands):
        tempDic = {}
        goldAnswer = que2answer[que]
        if((i + 1) % 100 == 0):
            # break
            print(i)
        if(que not in que2seqs):
            que2seqs[que] = []
        for cand in que2cands[que]:
            key = getKey(cand)
            if(key in tempDic):
                continue
            else:
                tempDic[key] = cand
            predictionAnswer = cand['answer']
            newGoldAnswer = processAnswerEntity(goldAnswer)
            predictionAnswer = processAnswerEntity(predictionAnswer)
            f1 = f1Item(newGoldAnswer, predictionAnswer)
            if(f1 < 0.00001):
                newGoldAnswer = processAnswerValue(goldAnswer)
                f1 = f1Item(newGoldAnswer, predictionAnswer)
            candSeq = transQueryGraph2Seq(cand)
            # print(candSeq)
            # if('\"天下布武；设立乐市乐座\"' in cand['path']):
            #     print(f1, cand, newGoldAnswer)
            #     import pdb; pdb.set_trace()
            candSeq["f1"] = f1
            que2seqs[que].append(candSeq)
        # if('第六天魔王"有哪些成就' in que):
        #     import pdb; pdb.set_trace()
    seqsFile = BASE_DIR + '/data/candidates/seq/testset_seq_y_6_5_1102.txt'
    writeSeq(que2seqs=que2seqs, fileName = seqsFile)

def generateTrainSeqNoSparql(normElement):
    ############## 用于分析NER对KBQA系统的影响 ##############
    # train
    # fileName = BASE_DIR + '/data/data_from_pzhou/trainset/train_query_triple.txt'
    # candsFile = BASE_DIR + '/data/candidates/9416_train_analysis_ner_query_graph_cands_1116_expand10_add_100_1000_10000_100000_1000.txt'
    # seqsFile = BASE_DIR + '/data/candidates/seq/analysis_ner_9416_trainset_seq_1116.txt'
    # dev
    # fileName = BASE_DIR + '/data/data_from_pzhou/devset/dev_query_triple.txt'
    # candsFile = BASE_DIR + '/data/candidates/9322_dev_analysis_ner_query_graph_cands_1116_expand10_add_100_1000_10000_100000_1000.txt'
    # seqsFile = BASE_DIR + '/data/candidates/seq/analysis_ner_9322_devset_seq_1116.txt'
    ####################################
    fileName = BASE_DIR + '/dataset/ccks2021/ccks2021_task13_train.txt'
    # candsFile = BASE_DIR + '/data/candidates/9375_query_graph_cands_1031_expand10_add_100_1000_10000_100000_1000.txt'
    candsFile = BASE_DIR + '/data/candidates/9727_4hop3_filter_rel_top300_query_graph_cands_1228_expand10_add_100_1000_10000_100000_1000.txt'
    seqsFile = BASE_DIR + '/data/candidates/seq/trainset_seq_new1_1229.txt'
    # seqsFile = BASE_DIR + '/data/candidates/seq/stagg_9342_trainset_seq_1115.txt'
    que2sparql = readTrainFileForSparql(fileName)
    que2cands = readCandsInfo(candsFile)
    que2seqs = {}
    # fileName = BASE_DIR + '/dataset/ccks2021/ccks2021_task13_train.txt'
    que2answer = readTrainFile(fileName)
    for i, que in enumerate(que2cands):
        tempDic = {}
        goldAnswer = que2answer[que]
        if(i % 100 == 0):
            print(i)
        if(que not in que2seqs):
            que2seqs[que] = []
        for cand in que2cands[que]:
            key = getKey(cand)
            if(key in tempDic):
                continue
            else:
                tempDic[key] = cand
            predictionAnswer = cand['answer']
            newGoldAnswer = processAnswerEntity(goldAnswer)
            predictionAnswer = processAnswerEntity(predictionAnswer)
            f1 = f1Item(newGoldAnswer, predictionAnswer)
            if(f1 < 0.00001):
                newGoldAnswer = processAnswerValue(goldAnswer)
                f1 = f1Item(newGoldAnswer, predictionAnswer)
            candSeq = transQueryGraph2Seq(cand, normElement)
            # print(candSeq)
            candSeq["f1"] = f1
            if(f1 >= 0.5):
                candSeq["label"] = 1
            else:
                candSeq["label"] = 0
            # if(len(cand['path']) > 3 and len(cand['entityPath']) > 0 and candSeq["label"] == 1):
            #     import pdb; pdb.set_trace()
            que2seqs[que].append(candSeq)
    writeSeq(que2seqs=que2seqs, fileName = seqsFile)

def generateTrainSeqNoSparqlByItems(normElement, dataType):
    ############## 用于分析NER对KBQA系统的影响 ##############
    # train
    # fileName = BASE_DIR + '/data/data_from_pzhou/trainset/train_query_triple.txt'
    # candsFile = BASE_DIR + '/data/candidates/9416_train_analysis_ner_query_graph_cands_1116_expand10_add_100_1000_10000_100000_1000.txt'
    # seqsFile = BASE_DIR + '/data/candidates/seq/analysis_ner_9416_trainset_seq_1116.txt'
    # dev
    # fileName = BASE_DIR + '/data/data_from_pzhou/devset/dev_query_triple.txt'
    # candsFile = BASE_DIR + '/data/candidates/9322_dev_analysis_ner_query_graph_cands_1116_expand10_add_100_1000_10000_100000_1000.txt'
    # seqsFile = BASE_DIR + '/data/candidates/seq/analysis_ner_9322_devset_seq_1116.txt'
    ####################################
    if(dataType == 'train'):
        fileName = BASE_DIR + '/dataset/ccks2021/ccks2021_task13_train.txt'
        # candsFile = BASE_DIR + '/data/candidates/xx4hop2_filter_rel_top300_query_graph_cands_1228_expand10_add_100_1000_10000_100000_1000.txt'
        candsFile = BASE_DIR + '/data/candidates/9727_4hop3_filter_rel_top300_query_graph_cands_1228_expand10_add_100_1000_10000_100000_1000.txt'
        seqsFile = BASE_DIR + '/data/candidates/seq/trainset_seq_9727_new1_1229.txt'
        # seqsFile = BASE_DIR + '/data/candidates/seq/stagg_9342_trainset_seq_1115.txt'
        que2sparql = readTrainFileForSparql(fileName)
        que2cands = readCandsInfo(candsFile)
        # fileName = BASE_DIR + '/dataset/ccks2021/ccks2021_task13_train.txt'
        que2answer = readTrainFile(fileName)
    else:
        fileName = BASE_DIR + '/dataset/ccks2021/testset_zhengqiu.json'
        que2answer = readQue2AnswerFromTestset(fileName)
        candsFile = BASE_DIR + '/data/candidates/8590_testset_filter_rel_top1000_query_graph_cands_1229_expand10_add_100_1000_10000_100000_1000.txt'
        que2cands = readCandsInfo(candsFile)
        seqsFile = BASE_DIR + '/data/candidates/seq/testset_seq_8590_new1_1229.txt'
    que2seqs = {}
    for i, que in enumerate(que2cands):
        tempDic = {}
        goldAnswer = que2answer[que]
        if((i + 1) % 100 == 0):
            print(i)
            # break
        if(que not in que2seqs):
            que2seqs[que] = []
        for cand in que2cands[que]:
            key = getKey(cand)
            if(key in tempDic):
                continue
            else:
                tempDic[key] = cand
            predictionAnswer = cand['answer']
            newGoldAnswer = processAnswerEntity(goldAnswer)
            predictionAnswer = processAnswerEntity(predictionAnswer)
            f1 = f1Item(newGoldAnswer, predictionAnswer)
            if(f1 < 0.00001):
                newGoldAnswer = processAnswerValue(goldAnswer)
                f1 = f1Item(newGoldAnswer, predictionAnswer)
            # candSeq = transQueryGraph2Seq(cand, normElement)
            candSeq = transQueryGraph2SeqItems(cand, normElement)
            # print(candSeq)
            candSeq["f1"] = f1
            if(f1 >= 0.5):
                candSeq["label"] = 1
            else:
                candSeq["label"] = 0
            # if(len(cand['path']) > 3 and len(cand['entityPath']) > 0 and candSeq["label"] == 1):
            #     import pdb; pdb.set_trace()
            que2seqs[que].append(candSeq)
    writeSeq(que2seqs=que2seqs, fileName = seqsFile)

def generateTrainSeqSparqlByItems(normElement, dataType):
    ####################################
    if(dataType == 'train'):
        fileName = BASE_DIR + '/dataset/ccks2021/ccks2021_task13_train.txt'
        # candsFile = BASE_DIR + '/data/candidates/xx4hop2_filter_rel_top300_query_graph_cands_1228_expand10_add_100_1000_10000_100000_1000.txt'
        candsFile = BASE_DIR + '/data/candidates/xxtrainset_4hop4_with_var_noentity_with_multiType_filter_rel_top300_Graph_0304_100_1000_10000_100000_1000.txt'
        seqsFile = BASE_DIR + '/data/candidates/seq/trainset_seq_9727_pos_f109_neg03_0304.txt'
        # seqsFile = BASE_DIR + '/data/candidates/seq/stagg_9342_trainset_seq_1115.txt'
        que2goldTriples = json2elements2(fileName, BASE_DIR + '/src/querygraph2seq/elements.txt')
        que2cands = readCandsInfo(candsFile)
        # fileName = BASE_DIR + '/dataset/ccks2021/ccks2021_task13_train.txt'
        que2answer = readTrainFile(fileName)
    que2seqs = {}
    que2trueNum = {}
    for i, que in enumerate(que2cands):
        tempDic = {}
        if(que in que2answer):
            goldAnswer = que2answer[que]
        else:
            print(que, 'not in que2answer')
            break
            # continue
        # if((i + 1) % 100 == 0):
        #     print(i)
        #     # break
        if(que not in que2seqs):
            que2seqs[que] = []
        print(que)
        if(que not in que2trueNum):
            que2trueNum[que] = 0
        cands2f1 = {}
        for cand in que2cands[que]:
            # key = getKey(cand)
            # import pdb; pdb.set_trace()
            # if(key in tempDic):
            #     continue
            # else:
            #     tempDic[key] = cand
            predictionAnswer = cand['answer']
            newGoldAnswer = processAnswerEntity(goldAnswer)
            predictionAnswer = processAnswerEntity(predictionAnswer)
            f1 = f1Item(newGoldAnswer, predictionAnswer)
            if(f1 < 0.00001):
                newGoldAnswer = processAnswerValue(goldAnswer)
                f1 = f1Item(newGoldAnswer, predictionAnswer)
            # candSeq = transQueryGraph2Seq(cand, normElement)
            candSeq = transQueryGraph2SeqItems(cand, normElement)
            # print(candSeq)
            candSeq["f1"] = f1
            goldTriples = copy.deepcopy(que2goldTriples[que])
            if(f1 >= 0.9):
                mainPathFlag = isTrueMainPath(candSeq['mainPath'], goldTriples)
                basePathFlag = isTrueBasePath(candSeq['basePath'], goldTriples)
                # import pdb; pdb.set_trace()
                entityConstrainFlag = isTrueEntityPath(candSeq['entityPath'], goldTriples)
                virtualConstrainFlag = isTrueVirtualPath(candSeq['virtualConstrain'], goldTriples)
                higherOrderFlag = isTrueHigherOrderPath(candSeq['higherOrderConstrain'], goldTriples)
                relConstrainFlag = isTrueRelConstrainPath(candSeq['relConstrain'], goldTriples)
                if(mainPathFlag and basePathFlag and entityConstrainFlag and virtualConstrainFlag\
                    and higherOrderFlag and relConstrainFlag):
                    candSeq["label"] = 1
                else:
                    candSeq["label"] = 0
            elif(f1 < 0.3):
                candSeq["label"] = 0
                que2seqs[que].append(candSeq)
            else:
                candSeq["label"] = -1
            cands2f1[str(candSeq)] = f1
            # que2seqs[que].append(candSeq)
        cands2f1Sorted = sorted(cands2f1.items(), key= lambda x:x[1], reverse=True)
        preScore = -1
        for cand in cands2f1Sorted:
            if(preScore == -1):
                preScore = cand[1]
            candSeq = eval(cand[0])
            # import pdb; pdb.set_trace()
            if(candSeq["label"] == 1):
                # print(cand[1], candSeq)
                # print(que2goldTriples[que])
                if(que2trueNum[que] >= 0):
                    que2trueNum[que] += 1
                    que2seqs[que].append(candSeq)
            # elif(cand[1] < preScore): # 是否只保留得分排在第一位的正例候选候选
            #     break
        # import pdb; pdb.set_trace()
        print('#'* 10)
    numTrue = 0
    num2count = {}
    for que in que2trueNum:
        num = que2trueNum[que]
        numTrue += num        
        if(num not in num2count):
            num2count[num] = 1
        else:
            num2count[num] += 1
    print('总共有多少个正例：', numTrue)
    num2countSorted = sorted(num2count.items(), key=lambda x: x[0])
    print('正例查询图个数及其出现的频次：', num2countSorted)
    writeSeq(que2seqs=que2seqs, fileName = seqsFile)

def removePad(que2seqs):
    index2Label = ['<A>', '<B>', '<C>', '<D>', '<E>', '<F>', '<G>', '<H>', '<I>']
    for que in que2seqs:
        for seq in que2seqs[que]:
            for key in seq:
                if(key in ["f1", "label"]):
                    continue
                # import pdb; pdb.set_trace()
                for pad in index2Label:
                    while(pad in seq[key]):
                        seq[key].remove(pad)
            # import pdb; pdb.set_trace()


def generateSeqNoSparqlByItemsForMultiTypes(normElement, dataType):
    if(dataType == 'train'):
        fileName = BASE_DIR + '/dataset/CCKS2019/train.txt'
        candsFile = BASE_DIR + '/data/candidates/0504_CCKS2019_CompType_Luo/0504_CCKS2019_CompType_Luo_train_300000_0429DNS_100_1000_10000_100000_1000.txt'
        seqsFile = BASE_DIR + '/data/candidates/seq/0504_CCKS2019_CompType_Luo/CCKS2019_CompType_Luo_train_0429DNS30w.txt'
        que2goldTriples = json2elements2(fileName, BASE_DIR + '/src/querygraph2seq/elements.txt')
        # import pdb; pdb.set_trace()
        que2answer = readTrainFile(fileName)
        fileName = BASE_DIR + '/dataset/ccks2021/new_ex.json'
        que2answer = addQue2AnswerOtherTypes(fileName, que2answer)
        que2goldTriples = addQue2GoldTriples(fileName, que2goldTriples)
        que2cands = readCandsInfo(candsFile)
    que2seqs = {}
    numFilter = 0
    for i, que in enumerate(que2cands):
        tempDic = {}
        if(que in que2answer):
            goldAnswer = que2answer[que]
        else:
            # print()
            print('答案不存在error: %s'%que)
            continue
        if((i + 1) % 100 == 0):
            print(i)
            # break
        if(que not in que2seqs):
            que2seqs[que] = []
        f1True = False
        tagTrue = False
        for cand in que2cands[que]:
            key = getKey(cand)
            # if(que == '与王永林星座和出生地点都一样的有哪些人？'):
            #     print(key)
            if(key in tempDic):
                continue
            else:
                tempDic[key] = cand
            predictionAnswer = cand['answer']
            newGoldAnswer = processAnswerEntity(goldAnswer)
            predictionAnswer = processAnswerEntity(predictionAnswer)
            f1 = f1Item(newGoldAnswer, predictionAnswer)
            if(f1 < 0.00001):
                newGoldAnswer = processAnswerValue(goldAnswer)
                f1 = f1Item(newGoldAnswer, predictionAnswer)
            # candSeq = transQueryGraph2Seq(cand, normElement)
            candSeq = transQueryGraph2SeqItems(cand, normElement)
            # print(candSeq)
            goldTriples = copy.deepcopy(que2goldTriples[que])
            # if(len(goldTriples) == 0)
            # import pdb; pdb.set_trace()
            candSeq["f1"] = f1
            if(f1 >= 0.5):
                f1True = True
                if(dataType == 'train'):
                    mainPathFlag = isTrueMainPath(candSeq['mainPath'], goldTriples)
                    basePathFlag = isTrueBasePath(candSeq['basePath'], goldTriples)
                    # basePathFlag = True
                else:
                    mainPathFlag = True
                # if(que == '和自己的女儿星座一样的人有哪些？'):
                #     import pdb; pdb.set_trace()
                if(mainPathFlag and basePathFlag):
                    candSeq["label"] = 1
                    tagTrue = True
                    que2seqs[que].append(candSeq)
                else:
                    # print(que)
                    # print(f1, candSeq)
                    # print(que2goldTriples[que])
                    candSeq["label"] = 0
            else:
                candSeq["label"] = 0
                que2seqs[que].append(candSeq)
            # que2seqs[que].append(candSeq)  # 候选是否需要部分丢弃，不参与训练
            # import pdb; pdb.set_trace()
        if(f1True != tagTrue):
            print('正确答案被sparql过滤：', que)
            numFilter += 1
    print('正确答案被filter过滤的个数：', numFilter)
    removePad(que2seqs)
    writeSeq(que2seqs=que2seqs, fileName = seqsFile)

def generateSeqNoSparqlByItemsForMultiTypesEval(fileName, candsFile, seqsFile, normElement):

    new_data_file = BASE_DIR + '/data/dataset/CCKS2019_CompType/new_ex.json'
    que2answer = readTrainFile(fileName)
    que2answer = addQue2AnswerOtherTypes(new_data_file, que2answer)
    que2cands = readCandsInfo(candsFile)
    que2seqs = {}
    numFilter = 0
    for i, que in enumerate(que2cands):
        tempDic = {}
        if(que in que2answer):
            goldAnswer = que2answer[que]
        else:
            print('问句对应的答案不存在error', que)
            continue
        if((i + 1) % 100 == 0):
            print(i)
            # break
        if(que not in que2seqs):
            que2seqs[que] = []
        f1True = False
        tagTrue = False
        for cand in que2cands[que]:
            key = getKey(cand)
            if(key in tempDic):
                # print('key', key)
                continue
            else:
                tempDic[key] = cand
            predictionAnswer = cand['answer']
            newGoldAnswer = processAnswerEntity(goldAnswer)
            predictionAnswer = processAnswerEntity(predictionAnswer)
            f1 = f1Item(newGoldAnswer, predictionAnswer)
            if(f1 < 0.00001):
                newGoldAnswer = processAnswerValue(goldAnswer)
                f1 = f1Item(newGoldAnswer, predictionAnswer)
            # candSeq = transQueryGraph2Seq(cand, normElement)
            candSeq = transQueryGraph2SeqItems(cand, normElement)
            candSeq["f1"] = f1
            if(f1 >= 0.5):
                f1True = True
                candSeq["label"] = 1
                tagTrue = True
            else:
                candSeq["label"] = 0
            que2seqs[que].append(candSeq)  # 候选是否需要部分丢弃，不参与训练
        if(f1True != tagTrue):
            print('正确答案被sparql过滤：', que)
            numFilter += 1
    print('正确答案被filter过滤的个数：', numFilter)
    removePad(que2seqs)
    writeSeq(que2seqs=que2seqs, fileName = seqsFile)

def generateTestSeqNoSparql(normElement):
    ############## 用于分析NER对KBQA系统的影响 ##############
    # test
    fileName = BASE_DIR + '/data/data_from_pzhou/testset/test_query_triple.txt'
    # candsFile = BASE_DIR + '/data/candidates/9299_test_analysis_ner_query_graph_cands_1116_expand10_add_100_1000_10000_100000_1000.txt'
    # seqsFile = BASE_DIR + '/data/candidates/seq/analysis_ner_9299_testset_seq_1116.txt'
    candsFile = BASE_DIR + '/data/candidates/7903_test_prediction_analysis_ner_query_graph_cands_1116_expand10_add_100_1000_10000_100000_1000.txt'
    seqsFile = BASE_DIR + '/data/candidates/seq/analysis_ner_7903_testset_prediction_seq_1116.txt'
    que2answer = readTrainFile(fileName)
    ####################################
    # fileName = BASE_DIR + '/dataset/ccks2021/testset_zhengqiu.json'
    # que2answer = readQue2AnswerFromTestset(fileName)
    # candsFile = BASE_DIR + '/data/candidates/8386_testset_query_graph_cands_1031_expand10_add_100_1000_10000_100000_1000.txt'
    # candsFile = BASE_DIR + '/data/candidates/8263_STAGG_testset_query_graph_cands_1112_3hop_expand10_add_100_1000_10000_100000_1000.txt'
    que2cands = readCandsInfo(candsFile)
    que2seqs = {}
    for i, que in enumerate(que2cands):
        tempDic = {}
        goldAnswer = que2answer[que]
        if((i + 1) % 100 == 0):
            # break
            print(i)
        if(que not in que2seqs):
            que2seqs[que] = []
        for cand in que2cands[que]:
            key = getKey(cand)
            if(key in tempDic):
                continue
            else:
                tempDic[key] = cand
            predictionAnswer = cand['answer']
            newGoldAnswer = processAnswerEntity(goldAnswer)
            predictionAnswer = processAnswerEntity(predictionAnswer)
            f1 = f1Item(newGoldAnswer, predictionAnswer)
            if(f1 < 0.00001):
                newGoldAnswer = processAnswerValue(goldAnswer)
                f1 = f1Item(newGoldAnswer, predictionAnswer)
            candSeq = transQueryGraph2Seq(cand, normElement)
            # print(candSeq)
            # if('\"天下布武；设立乐市乐座\"' in cand['path']):
            #     print(f1, cand, newGoldAnswer)
            #     import pdb; pdb.set_trace()
            candSeq["f1"] = f1
            que2seqs[que].append(candSeq)
        # if('第六天魔王"有哪些成就' in que):
        #     import pdb; pdb.set_trace()
    # seqsFile = BASE_DIR + '/data/candidates/seq/testset_seq_noexplanation_y_8_3_1105.txt'
    # seqsFile = BASE_DIR + '/data/candidates/seq/stagg_8263_testset_seq_1115.txt'
    writeSeq(que2seqs=que2seqs, fileName = seqsFile)



if __name__ == "__main__":
    '''转化train'''
    '''转化dev和test'''
    # CCKS2019
    fileName = BASE_DIR + '/data/dataset/CCKS2019/test.txt'
    candsFile = BASE_DIR + '/data/candidates/CCKS2019_Luo_test_top300000_0429DNS_100_1000_10000_100000_1000.txt'
    seqsFile = BASE_DIR + '/data/candidates/seq/CCKS2019_Luo_test.seq'

    fileName = BASE_DIR + '/data/dataset/CCKS2019/test.txt'
    candsFile = BASE_DIR + '/data/candidates/CCKS2019_Yhi_test_top300000_0429DNS_100_1000_10000_100000_1000.txt'
    seqsFile = BASE_DIR + '/data/candidates/seq/CCKS2019_Yih_test.seq'

    normElement = False
    # generateSeqNoSparqlByItemsForMultiTypes(normElement, dataType = 'train')
    # generateSeqNoSparqlByItemsForMultiTypesEval(normElement, dataType = 'dev')
    generateSeqNoSparqlByItemsForMultiTypesEval(fileName, candsFile, seqsFile, normElement)    # dev和test使用
    ##########################################################
    