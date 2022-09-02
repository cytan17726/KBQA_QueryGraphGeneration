import pymysql
import sys
import os
import json
from typing import List, Dict, Tuple, Optional
import copy

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from src.QueryGraph import QueryGraph
from config.MysqlConfig import CCKSConfig
from sqlalchemy import create_engine
from src.utils.data_processing import getFloat


# MAX_TRIPLE = 1000
# MAX_ANSWER_NUM = 1000
# MAX_VARIABLE_NUM = 300
# MAX_REL_NUM = 100

# 测试超参
B = 5
# 单跳相关
MAX_TRIPLES_NUM_INIT = int(20000 * B)
MAX_ANSWER_NUM = int(2000 * B) # 进行答案搜索时一个变量节点包含的实体个数
MAX_VARIABLE_NUM = int(200 * B) # 一个变量节点包含的实体个数
MAX_CONSTRAIN_NUM = int(100000)
MAX_REL_NUM = 100
# 多跳相关
MAX_TRIPLES_NUM_EXPAND = int(200 * B)

NAME_PROP = ['<中文名>', '<中文名称>', '<外文名>', '<外文名称>', '<本名>', '<别名>']


class MysqlConnection:
    def __init__(self, config: CCKSConfig) -> None:
        # 建立数据库连接
        # import pdb; pdb.set_trace()
        self.db = pymysql.connect(
            host=config.host, port = config.port, user=config.user, password=config.password, \
            database = config.database, charset=config.charset \
        )
        # 获取游标对象，用于执行sql语句
        self.cursor = self.db.cursor()
        # 用于快速导入知识库
        self.save_con = create_engine('mysql+pymysql://%s:%s@%s:%s/%s?charset=%s' % \
                        (config.user, config.password, config.host, str(config.port), config.database, config.charset))
        self.sql2results = {}

    def normNodeForSql(self, node: str):
        if('\\' in node):
            node = node.replace('\\', '')
        if('\\\'' not in node):
            node = node.replace('\'', '\\\'')
        return node

   
    def generate2HopChainWithEntity(self, entity: str, limitNum: int = 100) -> List[QueryGraph]:
        entity = self.normNodeForSql(entity)
        triples = self.searchWithEntity(entity, limitNum)
        # print(triples)
        resultTriples = copy.deepcopy(list(triples))
        for triple in triples:
            triplesIn2Hop = self.searchWithEntityID(triple[2])
            for tripleIn2Hop in triplesIn2Hop:
                resultTriples.append(triple + tripleIn2Hop)
                # import pdb; pdb.set_trace()
        queryGraphs: List[QueryGraph] = []
        for path in resultTriples:
            queryGraph = QueryGraph(path, path[-1], 0, len(path) - 1, [2, 5])
            queryGraphs.append(queryGraph)
        return queryGraphs


    def generate2HopChainWithEntityBasedProp(self, entity: str, limitNum: int = MAX_REL_NUM) -> List[QueryGraph]:
        entity = self.normNodeForSql(entity)
        mergeTriples = self.searchWithEntityBasedProp(entity, limitNum)
        resultMergeTriples = copy.deepcopy(list(mergeTriples))
        for mergeTriple in mergeTriples: # 链式两跳搜索
            if(len(mergeTriple[2]) < 2):
                for entity in mergeTriple[2]:
                    # import pdb; pdb.set_trace()
                    nextMergeTriples = self.searchWithEntityIDBasedProp(entity)
                    for nextMergeTriple in nextMergeTriples:
                        # resultMergeTriples.append(mergeTriple[0: 2] + [mergeTriple[2][0]] + nextMergeTriple[0: 2] + [nextMergeTriple[2][0]])
                        resultMergeTriples.append(mergeTriple + nextMergeTriple)
        queryGraphs: List[QueryGraph] = []
        for path in resultMergeTriples:
            pathTriples = []
            for i in range(len(path)):
                if((i + 1) % 3 == 0):
                    # import pdb; pdb.set_trace()
                    # print(path)
                    pathTriples.append(path[i][0])
                else:
                    pathTriples.append(path[i])
            # pathTriples = [path[0], path[1], path[2][0], path[3], path[4], path[5][0]]
            answer = '\t'.join(path[-1])
            if(len(pathTriples) == 6):
                queryGraph = QueryGraph(pathTriples, answer, 0, len(path) - 1, [2, 5], [path[2], path[5]])
            elif(len(pathTriples) == 3):
                queryGraph = QueryGraph(pathTriples, answer, 0, len(path) - 1, [2], [path[2]])
            queryGraphs.append(queryGraph)
        return queryGraphs

    def isSameWithLastTriple(self, queryGraph, triple):
        variableNodes = queryGraph.variableNodes[-1]
        if(len(variableNodes) > 1):
            return False
        variableNodes.sort()
        nodesStr = '\t'.join(variableNodes)
        variableId = queryGraph.variableNodeIds[-1]
        pathTriples = copy.deepcopy(queryGraph.pathTriples)
        pathTriples[variableId] = nodesStr
        # import pdb; pdb.set_trace()
        if(pathTriples[-1] == triple[-1] and pathTriples[-2] == triple[-2] and pathTriples[-3] == triple[-3]):
            return True
        else:
            return False

    def combineDouble(self, prop1, value1, prop2, value2, limitNum = 50):
        '''
        功能：进行变量约束的组合
        '''
        value1 = self.normNodeForSql(value1)
        value2 = self.normNodeForSql(value2)
        # print(prop1, value1, prop2, value2, str(limitNum))
        # sql = "select distinct a.entry from pkubase as a where a.prop = '%s' and a.value='%s' and a.entry in (select b.entry from pkubase as b where b.prop = '%s' and b.value = '%s') limit %s"\
        #                                  % (prop1, value1, prop2, value2, str(limitNum))
        # select distinct a.entry from pkubase as a, pkubase as b where a.value = '<阿根廷_（阿根廷共和国）>' and b.value = '<作家_（汉语词语）>' and a.entry = b.entry
        if(value1 != value2):
            sql = "select distinct a.entry from pkubase as a, pkubase as b where a.value = '%s' and b.value = '%s' and a.entry = b.entry limit %s"\
                                            % (value1, value2, str(limitNum))
        else:
            sql = "select distinct a.entry from pkubase as a, pkubase as b where a.prop = '%s' and a.value = '%s' and b.prop = '%s' and b.value = '%s' and a.entry = b.entry limit %s"\
                                            % (prop1, value1, prop2, value2, str(limitNum))
        # print(sql)
        if(sql not in self.sql2results):
            self.cursor.execute(sql)
            results = self.cursor.fetchall()
            self.sql2results[sql] = results
        else:
            results = self.sql2results[sql]
        return results

    def queryGraphsCombine(self, queryGraphs, candRels):
        newQueryGraphs = []
        # print(candRels)
        for i, queryGraph in enumerate(queryGraphs):
            pathTriples1 = queryGraph.pathTriples
            # import pdb; pdb.set_trace()
            if(len(pathTriples1) == 3 and pathTriples1[1] in candRels and 2 in queryGraph.variableNodeIds): # 单跳正向且符合关系选择
                for queryGraph2 in queryGraphs[i + 1:]:
                    pathTriples2 = queryGraph2.pathTriples
                    if(len(pathTriples2) == 3 and pathTriples1[1] != pathTriples2[1] and pathTriples2[1] in candRels and 2 in queryGraph2.variableNodeIds):
                        answers = []
                        for currentVar1 in queryGraph.variableNodes[-1][0:3]:
                            for currentVar2 in queryGraph2.variableNodes[-1][0:3]:
                                tempAnswers = self.combineDouble(pathTriples1[1], currentVar1, pathTriples2[1], currentVar2)
                                answers.extend(tempAnswers)
                        if(len(answers) > 0):   # 有答案就更新
                            answersList = [answer[0] for answer in answers]
                            topic_entity = pathTriples1[0]
                            if topic_entity in answersList:
                                # import pdb;pdb.set_trace()
                                answersList.remove(topic_entity)
                                # import pdb;pdb.set_trace()
                            if not answersList: # 答案为空了，就跳过
                                continue
                            newQueryGraph = QueryGraph()
                            newQueryGraph.updateBasePath(pathTriples1, [2])
                            newQueryGraph.updateBasePath(pathTriples2, [2])
                            newQueryGraph.updateAnswer('\t'.join(answersList))
                            newQueryGraphs.append(newQueryGraph)
                            # import pdb; pdb.set_trace()
        return newQueryGraphs + queryGraphs


    def generateOneHopFromQueryGraphs(self, queryGraphs, candRels = None, candRelsList: List[str] = None, forwardFlag: bool = True, backwardFlag: bool = True):
        '''
        遍历给定查询图，再扩展1hop
        :param queryGraphs: 待扩展查询图
        :param candRels:
        :param candRelsList:
        :param forwardFlag: 正向检索
        :param backwardFlag: 反向检索
        :return: 
        '''
        newQueryGraphs = []
        # import pdb; pdb.set_trace()
        for queryGraph in queryGraphs:
            variableNodes = copy.deepcopy(queryGraph.variableNodes[-1])
            # if(len(variableNodes) <= 50):
            if(len(variableNodes) <= 10 or (len(variableNodes) <= 50 and ("名称" in queryGraph.pathTriples[-2] or queryGraph.pathTriples[-2] in NAME_PROP)) \
                or '<CVT' in variableNodes[0]):
                forwardRels = []
                for entity in variableNodes[0:100]:
                    rels = self.searchRelWithEntryId(entity)
                    forwardRels.extend(rels)
                    if(len(forwardRels) > 100):
                        break
                forwardRels = set(forwardRels)
                # print('variableNodes:', variableNodes)
                # import pdb; pdb.set_trace()
                if(forwardFlag):
                    if(backwardFlag):
                        nextMergeTriples = self.searchWithEntityIDsBasedProp(variableNodes, forwardRels, candRels= candRels, candRelsList = candRelsList)
                    else:
                        nextMergeTriples = self.searchWithEntityIDsBasedTopProp(variableNodes, forwardRels, candRelsList = candRelsList, topn = 3)
                    for nextMergeTriple in nextMergeTriples:
                        queryGraphTemp = copy.deepcopy(queryGraph)
                        tripleTemp = copy.deepcopy(nextMergeTriple)
                        tripleTemp = [item for item in tripleTemp]
                        tripleTemp[-1].sort()
                        tripleTemp[-1] = '\t'.join(tripleTemp[-1])
                        if(self.isSameWithLastTriple(queryGraph, tripleTemp)): # 去除相同的三元组
                            continue
                        queryGraphTemp.pathTriples += [variableNodes[0], nextMergeTriple[1], nextMergeTriple[2][0]]
                        queryGraphTemp.answerNodeId = len(queryGraphTemp.pathTriples) - 1
                        queryGraphTemp.variableNodeIds.append(queryGraphTemp.answerNodeId)
                        queryGraphTemp.variableNodes.append(nextMergeTriple[2])
                        queryGraphTemp.updateAnswer('\t'.join(nextMergeTriple[2]))
                        newQueryGraphs.append(queryGraphTemp)
                        # if('宗教信仰' in nextMergeTriple[1]):
                        #     import pdb; pdb.set_trace()
                if(backwardFlag):
                    backwardRels = []
                    for entity in variableNodes[0:100]:
                        rels = self.searchRelWithValueId(entity)
                        backwardRels.extend(rels)
                        # if(len(backwardRels) > 100):
                        #     break
                    backwardRels = set(backwardRels)
                    # import pdb; pdb.set_trace()
                    nextMergeTriples = self.searchWithValueIDsBasedProp(variableNodes, backwardRels, candRels= candRels) # 反向单跳
                    for nextMergeTriple in nextMergeTriples:
                        queryGraphTemp = copy.deepcopy(queryGraph)
                        tripleTemp = copy.deepcopy(nextMergeTriple)
                        tripleTemp = [item for item in tripleTemp]
                        tripleTemp[0].sort()
                        tripleTemp[0] = '\t'.join(tripleTemp[0])
                        if(self.isSameWithLastTriple(queryGraph, tripleTemp)): # 去除相同的三元组
                            continue
                        queryGraphTemp.pathTriples += [nextMergeTriple[0][0], nextMergeTriple[1], variableNodes[0]]
                        queryGraphTemp.answerNodeId = len(queryGraphTemp.pathTriples) - 3
                        queryGraphTemp.variableNodeIds.append(queryGraphTemp.answerNodeId)
                        queryGraphTemp.variableNodes.append(nextMergeTriple[0])
                        queryGraphTemp.updateAnswer('\t'.join(nextMergeTriple[0]))
                        newQueryGraphs.append(queryGraphTemp)
                        # if('天津_' in entity):
                        #     print(nextMergeTriples)
                        #     import pdb; pdb.set_trace()
        return newQueryGraphs
    
    def generateOneHopFromQueryGraphs_based(self, queryGraphs, candRels = None, candRelsList: List[str] = None, forwardFlag: bool = True, backwardFlag: bool = True):
        '''
        遍历给定查询图，再扩展1hop=目前和之前是一样的
        :param queryGraphs: 待扩展查询图
        :param candRels:
        :param candRelsList:
        :param forwardFlag: 正向检索
        :param backwardFlag: 反向检索
        :return: 
        '''
        newQueryGraphs = []
        # import pdb; pdb.set_trace()
        for queryGraph in queryGraphs:
            variableNodes = copy.deepcopy(queryGraph.variableNodes[-1])
            # if(len(variableNodes) <= 50):
            if(len(variableNodes) <= 10 or (len(variableNodes) <= 50 and ("名称" in queryGraph.pathTriples[-2] or queryGraph.pathTriples[-2] in NAME_PROP)) \
                or '<CVT' in variableNodes[0]):
                forwardRels = []
                for entity in variableNodes[0:100]:
                    rels = self.searchRelWithEntryId(entity)
                    forwardRels.extend(rels)
                    if(len(forwardRels) > 100):
                        break
                forwardRels = set(forwardRels)
                if '' in forwardRels:
                    forwardRels.remove('')
                # print('variableNodes:', variableNodes)
                # import pdb; pdb.set_trace()
                if(forwardFlag):
                    nextMergeTriples = self.searchWithEntityIDsBasedProp(variableNodes, forwardRels, candRels= candRels, candRelsList = candRelsList)
                    # if(backwardFlag):
                    #     nextMergeTriples = self.searchWithEntityIDsBasedProp(variableNodes, forwardRels, candRels= candRels, candRelsList = candRelsList)
                    # else:
                    #     nextMergeTriples = self.searchWithEntityIDsBasedTopProp(variableNodes, forwardRels, candRelsList = candRelsList, topn = 3)
                    for nextMergeTriple in nextMergeTriples:
                        queryGraphTemp = copy.deepcopy(queryGraph)
                        tripleTemp = copy.deepcopy(nextMergeTriple)
                        tripleTemp = [item for item in tripleTemp]
                        tripleTemp[-1].sort()
                        tripleTemp[-1] = '\t'.join(tripleTemp[-1])
                        if(self.isSameWithLastTriple(queryGraph, tripleTemp)): # 去除相同的三元组
                            continue
                        queryGraphTemp.pathTriples += [variableNodes[0], nextMergeTriple[1], nextMergeTriple[2][0]]
                        queryGraphTemp.answerNodeId = len(queryGraphTemp.pathTriples) - 1
                        queryGraphTemp.variableNodeIds.append(queryGraphTemp.answerNodeId)
                        queryGraphTemp.variableNodes.append(nextMergeTriple[2])
                        queryGraphTemp.updateAnswer('\t'.join(nextMergeTriple[2]))
                        newQueryGraphs.append(queryGraphTemp)
                        # if('宗教信仰' in nextMergeTriple[1]):
                        #     import pdb; pdb.set_trace()
                if(backwardFlag):
                    backwardRels = []
                    for entity in variableNodes[0:100]:
                        rels = self.searchRelWithValueId(entity)
                        backwardRels.extend(rels)
                        # if(len(backwardRels) > 100):
                        #     break
                    backwardRels = set(backwardRels)
                    # import pdb; pdb.set_trace()
                    if backwardRels:    # 非空
                        nextMergeTriples = self.searchWithValueIDsBasedProp(variableNodes, backwardRels, candRels= candRels) # 反向单跳
                    else:
                        nextMergeTriples = []
                    for nextMergeTriple in nextMergeTriples:
                        queryGraphTemp = copy.deepcopy(queryGraph)
                        tripleTemp = copy.deepcopy(nextMergeTriple)
                        tripleTemp = [item for item in tripleTemp]
                        tripleTemp[0].sort()
                        tripleTemp[0] = '\t'.join(tripleTemp[0])
                        if(self.isSameWithLastTriple(queryGraph, tripleTemp)): # 去除相同的三元组
                            continue
                        queryGraphTemp.pathTriples += [nextMergeTriple[0][0], nextMergeTriple[1], variableNodes[0]]
                        queryGraphTemp.answerNodeId = len(queryGraphTemp.pathTriples) - 3
                        queryGraphTemp.variableNodeIds.append(queryGraphTemp.answerNodeId)
                        queryGraphTemp.variableNodes.append(nextMergeTriple[0])
                        queryGraphTemp.updateAnswer('\t'.join(nextMergeTriple[0]))
                        newQueryGraphs.append(queryGraphTemp)
                        # if('天津_' in entity):
                        #     print(nextMergeTriples)
                        #     import pdb; pdb.set_trace()
        return newQueryGraphs

    def searchWithEntity(self, entity: str, limitNum: int = 100) -> Tuple[Tuple[str]]:
        entity = self.normNodeForSql(entity)
        sql = "select entry, prop, value from `pkubase` where `entry` = '%s' or `entry`='<%s>' or `entry`='\"%s\"' limit %s"\
                                         % (entity, entity, entity, str(limitNum))
        if(sql not in self.sql2results):
            self.cursor.execute(sql)
            triples = self.cursor.fetchall()
            self.sql2results[sql] = triples
        else:
            triples = self.sql2results[sql]
        # self.cursor.execute(sql)
        # triples = self.cursor.fetchall()
        # import pdb; pdb.set_trace()
        return triples


    def searchWithEntityAndValueId(self, entity: str, valueId, limitNum: int = 100) -> Tuple[Tuple[str]]:
        entity = self.normNodeForSql(entity)
        valueId = self.normNodeForSql(valueId)
        sql = "select entry, prop, value from `pkubase` where `entry` = '%s' or `entry`='<%s>' and value = '%s' limit %s"\
                                         % (entity, entity, valueId, str(limitNum))
        triples = self.searchSQL(sql)
        # if(sql not in self.sql2results):
        #     self.cursor.execute(sql)
        #     triples = self.cursor.fetchall()
        #     self.sql2results[sql] = triples
        # else:
        #     triples = self.sql2results[sql]
        # self.cursor.execute(sql)
        # triples = self.cursor.fetchall()
        # import pdb; pdb.set_trace()
        return triples

    def searchWithValue(self, entity: str, limitNum: int = 100) -> Tuple[Tuple[str]]:
        entity = self.normNodeForSql(entity)
        sql = "select entry, prop, value from `pkubase` where `value` = '%s' or `value`='<%s>' or `value`='\"%s\"' limit %s"\
                                         % (entity, entity, entity, str(limitNum))
        # if(sql not in self.sql2results):
        #     self.cursor.execute(sql)
        #     triples = self.cursor.fetchall()
        #     self.sql2results[sql] = triples
        # else:
        #     triples = self.sql2results[sql]
        triples = self.searchSQL(sql)
        # self.cursor.execute(sql)
        # triples = self.cursor.fetchall()
        # import pdb; pdb.set_trace()
        return triples

    def searchWithEntryIdAndValue(self, entryId: str, value: str, limitNum: int = 100) -> Tuple[Tuple[str]]:
        entryId = self.normNodeForSql(entryId)
        value = self.normNodeForSql(value)
        sql = "select entry, prop, value from `pkubase` where entry = '%s' and (`value` = '%s' or `value`='<%s>' or `value`='\"%s\"') limit %s"\
                                         % (entryId, value, value, value, str(limitNum))
        triples = self.searchSQL(sql)
        return triples

    def searchSQL(self, sql):
        '''
        功能：根据sql语句获取搜索结果
        '''
        if(sql not in self.sql2results):
            # print(sql)
            self.cursor.execute(sql)
            results = self.cursor.fetchall()
            self.sql2results[sql] = results
        else:
            results = self.sql2results[sql]
        return results

    def generateRelWithEntity(self, entitys: List[str], conflictMatrix):
        '''
        功能：生成询问关系的查询图
        输入：
            entitys: 经实体链接得到的实体集合
            conflictMatrix: 实体集合之间的冲突矩阵
        输出：
            queryGraphs：关系作为最终答案的查询图集合
        '''
        queryGraphs = []
        if(len(entitys) < 2):
            return []
        for i, entity in enumerate(entitys):
            for j, item in enumerate(conflictMatrix[i]):
                if(item != 1):
                    sql = "select distinct entry, prop, value from `pkubase` where (`entry` = '%s' or `entry`='<%s>') and (`value` = '%s' or `value` = '<%s>') "\
                                         % (entity, entity, entitys[j], entitys[j])
                    results = self.searchSQL(sql)
                    if(len(results) < 1):
                        continue
                    props = [item[1] for item in results]
                    props = list(set(props))
                    # print('results:', results)
                    queryGraph = QueryGraph([results[0][0], results[0][1], results[0][2]],
                                results[0][1], 0, 1, [1], [results[0][1]])
                    queryGraph.updateAnswer('\t'.join(props))
                    queryGraphs.append(queryGraph)
        return queryGraphs

    def searchRelWithEntryValue(self, entry: str, value: str):
        entry = self.normNodeForSql(entry)
        value = self.normNodeForSql(value)
        sql = "select distinct prop from `pkubase` where (`entry` = '%s' or `entry`='<%s>') and (`value` = '%s' or `value` = '<%s>') "\
                                % (entry, entry, value, value)
        results = self.searchSQL(sql)
        if(len(results) > 0):
            return [item[0] for item in results]
        else:
            return ['']

    def searchRelWithEntryId(self, entry: str):
        entry = self.normNodeForSql(entry)
        sql = "select distinct prop from `pkubase` where `entry`='%s' limit 100 "\
                                % (entry)
        results = self.searchSQL(sql)
        if(len(results) > 0):
            return [item[0] for item in results]
        else:
            return ['']

    def searchRelWithValueId(self, entry: str):
        entry = self.normNodeForSql(entry)
        sql = "select distinct prop from `pkubase` where `value`='%s' limit 100 "\
                                % (entry)
        results = self.searchSQL(sql)
        if(len(results) > 0):
            return [item[0] for item in results]
        else:
            return ['']

    def searchWithEntityBasedProp(self, entity: str, limitNum: int = MAX_REL_NUM) -> Tuple[Tuple[str]]:
        entity = self.normNodeForSql(entity)
        sql = "select distinct prop, entry from `pkubase` where `entry` = '%s' or `entry`='<%s>' or `entry`='\"%s\"' limit %s"\
                                         % (entity, entity, entity, str(limitNum))
        results = self.searchSQL(sql)
        props = []
        entitys = []
        for item in results:
            props.append(item[0])
            entitys.append(item[1])
        queryGraphs: List[QueryGraph] = []
        mergeTriples = []
        numTriples = 0
        for i, propItem in enumerate(props):
            variables = self.searchObjectWithSubjectProp(entitys[i], propItem, limitNum = MAX_VARIABLE_NUM)
            variablesList = []
            for item in variables:
                if(item[0] not in variablesList):
                    variablesList.append(item[0])
            # variablesList = [item[0] for item in variables]
            answer = '\t'.join(variablesList)
            mergeTriples.append((entitys[i], propItem, variablesList))
            # import pdb; pdb.set_trace()
            queryGraph = QueryGraph([entitys[i], propItem, variablesList[0]],
                                answer, 0, 2, [2], [variablesList])
            queryGraphs.append(queryGraph)
            numTriples += len(variables)
            if(numTriples > MAX_TRIPLES_NUM_INIT):
                break
        return queryGraphs

    def addAnswerType(self, queryGraph, limitNum: int = MAX_REL_NUM) -> Tuple[Tuple[str]]:
        entitys = queryGraph.answer.split('\t')
        numTriples = 0
        # print('answer:', queryGraph.answer)
        type2num = {}
        for i, entity in enumerate(entitys):
            variables = self.searchObjectWithSubjectProp(entity, '<类型>', limitNum = MAX_VARIABLE_NUM)
            variablesList = []
            for item in variables:
                # import pdb; pdb.set_trace()
                if(item[0] not in variablesList):
                    variablesList.append(item[0])
            for item in variablesList:
                if(item not in type2num):
                    type2num[item] = 1
                else:
                    type2num[item] += 1
            # # variablesList = [item[0] for item in variables]
            # answer = '\t'.join(variablesList)
            # mergeTriples.append((entitys[i], propItem, variablesList))
            # # import pdb; pdb.set_trace()
            # queryGraph = QueryGraph([entitys[i], propItem, variablesList[0]],
            #                     answer, 0, 2, [2], [variablesList])
            # queryGraphs.append(queryGraph)
            numTriples += len(variables)
            if(numTriples > MAX_TRIPLES_NUM_INIT):
                break
        type2numList = sorted(type2num.items(), key = lambda x: x[1], reverse=True)
        if(len(type2numList) > 0):
            # print(type2numList)
            answerType = ''
            for item in type2numList[0:3]:
                answerType += item[0] + ','
            answerType = answerType[0: -1]
            queryGraph.updateAnswerType(answerType)
            # print(answerType)
        else:
            # print('answer', entitys)
            queryGraph.updateAnswerType(queryGraph.answer)
            # import pdb; pdb.set_trace()
        return queryGraph

    def searchWithValueBasedProp(self, entity: str, limitNum: int = MAX_REL_NUM):
        entity = self.normNodeForSql(entity)
        # sql = "select distinct prop from `pkubase` where `value` = '%s' or `value`='<%s>' or `value`='\"%s\"' limit %s"\
        #                                  % (entity, entity, entity, str(limitNum))
        sql = "select distinct prop, value from `pkubase` where `value` = '%s' or `value`='<%s>' or `value` = '\"%s\"' limit %s"\
                                         % (entity, entity, entity, str(limitNum))
        results = self.searchSQL(sql)
        # self.cursor.execute(sql)
        # results = self.cursor.fetchall()
        props = []
        values = []
        for item in results:
            props.append(item[0])
            values.append(item[1])
        # import pdb; pdb.set_trace()
        numTriples = 0
        queryGraphs: List[QueryGraph] = []
        for i, propItem in enumerate(props):
            # import pdb; pdb.set_trace()
            variables = self.searchSubjectWithPropObject(propItem, values[i], limitNum=MAX_ANSWER_NUM)
            variablesList = []
            for item in variables:
                if(item[0] not in variablesList):
                    variablesList.append(item[0])
            answer = '\t'.join(variablesList)
            # print(variablesList)
            # if(values[i] == '<导演>' ):
            #     import pdb; pdb.set_trace()
            queryGraph = QueryGraph([variablesList[0], propItem, values[i]],
                                answer, 2, 0, [0], [variablesList])
            queryGraphs.append(queryGraph)
            numTriples += len(variables)
            # print(propItem, numTriples)
            if(numTriples > MAX_TRIPLES_NUM_INIT):
                break
        return queryGraphs

    def searchWithEntityID(self, entity: str, limitNum: int = 100):
        triples = None
        try:
            entity = self.normNodeForSql(entity)
            sql = "select entry, prop, value from `pkubase` where `entry` = '%s' limit %s"\
                                            % (entity, str(limitNum))
            triples = self.searchSQL(sql)
        except:
            print(sql)
        return triples

    def searchWithEntityIDBasedProp(self, entity: str, limitNum: int = 100, candRels = None, candRelsList: List[str] = None):
        entity = self.normNodeForSql(entity)
        sql = "select distinct prop from `pkubase` where `entry` = '%s' limit %s"\
                                        % (entity, str(limitNum))
        props = self.searchSQL(sql)
        mergeTriples = []
        numTriples = 0
        for propItem in props:
            # import pdb; pdb.set_trace()
            if(candRels != None and propItem[0] not in candRels):
                continue
            variables = self.searchObjectWithSubjectProp(entity, propItem[0])
            variablesList = [item[0] for item in variables]
            if(len(variablesList) > 0):
                # import pdb; pdb.set_trace()
                mergeTriples.append((entity, propItem[0], variablesList))
            else:
                import pdb;pdb.set_trace()
                print('关系有误：', propItem)
            numTriples += len(variables)
            if(numTriples > MAX_TRIPLES_NUM_EXPAND):
                break
        return mergeTriples
    
    def searchWithEntityIDsBasedProp(self, entitys: List[str], props: List[str], limitNum: int = 100, candRels = None, candRelsList: List[str] = None,):
        entitys = copy.deepcopy(entitys)
        # for i in range(len(entitys)):
        #     entitys[i] = self.normNodeForSql(entitys[i])
        mergeTriples = []
        numTriples = 0
        for propItem in props:
            # import pdb; pdb.set_trace()
            if(candRels != None and propItem not in candRels):
                continue
            variablesList = []
            for entity in entitys:
                variables = self.searchObjectWithSubjectIdProp(entity, propItem, MAX_ANSWER_NUM)
                variablesList.extend([item[0] for item in variables])
            if(len(variablesList) > 0):
                # import pdb; pdb.set_trace()
                mergeTriples.append((entity, propItem, variablesList))
            else:
                import pdb;pdb.set_trace()
                print('关系有误：', propItem)
            numTriples += len(variables)
            if(numTriples > MAX_TRIPLES_NUM_EXPAND):
                break
        return mergeTriples

    def searchWithEntityIDsBasedTopProp(self, entitys: List[str], props: List[str], limitNum: int = 100, candRelsList: List[str] = None, topn = 3):
        entitys = copy.deepcopy(entitys)
        # for i in range(len(entitys)):
        #     entitys[i] = self.normNodeForSql(entitys[i])
        propDic = {item: 1 for item in props}
        mergeTriples = []
        numTriples = 0
        selectNum = 0
        # print(candRelsList)
        for rel in candRelsList:
            if(rel in propDic):
                # print(rel)
                selectNum += 1
                variablesList = []
                for entity in entitys:
                    variables = self.searchObjectWithSubjectProp(entity, rel, 100)
                    variablesList.extend([item[0] for item in variables])
                if(len(variablesList) > 0):
                    # import pdb; pdb.set_trace()
                    mergeTriples.append((entity, rel, variablesList))
                else:
                    # import pdb;pdb.set_trace()
                    print('关系有误：', rel)
                numTriples += len(variables)
                if(selectNum > topn): # 最多取topn个关系进行搜索
                    break
                if(numTriples > MAX_TRIPLES_NUM_EXPAND):
                    break
                # if('宗教信仰' in rel):
                #     import pdb; pdb.set_trace()
        return mergeTriples

    def searchWithEntityIDBasedTopProp(self, entity: str, limitNum: int = 100, candRelsList: List[str] = None, topn = 3):
        entity = self.normNodeForSql(entity)
        sql = "select distinct prop from `pkubase` where `entry` = '%s' limit %s"\
                                        % (entity, str(limitNum))
        props = self.searchSQL(sql)
        propDic = {item[0]: 1 for item in props}
        mergeTriples = []
        numTriples = 0
        selectNum = 0
        # print(candRelsList)
        for rel in candRelsList:
            if(rel in propDic):
                # print(rel)
                selectNum += 1
                variables = self.searchObjectWithSubjectProp(entity, rel)
                variablesList = [item[0] for item in variables]
                if(len(variablesList) > 0):
                    # import pdb; pdb.set_trace()
                    mergeTriples.append((entity, rel, variablesList))
                else:
                    import pdb;pdb.set_trace()
                    print('关系有误：', rel)
                numTriples += len(variables)
                if(selectNum > topn): # 最多取topn个关系进行搜索
                    break
                if(numTriples > MAX_TRIPLES_NUM_EXPAND):
                    break
                # if('宗教信仰' in rel):
                #     import pdb; pdb.set_trace()
        return mergeTriples


    def searchWithValueIDBasedProp(self, entity, limitNum: int = 100, candRels = None):
        '''
        功能：根据尾实体，逆向扩展三元组
        输入：
            entity:尾实体
        输出：
            mergeTriples: 与尾实体相关的三元组集合
        '''
        entity = self.normNodeForSql(entity)
        sql = "select distinct prop from `pkubase` where `value` = '%s' limit %s"\
                                        % (entity, str(limitNum))
        numTriples = 0
        props = self.searchSQL(sql)
        # self.cursor.execute(sql)
        # props = self.cursor.fetchall()
        mergeTriples = []
        for propItem in props:
            if(candRels != None and propItem[0] not in candRels):
                continue
            variables = self.searchSubjectWithPropObject(propItem[0], entity)
            variablesList = [item[0] for item in variables]
            if(len(variablesList) > 0):
                # import pdb; pdb.set_trace()
                mergeTriples.append((variablesList, propItem[0], entity))
            else:
                import pdb;pdb.set_trace()
                print('关系有误：', propItem)
            numTriples += len(variables)
            if(numTriples > MAX_TRIPLES_NUM_EXPAND):
                break
        return mergeTriples

    def searchWithValueIDsBasedProp(self, entitys: List[str], props: List[str], limitNum: int = 100, candRels = None):
        '''
        功能：根据尾实体，逆向扩展三元组
        输入：
            entity:尾实体
        输出：
            mergeTriples: 与尾实体相关的三元组集合
        '''
        numTriples = 0
        entitys = copy.deepcopy(entitys)
        # for i in range(len(entitys)):
        #     entitys[i] = self.normNodeForSql(entitys[i])
        mergeTriples = []
        for propItem in props:
            if propItem == '':
                continue
            # propItem = self.normNodeForSql(propItem)
            if(candRels != None and propItem not in candRels):
                continue
            # print(propItem, entitys[0:2])
            variablesList = []
            for entity in entitys:
                # entity = self.normNodeForSql(entity)
                variables = self.searchSubjectWithPropObjectId(propItem, entity, MAX_ANSWER_NUM)
                variablesList.extend([item[0] for item in variables])
            # print(variablesList)
            if(len(variablesList) > 0):
                # import pdb; pdb.set_trace()
                mergeTriples.append((variablesList, propItem, entity))
            else:
                import pdb;pdb.set_trace()
                print('关系有误：', propItem)
            numTriples += len(variables)
            if(numTriples > MAX_TRIPLES_NUM_EXPAND):
                break
        return mergeTriples
    
    def searchMentionWithEntity(self, entity: str, limitNum: int = 100):
        entity = self.normNodeForSql(entity)
        sql = "select entry, prop from `pkuorder` where `prop` = '%s' limit %s"\
                                         % (entity, str(limitNum))
        triples = self.searchSQL(sql)
        # import pdb; pdb.set_trace()
        return triples
    
    def searchObjectWithSubjectProp(self, subject: str, prop: str, limitNum: int = MAX_VARIABLE_NUM):
        subject = subject.replace('\\', '\\\\')
        subject = subject.replace('\'', '\\\'')
        prop = self.normNodeForSql(prop)
        sql = "select value from `pkubase` where (`entry` = '%s' or `entry`='<%s>' or `entry`='\"%s\"') and `prop` = '%s' limit %s"\
                                            % (subject, subject, subject, prop, str(limitNum))
        objects = self.searchSQL(sql)
        return objects

    def searchObjectWithSubjectIdProp(self, subject: str, prop: str, limitNum: int = MAX_VARIABLE_NUM):
        subject = self.normNodeForSql(subject)
        prop = self.normNodeForSql(prop)
        sql = "select value from `pkubase` where (`entry` = '%s') and `prop` = '%s' limit %s"\
                                            % (subject, prop, str(limitNum))
        objects = self.searchSQL(sql)
        return objects

    def searchTripleWithSubjectProp(self, subject: str, prop: str, limitNum: int = MAX_VARIABLE_NUM):
        subject = subject.replace('\\', '\\\\')
        subject = subject.replace('\'', '\\\'')
        sql = "select entry, prop, value from `pkubase` where (`entry` = '%s' or `entry`='<%s>' or `entry`='\"%s\"') and `prop` = '%s' limit %s"\
                                            % (subject, subject, subject, prop, str(limitNum))
        objects = self.searchSQL(sql)
        return objects

    def searchSubjectWithPropObject(self, prop: str, objectItem: str, limitNum: int = MAX_VARIABLE_NUM):
        entity = self.normNodeForSql(objectItem)
        prop = self.normNodeForSql(prop)
        sql = "select entry from `pkubase` where `prop` = '%s' and (`value` = '%s' or `value`='<%s>') limit %s"\
                                            % (prop, entity, entity, str(limitNum))
        subjects = self.searchSQL(sql)
        return subjects
        
    def searchSubjectWithPropObjectId(self, prop: str, objectItem: str, limitNum: int = MAX_VARIABLE_NUM):
        prop = self.normNodeForSql(prop)
        entity = self.normNodeForSql(objectItem)
        sql = "select entry from `pkubase` where `prop` = '%s' and `value` = '%s' limit %s"\
                                            % (prop, entity, str(limitNum))
        subjects = self.searchSQL(sql)
        return subjects

    def isExistTriple(self, entry, prop, value):
        value = value.replace('\'', '\\\'')
        entry = entry.replace('\'', '\\\'')
        sql = "select entry from `pkubase` where `entry` = '%s' and `prop` = '%s' and `value` = '%s'"\
                                            % (entry, prop, value)
        # print(sql)
        # sql = "select entry from `pkubase` where `entry` = '<Don\\'t Leave Me>' and `prop` = '<原产地>' and `value` = '<非洲_（世界七大洲之一）>'"
        # import pdb; pdb.set_trace()
        if(sql not in self.sql2results):
            self.cursor.execute(sql)
            results = self.cursor.fetchall()
            self.sql2results[sql] = results
        else:
            results = self.sql2results[sql]
        
        if(len(results) > 0):
            # print(results, sql)
            return True
        else:
            # print('err', results, sql)
            return False

    def isValidTriple(self, entry, prop, value):
        value = value.replace('\'', '\\\'')
        entry = entry.replace('\'', '\\\'')
        # print(value)
        # value = self.norm
        # import pdb; pdb.set_trace()
        if('<' in value):
            sql = "select value from `pkubase` where `entry` = '%s' and `prop` = '%s'"\
                                                % (entry, prop)
            if(sql not in self.sql2results):
                self.cursor.execute(sql)
                results = self.cursor.fetchall()
                self.sql2results[sql] = results
            else:
                results = self.sql2results[sql]
            # import pdb; pdb.set_trace()
            if(len(results) > 0):
                baseValue = float(value[2:])
                pos = results[0][0].find('<http://www')
                intValue = float(results[0][0][1: pos - 3])
                if(intValue <= baseValue):
                    return True
                # print(results, sql)
                return False
            else:
                # print('err', results, sql)
                return False
        else:
            # print('非实体约束未覆盖')
            # import pdb; pdb.set_trace()
            sql = "select entry from `pkubase` where `entry` = '%s' and `prop` = '%s'"\
                                                % (entry, prop, value)
            if(sql not in self.sql2results):
                self.cursor.execute(sql)
                results = self.cursor.fetchall()
                self.sql2results[sql] = results
            else:
                results = self.sql2results[sql]
            # import pdb; pdb.set_trace()
            if(len(results) > 0):
                # print(results, sql)
                return True
            else:
                # print('err', results, sql)
                return False
        print('error:', sql)
        return False


    def constrainVariable(self, objectsList, queryGraph, orderEntityConstrain):
        '''
        功能：根据约束路径过滤变量节点值
        输入：
            objectsList：未过滤前的节点值,
            queryGraph：当前查询图；
            orderEntityConstrain: 第几个实体约束信息
        输出：
            newEntitys：约束后的实体集合
        '''
        isForward = (queryGraph.entityIDs[orderEntityConstrain] + 1) % 3 # 为0说明实体在宾语位置，为1说明约束实体在主语位置
        newEntitys = []
        for entity in objectsList:
            prop = queryGraph.entityConstrainTriples[orderEntityConstrain][1]
            if(isForward == 0):
                entry = entity
                value = queryGraph.entityConstrainTriples[orderEntityConstrain][2]
            else:
                entry = queryGraph.entityConstrainTriples[orderEntityConstrain][0]
                value = entity
            # print(queryGraph.entityConstrainTriples[orderEntityConstrain], entry, prop, value)
            if(self.isExistTriple(entry, prop, value)):
                newEntitys.append(entity)
        return newEntitys

    def virtualConstrainVariable(self, objectsList, queryGraph, orderEntityConstrain):
        '''
        功能：根据非实体约束路径过滤变量节点值
        输入：
            objectsList：未过滤前的节点值,
            queryGraph：当前查询图；
            orderEntityConstrain: 第几个实体约束信息
        输出：
            newEntitys：约束后的实体集合
        '''
        newEntitys = []
        for entity in objectsList:
            prop = queryGraph.virtualConstrainTriples[orderEntityConstrain][1]
            entry = entity
            value = queryGraph.virtualConstrainTriples[orderEntityConstrain][2]
            # print(queryGraph.entityConstrainTriples[orderEntityConstrain], entry, prop, value)
            if(self.isValidTriple(entry, prop, value)):
                newEntitys.append(entity)
        # if(len(objectsList) != len(newEntitys)):
        #     import pdb; pdb.set_trace()
        return newEntitys

    def cvt2name(self, cvts: List[str]):
        names = []
        for cvt in cvts:
            objects = self.searchObjectWithSubjectProp(cvt, '<实体名称>')
            # print(objects, cvt)
            if(len(objects) > 0):
                names.append(objects[0][0])
            # import pdb; pdb.set_trace()
        return names

    def searchAnswer(self, queryGraph: QueryGraph, higherOrder = []):
        initNodeId = queryGraph.initNodeId
        variableNodeIds = queryGraph.variableNodeIds
        pathTriples = queryGraph.pathTriples
        # answerNodeId = queryGraph.answerNodeId
        entityConstrainIds = queryGraph.entityConstrainIDs
        virtualEntityConstrainIds = queryGraph.virtualConstrainIDs
        higherOrderConstrainIDs = queryGraph.higherOrderConstrainIDs
        i = 0
        initEntitys = []
        initEntitys.append(pathTriples[initNodeId])
        while(i < len(pathTriples)): # 路径延伸
            tripleOrder = i // 3 # 主路径上的第几个三元组
            forwardFlag = (variableNodeIds[tripleOrder] + 1) % 3
            # import pdb; pdb.set_trace()
            if(forwardFlag == 0): # 正向
                objectsList = []
                for subject in initEntitys:
                    objects = self.searchObjectWithSubjectProp(subject, \
                                                                pathTriples[variableNodeIds[tripleOrder] - 1],
                                                                limitNum=MAX_ANSWER_NUM)
                    objectsList.extend([item[0] for item in objects])
                if(variableNodeIds[tripleOrder] in entityConstrainIds):
                    for orderEntityConstrain, item in enumerate(entityConstrainIds):
                        if(item == variableNodeIds[tripleOrder]):
                            objectsList = self.constrainVariable(objectsList, queryGraph, orderEntityConstrain)
                if(variableNodeIds[tripleOrder] in virtualEntityConstrainIds):
                    for orderEntityConstrain, item in enumerate(virtualEntityConstrainIds):
                        if(item == variableNodeIds[tripleOrder]):
                            objectsList = self.virtualConstrainVariable(objectsList, queryGraph, orderEntityConstrain)
                    # import pdb; pdb.set_trace()
                initEntitys = objectsList
                if(variableNodeIds[tripleOrder] in higherOrderConstrainIDs):
                    for order, item in enumerate(higherOrderConstrainIDs):
                        if(item == higherOrderConstrainIDs[order]):
                            objectsList = self.higherOrderConstrainVariable(objectsList, queryGraph, order)
                initEntitys = objectsList
                # import pdb; pdb.set_trace()
            elif(forwardFlag == 2):
                initEntitys = self.searchRelWithEntryValue(pathTriples[i], pathTriples[i + 2])
                # import pdb; pdb.set_trace()
            else:
                entitysList = []
                for objectItem in initEntitys:
                    subjects = self.searchSubjectWithPropObject(pathTriples[variableNodeIds[tripleOrder] + 1], objectItem,
                        limitNum=MAX_ANSWER_NUM)
                    # print(subjects)
                    entitysList.extend([item[0] for item in subjects])
                if(variableNodeIds[tripleOrder] in entityConstrainIds):
                    for orderEntityConstrain, item in enumerate(entityConstrainIds):
                        if(item == variableNodeIds[tripleOrder]):
                            entitysList = self.constrainVariable(entitysList, queryGraph, orderEntityConstrain)
                if(variableNodeIds[tripleOrder] in virtualEntityConstrainIds):
                    for orderEntityConstrain, item in enumerate(virtualEntityConstrainIds):
                        if(item == variableNodeIds[tripleOrder]):
                            entitysList = self.virtualConstrainVariable(entitysList, queryGraph, orderEntityConstrain)
                initEntitys = entitysList
            i += 3
        # 处理高阶函数
        # if(len(higherOrder) > 0):
        #     self.addHigherOrderConstrain(initEntitys, higherOrder)
        #     # import pdb; pdb.set_trace()
        if(len(initEntitys) > 0 and '<CVT' in initEntitys[0]):
            newInitEntitys = self.cvt2name(initEntitys)
            if(len(newInitEntitys) > 0):
                initEntitys = newInitEntitys
            # import pdb; pdb.set_trace()
        return initEntitys

    def searchAnswerBySQL(self, queryGraph: QueryGraph, higherOrder = []):
        initNodeId = queryGraph.initNodeId
        variableNodeIds = queryGraph.variableNodeIds
        pathTriples = copy.deepcopy(queryGraph.pathTriples)
        # answerNodeId = queryGraph.answerNodeId
        entityConstrainIds = queryGraph.entityConstrainIDs
        virtualEntityConstrainIds = queryGraph.virtualConstrainIDs
        higherOrderConstrainIDs = queryGraph.higherOrderConstrainIDs
        if(len(virtualEntityConstrainIds) > 0 or len(higherOrderConstrainIDs) > 0 or len(entityConstrainIds) > 0): # mysql不方便处理这种类型的查询
            return self.searchAnswer(queryGraph=queryGraph, higherOrder= higherOrder)
        i = 0
        initEntitys = []
        initEntitys.append(pathTriples[initNodeId])
        # import pdb; pdb.set_trace()
        node2kb = {}
        # while(i < len(pathTriples)):
        #     i + 3
        # i = 0
        kgSection = ''''''
        whereSection = ''''''
        constrainNum = 0
        answerCol = ''''''
        preVar = ''''''
        while(i < len(pathTriples)): # 路径延伸
            tripleOrder = i // 3 # 主路径上的第几个三元组
            forwardFlag = (variableNodeIds[tripleOrder] + 1) % 3
            currentKg = "kg" + str(tripleOrder)
            kgSection += " pkubase as " + currentKg + ","
            pathTriples[i + 0] = self.normNodeForSql(pathTriples[i + 0])
            pathTriples[i + 1] = self.normNodeForSql(pathTriples[i + 1])
            pathTriples[i + 2] = self.normNodeForSql(pathTriples[i + 2])
            if(forwardFlag == 0): # 正向
                answerCol = ".value"
                if(whereSection != ''):
                    # whereSection += " and "
                    whereSection += ''' and %s.entry = %s and %s.prop = '%s' ''' \
                                    %(currentKg, preVar,\
                                    currentKg, pathTriples[variableNodeIds[tripleOrder] - 1])
                else:
                    whereSection += '''%s.entry = '%s' and %s.prop = '%s' ''' \
                                    %(currentKg, pathTriples[variableNodeIds[tripleOrder] - 2],\
                                        currentKg, pathTriples[variableNodeIds[tripleOrder] - 1])
                preVar = currentKg + '.value'
                if(variableNodeIds[tripleOrder] in entityConstrainIds):
                    for orderEntityConstrain, item in enumerate(entityConstrainIds):
                        if(item == variableNodeIds[tripleOrder]):
                            currentConstrainKg = 'conKg' + str(constrainNum)
                            kgSection += " pkubase as " + currentConstrainKg + ","
                            entityPathTriples = queryGraph.entityConstrainTriples[orderEntityConstrain]
                            entityPathTriples[0] = self.normNodeForSql(entityPathTriples[0])
                            entityPathTriples[1] = self.normNodeForSql(entityPathTriples[1])
                            entityPathTriples[2] = self.normNodeForSql(entityPathTriples[2])
                            isForward = (queryGraph.entityIDs[orderEntityConstrain] + 1) % 3 # 为0说明实体在宾语位置，为1说明约束实体在主语位置
                            if(whereSection != ''):
                                whereSection += " and "
                            if(isForward == 1): # 正向搜索三元组
                                whereSection += ''' %s.entry = '%s' and %s.prop = '%s' and %s.value = %s.value''' \
                                                %(currentConstrainKg, entityPathTriples[0], currentConstrainKg, entityPathTriples[1], currentConstrainKg, currentKg)
                            elif(isForward == 0):
                                whereSection += ''' %s.value = '%s' and %s.prop = '%s' and %s.entry = %s.value''' \
                                                %(currentConstrainKg, entityPathTriples[2], currentConstrainKg, entityPathTriples[1], currentConstrainKg, currentKg)
                            constrainNum += 1
                            # import pdb; pdb.set_trace()
                            break
            elif(forwardFlag == 2):
                answerCol = ".prop"
                if(whereSection != ''):
                    whereSection += " and "
                whereSection += ''' %s.entry = '%s' and %s.value = '%s' ''' \
                                    %(currentKg, pathTriples[i], currentKg, pathTriples[i + 2])
                # initEntitys = self.searchRelWithEntryValue(pathTriples[i], pathTriples[i + 2])
            else:
                answerCol = ".entry"
                if(whereSection != ''):
                    # whereSection += " and "
                    whereSection += ''' and %s.value = %s and %s.prop = '%s' ''' \
                                    %(currentKg, preVar,\
                                        currentKg, pathTriples[variableNodeIds[tripleOrder] + 1])
                else:
                    whereSection += ''' %s.value = '%s' and %s.prop = '%s' ''' \
                                    %(currentKg, pathTriples[variableNodeIds[tripleOrder] + 2],\
                                        currentKg, pathTriples[variableNodeIds[tripleOrder] + 1])
                preVar = currentKg + '.entry'
                if(variableNodeIds[tripleOrder] in entityConstrainIds):
                    for orderEntityConstrain, item in enumerate(entityConstrainIds):
                        if(item == variableNodeIds[tripleOrder]):
                            currentConstrainKg = 'conKg' + str(constrainNum)
                            kgSection += " pkubase as " + currentConstrainKg + ","
                            entityPathTriples = queryGraph.entityConstrainTriples[orderEntityConstrain]
                            entityPathTriples[0] = self.normNodeForSql(entityPathTriples[0])
                            entityPathTriples[1] = self.normNodeForSql(entityPathTriples[1])
                            entityPathTriples[2] = self.normNodeForSql(entityPathTriples[2])
                            isForward = (queryGraph.entityIDs[orderEntityConstrain] + 1) % 3 # 为0说明实体在宾语位置，为1说明约束实体在主语位置
                            if(whereSection != ''):
                                whereSection += " and "
                            if(isForward == 1): # 正向搜索三元组
                                whereSection += ''' %s.entry = '%s' and %s.prop = '%s' and %s.value = %s.entry''' \
                                                %(currentConstrainKg, entityPathTriples[0], currentConstrainKg, entityPathTriples[1], currentConstrainKg, currentKg)
                            elif(isForward == 0):
                                whereSection += ''' %s.value = '%s' and %s.prop = '%s' and %s.entry = %s.entry''' \
                                                %(currentConstrainKg, entityPathTriples[2], currentConstrainKg, entityPathTriples[1], currentConstrainKg, currentKg)
                            constrainNum += 1
                            # import pdb; pdb.set_trace()
                            break
            i += 3
        answerKg = "kg" + str(queryGraph.answerNodeId // 3) + answerCol
        sql = '''select distinct ''' + answerKg + ''' from ''' + kgSection[0: -1] + ''' where ''' + whereSection + ''' limit 10000'''
        # print(sql)
        # print(queryGraph.answerNodeId, pathTriples, queryGraph.entityConstrainTriples)
        results = self.searchSQL(sql)
        initEntitys = [item[0] for item in results]
        # import pdb; pdb.set_trace()
        if(len(initEntitys) > 0 and '<CVT' in initEntitys[0]):
            newInitEntitys = self.cvt2name(initEntitys)
            if(len(newInitEntitys) > 0):
                initEntitys = newInitEntitys
            # import pdb; pdb.set_trace()
        return initEntitys

    def higherOrderConstrainVariable(self, objectsList, queryGraph, order):
        newEntitys = []
        triples = []
        prop = queryGraph.higherOrderTriples[order][1]
        value = queryGraph.higherOrderTriples[order][2]
        for entity in objectsList:
            entry = entity
            # print(queryGraph.entityConstrainTriples[orderEntityConstrain], entry, prop, value)
            triples.extend(self.searchTripleWithSubjectProp(entry, prop))
        if(value == 'argmax'):
            newEntitys = self.getArgMaxEntity(triples)
            # if('<名人国际大酒店>' in newEntitys):
            #     import pdb; pdb.set_trace()
            # if(self.isValidTriple(entry, prop, value)):
            #     newEntitys.append(entity)
        # if(len(objectsList) != len(newEntitys)):
        #     import pdb; pdb.set_trace()
        return newEntitys

    def addHigherOrderConstrainForQueryGraph(self, queryGraph, higherOrder, candRelsList, constrainVarIndex = -1):
        '''
        功能：为查询图增加高阶约束
        '''
        flag = False
        pathTriples = queryGraph.pathTriples
        variableNodeIds = queryGraph.variableNodeIds
        if(constrainVarIndex == -1):
            variableNodes = queryGraph.variableNodes[constrainVarIndex]
            nodeId = variableNodeIds[-1]
        else:
            nodeId = constrainVarIndex
            # import pdb; pdb.set_trace()
            pos = variableNodeIds.index(constrainVarIndex)
            variableNodes = queryGraph.variableNodes[pos]
        # import pdb; pdb.set_trace()
        forwardConstrain = self.addHigherOrderConstrain(variableNodes, higherOrder, candRelsList)
        if(len(forwardConstrain) > 0):
            flag = True
            # print('forward:', [forwardConstrain[0], forwardConstrain[1], forwardConstrain[2][0]])
            # print(queryGraph.pathTriples)
            # import pdb; pdb.set_trace()
            queryGraph.updateHigherOrderConstrainInfo([forwardConstrain[0][0], forwardConstrain[1], forwardConstrain[2]], nodeId)
            queryGraph.updateVariableNodes(forwardConstrain[0])
            queryGraph.updateAnswer('\t'.join(forwardConstrain[0]))
            return flag, queryGraph
        return flag, queryGraph

    
    def searchQueryGraphsWithBasePath(self, entitys, queType = ''):
        graphs = []
        for entity in entitys:
            queryGraphs = self.searchWithEntityBasedProp(entity)
            graphs.append(queryGraphs)
        if(queType == 'compareJudge'):  # A和B相等么？
            queryGraphs = self.judgeBasePaths(graphs)
        elif(queType == 'NumericalCalculation'):
            queryGraphs = self.calculationBasePaths(graphs)
        elif(queType == '比较推理(计算)'):
            queryGraphs = self.calculationBasePaths(graphs)
        elif(queType == 'ComparisonSelectionMax'):
            queryGraphs = self.comparisonSelectionMax(graphs)
        elif(queType == 'ComparisonSelectionMin'):
            queryGraphs = self.comparisonSelectionMin(graphs)
        return queryGraphs

    def twoNodesCircle(self, rel1, rel2, limitNum = 10):
        results = self.searchCount('prop', rel1)
        num1 = int(results[0][0]) + 1
        results = self.searchCount('prop', rel2)
        num2 = int(results[0][0]) + 1
        # import pdb; pdb.set_trace()
        # print(num1 * num2)
        if(num1 * num2 > 1000000):
            return False, ''
            # import pdb; pdb.set_trace()
        # sql = "select a.entry, a.value from pkubase as a, pkubase as b where a.prop = '%s' and b.prop = '%s' and a.entry = b.entry and a.value = b.value limit %s"\
        #         %(rel1, rel2, str(limitNum))
        sql = "select a.entry, a.value from pkubase as a, pkubase as b where a.prop = '%s' and b.prop = '%s' and a.entry = b.entry and a.value = b.value limit %s"\
                %(rel1, rel2, str(limitNum))
        if(sql not in self.sql2results):
            self.cursor.execute(sql)
            results = self.cursor.fetchall()
            self.sql2results[sql] = results
        else:
            results = self.sql2results[sql]
        # print(sql, '完成')
        if(len(results) > 0):
            queryGraph = QueryGraph()
            basePath1 = [(results[0][0], rel1, results[0][1])]
            varIds = [0, 2]
            queryGraph.updateBasePath(basePath1, varIds)
            basePath2 = [(results[0][0], rel2, results[0][1])]
            queryGraph.updateBasePath(basePath2, varIds)
            answers = [item[0] for item in results]
            queryGraph.updateAnswer('\t'.join(answers))
            return True, queryGraph
        return False, ''

    def searchCount(self, key: str, value: str):
        sql = "select count(*) from pkubase where %s = '%s'" % (key, value)
        results = self.searchSQL(sql)
        return results

    def searchTriplesWithRel(self, rel: str):
        sql = "select entry, prop, value from pkubase where prop = '%s'" % (rel)
        results = self.searchSQL(sql)
        return results

    def threeNodesCircleIn(self, rel1, rel2, limitNum = 10):
        '''
        功能：实现双关系约束，引入的第三个实体作为头实体
        # 时间复杂度：
        # num1>10000时， 
        '''
        results = self.searchCount('prop', rel1)
        num1 = int(results[0][0]) + 1
        # results = self.searchCount('prop', rel2)
        # num2 = int(results[0][0]) + 1
        # import pdb; pdb.set_trace()
        # print(num1 * num2)
        if(num1 > 10000):
            # print('too much candidate'.format(rel1))
            return False, ''
        # sql = "select a.entry, a.value, b.entry from (select entry, value from pkubase where prop = '%s' ) as a , pkubase as b, pkubase as c where b.prop = '%s' and c.prop = b.prop and b.entry = c.entry and  (a.entry = b.value  and a.value = c.value) limit %s"\
        #         %(rel1, rel2, str(limitNum))
        sql = "select a.entry, a.value, b.entry from (select entry, value from pkubase where prop = '%s' ) as a , (select entry, value from pkubase where prop = '%s' ) as b, (select entry, value from pkubase where prop = '%s' ) as c where b.entry = c.entry and  (a.entry = b.value  and a.value = c.value) limit %s"\
                %(rel1, rel2, rel2, str(limitNum))
        results = self.searchSQL(sql)
        if(len(results) > 0):
            # import pdb; pdb.set_trace()
            queryGraph = QueryGraph()
            basePath1 = [(results[0][0], rel1, results[0][1])]
            varIds = [0, 2]
            queryGraph.updateBasePath(basePath1, varIds)
            basePath2 = [(results[0][2], rel2, results[0][0])]
            queryGraph.updateBasePath(basePath2, varIds)
            basePath2 = [(results[0][2], rel2, results[0][1])]
            queryGraph.updateBasePath(basePath2, varIds)
            answers = [item[0] for item in results]
            queryGraph.updateAnswer('\t'.join(answers))
            return True, queryGraph
        return False, ''

    def threeNodesCircleOut(self, rel1, rel2, limitNum = 10):
        '''
        功能：实现双关系约束，引入的第三个实体作为尾实体
        # 时间复杂度：
        # num1>10000时， 
        '''
        results = self.searchCount('prop', rel1)
        num1 = int(results[0][0]) + 1
        # results = self.searchCount('prop', rel2)
        # num2 = int(results[0][0]) + 1
        # import pdb; pdb.set_trace()
        # print(num1 * num2)
        if(num1 > 10000):
            return False, ''
        # sql = "select a.entry, a.value, b.value from (select entry, value from pkubase where prop = '%s' ) as a , pkubase as b, pkubase as c where b.prop = '%s' and c.prop = b.prop and b.value = c.value and  (a.entry = b.entry  and a.value = c.entry) limit %s"\
        #         %(rel1, rel2, str(limitNum))
        sql = "select a.entry, a.value, b.value from (select entry, value from pkubase where prop = '%s' ) as a , (select entry, value from pkubase where prop = '%s' ) as b, (select entry, value from pkubase where prop = '%s' ) as c where b.value = c.value and  (a.entry = b.entry  and a.value = c.entry) limit %s"\
                %(rel1, rel2, rel2, str(limitNum))
        results = self.searchSQL(sql)
        # if(sql not in self.sql2results):
        #     self.cursor.execute(sql)
        #     results = self.cursor.fetchall()
        #     self.sql2results[sql] = results
        # else:
        #     results = self.sql2results[sql]
        # print(sql, '完成')
        if(len(results) > 0):
            # import pdb; pdb.set_trace()
            queryGraph = QueryGraph()
            basePath1 = [(results[0][0], rel1, results[0][1])]
            varIds = [0, 2]
            queryGraph.updateBasePath(basePath1, varIds)
            basePath2 = [(results[0][0], rel2, results[0][2])]
            queryGraph.updateBasePath(basePath2, varIds)
            basePath2 = [(results[0][1], rel2, results[0][2])]
            queryGraph.updateBasePath(basePath2, varIds)
            answers = [item[0] for item in results]
            queryGraph.updateAnswer('\t'.join(answers))
            return True, queryGraph
        return False, ''
        

    def generateQueryGraphsNoEntity(self, entitys, candRelsList, queType = ''):
        # import pdb;pdb.set_trace()
        graphs = []
        # self.threeNodesCircleOut('<母亲>','<出演>', limitNum=100)
        # self.threeNodesCircleIn('<母亲>','<主演>', limitNum=100)
        # self.threeNodesCircleOut('<母亲>','<出演>', limitNum=100)
        print('no entity, use rel:', candRelsList)
        for i, rel1 in enumerate(candRelsList):
            for j, rel2 in enumerate(candRelsList[0:]):
                # self.twoNodesCircle(rel1, rel2)
                # import pdb; pdb.set_trace()
                if(j > i):
                    flag, queryGraph = self.twoNodesCircle(rel1, rel2)
                    if(flag):
                        graphs.append(queryGraph)
                    flag, queryGraph = self.threeNodesCircleIn(rel1, rel2)
                    if(flag):
                        graphs.append(queryGraph)
                    flag, queryGraph = self.threeNodesCircleOut(rel1, rel2)
                    if(flag):
                        graphs.append(queryGraph)
                elif(j < i):
                    flag, queryGraph = self.threeNodesCircleIn(rel1, rel2)
                    if(flag):
                        graphs.append(queryGraph)
                    flag, queryGraph = self.threeNodesCircleOut(rel1, rel2)
                    if(flag):
                        graphs.append(queryGraph)
        return graphs


    def comparisonSelectionMin(self, graphs):
        '''
        功能：针对两个实体涉及的基础路径进行数值计算并选择正确的实体
        '''
        queryGraphs = []
        if(len(graphs) == 2):
            for item1 in graphs[0]:
                for item2 in graphs[1]:
                    answer1 = item1.answer
                    answer2 = item2.answer
                    # print(answer1)
                    float1 = getFloat(answer1)
                    float2 = getFloat(answer2)
                    # print(float1, float2)
                    pathTriples1 = item1.pathTriples
                    variableNodesIds1 = item1.variableNodeIds
                    pathTriples2 = item2.pathTriples
                    variableNodesIds2 = item2.variableNodeIds
                    if(pathTriples1[1] == pathTriples2[1] and float1 != '' and float2 != ''):
                        subValue = float(float1) - float(float2)
                        if(subValue > 0):
                            answer = pathTriples2[0]
                        else:
                            answer = pathTriples1[0]
                        queryGraph = QueryGraph(answer = answer)
                        # import pdb; pdb.set_trace()
                        queryGraph.updateBasePath(pathTriples1, variableNodesIds1)
                        queryGraph.updateBasePath(pathTriples2, variableNodesIds2)
                        queryGraphs.append(queryGraph)
            return queryGraphs
        return queryGraphs

    def comparisonSelectionMax(self, graphs):
        '''
        功能：针对两个实体涉及的基础路径进行数值计算并选择正确的实体
        '''
        queryGraphs = []
        if(len(graphs) == 2):
            for item1 in graphs[0]:
                for item2 in graphs[1]:
                    answer1 = item1.answer
                    answer2 = item2.answer
                    # print(answer1)
                    float1 = getFloat(answer1)
                    float2 = getFloat(answer2)
                    # print(float1, float2)
                    pathTriples1 = item1.pathTriples
                    variableNodesIds1 = item1.variableNodeIds
                    pathTriples2 = item2.pathTriples
                    variableNodesIds2 = item2.variableNodeIds
                    if(pathTriples1[1] == pathTriples2[1] and float1 != '' and float2 != ''):
                        subValue = float(float1) - float(float2)
                        if(subValue > 0):
                            answer = pathTriples1[0]
                        else:
                            answer = pathTriples2[0]
                        queryGraph = QueryGraph(answer = answer)
                        # import pdb; pdb.set_trace()
                        queryGraph.updateBasePath(pathTriples1, variableNodesIds1)
                        queryGraph.updateBasePath(pathTriples2, variableNodesIds2)
                        queryGraphs.append(queryGraph)
            return queryGraphs
        return queryGraphs

    def calculationBasePaths(self, graphs):
        '''
        功能：针对两个实体涉及的基础路径进行数值计算
        '''
        queryGraphs = []
        if(len(graphs) == 2):
            for item1 in graphs[0]:
                for item2 in graphs[1]:
                    answer1 = item1.answer
                    answer2 = item2.answer
                    # print(answer1)
                    float1 = getFloat(answer1)
                    float2 = getFloat(answer2)
                    # print(float1, float2)
                    pathTriples1 = item1.pathTriples
                    variableNodesIds1 = item1.variableNodeIds
                    pathTriples2 = item2.pathTriples
                    variableNodesIds2 = item2.variableNodeIds
                    if(pathTriples1[1] == pathTriples2[1] and float1 != '' and float2 != ''):
                        answer = float(float1) - float(float2)
                        if answer-int(answer) == 0:
                            answer = int(answer)
                        # unit = answer.find(float1)
                        queryGraph = QueryGraph(answer = str(answer))
                        # import pdb; pdb.set_trace()
                        queryGraph.updateBasePath(pathTriples1, variableNodesIds1)
                        queryGraph.updateBasePath(pathTriples2, variableNodesIds2)
                        queryGraphs.append(queryGraph)
            return queryGraphs
        return queryGraphs

    

    def judgeBasePaths(self, graphs):
        '''
        功能：针对两个实体涉及的基础路径进行逻辑判断，答案为“是”和“否”
        输入：
            graphs:目前是查询图形式，后续会修改为base路径
        '''
        queryGraphs = []
        if(len(graphs) == 2):
            for item1 in graphs[0]:
                for item2 in graphs[1]:
                    answer1 = item1.answer
                    answer2 = item2.answer
                    if(answer1 == answer2):
                        answer = '是'
                    else:
                        answer = '否'
                    queryGraph = QueryGraph(answer = answer)
                    pathTriples1 = item1.pathTriples
                    variableNodesIds1 = item1.variableNodeIds
                    pathTriples2 = item2.pathTriples
                    variableNodesIds2 = item2.variableNodeIds
                    queryGraph.updateBasePath(pathTriples1, variableNodesIds1)
                    queryGraph.updateBasePath(pathTriples2, variableNodesIds2)
                    queryGraphs.append(queryGraph)
            return queryGraphs
        return queryGraphs

    
    def addHigherOrderLogic(self, entitys, logic = '>'):
        pass


    def addHigherOrderConstrain(self, entitys, higherOrder, candRelsList):
        tripleList = []
        for node in entitys:
            newNode = node.replace('\\', '\\\\')
            newNode = newNode.replace('\'', '\\\'')
            sql = "select distinct entry, prop, value from `pkubase` where (`entry` = '%s' and `value` like '%%<http://www.w3.org/2001/XMLSchema#float>') limit %s" % (newNode, str(10))
            # import pdb; pdb.set_trace()
            if(sql not in self.sql2results):
                self.cursor.execute(sql)
                triples = self.cursor.fetchall()
                self.sql2results[sql] = triples
            else:
                triples = self.sql2results[sql]
            if(len(triples) > 0):
                tripleList.extend([triple for triple in triples])
        prop2triples = {}
        if(len(tripleList) > 0):
            for triple in tripleList:
                if(triple[1] not in prop2triples):
                    prop2triples[triple[1]] = [triple]
                else:
                    prop2triples[triple[1]].append(triple)
            selectProp = ''
            maxNum = 0
            for prop in prop2triples:
                if(len(prop2triples[prop]) > maxNum):
                    maxNum = len(prop2triples[prop])
                    selectProp = prop
            if(len(prop2triples) > 1):
                minIndex = float('inf')
                for prop in prop2triples:
                    if(prop in candRelsList):
                        index = candRelsList.index(prop)
                        if(index < minIndex):
                            minIndex = index
                            selectProp = prop
            if(higherOrder[0][1] == 'argmax'): # 取最大值
                triples = prop2triples[selectProp]
                newNode = self.getArgMaxEntity(triples)
                return [newNode, selectProp, 'argmax']
            elif(higherOrder[0][1] == 'argmin'):
                triples = prop2triples[selectProp]
                newNode = self.getArgMinEntity(triples)
                # print(newNode)
                # if('<平均价格>' in triples[0] and len(triples) > 1 and '<北京千禧大酒店>' in newNode):
                #     import pdb; pdb.set_trace()
                return [newNode, selectProp, 'argmin']
                # import pdb; pdb.set_trace()
        return []

    def getArgMaxEntity(self, triples):
        newNode = ['']
        currentValue = 0
        for triple in triples:
            value = self.value2float(triple[2])
            if(value > currentValue):
                newNode[0] = triple[0]
                currentValue = value
        return newNode

    def getArgMinEntity(self, triples):
        newNode = ['']
        currentValue = float('inf')
        for triple in triples:
            value = self.value2float(triple[2])
            # import pdb; pdb.set_trace()
            if(value < currentValue):
                newNode[0] = triple[0]
                currentValue = value
        return newNode

    def value2float(self, value: str):
        pos = value.find('<')
        intValue = float(value[1: pos - 3])
        return intValue

    def findForwardPropEntityConstrain(self, entity, node, limitNum = 20):
        entity = entity.replace('\'', '\\\'')
        node = node.replace('\'', '\\\'')
        try:
            sql = "select distinct prop, entry from `pkubase` where (`entry` = '%s' or `entry`='<%s>' or `entry`='\"%s\"') and `value` = '%s' limit %s"\
                                                % (entity, entity, entity, node, str(limitNum))
            if(sql not in self.sql2results):
                self.cursor.execute(sql)
                props = self.cursor.fetchall()
            else:
                props = self.sql2results[sql]
        except:
            props = ()
            print('sql error:', sql)
        rels = []
        entity = ''
        for prop in props:
            rels.append(prop[0])
            entity = prop[1]
        # print('rels entity:', rels, entity)
        return rels, entity

    def findBackwardPropEntityConstrain(self, entity, node, limitNum = 20):
        entity = entity.replace('\'', '\\\'')
        node = node.replace('\'', '\\\'')
        sql = "select distinct prop, value from `pkubase` where (`value` = '%s' or `value`='<%s>' or `value`='\"%s\"') and `entry` = '%s' limit %s"\
                                            % (entity, entity, entity, node, str(limitNum))
        if(sql not in self.sql2results):
            self.cursor.execute(sql)
            props = self.cursor.fetchall()
        else:
            props = self.sql2results[sql]
        rels = []
        entity = ''
        for prop in props:
            # import pdb; pdb.set_trace()
            rels.append(prop[0])
            entity = prop[1]
        return rels, entity

    def addEntityConstrain(self, currentQueryGraphs, entity):
        flag = False
        for queryGraph in currentQueryGraphs:
            pathTriples = queryGraph.pathTriples
            variableNodeIds = queryGraph.variableNodeIds
            variableNodes = queryGraph.variableNodes
            for i, nodeId in enumerate(variableNodeIds):
                # if('科学家' in pathTriples[2]):
                #     print(pathTriples, variableNodes[i])
                for node in variableNodes[i]:
                    # print(node, entity)
                    propsForward, entityLink1 = self.findForwardPropEntityConstrain(entity, node)
                    propsBackward, entityLink2 = self.findBackwardPropEntityConstrain(entity, node)
                    if(len(propsForward) > 0):
                        # print(propsForward, propsBackward, entityLink1, entity, node)
                        flag = True
                        queryGraph.updateEntityConstrainInfo([entityLink1, propsForward[0], node], nodeId, 0)
                        break
                        # import pdb; pdb.set_trace()
                    elif(len(propsBackward) > 0):
                        # print(propsForward, propsBackward, entityLink2, entity, node)
                        flag = True
                        # import pdb; pdb.set_trace()
                        queryGraph.updateEntityConstrainInfo([node, propsBackward[0], entityLink2], nodeId, 2)
                        # print(queryGraph.entityConstrainTriples)
                        # import pdb; pdb.set_trace()
                        break
        return flag

    def forwardEntityConstrain2(self, entity, variableNodes):
        '''
        功能：在变量节点上增加实体约束
        输入：
            entity:用于约束的实体
            variableNodes:被约束的节点值集合
        输出：
            props:约束路径中的关系
            nodes:约束后的节点值集合
        '''
        entityProp = {}
        key2answer = {}
        for varItem in variableNodes[0: MAX_ANSWER_NUM]:
            triples = self.searchWithEntityAndValueId(entity, varItem)
            for triple in triples:
                key = (triple[0], triple[1])
                if(key not in entityProp):
                    entityProp[key] = 1
                    key2answer[key] = [triple[2]]
                else:
                    entityProp[key] += 1
                    key2answer[key].append(triple[2])
        if(len(entityProp) > 0):
            sortedEntityProp = sorted(entityProp.items(), key=lambda x:x[1], reverse=True)
            return [sortedEntityProp[0][0][0], sortedEntityProp[0][0][1]] + [key2answer[sortedEntityProp[0][0]]]
        else:
            # import pdb; pdb.set_trace()
            return []
    def forwardEntityConstrain(self, entity, variableNodes):
        '''
        功能：在变量节点上增加实体约束
        输入：
            entity:用于约束的实体
            variableNodes:被约束的节点值集合
        输出：
            props:约束路径中的关系
            nodes:约束后的节点值集合
        '''
        entityProp = {}
        key2answer = {}
        # print('forward:' , len(variableNodes))
        # if(len(variableNodes) < MAX_VARIABLE_NUM):
        #     return self.forwardEntityConstrain2(entity, variableNodes)
        triples = self.searchWithEntity(entity, limitNum= MAX_CONSTRAIN_NUM)
        values = [item[2] for item in triples]
        # import pdb; pdb.set_trace()
        sameEntitys = set(values) & set(variableNodes)
        # print(len(sameEntitys), sameEntitys)
        
        if(len(sameEntitys) > 0):
            for triple in triples:
                if(triple[2] in sameEntitys):
                    key = (triple[0], triple[1])
                    if(key not in entityProp):
                        entityProp[key] = 1
                        key2answer[key] = [triple[2]]
                    else:
                        entityProp[key] += 1
                        key2answer[key].append(triple[2])
            sortedEntityProp = sorted(entityProp.items(), key=lambda x:x[1], reverse=True)
            return [sortedEntityProp[0][0][0], sortedEntityProp[0][0][1]] + [key2answer[sortedEntityProp[0][0]]]
        else:
            # import pdb; pdb.set_trace()
            return []

    def forwardRelConstrain(self, entity, rel, variableNodes):
        '''
        功能：在变量节点上增加关系约束
        输入：
            entity:入口实体
            rel:关系约束
            variableNodes:被约束的节点值集合
        输出：
            
        '''
        entityProp = {}
        key2answer = {}
        # triples = self.searchWithEntityBasedProp(entity, limitNum= MAX_ANSWER_NUM)
        triples = self.searchWithEntityidRel(entity, rel)
        values = [item[2] for item in triples]
        # import pdb; pdb.set_trace()
        sameEntitys = set(values) & set(variableNodes)
        if(len(sameEntitys) > 0):
            return [entity, rel] + [[item for item in sameEntitys]]
        else:
            # import pdb; pdb.set_trace()
            return []

    def backwardRelConstrain(self, value, rel, variableNodes):
        '''
        功能：在变量节点上增加关系约束
        输入：
            entity:入口实体
            rel:关系约束
            variableNodes:被约束的节点值集合
        输出：
            
        '''
        entityProp = {}
        key2answer = {}
        # triples = self.searchWithEntityBasedProp(entity, limitNum= MAX_ANSWER_NUM)
        triples = self.searchWithValueIdRel(value, rel)
        entitys = [item[0] for item in triples]
        # import pdb; pdb.set_trace()
        sameEntitys = set(entitys) & set(variableNodes)
        if(len(sameEntitys) > 0):
            return [[item for item in sameEntitys]] + [rel, value]
        else:
            # import pdb; pdb.set_trace()
            return []

    def backwardEntityConstrain2(self, entity, variableNodes):
        '''
        功能：在变量节点上增加实体约束
        输入：
            entity:用于约束的实体
            variableNodes:被约束的节点值集合
        输出：
            props:约束路径中的关系
            nodes:约束后的节点值集合
        '''
        entityProp = {}
        key2answer = {}
        # triples = self.searchWithValue(entity, limitNum= MAX_ANSWER_NUM)  # 待改进：当entity对应的三元组较多，而variableNodes较少时当前方式速度慢
        # entries = [item[0] for item in triples]
        # sameEntitys = set(entries) & set(variableNodes)
        # if(len(sameEntitys) > 0):
        for varItem in variableNodes[0:MAX_ANSWER_NUM]:
            triples = self.searchWithEntryIdAndValue(varItem, entity)
            for triple in triples:
                key = (triple[1], triple[2])
                if(key not in entityProp):
                    entityProp[key] = 1
                    key2answer[key] = [triple[0]]
                else:
                    entityProp[key] += 1
                    key2answer[key].append(triple[0])
        if(len(entityProp) > 0):
            sortedEntityProp = sorted(entityProp.items(), key=lambda x:x[1], reverse=True)
            return [key2answer[sortedEntityProp[0][0]]] + [sortedEntityProp[0][0][0], sortedEntityProp[0][0][1]]
        else:
            return []

    def backwardEntityConstrain(self, entity, variableNodes):
        '''
        功能：在变量节点上增加实体约束
        输入：
            entity:用于约束的实体
            variableNodes:被约束的节点值集合
        输出：
            props:约束路径中的关系
            nodes:约束后的节点值集合
        '''
        entityProp = {}
        key2answer = {}
        # print('backward:' , len(variableNodes))
        # if(len(variableNodes) < MAX_VARIABLE_NUM):
        #     return self.backwardEntityConstrain2(entity, variableNodes)
        triples = self.searchWithValue(entity, limitNum= MAX_CONSTRAIN_NUM)  # 待改进：当entity对应的三元组较多，而variableNodes较少时当前方式速度慢
        entries = [item[0] for item in triples]
        sameEntitys = set(entries) & set(variableNodes)
        # print('over')
        if(len(sameEntitys) > 0):
            for triple in triples:
                if(triple[0] in sameEntitys):
                    key = (triple[1], triple[2])
                    # print(key)
                    # import pdb; pdb.set_trace()
                    if(key not in entityProp):
                        entityProp[key] = 1
                        key2answer[key] = [triple[0]]
                    else:
                        entityProp[key] += 1
                        key2answer[key].append(triple[0])
            sortedEntityProp = sorted(entityProp.items(), key=lambda x:x[1], reverse=True)
            # import pdb; pdb.set_trace()
            return [key2answer[sortedEntityProp[0][0]]] + [sortedEntityProp[0][0][0], sortedEntityProp[0][0][1]]
        else:
            return []


    def addEntityConstrainForQueryGraph(self, queryGraph, entity, constrainVarIndex = -1):
        '''
        功能：在给定查询图的变量节点上增加实体约束
        输入：
            queryGraph:待约束的查询图
            entity:约束实体
        输出：
            flag:约束挂载是否成功，True为成功，False为失败
            queryGraph:约束挂载后的查询图
        '''
        flag = False
        pathTriples = queryGraph.pathTriples
        variableNodeIds = queryGraph.variableNodeIds
        if(constrainVarIndex == -1):
            variableNodes = queryGraph.variableNodes[constrainVarIndex]
            nodeId = variableNodeIds[-1]
        else:
            nodeId = constrainVarIndex
            # import pdb; pdb.set_trace()
            pos = variableNodeIds.index(constrainVarIndex)
            variableNodes = queryGraph.variableNodes[pos]
        # if()
        forwardConstrain = self.forwardEntityConstrain(entity, variableNodes)
        # print('over')
        if(len(forwardConstrain) > 0):
            flag = True
            # print('forward:', [forwardConstrain[0], forwardConstrain[1], forwardConstrain[2][0]])
            # print(queryGraph.pathTriples)
            queryGraph.updateEntityConstrainInfo([forwardConstrain[0], forwardConstrain[1], forwardConstrain[2][0]], nodeId, 0)
            queryGraph.updateVariableNodes(forwardConstrain[2])
            queryGraph.updateAnswer('\t'.join(forwardConstrain[2]))
            return flag, queryGraph
        backwardConstrain = self.backwardEntityConstrain(entity, variableNodes)
        # print('over')
        # if('酒店' in entity and '<北京天伦王朝酒店>' in variableNodes):
        #     print(backwardConstrain)
        #     import pdb; pdb.set_trace()
        # if('<天津_（中华人民共和国直辖市）>' in pathTriples and len(pathTriples) == 6 and entity == '主持人' and len(queryGraph.entityConstrainTriples) > 0):
        #     print(pathTriples, entity, backwardConstrain)
        #     import pdb; pdb.set_trace()
        if(len(backwardConstrain) > 0):
            flag = True
            # print('back:', [backwardConstrain[0][0], backwardConstrain[1], backwardConstrain[2]])
            # print(queryGraph.pathTriples)
            # import pdb; pdb.set_trace()
            queryGraph.updateEntityConstrainInfo([backwardConstrain[0][0], backwardConstrain[1], backwardConstrain[2]], nodeId, 2)
            queryGraph.updateVariableNodes(backwardConstrain[0])
            queryGraph.updateAnswer('\t'.join(backwardConstrain[0]))
            
            return flag, queryGraph
        return flag, queryGraph

    def searchWithEntityidRel(self, entityId, rel, limitNum = 100):
        # print(entityId, rel)
        entityId = self.normNodeForSql(entityId)
        rel = self.normNodeForSql(rel)
        sql = "select entry, prop, value from `pkubase` where `entry` = '%s' and `prop`='%s' limit %s"\
                                         % (entityId, rel, str(limitNum))
        triples = self.searchSQL(sql)
        return triples
    
    def searchWithValueIdRel(self, entityId, rel, limitNum = 100):
        entityId = self.normNodeForSql(entityId)
        rel = self.normNodeForSql(rel)
        sql = "select entry, prop, value from `pkubase` where `value` = '%s' and `prop`='%s' limit %s"\
                                         % (entityId, rel, str(limitNum))
        triples = self.searchSQL(sql)
        return triples

    def getRelationsForRelConstrain(self, queryGraph, candRelList):
        pathTriples = queryGraph.pathTriples
        variableNodeIds = queryGraph.variableNodeIds
        if(len(variableNodeIds) > 1): # 说明不是单跳
            return ()
        nodeId = variableNodeIds[-1]
        # print(pathTriples, rel)
        if(nodeId == 2):
            props = self.searchRelWithEntryId(pathTriples[0])
            return set(props) & set(candRelList)
        elif(nodeId == 0):
            props = self.searchRelWithValueId(pathTriples[2])
            return set(props) & set(candRelList)
    

    def addRelationConstrainForQueryGraph(self, queryGraph, rel, constrainVarIndex = -1):
        '''
        功能：在给定查询图的变量节点上增加关系约束
        输入：
            queryGraph:待约束的查询图
            rel:约束关系
        输出：
            flag:约束挂载是否成功，True为成功，False为失败
            queryGraph:约束挂载后的查询图
        '''
        flag = False
        pathTriples = queryGraph.pathTriples
        variableNodeIds = queryGraph.variableNodeIds
        if(len(variableNodeIds) > 1): # 说明不是单跳
            return flag, queryGraph
        if(rel == pathTriples[1] or rel in queryGraph.rels): # 说明关系已存在
            return flag, queryGraph
        # import pdb; pdb.set_trace()
        if(constrainVarIndex == -1):
            variableNodes = queryGraph.variableNodes[constrainVarIndex]
            nodeId = variableNodeIds[-1]
        # print(pathTriples, rel)
        if(nodeId == 2):
            forwardConstrain = self.forwardRelConstrain(pathTriples[0], rel, variableNodes)
            if(len(forwardConstrain) > 0):
                flag = True
                # print('forward:', [forwardConstrain[0], forwardConstrain[1], forwardConstrain[2][0]])
                # print(queryGraph.pathTriples)
                # import pdb; pdb.set_trace()
                queryGraph.updateRelConstrainTriples([forwardConstrain[0], forwardConstrain[1], forwardConstrain[2][0]], [2])
                queryGraph.updateVariableNodes(forwardConstrain[2])
                queryGraph.updateAnswer('\t'.join(forwardConstrain[2]))
                return flag, queryGraph
        elif(nodeId == 0):
            backwardConstrain = self.backwardRelConstrain(pathTriples[2], rel, variableNodes)
            # if('科学家' in entity and '毕业院校' in pathTriples[1]):
            #     print(backwardConstrain)
            #     import pdb; pdb.set_trace()
            if(len(backwardConstrain) > 0):
                flag = True
                # print('back:', [backwardConstrain[0][0], backwardConstrain[1], backwardConstrain[2]])
                # print(queryGraph.pathTriples)
                # import pdb; pdb.set_trace()
                queryGraph.updateRelConstrainTriples([backwardConstrain[0][0], backwardConstrain[1], backwardConstrain[2]], [0])
                queryGraph.updateVariableNodes(backwardConstrain[0])
                queryGraph.updateAnswer('\t'.join(backwardConstrain[0]))
                return flag, queryGraph
        return flag, queryGraph

    def forwardVirtualConstrain(self, virtualEntity, variableNodes):
        '''
        功能：进行非实体约束的挂载
        输入：非实体约束virtualEntity,
            变量：variableNodes
        输出：
            挂载的关系和约束后的实体:[关系，属性值，约束后的实体集]
        '''
        entityProp = {}
        key2answer = {}
        # print(virtualEntity)
        for node in variableNodes:
            newNode = self.normNodeForSql(node)
            # newNode = node.replace('\\', '\\\\')
            # newNode = newNode.replace('\'', '\\\'')
            sql = "select distinct prop, value from `pkubase` where (`entry` = '%s' and `value` like '%%<http://www.w3.org/2001/XMLSchema#float>') limit %s" % (newNode, str(10))
            # import pdb; pdb.set_trace()
            if(sql not in self.sql2results):
                self.cursor.execute(sql)
                triples = self.cursor.fetchall()
                self.sql2results[sql] = triples
            else:
                triples = self.sql2results[sql]
            if(len(triples) > 0):
                values = triples[0][1]
                pos = values.find('<')
                intValue = float(values[1: pos - 3])
                # import pdb; pdb.set_trace()
                if('<' == virtualEntity[0][4]): # 小于条件
                    if(intValue <= float(virtualEntity[1])):
                        # key = (triples[0][0], intValue)
                        key = (triples[0][0], virtualEntity[1])
                        if(key not in entityProp):
                            entityProp[key] = 1
                            key2answer[key] = [node]
                        else:
                            entityProp[key] += 1
                            key2answer[key].append(node)
                else:
                    if(intValue >= float(virtualEntity[1])):
                        # key = (triples[0][0], intValue)
                        key = (triples[0][0], virtualEntity[1])
                        if(key not in entityProp):
                            entityProp[key] = 1
                            key2answer[key] = [node]
                        else:
                            entityProp[key] += 1
                            key2answer[key].append(node)
        # if('<孤山公园>' in variableNodes):
        #     print(entityProp)
            # import pdb; pdb.set_trace()
        if(len(entityProp) > 0):
            sortedPropValue = sorted(entityProp.items(), key = lambda x:x[1], reverse=True)
            return [sortedPropValue[0][0][0], virtualEntity[0][4] + virtualEntity[1]] + [key2answer[sortedPropValue[0][0]]]
        return []

    def addVirtualConstrainForQueryGraph(self, queryGraph, virtualEntity, constrainVarIndex = -1):
        '''
        功能：在给定查询图的变量节点上增加非实体约束
        输入：
            queryGraph:待约束的查询图
            virtualEntity:约束实体
        输出：
            flag:约束挂载是否成功，True为成功，False为失败
            queryGraph:约束挂载后的查询图
        '''
        flag = False
        pathTriples = queryGraph.pathTriples
        variableNodeIds = queryGraph.variableNodeIds
        if(constrainVarIndex == -1):
            variableNodes = queryGraph.variableNodes[constrainVarIndex]
            nodeId = variableNodeIds[-1]
        else:
            nodeId = constrainVarIndex
            # import pdb; pdb.set_trace()
            pos = variableNodeIds.index(constrainVarIndex)
            variableNodes = queryGraph.variableNodes[pos]
        # import pdb; pdb.set_trace()
        forwardConstrain = self.forwardVirtualConstrain(virtualEntity, variableNodes)
        if(len(forwardConstrain) > 0):
            flag = True
            # print('forward:', [forwardConstrain[0], forwardConstrain[1], forwardConstrain[2][0]])
            # print(queryGraph.pathTriples)
            # import pdb; pdb.set_trace()
            queryGraph.updateVirtualConstrainInfo([forwardConstrain[2][0], forwardConstrain[0], forwardConstrain[1]], nodeId)
            queryGraph.updateVariableNodes(forwardConstrain[2])
            queryGraph.updateAnswer('\t'.join(forwardConstrain[2]))
            # import pdb; pdb.set_trace()
            return flag, queryGraph
        return flag, queryGraph

    
    def addEntityConstrainForQueryGraphSTAGG(self, queryGraph, entity):
        '''
        功能：在给定查询图的变量节点上增加实体约束
        输入：
            queryGraph:待约束的查询图
            entity:约束实体
        输出：
            flag:约束挂载是否成功，True为成功，False为失败
            queryGraph:约束挂载后的查询图
        '''
        flag = False
        variableNodeIds = queryGraph.variableNodeIds
        for variableNodeId in variableNodeIds:
            flag, queryGraph = self.addEntityConstrainForQueryGraph(queryGraph, entity, variableNodeId)
            if(flag):
                return flag, queryGraph
        return flag, queryGraph


    def done(self):
        self.cursor.close()
        self.db.commit()
        self.db.close()



if __name__ == "__main__":
    # connectionMysql = ConnectionMysql()
    # connectionMysql.generate2HopChainWithEntity('天使与猎人')
    pass

