'''
尝试构建一个比较通用的查询图生成框架，具体而言，整个过程按照人类的思维方式分为外延和内展两部分；
其中外延指以变量节点为中心向外延伸，实现的是关系路径的延长；内展指在已有的查询图路径节点上加约束信息。
外延和内展两个操作会不断迭代，直到满足终止条件。
'''
import sys
import os
import json
import copy
import math
import pdb
from typing import List, Dict, Tuple
from fuzzywuzzy import process

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
print(BASE_DIR)
# from src.gstore.GstoreSearch import GstoreConnection
from src.mysql.MysqlConnection import MysqlConnection
from src.QueryGraph import QueryGraph
from src.querygraph2seq.querygraph_to_seq import getMainPathSeq, getConstrainPathSeq
from config.MysqlConfig import CCKSConfig
from src.utils.data_processing import preprocessRel

# ccksConfig = CCKSConfig()
# exit()
def mergeForSameAnswer(queryGraphs: QueryGraph):
    '''
    针对具有相同答案节点的查询图进行合并，解决的是常见的二合一问题。
    '''
    newQueryGraphs = []
    for i, queryGraph in enumerate(queryGraphs):
        flag = 0
        for queryGraph2 in queryGraphs[i + 1:]:
            if(queryGraph.answer == queryGraph2.answer):
                flag = 1
                newQueryGraphs.append(QueryGraph())


class SearchPath(object):
    def __init__(self,ccksConfig = CCKSConfig()) -> None:
        self.mysqlConnection = MysqlConnection(ccksConfig)

    def updateAvailableEntityIds(self, availableEntityIds, pos, conflictMatrix):
        newEntityIds = []
        for entityId in availableEntityIds:
            if(conflictMatrix[pos][entityId] == 0):
                newEntityIds.append(entityId)
        return newEntityIds


    def generateQueryGraphComp(self, entitys=[],
                                conflictMatrix=[], candRels = None,\
                                queType = ''):
        '''
        功能：生成处理比较逻辑问题的查询图
        '''
        queryGraphs = self.mysqlConnection.searchQueryGraphsWithBasePath(entitys, queType = queType)
        for i in range(len(queryGraphs)):
            queryGraph = self.mysqlConnection.addAnswerType(queryGraphs[i])
        # import pdb;pdb.set_trace()
        return queryGraphs

    def generateQueryGraphNoEntity(self, entitys=[],
                                candRelsList = None,\
                                queType = ''):
        '''
        功能：生成处理复杂逻辑问题的查询图
        '''
        queryGraphs = self.mysqlConnection.generateQueryGraphsNoEntity(entitys, candRelsList, queType = queType)
        for i in range(len(queryGraphs)):
            queryGraph = self.mysqlConnection.addAnswerType(queryGraphs[i])

        return queryGraphs

    def generateQueryGraph(self, entitys: List[str] = ['北京大学', '天文学家'], \
                            virtualEntitys = [], higherOrder = [], \
                            conflictMatrix = [], candRels = None, candRelsList = None, MaxHop:int=4):
        '''
        virtualEntitys: [(('5', 6, 6, 'distance', '<'), '5')]
        higherOrder: [(('argmax', 10, 11, 'higher-order'), 'argmax')]
        '''
        queryGraphs = []
        usedLogs = []
        currentUsedEntitys = []
        virtualConflictMatrix = []
        for i in range(len(virtualEntitys)):
            virtualConflictMatrix.append([0] * len(virtualEntitys))
            virtualConflictMatrix[i][i] = 1
        higherOrderConflictMatrix = []
        for i in range(len(higherOrder)):
            higherOrderConflictMatrix.append([0] * len(higherOrder))
            higherOrderConflictMatrix[i][i] = 1
        # print(conflictMatrix)
        # 询问关系词
        currentQueryGraphs = self.mysqlConnection.generateRelWithEntity(entitys, conflictMatrix)
        queryGraphs.extend(copy.deepcopy(currentQueryGraphs))
        # print(entitys, conflictMatrix)
        # import pdb; pdb.set_trace()
        forwardFlag = True
        # print(entitys)
        for i, entity in enumerate(entitys):
            # import pdb; pdb.set_trace()
            backwardFlag = True
            initAvailableEntityIds = [entityId for entityId in range(len(entitys))]
            newAvailableEntityIds = copy.deepcopy(initAvailableEntityIds)
            initAvailableVirtualEntityIds = [entityId for entityId in range(len(virtualEntitys))]
            initAvailableHigherOrderIds = [higherOrderId for higherOrderId in range(len(higherOrder))]
            currentQueryGraphs = []
            for hopNum in range(MaxHop): # 最多4跳
                ############### 关系扩展###########################
                if(hopNum == 0): # 第一次从实体出发
                    queryGraphsForward = self.mysqlConnection.searchWithEntityBasedProp(entity)
                    currentQueryGraphs.extend(queryGraphsForward)
                    queryGraphsFor1HopBackward = self.mysqlConnection.searchWithValueBasedProp(entity)
                    currentQueryGraphs.extend(queryGraphsFor1HopBackward)
                    if(len(currentQueryGraphs) > 0):
                        newAvailableEntityIds = self.updateAvailableEntityIds(initAvailableEntityIds, i, conflictMatrix)
                    for indexI in range(len(currentQueryGraphs)):
                        currentQueryGraphs[indexI].setAvailableEntityIds(newAvailableEntityIds)
                        currentQueryGraphs[indexI].setAvailableVirtualEntityIds(initAvailableVirtualEntityIds)
                        currentQueryGraphs[indexI].setAvailableHigherOrderIds(initAvailableHigherOrderIds)
                else: # 从当前所在的查询图集合出发
                    # import pdb; pdb.set_trace()
                    candRelsListNew = candRelsList[0: math.ceil(len(candRels) / pow(2, hopNum - 1))]
                    currentCandRels = {item: candRels[item] for item in candRelsListNew}
                    # print(hopNum, len(currentCandRels))
                    if(hopNum > 1):
                        backwardFlag = False
                    # print('开始延伸', len(currentQueryGraphs))
                    currentQueryGraphs = self.mysqlConnection.generateOneHopFromQueryGraphs(currentQueryGraphs, \
                                    currentCandRels, candRelsList = candRelsListNew, forwardFlag=forwardFlag, backwardFlag=backwardFlag)
                    # print('延伸结束')
                ##################### 约束挂载 ##################
                for j, queryGraph in enumerate(currentQueryGraphs):
                    constrainQueryGraphs = [copy.deepcopy(queryGraph)]
                    # print('开始挂载约束')
                    while(len(constrainQueryGraphs) > 0):
                        queryGraph = constrainQueryGraphs.pop()
                        availableEntityIds = queryGraph.availableEntityIds
                        for entityId in availableEntityIds:
                            entity = entitys[entityId]
                            # 判断是否可以增加约束，可以则更新newAvailableEntityIds
                            operationFlag, queryGraph = self.mysqlConnection.addEntityConstrainForQueryGraph(copy.deepcopy(queryGraph), entity)
                            if(operationFlag):
                                availableEntityIds = self.updateAvailableEntityIds(availableEntityIds, entityId, conflictMatrix)
                                queryGraph.setAvailableEntityIds(availableEntityIds)
                                constrainQueryGraphs.append(queryGraph)
                                currentQueryGraphs.append(queryGraph)
                                break       # 每次只加一个约束，循环加到无法再加
                        # print('一个查询图挂载结束')
                        availableVirtualEntityIds = queryGraph.availableVirtualEntityIds
                        # 增加关系约束(只在单跳查询图上加)
                        candRelsSet = self.mysqlConnection.getRelationsForRelConstrain(queryGraph, candRelsList)
                        for rel in candRelsSet:
                            operationFlag, queryGraph = self.mysqlConnection.addRelationConstrainForQueryGraph(copy.deepcopy(queryGraph), rel)
                            if(operationFlag):
                                constrainQueryGraphs.append(queryGraph)
                                currentQueryGraphs.append(queryGraph)
                                break       # 每次只加一个约束，循环加到无法再加
                        # import pdb; pdb.set_trace()
                        # 增加非实体约束    # 特殊格式：like '%%<http://www.w3.org/2001/XMLSchema#float>'
                        for virtualEntityId in availableVirtualEntityIds:
                            virtualEntity = virtualEntitys[virtualEntityId]
                            # import pdb; pdb.set_trace()
                            operationFlag, queryGraph = self.mysqlConnection.addVirtualConstrainForQueryGraph(copy.deepcopy(queryGraph), virtualEntity)
                            if(operationFlag):
                                availableVirtualEntityIds = self.updateAvailableEntityIds(availableVirtualEntityIds, virtualEntityId, virtualConflictMatrix)
                                queryGraph.setAvailableVirtualEntityIds(availableVirtualEntityIds)
                                # print(queryGraph.serialization())
                                # print(availableEntityIds, entitys)
                                constrainQueryGraphs.append(queryGraph)
                                currentQueryGraphs.append(queryGraph)
                                break       # 每次只加一个约束，循环加到无法再加
                        # 增加高阶约束
                        availableHigherOrderIds = queryGraph.availableHigherOrderIds
                        for higherOrderId in availableHigherOrderIds:
                            # print(higherOrderId, availableHigherOrderIds, len(constrainQueryGraphs))
                            higherOrderItem = higherOrder[higherOrderId]
                            # import pdb; pdb.set_trace()
                            operationFlag, queryGraph = self.mysqlConnection.addHigherOrderConstrainForQueryGraph(copy.deepcopy(queryGraph),\
                                                                             [higherOrderItem], candRelsList)
                            if(operationFlag):
                                # print('更新前：', availableHigherOrderIds)
                                availableHigherOrderIds = self.updateAvailableEntityIds(availableHigherOrderIds, higherOrderId, higherOrderConflictMatrix)
                                queryGraph.setAvailableHigherOrderIds(availableHigherOrderIds)
                                constrainQueryGraphs.append(queryGraph)
                                currentQueryGraphs.append(queryGraph)
                                break
                queryGraphs.extend(currentQueryGraphs)
        # 组合约束
        # processCandRelsList = preprocessRel(candRelsList[0:100])
        # print(processCandRelsList)
        # import pdb; pdb.set_trace()
        queryGraphs = self.mysqlConnection.queryGraphsCombine(queryGraphs, candRelsList[0:100])
        # 根据查询图结构从知识库中重新检索答案
        keys = {}
        newQueryGraphs = []
        for queryGraph in queryGraphs:
            queryGraph = self.mysqlConnection.addAnswerType(queryGraph) # 永辉师兄新加的内容
            queryGraph.getKey()
            if(queryGraph.key not in keys):
                # if(len(queryGraph.basePathTriples) == 0 and len(queryGraph.higherOrderTriples) == 0 and len(queryGraph.virtualConstrainTriples) == 0):
                #     answerList = self.mysqlConnection.searchAnswer(queryGraph=queryGraph, higherOrder= higherOrder)
                #     # answerList = self.mysqlConnection.searchAnswerBySQL(queryGraph=queryGraph)
                #     queryGraph.updateAnswer('\t'.join(answerList))
                keys[queryGraph.key] = 1
                newQueryGraphs.append(queryGraph)
        return newQueryGraphs

    
    def generateQueryGraphSTAGGUpdate(self, entitys: List[str] = ['北京大学', '天文学家'], \
                            virtualEntitys = [], higherOrder = [], \
                            conflictMatrix = [], candRels = None, candRelsList = None):
        '''
        virtualEntitys: [(('5', 6, 6, 'distance', '<'), '5')]
        higherOrder: [(('argmax', 10, 11, 'higher-order'), 'argmax')]
        '''
        queryGraphs = []
        usedLogs = []
        currentUsedEntitys = []
        virtualConflictMatrix = []
        for i in range(len(virtualEntitys)):
            virtualConflictMatrix.append([0] * len(virtualEntitys))
            virtualConflictMatrix[i][i] = 1
        higherOrderConflictMatrix = []
        for i in range(len(higherOrder)):
            higherOrderConflictMatrix.append([0] * len(higherOrder))
            higherOrderConflictMatrix[i][i] = 1
        forwardFlag = True
        # print(entitys)
        for i, entity in enumerate(entitys):
            # import pdb; pdb.set_trace()
            backwardFlag = True
            initAvailableEntityIds = [entityId for entityId in range(len(entitys))]
            newAvailableEntityIds = copy.deepcopy(initAvailableEntityIds)
            initAvailableVirtualEntityIds = [entityId for entityId in range(len(virtualEntitys))]
            initAvailableHigherOrderIds = [higherOrderId for higherOrderId in range(len(higherOrder))]
            currentQueryGraphs = []
            for hopNum in range(2):
                ############### 关系扩展###########################
                if(hopNum == 0): # 第一次从实体出发
                    queryGraphsForward = self.mysqlConnection.searchWithEntityBasedProp(entity)
                    currentQueryGraphs.extend(queryGraphsForward)
                    queryGraphsFor1HopBackward = self.mysqlConnection.searchWithValueBasedProp(entity)
                    currentQueryGraphs.extend(queryGraphsFor1HopBackward)
                    if(len(currentQueryGraphs) > 0):
                        newAvailableEntityIds = self.updateAvailableEntityIds(initAvailableEntityIds, i, conflictMatrix)
                    for indexI in range(len(currentQueryGraphs)):
                        currentQueryGraphs[indexI].setAvailableEntityIds(newAvailableEntityIds)
                        currentQueryGraphs[indexI].setAvailableVirtualEntityIds(initAvailableVirtualEntityIds)
                        currentQueryGraphs[indexI].setAvailableHigherOrderIds(initAvailableHigherOrderIds)
                else: # 从当前所在的查询图集合出发
                    # import pdb; pdb.set_trace()
                    candRelsListNew = candRelsList[0: ]
                    currentCandRels = {item: candRels[item] for item in candRelsListNew}
                    # print(hopNum, len(currentCandRels))
                    if(hopNum >= 1):
                        backwardFlag = False
                    # print('开始延伸', len(currentQueryGraphs))
                    currentQueryGraphs = self.mysqlConnection.generateOneHopFromQueryGraphs(currentQueryGraphs, \
                                    currentCandRels, candRelsList = candRelsListNew, forwardFlag=forwardFlag, backwardFlag=backwardFlag)
                    # print('延伸结束')
                ##################### 约束挂载 ##################
                for j, queryGraph in enumerate(currentQueryGraphs):
                    constrainQueryGraphs = [copy.deepcopy(queryGraph)]
                    # print('开始挂载约束')
                    while(len(constrainQueryGraphs) > 0):
                        queryGraph = constrainQueryGraphs.pop()
                        availableEntityIds = queryGraph.availableEntityIds
                        for entityId in availableEntityIds:
                            entity = entitys[entityId]
                            # 判断是否可以增加约束，可以则更新newAvailableEntityIds
                            operationFlag, queryGraph = self.mysqlConnection.addEntityConstrainForQueryGraph(copy.deepcopy(queryGraph), entity)
                            if(operationFlag):
                                availableEntityIds = self.updateAvailableEntityIds(availableEntityIds, entityId, conflictMatrix)
                                queryGraph.setAvailableEntityIds(availableEntityIds)
                                constrainQueryGraphs.append(queryGraph)
                                currentQueryGraphs.append(queryGraph)
                                break       # 每次只加一个约束，循环加到无法再加
                        # print('一个查询图挂载结束')
                        availableVirtualEntityIds = queryGraph.availableVirtualEntityIds
                        # import pdb; pdb.set_trace()
                        # 增加非实体约束
                        for virtualEntityId in availableVirtualEntityIds:
                            virtualEntity = virtualEntitys[virtualEntityId]
                            # import pdb; pdb.set_trace()
                            operationFlag, queryGraph = self.mysqlConnection.addVirtualConstrainForQueryGraph(copy.deepcopy(queryGraph), virtualEntity)
                            if(operationFlag):
                                availableVirtualEntityIds = self.updateAvailableEntityIds(availableVirtualEntityIds, virtualEntityId, virtualConflictMatrix)
                                queryGraph.setAvailableVirtualEntityIds(availableVirtualEntityIds)
                                # print(queryGraph.serialization())
                                # print(availableEntityIds, entitys)
                                constrainQueryGraphs.append(queryGraph)
                                currentQueryGraphs.append(queryGraph)
                                break       # 每次只加一个约束，循环加到无法再加
                        # 增加高阶约束
                        availableHigherOrderIds = queryGraph.availableHigherOrderIds
                        for higherOrderId in availableHigherOrderIds:
                            # print(higherOrderId, availableHigherOrderIds, len(constrainQueryGraphs))
                            higherOrderItem = higherOrder[higherOrderId]
                            # import pdb; pdb.set_trace()
                            operationFlag, queryGraph = self.mysqlConnection.addHigherOrderConstrainForQueryGraph(copy.deepcopy(queryGraph),\
                                                                             [higherOrderItem], candRelsList)
                            if(operationFlag):
                                # print('更新前：', availableHigherOrderIds)
                                availableHigherOrderIds = self.updateAvailableEntityIds(availableHigherOrderIds, higherOrderId, higherOrderConflictMatrix)
                                queryGraph.setAvailableHigherOrderIds(availableHigherOrderIds)
                                constrainQueryGraphs.append(queryGraph)
                                currentQueryGraphs.append(queryGraph)
                                break
                queryGraphs.extend(currentQueryGraphs)
        # 根据查询图结构从知识库中重新检索答案
        keys = {}
        newQueryGraphs = []
        for queryGraph in queryGraphs:
            queryGraph = self.mysqlConnection.addAnswerType(queryGraph) # 永辉师兄新加的内容
            queryGraph.getKey()
            if(queryGraph.key not in keys):
                # if(len(queryGraph.basePathTriples) == 0 and len(queryGraph.higherOrderTriples) == 0 and len(queryGraph.virtualConstrainTriples) == 0):
                #     answerList = self.mysqlConnection.searchAnswer(queryGraph=queryGraph, higherOrder= higherOrder)
                #     # answerList = self.mysqlConnection.searchAnswerBySQL(queryGraph=queryGraph)
                #     queryGraph.updateAnswer('\t'.join(answerList))
                keys[queryGraph.key] = 1
                newQueryGraphs.append(queryGraph)
        return newQueryGraphs

    def generateQueryGraphSTAGG(self, entitys: List[str] = ['北京大学', '天文学家'], conflictMatrix = []):
        queryGraphs = []
        usedLogs = []
        currentUsedEntitys = []
        # print(conflictMatrix)
        for i, entity in enumerate(entitys):
            availableEntityIds = [entityId for entityId in range(len(entitys))]
            newAvailableEntityIds = copy.deepcopy(availableEntityIds)
            currentQueryGraphs = []
            ############### 主路径搜索 ##########################
            for hopNum in range(2):
                if(hopNum == 0): # 第一次从实体出发
                    queryGraphsForward = self.mysqlConnection.searchWithEntityBasedProp(entity)
                    currentQueryGraphs.extend(queryGraphsForward)
                    queryGraphsFor1HopBackward = self.mysqlConnection.searchWithValueBasedProp(entity)
                    currentQueryGraphs.extend(queryGraphsFor1HopBackward)
                else: # 从当前所在的查询图集合出发
                    # currentQueryGraphs = self.mysqlConnection.searchOneHopFromQueryGraphs(currentQueryGraphs)
                    currentQueryGraphs = self.mysqlConnection.generateOneHopFromQueryGraphs(currentQueryGraphs)
                if(len(currentQueryGraphs) > 0):
                    newAvailableEntityIds = self.updateAvailableEntityIds(availableEntityIds, i, conflictMatrix)
                for indexI in range(len(currentQueryGraphs)):
                    currentQueryGraphs[indexI].setAvailableEntityIds(newAvailableEntityIds)
                queryGraphs.extend(currentQueryGraphs)
                # if(len(queryGraphs) > 10000):
                #     print(len(queryGraphs))
                #     import pdb; pdb.set_trace()
                # currentQueryGraphs = []
            ################## 约束挂载 ###########################
            constrainQueryGraphs = copy.deepcopy(queryGraphs)
            while(len(constrainQueryGraphs) > 0):
                queryGraph = constrainQueryGraphs.pop()
                availableEntityIds = queryGraph.availableEntityIds
                # print('1:', queryGraph.serialization())
                # print('1:', availableEntityIds, entitys)
                for entityId in availableEntityIds:
                    entity = entitys[entityId]
                    # print('entity:', entity)
                    # 判断是否可以增加约束，可以则更新newAvailableEntityIds
                    operationFlag, queryGraph = self.mysqlConnection.addEntityConstrainForQueryGraphSTAGG(queryGraph, entity)
                    if(operationFlag):
                        availableEntityIds = self.updateAvailableEntityIds(availableEntityIds, entityId, conflictMatrix)
                        queryGraph.setAvailableEntityIds(availableEntityIds)
                        # print(queryGraph.serialization())
                        # print(availableEntityIds, entitys)
                        constrainQueryGraphs.append(queryGraph)
                        queryGraphs.append(queryGraph)
                        break       # 每次只加一个约束，循环加到无法再加
        keys = {}
        newQueryGraphs = []
        for queryGraph in queryGraphs:
            queryGraph.getKey()
            if(queryGraph.key not in keys):
                # answerList = self.mysqlConnection.searchAnswer(queryGraph=queryGraph)
                answerList = self.mysqlConnection.searchAnswerBySQL(queryGraph=queryGraph)  # 根据SQL语句找答案
                if(len(answerList) == 0):
                    continue
                queryGraph.updateAnswer('\t'.join(answerList))
                # queryGraph.serialization()
                keys[queryGraph.key] = 1
                newQueryGraphs.append(queryGraph)
        return newQueryGraphs

    def generateQueryGraphYhi(self, entitys: List[str] = ['北京大学', '天文学家'], \
                            virtualEntitys = [], higherOrder = [], \
                            conflictMatrix = [], candRels = None, candRelsList = None):
        '''
        virtualEntitys: [(('5', 6, 6, 'distance', '<'), '5')]
        higherOrder: [(('argmax', 10, 11, 'higher-order'), 'argmax')]
        '''
        queryGraphs = []
        usedLogs = []
        currentUsedEntitys = []
        virtualConflictMatrix = []
        for i in range(len(virtualEntitys)):
            virtualConflictMatrix.append([0] * len(virtualEntitys))
            virtualConflictMatrix[i][i] = 1
        higherOrderConflictMatrix = []
        for i in range(len(higherOrder)):
            higherOrderConflictMatrix.append([0] * len(higherOrder))
            higherOrderConflictMatrix[i][i] = 1
        forwardFlag = True
        # print(entitys)
        for i, entity in enumerate(entitys):
            # import pdb; pdb.set_trace()
            backwardFlag = True
            initAvailableEntityIds = [entityId for entityId in range(len(entitys))]
            newAvailableEntityIds = copy.deepcopy(initAvailableEntityIds)
            initAvailableVirtualEntityIds = [entityId for entityId in range(len(virtualEntitys))]
            initAvailableHigherOrderIds = [higherOrderId for higherOrderId in range(len(higherOrder))]
            currentQueryGraphs = []
            for hopNum in range(2):
                ############### 关系扩展###########################
                if(hopNum == 0): # 第一次从实体出发
                    queryGraphsForward = self.mysqlConnection.searchWithEntityBasedProp(entity)
                    currentQueryGraphs.extend(queryGraphsForward)
                    queryGraphsFor1HopBackward = self.mysqlConnection.searchWithValueBasedProp(entity)
                    currentQueryGraphs.extend(queryGraphsFor1HopBackward)
                    if(len(currentQueryGraphs) > 0):
                        newAvailableEntityIds = self.updateAvailableEntityIds(initAvailableEntityIds, i, conflictMatrix)
                    for indexI in range(len(currentQueryGraphs)):
                        currentQueryGraphs[indexI].setAvailableEntityIds(newAvailableEntityIds)
                        currentQueryGraphs[indexI].setAvailableVirtualEntityIds(initAvailableVirtualEntityIds)
                        currentQueryGraphs[indexI].setAvailableHigherOrderIds(initAvailableHigherOrderIds)
                else: # 从当前所在的查询图集合出发
                    # import pdb; pdb.set_trace()
                    candRelsListNew = candRelsList[0: ]
                    currentCandRels = {item: candRels[item] for item in candRelsListNew}
                    # print(hopNum, len(currentCandRels))
                    if(hopNum >= 1):
                        backwardFlag = False
                    # print('开始延伸', len(currentQueryGraphs))
                    currentQueryGraphs = self.mysqlConnection.generateOneHopFromQueryGraphs(currentQueryGraphs, \
                                    currentCandRels, candRelsList = candRelsListNew, forwardFlag=forwardFlag, backwardFlag=backwardFlag)
                    # print('延伸结束')
                ##################### 约束挂载 ##################
                for j, queryGraph in enumerate(currentQueryGraphs):
                    constrainQueryGraphs = [copy.deepcopy(queryGraph)]
                    # print('开始挂载约束')
                    while(len(constrainQueryGraphs) > 0):
                        queryGraph = constrainQueryGraphs.pop()
                        availableEntityIds = queryGraph.availableEntityIds
                        for entityId in availableEntityIds:
                            entity = entitys[entityId]
                            # 判断是否可以增加约束，可以则更新newAvailableEntityIds
                            operationFlag, queryGraph = self.mysqlConnection.addEntityConstrainForQueryGraph(copy.deepcopy(queryGraph), entity)
                            if(operationFlag):
                                availableEntityIds = self.updateAvailableEntityIds(availableEntityIds, entityId, conflictMatrix)
                                queryGraph.setAvailableEntityIds(availableEntityIds)
                                constrainQueryGraphs.append(queryGraph)
                                currentQueryGraphs.append(queryGraph)
                                break
                queryGraphs.extend(currentQueryGraphs)
        # 根据查询图结构从知识库中重新检索答案
        keys = {}
        newQueryGraphs = []
        for queryGraph in queryGraphs:
            queryGraph = self.mysqlConnection.addAnswerType(queryGraph) # 永辉师兄新加的内容
            queryGraph.getKey()
            if(queryGraph.key not in keys):
                keys[queryGraph.key] = 1
                newQueryGraphs.append(queryGraph)
        return newQueryGraphs

    '''Lan 的baseline'''
    def generateQueryGraphLan(self, entitys: List[str] = ['北京大学', '天文学家'], \
                            virtualEntitys = [], higherOrder = [], \
                            conflictMatrix = [], candRels = None, simModel = None, beam_size = 2,
                            max_hop = 2, question = ''):
        '''
        :parma entitys: 可用的实体 # TODO 将通过规则得到的约束单独存放
        :param virtualEntitys: [(('5', 6, 6, 'distance', '<'), '5')]
        :param higherOrder: [(('argmax', 10, 11, 'higher-order'), 'argmax')]
        :parma conflictMatrix:
        :parmar candRels: 关系识别得到的关系字典？ 后面的value是KB中的频次统计
        :param simModel: 用于beamsearch的相似度模型
        :param beam_size:
        '''
        # return  
        queryGraphs = []
        candRelsList = []
        usedLogs = []
        currentUsedEntitys = []
        virtualConflictMatrix = []
        for i in range(len(virtualEntitys)):
            virtualConflictMatrix.append([0] * len(virtualEntitys))
            virtualConflictMatrix[i][i] = 1
        higherOrderConflictMatrix = []
        for i in range(len(higherOrder)):
            higherOrderConflictMatrix.append([0] * len(higherOrder))
            higherOrderConflictMatrix[i][i] = 1
        # print(conflictMatrix)
        '''询问关系词,Lan不做处理'''
        # currentQueryGraphs = self.mysqlConnection.generateRelWithEntity(entitys, conflictMatrix)
        # queryGraphs.extend(copy.deepcopy(currentQueryGraphs))

        # print(entitys, conflictMatrix)
        # import pdb; pdb.set_trace()
        forwardFlag = True
        # print(entitys)
        for i, entity in enumerate(entitys):    # 每个实体作为TopicEntity，走一遍流程
            # import pdb; pdb.set_trace()
            backwardFlag = True
            initAvailableEntityIds = [entityId for entityId in range(len(entitys))] # 最开始可用的 实体
            newAvailableEntityIds = copy.deepcopy(initAvailableEntityIds)   # 上面的deepcopy
            initAvailableVirtualEntityIds = [entityId for entityId in range(len(virtualEntitys))]   # virtual实体
            initAvailableHigherOrderIds = [higherOrderId for higherOrderId in range(len(higherOrder))]  # higherOrder实体

            # currentQueryGraphs = [] # 存储目前这1hop筛选出来的
            
            ito_topK_QueryGraphs = []   # 用于迭代 最佳的k个
            done = False
            hopNum = 0    # 这个是跳数-1
            max_score = 0   # 最佳得分，用于提前终止
            cache_QueryGraph_of_hop = []
            while hopNum < max_hop: # 同样最多2跳 目前hop = 2
                cur_hop_QueryGraphs = []    # 本轮生成的查询图
                extendQueryGraphs = []  # 延1hop
                connectQueryGraphs = [] # 挂约束（ground entity）
                ############### 关系扩展###########################
                if(hopNum == 0): # 第一次从实体出发
                    # import pdb;pdb.set_trace()
                    queryGraphsForward = self.mysqlConnection.searchWithEntityBasedProp(entity) # 正向1hop检索
                    extendQueryGraphs.extend(queryGraphsForward)
                    queryGraphsFor1HopBackward = self.mysqlConnection.searchWithValueBasedProp(entity)  # 反向1hop检索
                    extendQueryGraphs.extend(queryGraphsFor1HopBackward)
                    if(len(extendQueryGraphs) > 0):    # 新增了候选
                        newAvailableEntityIds = self.updateAvailableEntityIds(initAvailableEntityIds, i, conflictMatrix)
                    for indexI in range(len(extendQueryGraphs)):   # 更新每个查询图的 XXX 状态
                        extendQueryGraphs[indexI].setAvailableEntityIds(newAvailableEntityIds)
                        extendQueryGraphs[indexI].setAvailableVirtualEntityIds(initAvailableVirtualEntityIds)
                        extendQueryGraphs[indexI].setAvailableHigherOrderIds(initAvailableHigherOrderIds)
                    cur_hop_QueryGraphs.extend(extendQueryGraphs)
                    # print('finish extend')
                    # pdb.set_trace()
                else: # 从当前所在的查询图集合出发，extend 1 hop
                    # print(len(ito_topK_QueryGraphs))
                    # candRelsListNew = candRelsList[0: math.ceil(len(candRels) / pow(2, hopNum - 1))]    # 靠前的关系列表再做切片
                    # currentCandRels = {item: candRels[item] for item in candRelsListNew}
                    # print(hopNum, len(currentCandRels))
                    if(hopNum > 1): # 2hop及以上，不做反向
                        backwardFlag = False
                    # print('开始延伸', len(currentQueryGraphs))
                    '''基于查询图 生成1hop'''
                    extendQueryGraphs = self.mysqlConnection.generateOneHopFromQueryGraphs_based(ito_topK_QueryGraphs, \
                                    forwardFlag=forwardFlag, backwardFlag=backwardFlag)
                    cur_hop_QueryGraphs.extend(extendQueryGraphs)
                    # print('延伸结束')
                    '''这些约束 暂且都先加到 connectQueryGraphs中'''
                    ##################### 约束挂载 ##################
                    for j, _queryGraph in enumerate(ito_topK_QueryGraphs): # 遍历现有查询图，挂载约束 # 对于Lan而言，挂约束和延展1hop是同步进行的，即 给上一波的QG加约束
                        availableEntityIds = _queryGraph.availableEntityIds
                        for entityId in availableEntityIds:
                            entity = entitys[entityId]
                            # 判断是否可以增加约束，可以则更新newAvailableEntityIds
                            operationFlag, queryGraph = self.mysqlConnection.addEntityConstrainForQueryGraph(copy.deepcopy(_queryGraph), entity)
                            if(operationFlag):
                                availableEntityIds = self.updateAvailableEntityIds(availableEntityIds, entityId, conflictMatrix)
                                queryGraph.setAvailableEntityIds(availableEntityIds)
                                connectQueryGraphs.append(queryGraph)
                                # break       # 每次只加一个约束，循环加到无法再加
                        # print('一个查询图挂载结束')
                        availableVirtualEntityIds = _queryGraph.availableVirtualEntityIds
                        # 增加关系约束(只在单跳查询图上加) TODO 这个关系约束 指的是什么
                        candRelsSet = self.mysqlConnection.getRelationsForRelConstrain(_queryGraph, candRelsList)
                        for rel in candRelsSet:
                            operationFlag, queryGraph = self.mysqlConnection.addRelationConstrainForQueryGraph(copy.deepcopy(_queryGraph), rel)
                            if(operationFlag):
                                connectQueryGraphs.append(queryGraph)
                                # break       # 每次只加一个约束，循环加到无法再加
                        # import pdb; pdb.set_trace()
                        # 增加非实体约束
                        for virtualEntityId in availableVirtualEntityIds:
                            virtualEntity = virtualEntitys[virtualEntityId]
                            # import pdb; pdb.set_trace()
                            operationFlag, queryGraph = self.mysqlConnection.addVirtualConstrainForQueryGraph(copy.deepcopy(_queryGraph), virtualEntity)
                            if(operationFlag):
                                availableVirtualEntityIds = self.updateAvailableEntityIds(availableVirtualEntityIds, virtualEntityId, virtualConflictMatrix)
                                queryGraph.setAvailableVirtualEntityIds(availableVirtualEntityIds)
                                connectQueryGraphs.append(queryGraph)
                                # break       # 每次只加一个约束，循环加到无法再加
                        # 增加高阶约束
                        availableHigherOrderIds = _queryGraph.availableHigherOrderIds
                        for higherOrderId in availableHigherOrderIds:
                            # print(higherOrderId, availableHigherOrderIds, len(constrainQueryGraphs))
                            higherOrderItem = higherOrder[higherOrderId]
                            operationFlag, queryGraph = self.mysqlConnection.addHigherOrderConstrainForQueryGraph(copy.deepcopy(_queryGraph),\
                                                                            [higherOrderItem], candRelsList)
                            if(operationFlag):
                                availableHigherOrderIds = self.updateAvailableEntityIds(availableHigherOrderIds, higherOrderId, higherOrderConflictMatrix)
                                queryGraph.setAvailableHigherOrderIds(availableHigherOrderIds)
                                connectQueryGraphs.append(queryGraph)
                                # break
                    cur_hop_QueryGraphs.extend(connectQueryGraphs)
                '''筛选Topk'''
                # print('\t{} extend + connect: {}+{}={}'.format(hopNum, len(extendQueryGraphs), len(connectQueryGraphs), len(cur_hop_QueryGraphs)))
                # pdb.set_trace()
                ito_topK_QueryGraphs = []   # 本阶段的Topk
                topK_QueryGraphs, ranked_cur_hop_QueryGraphs, score = self.generateTopk_QueryGraphLan(cur_hop_QueryGraphs, question, k=beam_size)
                if score < max_score:   # 没有更好的结果
                    done = True
                else:
                    max_score = score
                '''加入现阶段的查询图'''
                ito_topK_QueryGraphs = copy.deepcopy(topK_QueryGraphs)
                cache_QueryGraph_of_hop.append(ranked_cur_hop_QueryGraphs)
                
                if done: break   # When the best path in the previous iteration is same as the best path in current iteration
                hopNum += 1
                # pdb.set_trace()

        # # 组合约束
        # queryGraphs = self.mysqlConnection.queryGraphsCombine(queryGraphs, candRelsList[0:100]) # XXX 这个是什么
        '''最后需要的是什么？'''
        # 根据查询图结构从知识库中重新检索答案
        keys = {}
        newQueryGraphs = []
        # queryGraphs.extend(topK_QueryGraphs)    # 本轮的Topk加入结果
        for ito in cache_QueryGraph_of_hop: # 每个cache取topk个
            queryGraphs.extend(ito[:beam_size])
        # pdb.set_trace()
        for queryGraph in queryGraphs:
            queryGraph = self.mysqlConnection.addAnswerType(queryGraph) # 永辉师兄新加的内容
            queryGraph.getKey()
            if(queryGraph.key not in keys):
                # if(len(queryGraph.basePathTriples) == 0 and len(queryGraph.higherOrderTriples) == 0 and len(queryGraph.virtualConstrainTriples) == 0):
                #     answerList = self.mysqlConnection.searchAnswer(queryGraph=queryGraph, higherOrder= higherOrder)
                #     # answerList = self.mysqlConnection.searchAnswerBySQL(queryGraph=queryGraph)
                #     queryGraph.updateAnswer('\t'.join(answerList))
                keys[queryGraph.key] = 1
                newQueryGraphs.append(queryGraph)
        return newQueryGraphs

    def generateTopk_QueryGraphLan(self,queryGraphs,question,model=None,k=3):
        '''
            从3部分中，挑出Topk并返回（作为下一hop的外延）
            :param queryGraphs: 待选择的查询图
            :param model: 用于排序的模型
            :return: [TopK_Graph, ranked_Graph, max_score]
        '''
        
        newQueryGraphs = []
        # ranked_Graph = copy.deepcopy(queryGraphs)
        ranked_Graph = []
        max_score = 0
        seq2qg = {}
        if not queryGraphs:
            return newQueryGraphs, ranked_Graph, max_score

        ranked_Graph = copy.deepcopy(queryGraphs)
        
        # for qg in queryGraphs:
        #     MainPath = getMainPathSeq(eval(qg.serialization_raw()),False)
        #     ConstrainPath = getConstrainPathSeq(eval(qg.serialization_raw()),False)
        #     seq = '%s\t%s'%(MainPath, ConstrainPath)
        #     seq2qg[seq] = qg
        # # pdb.set_trace()
        # Top_fuzz = process.extract(question,list(seq2qg.keys()),limit = len(queryGraphs))
        # max_score = Top_fuzz[0][1]
        # for seq, score in Top_fuzz:
        #     ranked_Graph.append(seq2qg[seq])
        if len(queryGraphs)>=k:
            newQueryGraphs = ranked_Graph[:k]
        else:
            newQueryGraphs = ranked_Graph
        # pdb.set_trace()
        return newQueryGraphs, ranked_Graph, max_score
if __name__ == '__main__':
    searchPath = SearchPath()
    searchPath.searchQueryGraph()