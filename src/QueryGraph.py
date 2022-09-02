'''
功能：定义查询图结构
'''
import sys
import os
import json
from typing import List, Dict, Tuple
import copy

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)


# class QueryGraph


class QueryGraph:
    def __init__(self, pathTriples = (), answer = '', initNodeId = -1, answerNodeId = -1,\
                variableNodeIds = [], variableNodes = []) -> None:
        '''
        功能：查询图初始化
        输入：主路径三元组：pathTriples、答案字符串：answer、
                初始节点ID：initNodeId、答案节点ID：answerNodeId、
                中间变量节点ID：variableNodeIds
        输出：None
        '''
        # 主路径信息
        self.answer: str = answer
        self.pathTriples = pathTriples # 存储主路径三元组
        self.answerNodeId = answerNodeId # 存储答案节点ID
        self.initNodeId = initNodeId # 存储入口节点ID
        self.variableNodeIds = variableNodeIds # 存储中间的变量节点ID
        self.variableNodes = variableNodes # 存储所有变量节点的值
        self.schemaPath = self.getSchemaPath
        # self.sameNode
        # 实体约束信息
        self.entityConstrainTriples = []
        self.entityConstrainIDs = [] # 实体约束的主路径节点ID
        self.entityIDs = [] # 实体在实体约束路径中的位置ID，判断正向和逆向三元组
        self.key = ''
        self.availableEntityIds = []
        self.availableVirtualEntityIds = []
        self.availableHigherOrderIds = []
        self.virtualConstrainTriples = []
        self.virtualConstrainIDs = [] # 非实体约束主路径节点ID
        self.higherOrderTriples = []
        self.higherOrderConstrainIDs = []
        self.basePathTriples: List[List[str]] = []
        self.basePathVariableIds: List[List[int]] = []
        self.relConstrainTriples: List[List[str]] = []
        self.relConstrainIDs : List[List[int]] = []
        self.rels = {}

    def updateBasePath(self, basePathTriple, basePathVariableId):
        self.basePathTriples.append(basePathTriple)
        self.basePathVariableIds.append(basePathVariableId)

    def updateAnswerType(self, answerType):
        self.answerType = answerType

    def updateVariableNodes(self, nodes):
        self.variableNodes[-1] = copy.deepcopy(nodes)

    def setAvailableEntityIds(self, availableEntityIds):
        self.availableEntityIds = copy.deepcopy(availableEntityIds)

    def setAvailableVirtualEntityIds(self, availableEntityIds):
        self.availableVirtualEntityIds = copy.deepcopy(availableEntityIds)

    def setAvailableHigherOrderIds(self, availableEntityIds):
        self.availableHigherOrderIds = copy.deepcopy(availableEntityIds)

    def updateEntityConstrainInfo(self, entityConstrainTriple, entityConstrainID, entityID):
        self.entityConstrainTriples.append(entityConstrainTriple)
        self.entityConstrainIDs.append(entityConstrainID)
        self.entityIDs.append(entityID)

    def updateRelConstrainTriples(self, relConstrainTriple, relConstrainID):
        self.rels[relConstrainTriple[1]] = 1
        self.relConstrainTriples.append(relConstrainTriple)
        self.relConstrainIDs.append(relConstrainID)

    def updateVirtualConstrainInfo(self, virtualConstrainTriple, virtualConstrainID):
        self.virtualConstrainTriples.append(virtualConstrainTriple)
        self.virtualConstrainIDs.append(virtualConstrainID)

    def updateHigherOrderConstrainInfo(self, higherOrderConstrainTriple, higherOrderConstrainID):
        self.higherOrderTriples.append(higherOrderConstrainTriple)
        self.higherOrderConstrainIDs.append(higherOrderConstrainID)


    def getPathTriplesKey(self):
        pathKey = []
        tripleStr = ''
        for i in range(len(self.pathTriples)): 
            if(i not in self.variableNodeIds):
                tripleStr += self.pathTriples[i] + '\t'
            else:
                tripleStr += '<pad>\t'
            if((i + 1) % 3 == 0):
                pathKey.append(tripleStr)
                tripleStr = ''
        if(tripleStr != ''):
            pathKey.append(tripleStr)
        pathKey.sort()
        return '\t'.join(pathKey)

    def getBasePathTriplesKey(self):
        pathKey = []
        tripleStr = ''
        for i in range(len(self.basePathTriples)): 
            for j in range(len(self.basePathTriples[i])):
                if(j not in self.basePathVariableIds[i]):
                    tripleStr += self.basePathTriples[i][j] + '\t'
                else:
                    tripleStr += '<pad>\t'
                if((j + 1) % 3 == 0):
                    pathKey.append(tripleStr)
                    tripleStr = ''
            if(tripleStr != ''):
                pathKey.append(tripleStr)
        pathKey.sort()
        return '\t'.join(pathKey)

    def getEntityConstrainKey(self):
        pathKey = []
        tripleStr = ''
        for orderTriple, triple in enumerate(self.entityConstrainTriples):
            for orderIndex in range(len(triple)):
                if(orderIndex == self.entityIDs[orderTriple]):
                    tripleStr += '<pad>\t'
                else:
                    # import pdb; pdb.set_trace()
                    # print(tripleStr, triple[orderIndex])
                    tripleStr += triple[orderIndex] + '\t'
            pathKey.append(tripleStr)
            tripleStr = ''
        pathKey.sort()
        return '\t'.join(pathKey)

    def getRelConstrainKey(self):
        pathKey = []
        tripleStr = ''
        for orderTriple, triple in enumerate(self.relConstrainTriples):
            for orderIndex in range(len(triple)):
                if(orderIndex in self.relConstrainIDs[orderTriple]):
                    tripleStr += '<pad>\t'
                else:
                    # import pdb; pdb.set_trace()
                    # print(tripleStr, triple[orderIndex])
                    tripleStr += triple[orderIndex] + '\t'
            pathKey.append(tripleStr)
            tripleStr = ''
        pathKey.sort()
        return '\t'.join(pathKey)

    def getVirtualEntityConstrainKey(self):
        pathKey = []
        tripleStr = ''
        for orderTriple, triple in enumerate(self.virtualConstrainTriples):
            for orderIndex in range(len(triple)):
                tripleStr += triple[orderIndex] + '\t'
            pathKey.append(tripleStr)
            tripleStr = ''
        pathKey.sort()
        return '\t'.join(pathKey)

    def getHigherOrderConstrainKey(self):
        pathKey = []
        tripleStr = ''
        for orderTriple, triple in enumerate(self.higherOrderTriples):
            for orderIndex in range(len(triple)):
                tripleStr += triple[orderIndex] + '\t'
            pathKey.append(tripleStr)
            tripleStr = ''
        pathKey.sort()
        return '\t'.join(pathKey)

    def getKey(self) -> str:
        pathTriplesKey = self.getPathTriplesKey()
        entityConstrainKey = self.getEntityConstrainKey()
        virtualEntityConstrainKey = self.getVirtualEntityConstrainKey()
        higherOrderConstrainKey = self.getHigherOrderConstrainKey()
        basePathKey = self.getBasePathTriplesKey()
        relConstrainKey = self.getRelConstrainKey()
        self.key = pathTriplesKey + entityConstrainKey + virtualEntityConstrainKey + higherOrderConstrainKey + basePathKey + relConstrainKey

    
    def getSchemaPath(self) -> str:
        schemaPath = ''
        for i in range(len(self.pathTriples)):
            if(i in self.variableNodeIds):
                schemaPath += '\t'
            else:
                schemaPath += self.pathTriples[i] + '\t'
        return schemaPath[0:-1]

    def updateAnswer(self, answer) -> str:
        self.answer = answer

    def processAnswer(self):
        answer = self.answer.split('\t')
        for i, item in enumerate(answer):
            if '^^<http://www.w3.org' in item:
                p = item.index('^^<http://www.w3.org')
                answer[i] = answer[i][:p]
        return answer

    def serialization(self):
        answer = self.processAnswer()
        self.getKey()
        queryGraphInfo = json.dumps({"path": self.pathTriples,
            "basePathTriples": self.basePathTriples,
            "basePathVariableIds": self.basePathVariableIds,
            "entityPath": self.entityConstrainTriples,
            "entityIDs": self.entityIDs, 
            "relConstrainTriples": self.relConstrainTriples,
            "relConstrainIDs": self.relConstrainIDs,
            "virtualConstrainTriples": self.virtualConstrainTriples,
            "virtualConstrainIDs": self.virtualConstrainIDs,
            'higherOrderTriples': self.higherOrderTriples,
            "higherOrderConstrainIDs": self.higherOrderConstrainIDs,
            "entityConstrainIDs": self.entityConstrainIDs, "answer": '\t'.join(answer[0:]), 
            "initNodeId": self.initNodeId,
            "answerNodeId": self.answerNodeId,
            "variableNodeIds": self.variableNodeIds,
            "answerType": self.answerType,
            }, ensure_ascii=False)
        return queryGraphInfo

    def serialization_raw(self):
        answer = self.processAnswer()
        self.getKey()
        queryGraphInfo = json.dumps({"path": self.pathTriples,
            "basePathTriples": self.basePathTriples,
            "basePathVariableIds": self.basePathVariableIds,
            "entityPath": self.entityConstrainTriples,
            "entityIDs": self.entityIDs, 
            "relConstrainTriples": self.relConstrainTriples,
            "relConstrainIDs": self.relConstrainIDs,
            "virtualConstrainTriples": self.virtualConstrainTriples,
            "virtualConstrainIDs": self.virtualConstrainIDs,
            'higherOrderTriples': self.higherOrderTriples,
            "higherOrderConstrainIDs": self.higherOrderConstrainIDs,
            "entityConstrainIDs": self.entityConstrainIDs, "answer": '\t'.join(answer[0:]), 
            "initNodeId": self.initNodeId,
            "answerNodeId": self.answerNodeId,
            "variableNodeIds": self.variableNodeIds,
            }, ensure_ascii=False)
        return queryGraphInfo



    