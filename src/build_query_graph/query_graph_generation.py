'''
这里实现与知识库交互产生查询图的接口
'''
import sys
import os
from typing import List, Dict, Tuple
import copy
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.QueryGraph import QueryGraph
from config.MysqlConfig import CCKSConfig
from src.mysql.MysqlConnection import MysqlConnection
ccksConfig = CCKSConfig()
ExpansionLimitNum = 10000

class QueryGraphGeneration(object):
    '''
    逻辑：根据实体构建查询语句进行路径搜索，该过程可构建实体集到可执行语句的映射；
            由于关系实例化后的不同，一条查询语句可对应多个查询图，该过程可构建可执行语句到查询图的映射；
            最后从查询图中获取答案
    '''
    def __init__(self, entitys: List[str]) -> None:
        '''
        功能：查询图生成初始化
        '''
        super().__init__()
        self.mysqlConnection = MysqlConnection(ccksConfig)
        self.entitys = entitys
        self.cursor = self.mysqlConnection.cursor
        self.entitys2execution: Dict[str, List[str]] = {} # 实体集到知识库可执行语句集合的映射,一对多
        # 可执行语句到查询图的映射，
        # 一对多（这里的可执行语句是泛化的，只是用于快速构建查询图，而不是与查询图一一对应的可执行语句）
        self.execution2queryGraphs: Dict[str, List[QueryGraph]] = {} 
        self.usedEntitys2unusedEntitys: Dict[str, List[str]] = {} # 已经使用的实体和剩余未使用实体之间的对应关系
        self.FocusNodeExpansion()


    def entitys2HashKey(self, entitys: List[str]) -> str:
        entitys.sort()
        return '\t'.join(entitys)

    
    def FocusNodeExpansion(self) -> None:
        '''
        功能：根据每个实体节点，进行一跳搜索获取基础候选，包含正向和反向
        '''
        for entity in self.entitys:
            execution, triples = self.searchWithEntity(entity, ExpansionLimitNum) # 正向扩充
            if(entity not in self.entitys2execution):
                self.entitys2execution[entity] = [execution]
            else:
                self.entitys2execution[entity].append(execution)
            # import pdb; pdb.set_trace()
            keyExecution = self.execution2key([execution])
            queryGraphs = self.forwardTriples2queryGraph(triples)
            self.execution2queryGraphs[keyExecution] = queryGraphs
            execution, triples = self.searchWithValue(entity, ExpansionLimitNum) # 反向扩充
            if(entity not in self.entitys2execution):
                self.entitys2execution[entity] = [execution]
            else:
                self.entitys2execution[entity].append(execution)
            # import pdb; pdb.set_trace()
            keyExecution = self.execution2key([execution])
            queryGraphs = self.backwardTriples2queryGraph(triples)
            self.execution2queryGraphs[keyExecution] = queryGraphs
            unusedEntitys = copy.deepcopy(self.entitys)
            unusedEntitys.remove(entity)
            self.usedEntitys2unusedEntitys[entity] = unusedEntitys
            # import pdb; pdb.set_trace()

    def querygraphExpansion(self):
        '''
        功能：对生成的查询图进行深度扩展，即增加一跳深度
        输入：生成的查询图列表：querygraphs
        '''
        for usedEntitys in self.entitys2execution:
            executions = self.entitys2execution[usedEntitys]
            for execution in executions:
                executionKey = self.execution2key([execution])
                # print(execution)
                # import pdb; pdb.set_trace()
                newQueryGraphs = copy.deepcopy(self.execution2queryGraphs[executionKey])
                for querygraph in self.execution2queryGraphs[executionKey]:
                    # answers = querygraph.answer
                    querygraphs = self.relationExpansion(querygraph)
                    newQueryGraphs.extend(querygraphs)
                self.execution2queryGraphs[executionKey] = newQueryGraphs

    
    def forwardTriples2queryGraph(self, triples: Tuple[Tuple[str]]) -> List[QueryGraph]:
        '''
        功能：将正向三元组转成查询图结构，用于单跳查询图构建
        '''
        path2answer: Dict[str, List[str]] = {} # 每条路径对应的答案
        path2Tuple: Dict[str, Tuple[str]] = {}
        for triple in triples:
            key = triple[0] + triple[1]
            if(key not in path2answer):
                path2answer[key] = [triple[2]]
                path2Tuple[key] = triple
            else:
                path2answer[key].append(triple[2])
        queryGraphs: List[QueryGraph] = []
        for key in path2answer:
            queryGraph = QueryGraph(path2Tuple[key], path2answer[key], 0, 2, [2])
            queryGraph.variableNodes.append(copy.deepcopy(path2answer[key]))
            queryGraphs.append(queryGraph)
        # for queryGraph in queryGraphs:
        #     queryGraph.serialization()
        return queryGraphs

    def backwardTriples2queryGraph(self, triples: Tuple[Tuple[str]]) -> List[QueryGraph]:
        '''
        功能：将反向三元组转成查询图结构，用于单跳查询图构建
        '''
        path2answer: Dict[str, List[str]] = {} # 每条路径对应的答案
        path2Tuple: Dict[str, Tuple[str]] = {}
        for triple in triples:
            key = triple[1] + triple[2]
            if(key not in path2answer):
                path2answer[key] = [triple[0]]
                path2Tuple[key] = triple
            else:
                path2answer[key].append(triple[0])
        queryGraphs: List[QueryGraph] = []
        for key in path2answer:
            queryGraph = QueryGraph(path2Tuple[key], path2answer[key], 2, 0, [0])
            queryGraph.variableNodes.append(copy.deepcopy(path2answer[key]))
            queryGraphs.append(queryGraph)
        # for queryGraph in queryGraphs:
        #     queryGraph.serialization()
        return queryGraphs

    def forwardTriples2queryGraphMulHop(self, querygraph: QueryGraph, triples: Tuple[Tuple[str]]) -> List[QueryGraph]:
        '''
        功能：将正向三元组转成查询图结构，用于多跳查询图构建
        '''
        path2answer: Dict[str, List[str]] = {} # 每条路径对应的答案
        path2Tuple: Dict[str, Tuple[str]] = {}
        for triple in triples:
            key = triple[0] + triple[1]
            if(key not in path2answer):
                path2answer[key] = [triple[2]]
                path2Tuple[key] = triple
            else:
                path2answer[key].append(triple[2])
        queryGraphs: List[QueryGraph] = []
        for key in path2answer:
            queryGraph = QueryGraph(querygraph.pathTriples + path2Tuple[key], path2answer[key],\
                                    querygraph.initNodeId, len(querygraph.pathTriples) - 1, \
                                    querygraph.variableNodeIds + [querygraph.answerNodeId + 2])
            queryGraph.variableNodes.append(copy.deepcopy(path2answer[key]))
            queryGraphs.append(queryGraph)
        # for queryGraph in queryGraphs:
        #     queryGraph.serialization()
        return queryGraphs

    
    def backwardTriples2queryGraphMulHop(self, querygraph: QueryGraph, triples: Tuple[Tuple[str]]) -> List[QueryGraph]:
        '''
        功能：将反向三元组转成查询图结构，用于多跳查询图构建
        '''
        path2answer: Dict[str, List[str]] = {} # 每条路径对应的答案
        path2Tuple: Dict[str, Tuple[str]] = {}
        for triple in triples:
            key = triple[1] + triple[2]
            if(key not in path2answer):
                path2answer[key] = [triple[0]]
                path2Tuple[key] = triple
            else:
                path2answer[key].append(triple[0])
        queryGraphs: List[QueryGraph] = []
        for key in path2answer:
            queryGraph = QueryGraph(querygraph.pathTriples + path2Tuple[key],\
                                    path2answer[key],\
                                    querygraph.initNodeId,\
                                    len(querygraph.pathTriples) - 3,\
                                    querygraph.variableNodeIds + [len(querygraph.pathTriples) - 3])
            queryGraph.variableNodes.append(copy.deepcopy(path2answer[key]))
            queryGraphs.append(queryGraph)
        # for queryGraph in queryGraphs:
        #     queryGraph.serialization()
        return queryGraphs

    # def relationExpansion(self):
    #     pass
    def deleteTriples(self, triples, deleteRel):
        '''
        功能：删除具有某个关系的三元组，防止正向反向搜索中的重复问题
        输入：三元组：triples
                要删除的关系名字：deleteRel
        输出：新的三元组
        '''
        newTriples = []
        for triple in triples:
            if(triple[1] != deleteRel):
                newTriples.append(triple)
        return newTriples


    def relationExpansion(self, querygraph: QueryGraph) -> None:
        '''
        功能：根据每个实体节点，进行一跳搜索获取基础候选，包含正向和反向
        输入：一个查询图：querygraph
        输出：扩展后的查询图
        '''
        if(len(querygraph.answer) > 100):
            '''
            默认候选过多时的情况忽略，后期需要可以改进处理。
            '''
            return
        querygraphs = []
        for entity in querygraph.answer:
            if((querygraph.answerNodeId + 1) % 3 == 0): # 表示最后一个主路径是正向的
                execution, triples = self.searchWithEntity(entity, ExpansionLimitNum) # 正向扩充
                # import pdb; pdb.set_trace()
                keyExecution = self.execution2key([])
                queryGraphs = self.forwardTriples2queryGraphMulHop(querygraph, triples)
                querygraphs.extend(queryGraphs)
                execution, triples = self.searchWithValue(entity, ExpansionLimitNum) # 反向扩充
                # 去除某个关系的triples
                newTriples = self.deleteTriples(triples, querygraph.pathTriples[-2])
                # import pdb; pdb.set_trace()
                # keyExecution = self.execution2key([])
                queryGraphs = self.backwardTriples2queryGraphMulHop(querygraph, newTriples)
                querygraphs.extend(queryGraphs)
            else: # 表示最后一个主路径三元组是反向的
                execution, triples = self.searchWithEntity(entity, ExpansionLimitNum) # 正向扩充
                # import pdb; pdb.set_trace()
                newTriples = self.deleteTriples(triples, querygraph.pathTriples[-2])
                queryGraphs = self.forwardTriples2queryGraphMulHop(querygraph, newTriples)
                querygraphs.extend(queryGraphs)
                execution, triples = self.searchWithValue(entity, ExpansionLimitNum) # 反向扩充
                # import pdb; pdb.set_trace()
                # keyExecution = self.execution2key([])
                queryGraphs = self.backwardTriples2queryGraphMulHop(querygraph, triples)
                querygraphs.extend(queryGraphs)
        return querygraphs
            


    def execution2key(self, execution: List[str]) -> str:
        return '\t'.join(execution)


    def entityConstrainAddition(self):
        '''
        功能：对查询图进行实体约束挂载
        '''
        for usedEntitys in self.entitys2execution:
            executions = self.entitys2execution[usedEntitys]
            for execution in executions:
                executionKey = self.execution2key([execution])
                unusedEntitys = self.usedEntitys2unusedEntitys[usedEntitys]
                # import pdb; pdb.set_trace()
                queryGraphs = self.execution2queryGraphs[executionKey]
                copyQueryGraphs = copy.deepcopy(queryGraphs)
                i = 0
                for querygraph in copyQueryGraphs:
                    i += 1
                    # import pdb; pdb.set_trace()
                    # print(i, len(copyQueryGraphs), querygraph)
                    for entity in unusedEntitys:
                        for i, nodes in enumerate(querygraph.variableNodes):
                            constrainPath2Results = self.constrainNodes(nodes, entity)
                            for constrainPath in constrainPath2Results:
                                newQueryGraph = copy.deepcopy(querygraph)
                                newQueryGraph.entityConstrainIDs.append(querygraph.variableNodeIds[i])
                                newQueryGraph.entityConstrainTriples.append(constrainPath)
                                newQueryGraph.variableNodes[i] = constrainPath2Results[constrainPath]
                                if(querygraph.variableNodeIds[i] == querygraph.answerNodeId):
                                    newQueryGraph.answer = constrainPath2Results[constrainPath]
                                    # import pdb; pdb.set_trace()
                                queryGraphs.append(newQueryGraph)
                            # for node in nodes:
                            #     sql, triples = self.fromNodeToEntity(node, entity)
                            #     if(len(triples) != 0):
                            #         newQueryGraphs = self.addEntityConstrainToQueryGraphForward(
                            #             querygraph, triples, nodeId)
                            #         import pdb; pdb.set_trace()
                            #         queryGraphs.append(newQueryGraphs)
                            #     # print(sql, triples)
                            #     sql, triples = self.fromEntityToNode(node, entity)
                            #     if(len(triples) != 0):
                            #         newQueryGraphs = self.addEntityConstrainToQueryGraphForward(
                            #             querygraph, triples, nodeId)
                            #         queryGraphs.extend(newQueryGraphs)
                            #         import pdb; pdb.set_trace()
                            # print(sql, triples)

    def constrainNodes(self, nodeEntitys: List[str], constrainEntity: str) -> Dict[str, List[str]]:
        '''
        功能：在中间节点上加约束。加上约束实体后，如果返回结果不为空，说明该约束有效，否则，该约束无效
        '''
        constrainPath2Results = {}
        for node in nodeEntitys:
            sql, triples = self.fromNodeToEntity(node, constrainEntity)
            if(len(triples) ==  1):
                key = '\t'.join(triples[0][1:3])
                if(key not in constrainPath2Results):
                    constrainPath2Results[key] = [node]
                else:
                    constrainPath2Results[key].append(node)
            elif(len(triples) > 1):
                print('constrainNodes')
                key = '\t'.join(triples[0][1:3])
                if(key not in constrainPath2Results):
                    constrainPath2Results[key] = [node]
                else:
                    constrainPath2Results[key].append(node)
                # import pdb; pdb.set_trace()
            sql, triples = self.fromEntityToNode(node, constrainEntity)
            if(len(triples) ==  1):
                key = '\t'.join(triples[0][0: 2])
                if(key not in constrainPath2Results):
                    constrainPath2Results[key] = [node]
                else:
                    constrainPath2Results[key].append(node)
            elif(len(triples) > 1):
                '''
                两个实体之间可能存在不同的关系，如‘郭芙’和‘郭靖’之间既是“家人”也是“师承”关系，
                这里只简单保留第一个，因为实体约束中的关系意义不大，如有需要可进行保留。
                '''
                print('constrainNodes')
                key = '\t'.join(triples[0][0: 2])
                if(key not in constrainPath2Results):
                    constrainPath2Results[key] = [node]
                else:
                    constrainPath2Results[key].append(node)
                # import pdb; pdb.set_trace()
        # if(len(constrainPath2Results) > 0):
        #     import pdb; pdb.set_trace()
        return constrainPath2Results 

                        
    def addEntityConstrainToQueryGraphForward(self, queryGraph, triples, nodeId):
        newQueryGraph = copy.deepcopy(queryGraph)
        if(len(triples) > 1):
            print('相同实体有多个关系')
            import pdb; pdb.set_trace()
        newQueryGraph.entityConstrainTriples = triples[0]
        newQueryGraph.entityConstrainIDs = nodeId # 实体约束的主路径节点ID
        self.entityID = 2
        return newQueryGraph


    def addEntityConstrainToQueryGraphBackward(self, queryGraph, triples, nodeId):
        newQueryGraph = copy.deepcopy(queryGraph)
        if(len(triples) > 1):
            print('相同实体有多个关系')
            import pdb; pdb.set_trace()
        newQueryGraph.entityConstrainTriples = triples[0]
        newQueryGraph.entityConstrainIDs = nodeId # 实体约束的主路径节点ID
        self.entityID = 0
        return newQueryGraph


    def fromNodeToEntity(self, node: str, entity2: str, limitNum: int = 10):
        sql = "select entry, prop, value from `pkubase` where `entry` = '%s' and (`value` = '%s' or `value`='<%s>' or `value`='\"%s\"') limit %s"\
                                         % (node, entity2, entity2, entity2, str(limitNum))
        # sql = "select entry, prop, value from `pkubase` where `value` = '<北京大学>' and `value` = '<天文学家>' "
        self.cursor.execute(sql)
        triples = self.cursor.fetchall()
        return (sql, triples)
    
    def fromEntityToNode(self, node: str, entity2: str, limitNum: int = 10):
        # import pdb; pdb.set_trace()
        sql = "select entry, prop, value from `pkubase` where `value` = '%s' and (`entry` = '%s' or `entry`='<%s>' or `entry`='\"%s\"') limit %s"\
                                         % (node, entity2, entity2, entity2, str(limitNum))
        self.cursor.execute(sql)
        triples = self.cursor.fetchall()
        # import pdb; pdb.set_trace()
        return (sql, triples)


    def constrainAddition(self):
        '''
        功能：对所有查询图进行约束挂载
        '''
        self.entityConstrainAddition()
        pass

    
    def searchWithEntity(self, entity: str, limitNum: int = 1000) -> Tuple[str, Tuple[Tuple[str]]]:
        sql = "select entry, prop, value from `pkubase` where `entry` = '%s' or `entry`='<%s>' or `entry`='\"%s\"' limit %s"\
                                         % (entity, entity, entity, str(limitNum))
        self.cursor.execute(sql)
        triples = self.cursor.fetchall()
        # import pdb; pdb.set_trace()
        return (sql, triples)

    def searchWithValue(self, entity: str, limitNum: int = 1000) -> Tuple[str, Tuple[Tuple[str]]]:
        sql = "select entry, prop, value from `pkubase` where `value` = '%s' or `value`='<%s>' or `value`='\"%s\"' limit %s"\
                                         % (entity, entity, entity, str(limitNum))
        self.cursor.execute(sql)
        triples = self.cursor.fetchall()
        return (sql, triples)

    def searchWithEntityID(self, entity: str, limitNum: int = 100):
        sql = "select entry, prop, value from `pkubase` where `entry` = '%s' limit %s"\
                                         % (entity, str(limitNum))
        self.cursor.execute(sql)
        triples = self.cursor.fetchall()
        
        # import pdb; pdb.set_trace()
        return triples

    def printInfo(self):
        for usedEntitys in self.entitys2execution:
            print(usedEntitys)
            print(self.entitys2execution[usedEntitys])
            
            executions = self.entitys2execution[usedEntitys]
            for execution in executions:

                executionKey = self.execution2key([execution])
                # print(execution)
                # import pdb; pdb.set_trace()
                for queryGraph in self.execution2queryGraphs[executionKey]:
                    queryGraph.serialization()


if __name__ == "__main__":
    # querygraphGeneration = QueryGraphGeneration(['北京大学', '<主持人>'])
    # querygraphGeneration = QueryGraphGeneration(['<郭靖_（金庸武侠小说《射雕英雄传》男主角）>',\
    #                         '<黄蓉_（金庸武侠小说《射雕英雄传》女主角）>'])
    querygraphGeneration = QueryGraphGeneration(['<杨康_（武侠小说《射雕英雄传》中的人物）>'])
    querygraphGeneration.constrainAddition()
    querygraphGeneration.querygraphExpansion()
    querygraphGeneration.printInfo()