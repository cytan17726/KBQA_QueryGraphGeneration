import pymysql
import sys
import os
import json
from typing import List, Dict, Tuple, Optional
import copy

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from src.QueryGraph import QueryGraph
from config.MysqlConfig import CCKSConfig, NLPCCConfig
from sqlalchemy import create_engine

# MAX_TRIPLE = 1000
# MAX_ANSWER_NUM = 1000
# MAX_VARIABLE_NUM = 300
# MAX_REL_NUM = 100

# 测试超参
MAX_ANSWER_NUM = 1000
MAX_VARIABLE_NUM = 300
MAX_REL_NUM = 100


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

    def search2HopChainWithEntity(self, entity: str, limitNum: int = 100) -> List[QueryGraph]:
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

    def search2HopChainWithEntityBasedProp(self, entity: str, limitNum: int = MAX_REL_NUM) -> List[QueryGraph]:
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

    
    def searchWithEntity(self, entity: str, limitNum: int = 100) -> Tuple[Tuple[str]]:
        sql = "select entry, prop, value from `pkubase` where `entry` = '%s' or `entry`='<%s>' or `entry`='\"%s\"' limit %s"\
                                         % (entity, entity, entity, str(limitNum))
        self.cursor.execute(sql)
        triples = self.cursor.fetchall()
        # import pdb; pdb.set_trace()
        return triples

    def searchWithEntityBasedProp(self, entity: str, limitNum: int = MAX_REL_NUM) -> Tuple[Tuple[str]]:
        sql = "select distinct prop from `pkubase` where `entry` = '%s' or `entry`='<%s>' or `entry`='\"%s\"' limit %s"\
                                         % (entity, entity, entity, str(limitNum))
        self.cursor.execute(sql)
        props = self.cursor.fetchall()
        queryGraphs: List[QueryGraph] = []
        mergeTriples = []
        for propItem in props:
            variables = self.searchObjectWithSubjectProp(entity, propItem[0])
            variablesList = [item[0] for item in variables]
            answer = '\t'.join(variablesList)
            mergeTriples.append((entity, propItem[0], variablesList))
            # queryGraph = QueryGraph((entity, propItem[0], variablesList[0]),
            #                     answer, 0, 2, [2], [variablesList])
            # queryGraphs.append(queryGraph)
        return mergeTriples

    def searchWithValue(self, entity: str, limitNum: int = MAX_REL_NUM):
        # sql = "select distinct prop from `pkubase` where `value` = '%s' or `value`='<%s>' or `value`='\"%s\"' limit %s"\
        #                                  % (entity, entity, entity, str(limitNum))
        sql = "select distinct prop from `pkubase` where `value` = '%s' or `value`='<%s>' or `value` = '\"%s\"' limit %s"\
                                         % (entity, entity, entity, str(limitNum))
        self.cursor.execute(sql)
        props = self.cursor.fetchall()
        # if(len(props) == 0):
        #     sql = "select distinct prop from `pkubase` where `value` = '\"%s\"' limit %s"\
        #                                  % (entity, str(limitNum))
        #     self.cursor.execute(sql)
        #     props = self.cursor.fetchall()
        # import pdb; pdb.set_trace()
        queryGraphs: List[QueryGraph] = []
        for propItem in props:
            variables = self.searchSubjectWithPropObject(propItem[0], entity, limitNum=MAX_VARIABLE_NUM)
            variablesList = [item[0] for item in variables]
            answer = '\t'.join(variablesList)
            # print(variablesList)
            # import pdb; pdb.set_trace()
            queryGraph = QueryGraph((variablesList[0], propItem[0], entity),
                                answer, 2, 0, [0], [variablesList])
            queryGraphs.append(queryGraph)
        return queryGraphs

    def searchWithEntityID(self, entity: str, limitNum: int = 100):
        try:
            entity = entity.replace('\'', '\\\'')
            sql = "select entry, prop, value from `pkubase` where `entry` = '%s' limit %s"\
                                            % (entity, str(limitNum))
            self.cursor.execute(sql)
            triples = self.cursor.fetchall()
        except:
            print(sql)
            # import pdb; pdb.set_trace()
        
        
        # import pdb; pdb.set_trace()
        return triples

    def searchWithEntityIDBasedProp(self, entity: str, limitNum: int = 100):
        entity = entity.replace('\'', '\\\'')
        sql = "select distinct prop from `pkubase` where `entry` = '%s' limit %s"\
                                        % (entity, str(limitNum))
        # print(sql)
        self.cursor.execute(sql)
        props = self.cursor.fetchall()
        mergeTriples = []
        for propItem in props:
            variables = self.searchObjectWithSubjectProp(entity, propItem[0])
            variablesList = [item[0] for item in variables]
            if(len(variablesList) > 0):
                # import pdb; pdb.set_trace()
                mergeTriples.append((entity, propItem[0], variablesList))
            else:
                print('关系有误：', propItem)
            # queryGraph = QueryGraph((entity, propItem[0], variablesList[0]),
            #                     answer, 0, 2, [2], [variablesList])
            # queryGraphs.append(queryGraph)
        return mergeTriples
    
    def searchMentionWithEntity(self, entity: str, limitNum: int = 100):
        sql = "select entry, prop from `pkuorder` where `prop` = '%s' limit %s"\
                                         % (entity, str(limitNum))
        if(sql not in self.sql2results):
            self.cursor.execute(sql)
            triples = self.cursor.fetchall()
            self.sql2results[sql] = triples
        else:
            triples = self.sql2results[sql]
        # import pdb; pdb.set_trace()
        return triples
    
    def searchObjectWithSubjectProp(self, subject: str, prop: str, limitNum: int = MAX_VARIABLE_NUM):
        subject = subject.replace('\\', '\\\\')
        subject = subject.replace('\'', '\\\'')
        sql = "select value from `pkubase` where (`entry` = '%s' or `entry`='<%s>' or `entry`='\"%s\"') and `prop` = '%s' limit %s"\
                                            % (subject, subject, subject, prop, str(limitNum))
        if(sql not in self.sql2results):
            self.cursor.execute(sql)
            objects = self.cursor.fetchall()
        else:
            objects = self.sql2results[sql]
        return objects

    def searchSubjectWithPropObject(self, prop: str, objectItem: str, limitNum: int = MAX_VARIABLE_NUM):
        entity = objectItem.replace('\'', '\\\'')
        sql = "select entry from `pkubase` where `prop` = '%s' and (`value` = '%s' or `value`='<%s>' or `value`='\"%s\"') limit %s"\
                                            % (prop, entity, entity, entity, str(limitNum))
        if(sql not in self.sql2results):
            self.cursor.execute(sql)
            subjects = self.cursor.fetchall()
            self.sql2results[sql] = subjects
        else:
            subjects = self.sql2results[sql]
        return subjects


    def isExistTriple(self, entry, prop, value):
        value = value.replace('\'', '\\\'')
        entry = entry.replace('\'', '\\\'')
        sql = "select entry from `pkubase` where `entry` = '%s' and `prop` = '%s' and `value` = '%s'"\
                                            % (entry, prop, value)
        # print(sql)
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


    def constrainVariable(self, objectsList, queryGraph, orderEntityConstrain):
        '''
        功能：根据约束路径过滤变量节点值
        输入：未过滤前的节点值：objectsList,
            orderEntityConstrain: 实体约束信息
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


    def searchAnswer(self, queryGraph: QueryGraph):
        initNodeId = queryGraph.initNodeId
        variableNodeIds = queryGraph.variableNodeIds
        pathTriples = queryGraph.pathTriples
        answerNodeId = queryGraph.answerNodeId
        entityConstrainIds = queryGraph.entityConstrainIDs
        i = 0
        initEntitys = []
        initEntitys.append(pathTriples[initNodeId])
        while(i < len(pathTriples)): # 路径延伸
            tripleOrder = i // 3
            forwardFlag = (variableNodeIds[tripleOrder] + 1) % 3
            if(forwardFlag == 0): # 正向
                objectsList = []
                for subject in initEntitys:
                    objects = self.searchObjectWithSubjectProp(subject, \
                                                                pathTriples[variableNodeIds[tripleOrder] - 1],
                                                                limitNum=MAX_ANSWER_NUM)
                    objectsList.extend([item[0] for item in objects])
                # print(objectsList)
                if(variableNodeIds[tripleOrder] in entityConstrainIds):
                    orderEntityConstrain = entityConstrainIds.index(variableNodeIds[tripleOrder])
                    # print()
                    objectsList = self.constrainVariable(objectsList, queryGraph, orderEntityConstrain)
                    # import pdb; pdb.set_trace()
                initEntitys = objectsList
                # import pdb; pdb.set_trace()
            else:
                entitysList = []
                for objectItem in initEntitys:
                    subjects = self.searchSubjectWithPropObject(pathTriples[variableNodeIds[tripleOrder] + 1], objectItem,
                    limitNum=MAX_ANSWER_NUM)
                    # print(subjects)
                    entitysList.extend([item[0] for item in subjects])
                if(variableNodeIds[tripleOrder] in entityConstrainIds):
                    orderEntityConstrain = entityConstrainIds.index(variableNodeIds[tripleOrder])
                    # print()
                    entitysList = self.constrainVariable(entitysList, queryGraph, orderEntityConstrain)
                # print(entitysList)
                initEntitys = entitysList
            i += 3
        # print(pathTriples, initEntitys)
        return initEntitys


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


    def done(self):
        self.cursor.close()
        self.db.commit()
        self.db.close()


class MysqlNLPCCConnection:
    def __init__(self, config: NLPCCConfig) -> None:
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

    
    # def searchWithEntity(self, entity: str, limitNum: int = 100) -> Tuple[Tuple[str]]:
    #     sql = "select entry, prop, value from `pkubase` where `entry` = '%s' or `entry`='<%s>' or `entry`='\"%s\"' limit %s"\
    #                                      % (entity, entity, entity, str(limitNum))
    #     self.cursor.execute(sql)
    #     triples = self.cursor.fetchall()
    #     # import pdb; pdb.set_trace()
    #     return triples

    # def searchWithValue(self, entity: str, limitNum: int = 100):
    #     sql = "select entry, prop, value from `pkubase` where `value` = '%s' or `value`='<%s>' or `value`='\"%s\"' limit %s"\
    #                                      % (entity, entity, entity, str(limitNum))
    #     self.cursor.execute(sql)
    #     triples = self.cursor.fetchall()
    #     # import pdb; pdb.set_trace()
    #     queryGraphs: List[QueryGraph] = []
    #     for path in triples:
    #         queryGraph = QueryGraph(path, path[0], 2, 0, [0])
    #         queryGraphs.append(queryGraph)
    #     return queryGraphs

    # def searchWithEntityID(self, entity: str, limitNum: int = 100):
    #     sql = "select entry, prop, value from `pkubase` where `entry` = '%s' limit %s"\
    #                                      % (entity, str(limitNum))
    #     self.cursor.execute(sql)
    #     triples = self.cursor.fetchall()
        
    #     # import pdb; pdb.set_trace()
    #     return triples
    
    def searchMentionWithEntity(self, entity: str, limitNum: int = 100):
        entity = entity.replace('\\', '\\\\')
        entity = entity.replace('\'', '\\\'')
        sql = "select entry, value from `nlpccmention2id` where `value` = '%s' limit %s"\
                                         % (entity, str(limitNum))
        # print(sql)
        self.cursor.execute(sql)
        triples = self.cursor.fetchall()
        # import pdb; pdb.set_trace()
        return triples



    def done(self):
        self.cursor.close()
        self.db.commit()
        self.db.close()


if __name__ == "__main__":
    connectionMysql = ConnectionMysql()
    connectionMysql.search2HopChainWithEntity('天使与猎人')

