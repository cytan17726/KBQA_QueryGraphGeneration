import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.build_query_graph.query_graph_generation import ExpansionLimitNum

    
def relationExpansion(entitys, forwardRel = '', backwardRel = '') -> None:
    '''
    功能：根据每个实体节点，进行一跳搜索获取基础候选，包含正向和反向
    '''
    if(len(entitys) > 100):
        '''
        默认候选过多时的情况忽略，后期需要可以改进处理。
        '''
        return
    for entity in self.entitys:
        execution, triples = self.searchWithEntity(entity, ExpansionLimitNum) # 正向扩充
        # import pdb; pdb.set_trace()
        keyExecution = self.execution2key(triples)
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