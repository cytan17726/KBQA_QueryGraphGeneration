import sys
import os
import json
from typing import Dict
import re
from typing import List, Dict, Tuple
import copy

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

def isTrueMainPath(triples, triplesGold):
    '''
    主路径合法条件：关系一致即可
    '''
    i = 0
    while(i < len(triples)):
        flag = False
        for j, triple in enumerate(triplesGold):
            if(triples[i + 1][0] != '#'):
                if(triples[i + 1] in triple):
                    triplesGold.remove(triple)
                    flag = True
                    break
            else:
                return True
        if(not flag):
            return False
        i += 3
    return True

def isTrueBasePath(triples, triplesGold):
    '''
    base路径合法条件：关系一致即可
    '''
    i = 0
    while(i < len(triples)):
        # import pdb; pdb.set_trace()
        flag = False
        for j, triple in enumerate(triplesGold):
            if(triples[i + 1][0] != '#'):
                if(triples[i + 1] in triple):
                    triplesGold.remove(triple)
                    flag = True
                    break
            else:
                return True
        if(not flag):
            return False
        if('<A>' == triples[0]):
            i += 3
        else:
            i += 2
    return True

def isTrueEntityPath(triples, triplesGold):
    '''
    实体约束合法条件：实体一致即可
    '''
    i = 0
    while(i < len(triples)):
        # import pdb; pdb.set_trace()
        flag = False
        for j, triple in enumerate(triplesGold):
            if(triples[i] in triple or triples[i + 2] in triple):
                triplesGold.remove(triple)
                flag = True
                break
        if(not flag):
            return False
        i += 3
    return True


def isTrueVirtualPath(triples, triplesGold):
    '''
    虚约束合法条件：关系要一致
    '''
    i = 0
    while(i < len(triples)):
        # import pdb; pdb.set_trace()
        flag = False
        for j, triple in enumerate(triplesGold):
            if(triples[i + 1] in triple):
                triplesGold.remove(triple)
                flag = True
                break
        if(not flag):
            return False
        i += 3
    return True


def isTrueHigherOrderPath(triples, triplesGold):
    '''
    高阶约束合法条件：关系要一致
    '''
    i = 0
    while(i < len(triples)):
        # import pdb; pdb.set_trace()
        flag = False
        for j, triple in enumerate(triplesGold):
            if(triples[i + 1] in triple):
                triplesGold.remove(triple)
                flag = True
                break
        if(not flag):
            return False
        i += 3
    return True


def isTrueRelConstrainPath(triples, triplesGold):
    '''
    高阶约束合法条件：关系要一致
    '''
    i = 0
    while(i < len(triples)):
        # import pdb; pdb.set_trace()
        flag = False
        for j, triple in enumerate(triplesGold):
            if(triples[i + 1] in triple):
                triplesGold.remove(triple)
                flag = True
                break
        if(not flag):
            return False
        i += 3
    return True