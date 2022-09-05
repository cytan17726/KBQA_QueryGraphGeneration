import sys
import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)


# 读取分词结果
def readSegment(fileName: str):
    with open(fileName, 'r', encoding='utf-8') as fread:
        que2segment = json.load(fread)
        newQue2segment = {}
        for que in que2segment:
            # if('\t' in que):
            #     import pdb; pdb.set_trace()
            newQue2segment[que.replace('\t', '')] = que2segment[que]
            newQue2segment[que] = que2segment[que]
            newQue2segment[que.lower()] = que2segment[que]
            # import pdb; pdb.set_trace()
        wordsNum = {}
        for que in que2segment:
            num = len(que2segment[que])
            if(num not in wordsNum):
                wordsNum[num] = 1
            else:
                wordsNum[num] += 1
        wordsNumSorted = sorted(wordsNum.items(), key= lambda x:x[0], reverse=True)
        # print(wordsNumSorted)
        '''
        [(28, 1), (26, 2), (25, 3), (24, 2), (23, 2), (22, 5), (21, 9), (20, 10), (19, 16), (18, 18), (17, 30), (16, 60), (15, 102), (14, 121), (13, 198), (12, 291), (11, 345), (10, 512), (9, 660), (8, 935), (7, 1051), (6, 1451), (5, 1843), (4, 1475), (3, 307), (2, 42), (1, 1)]
        '''
    return newQue2segment
        # import pdb; pdb.set_trace()
    
    

if __name__ == "__main__":
    fileName = BASE_DIR + '/data/sep_res_1206.json'
    readSegment(fileName)
    pass
