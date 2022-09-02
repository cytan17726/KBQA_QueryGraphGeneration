import linecache
import sys
import os
import json
from typing import List, Tuple, Dict
import re

# 时间正则
time_5 = re.compile(r'\d{1,2}[月]\d{1,2}[日]')
time_4 = re.compile(r'\d{4}-\d{1,2}-\d{1,2}')
time_3 = re.compile(r'\d{4}[年]')   
# time_2 = re.compile(r'\d{4}[年]\d{1,2}[月]')    # 62年8月 ; 1962年8月
time_2 = re.compile(r'\d{2,4}[年]\d{1,2}[月]')    # 62年8月 ; 1962年8月
time_1 = re.compile(r'\d{4}[年]\d{1,2}[月]\d{1,2}[日]') # 2011年11月17日
# 价格正则
price_1 = re.compile(r'\d+块')
price_2 = re.compile(r'\d+的')
# 距离正则
distance_1 = re.compile(r'\d+公里')
distance_2 = re.compile(r'\d+000米')
distance_3 = re.compile(r'\d+km')
distance_4 = re.compile(r'\d+千米')
# 年龄正则
age_1 = re.compile(r'\d{1,2}[岁]')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

def getQuestionType(que: str):
    type2key = {"compareJudge": ["同一个位置吗", "一样吗"],
                "NumericalCalculation": ["比.*高多少", "比.*胖多少"],
                "ComparisonSelectionMax": ["谁更高"],
                "ComparisonSelectionMin": ["谁更矮"]}
    for queType in type2key:
        keys = type2key[queType]
        for key in keys:
            pattern = re.compile(key)
            res = re.findall(pattern, que)
            if(len(res) > 0):
                return queType
    return ''

def readQuestionType(fileName:str) -> Dict[str, str]:
    with open(fileName, 'r', encoding='utf-8') as fread:
        que2type = {}
        for line in fread:
            lineCut = line.strip().split('\t')
            que = '\t'.join(lineCut[:-1])
            if(que not in que2type):
                que2type[que] = lineCut[-1]
        return que2type

def getQuestionsWithComplex(fileName: str) -> List[str]:
    with open(fileName, 'r', encoding='utf-8') as fread:
        questions = []
        for line in fread:
            lineCut = line.strip().split('\t')
            if line.count('\t')>1:
                questions.append('\t'.join(lineCut[:-1]))
            else:
                questions.append(lineCut[0])
        return questions

def getFloat(answer: str):
    pattern = re.compile(r'\d+\.\d+|\d+')
    result = re.findall(pattern, answer)
    # print(result)
    # if('226' in answer):
    #     import pdb; pdb.set_trace()
    if(len(result) == 1):
        return ''.join(result[0])
    else:
        return ''


def getQuestions(fileName: str) -> List[str]:
    questions = []
    with open(fileName, 'r', encoding='utf-8') as fread:
        lines = fread.readlines()
        i = 0
        while(i < len(lines)):
            line = lines[i].strip()
            pos = line.find(':')
            que = line[pos + 1:]
            questions.append(que)
            i += 4
    return questions


def getQuestionsAndTypes(fileName: str):
    questions = []
    que2type = {}
    with open(fileName, 'r', encoding='utf-8') as fread:
        data = json.load(fread)
        for item in data:
            questions.append(item['question'])
            que2type[item['question']] = item['tag']
    return questions, que2type


def getQuestionsFromRecallFalse(fileName: str) -> List[str]:
    questions = []
    with open(fileName, 'r', encoding='utf-8') as fread:
        lines = fread.readlines()
        i = 0
        while(i < len(lines)):
            que = lines[i].split('\t')[0]
            questions.append(que)
            i += 3
    return questions


def getQuestionsFromTestset(fileName: str) -> List[str]:
    questions = []
    with open(fileName, 'r', encoding='utf-8') as fread:
        originResults = json.load(fread)
        for key in originResults.keys():
            # ans = originResults[key]['ans']
            question = originResults[key]['que']
            # import pdb; pdb.set_trace()
            questions.append(question)
    if(len(questions) == 0):
        print('{%s}文件路径出错或者文件为空！'%(fileName))
    else:
        print('{%s}文件读取成功！'%(fileName))
    return questions

def getTypeQuestionsFromTestset(fileName: str) -> List[str]:
    questions = []
    with open(fileName, 'r', encoding='utf-8') as fread:
        originResults = json.load(fread)
        for key in originResults.keys():
            # ans = originResults[key]['ans']
            question = originResults[key]['que']
            # import pdb; pdb.set_trace()
            questions.append((question, 'Normal'))
    if(len(questions) == 0):
        print('{%s}文件路径出错或者文件为空！'%(fileName))
    else:
        print('{%s}文件读取成功！'%(fileName))
    return questions


def readQue2AnswerFromTestset(fileName: str) -> Dict[str, str]:
    que2answer = {}
    with open(fileName, 'r', encoding='utf-8') as fread:
        originResults = json.load(fread)
        for key in originResults.keys():
            ans = originResults[key]['ans']
            que = originResults[key]['que']
            # import pdb; pdb.set_trace()
            que2answer[que] = '\t'.join(ans)
    return que2answer

def addQue2AnswerOtherTypes(fileName: str, que2answer) -> Dict[str, str]:
    with open(fileName, 'r', encoding='utf-8') as fread:
        data = json.load(fread)
        for item in data:
            answer = item['answer']
            if(answer == True):
                answer = '是'
            elif(answer == False):
                answer = '否'
            if(item['question'] not in que2answer):
                que2answer[item['question']] = answer
            # print(answer)
    return que2answer

def addQue2GoldTriples(fileName: str, que2goldTriples):
    with open(fileName, 'r', encoding='utf-8') as fread:
        data = json.load(fread)
        for item in data:
            if(item['question'] not in que2goldTriples):
                triples = []
                for triple in item['triple']:
                    triples.append(tuple(triple))
                # import pdb; pdb.set_trace()
                que2goldTriples[item['question']] = triples
                # if(item['question'] == '与王永林星座和出生地点都一样的有哪些人？'):
                #     import pdb; pdb.set_trace()
        return que2goldTriples

def addQue2GoldTriples2(fileName: str, que2goldTriples):
    with open(fileName, 'r', encoding='utf-8') as fread:
        data = json.load(fread)
        for item in data:
            if(item['question'] not in que2goldTriples):
                triples = []
                triples.append(tuple(item['triple']))
                # import pdb; pdb.set_trace()
                que2goldTriples[item['question']] = triples
        return que2goldTriples

def readEntityLinking(fileName: str):
    with open(fileName, 'r', encoding='utf-8') as fread:
        que2entitys = json.load(fread)
        return que2entitys
        # import pdb; pdb.set_trace()

def addEntityLinking(que2entitys: Dict[str, str], fileName: str):
    with open(fileName, 'r', encoding='utf-8') as fread:
        que2entitysNew = json.load(fread)
        for que in que2entitysNew:
            # import pdb; pdb.set_trace()
            if(que['question'] not in que2entitys):
                el = {}
                for item in que['EL']:
                    el[item] = [que['EL'][item]]
                que2entitys[que['question']] = el
                # import pdb; pdb.set_trace()
        return que2entitys

def find_time_constrain(question) -> Tuple[str]:
    res = re.findall(time_1, question)  # 年月日
    # import pdb; pdb.set_trace()
    if not res:
        res = re.findall(time_2, question)  # XX年XX月
    if not res:
        res = re.findall(time_3, question)  # XXXX年
    if not res:
        res = re.findall(time_4, question)  # 1999-05-31
    if not res:
        res = re.findall(time_5, question)# XX月XX日
        # if res:
        #     return res[0]
    if not res:
        return ''
    res = res[0]
    begin = question.find(res)
    output = [(res, begin, begin + len(res) - 1)]
    return output

def normalize_time_constrain(time_mention:str)->List[str]:
    output = []
    '''
    返回分界符确定的约束
    位数不足就要填充
    2011年 -> "2011" / <2011年>
    2011年8月7日 -> "2011-08-07"
    '''
    ''' 确定标准化的属性值形式 '''
    year_reg = re.compile(r'(\d{2,4})年')
    month_reg = re.compile(r'(\d{1,2})月')
    day_reg = re.compile(r'(\d{1,2})日')
    time_4 = re.compile(r'\d{4}-\d{1,2}-\d{1,2}')
    if re.findall(time_4, time_mention):    # 标准形式
        return ['\"'+time_mention+'\"']
    reg_list = []
    tmp_y = re.findall(year_reg, time_mention)
    if len(tmp_y)>0:
        year = tmp_y[0]
        if len(year) == 2:
            year = '19'+year
        reg_list.append(year)
        output.append('<'+time_mention+'>') # 年的，加下原样
    tmp_m = re.findall(month_reg, time_mention)
    if len(tmp_m)>0:
        month = tmp_m[0]
        if len(month) == 1:
            month = '0'+month
        reg_list.append(month)
    tmp_d = re.findall(day_reg, time_mention)
    if len(tmp_d)>0:
        day = tmp_d[0]
        if len(day) == 1:
            day = '0'+day
        reg_list.append(day)
    output.append('\"'+'-'.join(reg_list)+'\"')
    # print('\"'+'-'.join(reg_list)+'\"')
    
    # 这边就抽取 年/月/日，构成属性值 "1999-05-31"
    # 还有对应的实体
    return output

def find_price_constrain(question):
    res = []
    temp_res = re.findall(price_1, question)
    if len(temp_res) > 0:
        res = temp_res
    else:
        temp_res = re.findall(price_2, question)        
        if len(temp_res) > 0:
            res = temp_res
    # import pdb; pdb.set_trace()
    if len(res) > 0:
        if '高于' in question:
            return [res[0][:-1], -1, -1, 'price', '>']
        else:
            return [res[0][:-1], -1, -1, 'price', '<']
    else:
        return ''

def find_distrance_constrain(question):
    res = re.findall(distance_1, question)
    if len(res) > 0:
        # import pdb; pdb.set_trace()
        res = res[0][: -2]
        begin = question.find(res)
        return (res, begin, begin + len(res) - 1, 'distance', '<')
    res = re.findall(distance_2, question)
    if len(res) > 0:
        res = res[0][: -4]
        begin = question.find(res)
        return (res, begin, begin + len(res) - 1, 'distance', '<')
    res = re.findall(distance_3, question)
    if len(res) > 0:
        res = res[0][: -2]
        begin = question.find(res)
        return (res, begin, begin + len(res) - 1, 'distance', '<')
    res = re.findall(distance_4, question)
    if len(res) > 0:
        res = res[0][: -2]
        begin = question.find(res)
        return (res, begin, begin + len(res) - 1, 'distance', '<')
    return ''


def find_argmax_constrain(question: str):
    '''
    功能：识别argmax高阶触发词
    输入：
        question:问句
    输出：
        长度为三的元组结果
    '''
    argmaxWords = ['最贵']
    for word in argmaxWords:
        if(word in question):
            begin = question.find(word)
            return ('argmax', begin, begin + len(word) - 1, 'higher-order')
    argminWords = ['最近', '最便宜', '最少', '便宜']
    for word in argminWords:
        if(word in question):
            begin = question.find(word)
            return ('argmin', begin, begin + len(word) - 1, 'higher-order')
    return ''

def find_age_constrain(question: str):
    res = re.findall(age_1, question)
    if not res:
        return ''
    res = res[0]
    begin = question.find(res)
    # import pdb;pdb.set_trace()
    output = [(res, begin, begin + len(res) - 1)]
    # if res[-1]== '岁':
    #     output.append(('\"'+res[:-1]+'\"', begin, begin + len(res) - 1))
    return output

def normalize_age_constrain(age_mention:str)->List[str]:
    output = []
    output.append(age_mention)
    output.append('\"'+ age_mention[:-1] +'\"')
    return output

def addMentionAndDigitToLinking(que2entitys: Dict[str, List[str]]):
    for que in que2entitys:
        # import pdb;pdb.set_trace()
        # if que != '导演诺兰2017年的哪部作品出品了？':
        #     continue
        # import pdb;pdb.set_trace()
        # if('1998-07-01' not in que):
        #     continue
        for mention in que2entitys[que]:
            mentionTuple = eval(mention)
            if(len(mentionTuple[0]) > 0 and mentionTuple[0] not in que2entitys[que][mention]):
                que2entitys[que][mention].append(mentionTuple[0])
        time_list = find_time_constrain(que) # 返回 时间约束的列表  # 找到对应的mention list
        if(len(time_list) > 0):
            for _time in time_list:
                timeStr = str(_time)
                que2entitys[que][timeStr] = normalize_time_constrain(_time[0])  # 标准化，记录  # XXX 调整为 直接覆盖
                # if(timeStr not in que2entitys[que]):    # 这个Span没有被找过
                #     que2entitys[que][timeStr] = normalize_time_constrain(_time[0])  # 标准化，记录
            # print(que)
            # print(que2entitys[que])
            # import pdb; pdb.set_trace()
        distance = find_distrance_constrain(que)
        if(len(distance) > 0):
            distanceStr = str(distance)
            que2entitys[que][distanceStr] = [distance[0]]
            # import pdb; pdb.set_trace()
        price = find_price_constrain(que)
        if(len(price) > 0):
            priceStr = str(price)
            que2entitys[que][priceStr] = [price[0]]
            # import pdb; pdb.set_trace()
        argmax = find_argmax_constrain(que)
        if(len(argmax) > 0):
            argmaxStr = str(argmax)
            que2entitys[que][argmaxStr] = [argmax[0]]
        age_list = find_age_constrain(que)
        if(len(age_list) > 0):
            for _age in age_list:
                ageStr = str(_age)
                if(ageStr not in que2entitys[que]):
                    que2entitys[que][ageStr] = normalize_age_constrain(_age[0])
    # import pdb;pdb.set_trace()
    return que2entitys

def addMentionToLinking(que2entitys: Dict[str, List[str]]):
    for que in que2entitys:
        for mention in que2entitys[que]:
            mentionTuple = eval(mention)
            if(len(mentionTuple[0]) > 0 and mentionTuple[0] not in que2entitys[que][mention]):
                que2entitys[que][mention].append(mentionTuple[0])
    return que2entitys

def readCands(fileName: str):
    que_num = 0
    with open(fileName, 'r', encoding='utf-8') as fread:
        lines = fread.readlines()
        i = 0
        que2cands = {}
        while(i < len(lines)):
            que = lines[i].strip()
            
            i += 1
            cands = []
            while(i < len(lines) and lines[i] != '\n'):
                # print('current:', lines[i])
                try:
                    cand = json.loads(lines[i].strip())
                except:
                    # import pdb; pdb.set_trace()
                    print('error:' + lines[i])
                    print(lines[i - 1])
                    # i += 1
                    break
                key = cand['path']
                answer = cand['answer']
                cands.append((key, answer))
                # import pdb; pdb.set_trace()
                i += 1
            while(i < len(lines) and lines[i] == '\n'):
                i += 1
            que2cands[que] = cands
            que_num+=1
            if(len(cands) == 0):
                print('候选为0', que)
    print('que_num: %d'%que_num)
    return que2cands

def readSequences(fileName: str):
    with open(fileName, 'r', encoding='utf-8') as fread:
        lines = fread.readlines()
        i = 0
        que2cands = {}
        while(i < len(lines)):
            que = lines[i].strip()
            i += 1
            cands = []
            while(i < len(lines) and lines[i] != '\n'):
                try:
                    cand = json.loads(lines[i].strip())
                except:
                    # import pdb; pdb.set_trace()
                    print('error:' + lines[i])
                    # i += 1
                    break
                key = cand['mainPath']
                answer = cand['entityPath']
                f1 = cand['f1']
                cands.append((key, answer, f1))
                # import pdb; pdb.set_trace()
                i += 1
            i += 1
            que2cands[que] = cands
    return que2cands


def readCandsInfo(fileName: str):
    with open(fileName, 'r', encoding='utf-8') as fread:
        lines = fread.readlines()
        i = 0
        que2cands = {}
        lenLines = len(lines)
        while(i < lenLines):
            # if((i + 1) > 100000):
            #     print('已读取%d行' %(i), end=' ')
            #     break
            que = lines[i].strip()
            i += 1
            cands = []
            while(i < lenLines and lines[i] != '\n'):
                if(i % 100000 == 0):
                    print('已读取%d行' %(i), end=' ')
                cand = json.loads(lines[i].strip())
                cands.append(cand)
                # import pdb; pdb.set_trace()
                i += 1
            i += 1
            que2cands[que] = cands
    return que2cands


def readTrainFile(fileName: str) -> List[List[str]]:
    que2answer = {}
    with open(fileName, 'r', encoding='utf-8') as fread:
        lines = fread.readlines()
        i = 0
        while(i < len(lines)):
            line = lines[i].strip()
            pos = line.find(':')
            que = line[pos + 1:].replace('\t','')
            answer = lines[i + 2].strip()
            # if('北京故宫博物院附近5公里' in que):
            #     import pdb; pdb.set_trace()
            que2answer[que] = answer
            que2answer[line[pos + 1:]] = answer
            i += 4
    return que2answer


def intersectionStr(str1, str2):
    '''之间的item有重叠，True；否则False'''
    for item in str1:
        if(item in str2):
            return True
    return False

# def preprocessRel(topkRelsList):
#     relDic = {}
#     rels = []
#     for rel in topkRelsList:
#         flag = True
#         for item in relDic:
#             if(intersectionStr(item[1:-1], rel[1:-1])): # 有交集
#                 flag = False
#                 if(relDic[item] < 5):
#                     # print(item, relDic[item])
#                     rels.append(rel)
#                     relDic[item] += 1
#         if(flag):
#             relDic[rel] = 1
#             rels.append(rel)
#     # print(rels)
#     # import pdb; pdb.set_trace()
#     return rels

def preprocessRel(topkRelsList, que):
    relDic = {}
    rels = []
    RelsList = []
    for rel in topkRelsList:
        if len(rel) <=3 or '\u200b' in rel:    # 过短关系舍弃
            continue
        elif rel[1:-1] in que:  # 完全匹配
            flag = True
            for item in relDic:
                if(intersectionStr(item[1:-1], rel[1:-1])): # 有交集
                    flag = False
                    if len(item)<len(rel):
                        index = rels.index(item)
                        rels[index] = rel
                        relDic[rel] = relDic[item]
                        relDic[rel].append(item)
                        relDic.pop(item)
                        break
            if(flag):
                relDic[rel] = []
                rels.append(rel)
        else:
            RelsList.append(rel)
            
    # import pdb;pdb.set_trace()
    for rel in RelsList:
        flag = True
        for item in relDic:
            if(intersectionStr(item[1:-1], rel[1:-1])): # 有交集
                flag = False
                if(len(relDic[item]) < 5):
                    # print(item, relDic[item])
                    # rels.append(rel)
                    relDic[item].append(rel)
                    break
        if(flag):
            relDic[rel] = [rel]
    
    for key in relDic:
        rels.extend(relDic[key][:5])
    # import pdb; pdb.set_trace()
    return rels


def readTrainFileForSparql(fileName: str) -> List[List[str]]:
    que2sparql = {}
    with open(fileName, 'r', encoding='utf-8') as fread:
        lines = fread.readlines()
        i = 0
        while(i < len(lines)):
            line = lines[i].strip()
            pos = line.find(':')
            que = line[pos + 1:]
            sparql = lines[i + 1].strip()
            que2sparql[que] = sparql
            i += 4
    return que2sparql

if __name__ == '__main__':
    import pdb
    que = '孙俪和蒋欣共同主演的在2011年11月17日上映的电视剧是？'
    que='在1968年被发明，用于代替键盘繁琐的指令的设备是？？'
    time_list = find_time_constrain(que) # 返回 时间约束的列表  # 找到对应的mention list
    # pdb.set_trace()
    if(len(time_list) > 0):
        for _time in time_list:
            timeStr = str(_time)
            pdb.set_trace()
            normalize_time_constrain(_time[0])  # 标准化，记录

    