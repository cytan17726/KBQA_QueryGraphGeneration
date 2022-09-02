import enum
import sys
import os
import json
from typing import List, Tuple, Dict
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)


def processAnswerEntity(answers: str):
    answers = answers.split('\t')
    newAnswers = []
    for answer in answers:
        if '^^<http://www.w3.org' in answer:
            p = answer.index('^^<http://www.w3.org')
            answer = answer[:p]
        # if(len(answer) <= 0):
        #     print(answers)
        #     import pdb; pdb.set_trace()
        if(len(answer) > 0 and answer[0] != '<' and answer[0] != '\"'):
            newAnswers.append('<' + answer + '>')
        else:
            newAnswers.append(answer)
    return newAnswers

def processAnswerValue(answers: str):
    answers = answers.split('\t')
    newAnswers = []
    for answer in answers:
        if '^^<http://www.w3.org' in answer:
            p = answer.index('^^<http://www.w3.org')
            answer = answer[:p]
        if(len(answer) > 0 and answer[0] != '<' and answer[0] != '\"'):
            newAnswers.append('\"' + answer + '\"')
        else:
            newAnswers.append(answer)
    return newAnswers


def f1Item(label: List[str], pred: List[str]):
    if len(label)<=0: # 有问题
        # import pdb; pdb.set_trace()
        return 0.0
    if len(pred)<=0:
        return 0.0
    # label = processAnswerEntity(label)
    pred = set(pred)
    label = set(label)
    common_set = pred.intersection(label)
    p = len(common_set) / len(pred)
    r = len(common_set) / len(label)
    if p<=0 or r<=0:
        f1 = 0.0
    else:
        f1 = 2 * p * r / (p + r)
    return f1

def calMultiClassification(file_name1, file_name2, data_type, label_list):
    label_map = {label : i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}
    f = open(file_name1, 'r', encoding='utf-8')
    lines_predict = f.readlines()
    f2 = open(file_name2, 'r', encoding='utf-8')
    lines = f2.readlines()
    # import pdb; pdb.set_trace()
    # assert len(lines) == len(lines_predict)
    trueList = [0] * len(label_list)
    predictList = [0] * len(label_list)
    goldList = [0] * len(label_list)
    i = 0
    while(i < len(lines_predict)):
        scoresItem = eval(lines_predict[i].strip())
        maxIndex = scoresItem.index(max(scoresItem))
        label = lines[i].strip().split('\t')[-1]
        if(label not in label_map):
            i += 1
            continue
        labelID = label_map[label]
        predictList[maxIndex] += 1
        goldList[labelID] += 1
        if(maxIndex == labelID):
            trueList[maxIndex] += 1
        # else:
        #     print(lines[i], id2label[maxIndex])
        i += 1
    recall = []
    for i, item in enumerate(trueList):
        recall.append(item * 1.0 / goldList[i])
    return sum(trueList) * 1.0 / sum(goldList), recall
        

# 计算预测文件与gold文件的得分
def cal_f1(file_name1, file_name2, data_type, actual_num = 0, log=0):
    f = open(file_name1, 'r', encoding='utf-8')
    lines_predict = f.readlines()
    f2 = open(file_name2, 'r', encoding='utf-8')
    lines = f2.readlines()
    assert len(lines) == len(lines_predict)
    begin = -1
    end = -1
    qid = ''
    i = 0
    sum_f1 = 0.0
    have_p_r = 0
    p = 0.0
    r = 0.0
    que_num = 0
    while i < len(lines):
        have_p_r = 1
        line = lines[i]
        # import pdb; pdb.set_trace()
        lineCut = eval(line.strip())
        qid_temp = lineCut[0]
        if(qid_temp != qid):
            qid = qid_temp
            if(begin == -1):
                begin = i
            else:
                end = i
        if(end != -1):
            scores = lines_predict[begin:end]
            max_score = -100000000
            num = 0
            for j, score in enumerate(scores):
                if(float(score.strip()) >= max_score):
                    max_score = float(score.strip())
                    num = j
            # num = len(scores) - 1
            # que = eval(lines[begin + num].strip())[0]
            f1 = float(eval(eval(lines[begin + num].strip())[-1])['f1'])
            # import pdb; pdb.set_trace()
            if(have_p_r):
                p += float(eval(eval(lines[begin + num].strip())[-1])['f1'])
                r += float(eval(eval(lines[begin + num].strip())[-1])['f1'])
            if(log == 1):
                print(p, r, f1)
            # if(f1 > 0):
            #     print(begin+num, f1)
            sum_f1 += f1
            que_num += 1
            # print(begin, end, num, f1)
            # import pdb; pdb.set_trace()
            begin = -1
            end = -1
            qid = ''
            i -= 1
            # import pdb; pdb.set_trace()
        i += 1
    if(begin != -1):
        # import pdb; pdb.set_trace()
        scores = lines_predict[begin:]
        max_score = -100000000
        num = 0
        for j, score in enumerate(scores):
            if(float(score.strip()) >= max_score):
                max_score = float(score.strip())
                num = j
        # num = len(scores) - 1
        f1 = float(eval(eval(lines[begin + num].strip())[-1])['f1'])
        if(have_p_r):
            p += float(eval(eval(lines[begin + num].strip())[-1])['f1'])
            r += float(eval(eval(lines[begin + num].strip())[-1])['f1'])
        if(log == 1):
            print(p, r, f1)
        sum_f1 += f1
        que_num += 1
    print('数据个数：', len(lines))
    print('问句个数：', que_num)
    if(data_type == 't'):
        print(p / que_num, r / que_num, sum_f1 / que_num)
        return p / que_num, r / que_num, sum_f1 / que_num
    else:
        print(p / que_num, r / que_num, sum_f1 / que_num)
        return p / que_num, r / que_num, sum_f1 / que_num
    


# 计算预测文件与gold文件的得分
def cal_f1_with_scores(file_name1, file_name2, data_type):
    f = open(file_name1, 'r', encoding='utf-8')
    lines_predict = f.readlines()
    f2 = open(file_name2, 'r', encoding='utf-8')
    lines = f2.readlines()
    print(len(lines), len(lines_predict))
    assert len(lines) == len(lines_predict)
    begin = -1
    end = -1
    qid = ''
    i = 0
    have_p_r = 0
    sum_f1 = 0.0
    p = 0.0
    r = 0.0
    que_num = 0
    while i < len(lines):
        line_cut = lines[i].strip().split('\t')
        if(len(line_cut) == 11):
            have_p_r = 1
        line = lines[i]
        qid_temp = line.strip().split('\t')[-2]
        if(qid_temp != qid):
            qid = qid_temp
            if(begin == -1):
                begin = i
            else:
                end = i
        if(end != -1):
            scores = lines_predict[begin:end]
            max_score = -100000000
            num = 0
            for j, score in enumerate(scores):
                if(float(score.strip()) >= max_score):
                    max_score = float(score.strip())
                    num = j
            # num = len(scores) - 1
            f1 = float(lines[begin + num].split('\t')[-3])
            if(have_p_r):
                p += float(lines[begin + num].split('\t')[-5])
                r += float(lines[begin + num].split('\t')[-4])
            # if(f1 > 0):
            #     print(begin+num, f1)
            sum_f1 += f1
            que_num += 1
            # print(begin, end, num, f1)
            # import pdb; pdb.set_trace()
            begin = -1
            end = -1
            qid = ''
            i -= 1
            # import pdb; pdb.set_trace()
        i += 1
    if(begin != -1):
        # import pdb; pdb.set_trace()
        scores = lines_predict[begin:]
        max_score = -100000000
        num = 0
        for j, score in enumerate(scores):
            if(float(score.strip()) >= max_score):
                max_score = float(score.strip())
                num = j
        # num = len(scores) - 1
        f1 = float(lines[begin + num].split('\t')[-3])
        if(have_p_r):
            p += float(lines[begin + num].split('\t')[-5])
            r += float(lines[begin + num].split('\t')[-4])
        sum_f1 += f1
        que_num += 1
    print('数据个数：', len(lines))
    print('问句个数：', que_num)
    if(data_type == 't'):
        print(sum_f1, p / 2032.0, r / 2032.0, sum_f1 / 2032.0)
        return sum_f1 / 2032.0
    else:
        print(sum_f1, p / 755.0, r / 755.0, sum_f1 / 755.0)
        return sum_f1 / 755.0


if __name__ == "__main__":
    # file_name1 = BASE_DIR + '/src/model_train/models/5789_8386_listwise_bert_quesame_max_layernorm_avg2_labelvar_entitymain_#answer#_trainexamples_devneg50_seq150_CE_group_5_1_42_100/prediction_dev'
    # file_name2 = BASE_DIR + '/data/train_data/y_6_4_1102/dev_20%_neg50.txt'
    file_name1 = BASE_DIR + '/prediction_max_sum_test_softmax1dim128'
    file_name1 = BASE_DIR + '/prediction_max_sum_test_nosoftmax128'
    file_name1 = '/home/chenwenliang/jiayonghui/ckbqa/src/model_train/models/6850_8386_listwise_quesegment_nosofmax2dim_alignscorea_sim_align_devneg50_seq150_CE_group_15_1_42_100/prediction_test128'
    file_name1 = '/home/chenwenliang/jiayonghui/ckbqa/src/model_train/models/6850_8386_listwise_quesegment_nosofmax2dim_alignscorea_sim_align_devneg50_seq150_CE_group_15_1_42_100/prediction_test128_process_maxnorm'
    file_name1 = '/home/chenwenliang/jiayonghui/ckbqa/src/model_train/models/6850_8386_listwise_quesegment_nosofmax2dim_alignscorea_sim_align_devneg50_seq150_CE_group_15_1_42_100/prediction_test128_process_minnorm'
    file_name1 = '/home/chenwenliang/jiayonghui/ckbqa/src/model_train/models/6850_8386_listwise_quesegment_nosofmax2dim_alignscorea_sim_align_devneg50_seq150_CE_group_15_1_42_100/prediction_test128_process_avgnorm'
    file_name1 = '/home/chenwenliang/jiayonghui/ckbqa/src/model_train/models/6850_8386_listwise_quesegment_nosofmax2dim_alignscorea_sim_align_devneg50_seq150_CE_group_15_1_42_100/prediction_test128_process_lastnorm'

    file_name2 = BASE_DIR + '/data/train_data/y_6_4_1102/testset_model_data.txt'
    cal_f1(file_name1, file_name2, 't', actual_num=0)