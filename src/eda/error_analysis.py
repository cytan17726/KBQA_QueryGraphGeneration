import sys
import os
import json
import copy
import re
from typing import List, Dict, Tuple
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.model_train.common.utils import readSegment


# que2segmentFileName = BASE_DIR + '/data/sep_res_1206.json'
# que2segment = readSegment(que2segmentFileName)


# 读取预测的得分文件
def read_prediction_scores(file_name: str) -> List[float]:
    f = open(file_name, 'r', encoding='utf-8')
    lines = f.readlines()
    scores: List[float] = []
    for i, line in enumerate(lines):
        scores.append(float(line.strip()))
    return scores


def readScoreLog(file_name: str) -> List[List[float]]:
    f = open(file_name, 'r', encoding='utf-8')
    lines = f.readlines()
    scores: List[float] = []
    for i, line in enumerate(lines):
        scores.append(eval(line.strip()))
    return scores


# 读取候选文件
def read_cands(file_name: str, qidIndex = -1):
    f = open(file_name, 'r', encoding='utf-8')
    lines = f.readlines()
    qid2cands = {}# 记录每个问句对应的候选信息
    qid2pos = {}# 记录每个问句对应候选的开始和结束位置
    qid_list = [] # 记录问句ID
    before_qid = ''
    current_qid = ''
    for i, line in enumerate(lines):
        # import pdb;pdb.set_trace()
        line_cut = eval(line.strip())
        line_cut = [line_cut[0], eval(line_cut[1])]
        current_qid = line_cut[0]
        if(current_qid not in qid2cands):
            qid2cands[current_qid] = []
            qid2cands[current_qid].append(line_cut)
            qid2pos[current_qid] = []
            qid2pos[current_qid].append(i) # 记录开始的位置
            qid_list.append(current_qid)
            if(before_qid != ''):
                qid2pos[before_qid].append(i - 1) # 记录结束的位置
            before_qid = current_qid
        else:
            qid2cands[current_qid].append(line_cut)
            # import pdb; pdb.set_trace()
    if(len(qid2pos[before_qid]) == 1):
        qid2pos[before_qid].append(i)
        print('before_qid:', before_qid, qid2pos[before_qid])
    # import pdb; pdb.set_trace()
    return qid2cands, qid2pos, qid_list


# 针对每个问句的候选进行排序，并输出选择出错的问句信息
def sorted_and_print(qid2cands, qid2pos, qid_list, scores):
    fwrite = open('./classify_5583_error_que.txt', 'w', encoding='utf-8')
    error_num = 0
    for qid in qid_list:
        cands = qid2cands[qid]
        pos = qid2pos[qid]
        current_scores = scores[pos[0]: pos[1] + 1]
        sorted_scores = sorted(enumerate(current_scores), key=lambda x: x[1], reverse=True)
        if(cands[sorted_scores[0][0]][0] == '0'):
            error_num += 1
            for item in sorted_scores:
                fwrite.write(cands[item[0]] + '\n')
                # import pdb; pdb.set_trace()
            fwrite.write('\n')
        # import pdb; pdb.set_trace()
    fwrite.flush()
    print('回答错误的问句数量：', error_num)


# 计算每个问句的f1值得分,根据模型预测得分进行排序选取
def get_qid2f1(qid2cands, qid2pos, qid_list, scores, qid2maxcand):
    qid2f1 = {}
    qid2onecand = {}
    for qid in qid_list:
        cands = qid2cands[qid]
        pos = qid2pos[qid]
        current_scores = scores[pos[0]: pos[1] + 1]
        sorted_scores = sorted(enumerate(current_scores), key=lambda x: x[1], reverse=True)
        # import pdb; pdb.set_trace()
        # if qid == '维力医疗哪个高管年龄最大?':
        #     import pdb;pdb.set_trace()
        try:
            # import pdb; pdb.set_trace()
            qid2f1[qid] = float(cands[sorted_scores[0][0]][1]['f1'])
            # import pdb; pdb.set_trace()
            qid2onecand[qid] = [sorted_scores[0][1], cands[sorted_scores[0][0]]]
        except:
            # print(que)
            import pdb; pdb.set_trace()
        for i, cand in enumerate(cands):
            if(cand == qid2maxcand[qid]):
                try:
                    qid2maxcand[qid] = [current_scores[i], cand]
                except:
                    import pdb;pdb.set_trace()
                break
        # import pdb; pdb.set_trace()
    return qid2f1, qid2onecand


def get_qid2scoreLog(qid2cands, qid2pos, qid_list, scores, qid2maxcand, scoreLog):
    '''
    功能：获取每个候选查询图对应的得分细节
    '''
    qid2f1 = {}
    qid2onecand = {}
    qid2scorelog = {}
    for qid in qid_list:
        cands = qid2cands[qid]
        pos = qid2pos[qid]
        current_scores = scores[pos[0]: pos[1] + 1]
        sorted_scores = sorted(enumerate(current_scores), key=lambda x: x[1], reverse=True)
        # qid2f1[qid] = float(cands[sorted_scores[0][0]][1]['f1'])
        qid2scorelog[qid] = []
        qid2scorelog[qid].append(scoreLog[pos[0] + sorted_scores[0][0]])
        for i, cand in enumerate(cands):
            if(cand == qid2maxcand[qid][1]):
                # qid2maxcand[qid] = [current_scores[i], cand]
                qid2scorelog[qid].append(scoreLog[pos[0] + i])
                break
            # import pdb; pdb.set_trace()
        if(len(qid2scorelog[qid]) != 2):
            import pdb; pdb.set_trace()
    return qid2scorelog


# 计算每个问句对应的f1值上限得分,根据查询图f1值得分进行排序选取
def get_qid2maxf1(qid2candidates, qid2pos, qid_list):
    qid2cands = copy.deepcopy(qid2candidates)
    qid2f1 = {}
    qid2maxcand = {}
    for qid in qid_list:
        cands = qid2cands[qid]
        pos = qid2pos[qid]
        cands.sort(key=lambda x:float(x[1]['f1']), reverse=True)
        # import pdb; pdb.set_trace()
        try:
            # import pdb; pdb.set_trace()
            qid2f1[qid] = float(cands[0][1]['f1'])
            qid2maxcand[qid] = cands[0]
        except:
            import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
    return qid2f1, qid2maxcand

    
def load_compq():
    compq_path = BASE_DIR + '/qa-corpus/MulCQA'
    qa_list = []
    for Tvt in ('train', 'test'):
        fp = '%s/compQ.%s.release' % (compq_path, Tvt)
        f = open(fp, 'r', encoding='utf-8')
        lines = f.readlines()
        for line in lines:
            qa = {}
            q, a_list_str = line.strip().split('\t')
            qa['utterance'] = q
            qa['targetValue'] = json.loads(a_list_str)
            qa_list.append(qa)
    print('%d CompQuesetions loaded.', len(qa_list))
    return qa_list
    
def load_webq():
    '''
    功能：读取webq数据集
    '''
    webq_path = BASE_DIR + '/qa-corpus/web-question'
    qa_list = []
    for Tvt in ('train', 'test'):
        webq_fp = webq_path + '/data/webquestions.examples.' + Tvt + '.json'
        print(webq_fp)
        f = open(webq_fp, 'r', encoding='utf-8')
        webq_data = json.load(f)
        # import pdb; pdb.set_trace()
        for raw_info in webq_data:
            qa = {}
            target_value = []
            ans_line = raw_info['targetValue']
            ans_line = ans_line[7: -2]      # remove '(list (' and '))'
            for ans_item in ans_line.split(') ('):
                ans_item = ans_item[12:]    # remove 'description '
                if ans_item.startswith('"') and ans_item.endswith('"'):
                    ans_item = ans_item[1: -1]
                target_value.append(ans_item)
            qa['utterance'] = raw_info['utterance']
            qa['targetValue'] = target_value
            qa_list.append(qa)
            
    # import pdb; pdb.set_trace()
    # qa_list中每个元素格式:{'utterance': 'what is the name of justin bieber brother?', 'targetValue': ['Jazmyn Bieber', 'Jaxon Bieber']}
    print('%d WebQuesetions loaded.', len(qa_list))
    return qa_list


# 分析查询图构建成功，而排序模型出错的情况
def sort_error_instance(qid2maxf1, qid2maxcand, qid2f1, qid2onecand):
    qid2diff = {}
    for qid in qid2maxf1:
        # import pdb; pdb.set_trace()
        if(qid2maxf1[qid] > qid2f1[qid] and qid2maxf1[qid] > 0.1):
            qid2diff[qid] = (qid2maxcand[qid], qid2onecand[qid])
            # import pdb; pdb.set_trace()
    return qid2diff

def sort_true_instance(qid2maxf1, qid2maxcand, qid2f1, qid2onecand):
    qid2diff = {}
    for qid in qid2maxf1:
        # import pdb; pdb.set_trace()
        if(abs(qid2maxf1[qid] - qid2f1[qid]) < 0.0001 and qid2maxf1[qid] > 0.1):
            qid2diff[qid] = (qid2maxcand[qid], qid2onecand[qid])
            # import pdb; pdb.set_trace()
    return qid2diff


def true_false(qid2f1_true, qid2onecand_true, qid2f1_false, qid2onecand_false, f):
    for qid in qid2f1_true:
        if(qid2f1_true[qid] > qid2f1_false[qid] and qid2f1_true[qid] > 0.1 and qid2f1_false[qid] < 0.1):
            f.write(qid2onecand_true[qid] + '\n')
            f.write(qid2onecand_false[qid] + '\n')
            # import pdb; pdb.set_trace()
            # print(qid2onecand_true[qid])
            # print(qid2onecand_false[qid])

def false_true(qid2f1_true, qid2onecand_true, qid2f1_false, qid2onecand_false, f):
    for qid in qid2f1_true:
        if(qid2f1_true[qid] < qid2f1_false[qid] and qid2f1_false[qid] > 0.1 and qid2f1_true[qid] < 0.1):
            f.write(qid2onecand_true[qid] + '\n')
            f.write(qid2onecand_false[qid] + '\n')


def webq_compare():
    file_name = './prediction_test_webq_top20.txt'
    file_name_pointwise = './prediction_test_webq_all.txt'
    scores = read_prediction_scores(file_name)
    scores_point = read_prediction_scores(file_name_pointwise)
    file_name = BASE_DIR + '/runnings/train_data/webq/webq_t_top20_from5244.txt'
    file_nameAll = BASE_DIR + '/runnings/train_data/webq/webq_with_answer_info_test_all.txt'
    qid2cands, qid2pos, qid_list = read_cands(file_name, -2)
    qid2f1, qid2onecand = get_qid2f1(qid2cands, qid2pos, qid_list, scores, -3)
    qid2candsAll, qid2posAll, qid_listAll = read_cands(file_nameAll, -1)
    qid2f1_point, qid2onecand_point = get_qid2f1(qid2candsAll, qid2posAll, qid_listAll, scores_point, -2)
    qid2maxf1, qid2maxcand = get_qid2maxf1(qid2cands, qid2pos, qid_list)
    f = open('webq_rerank_true_rank_false.txt', 'w', encoding='utf-8')
    true_false(qid2f1, qid2onecand, qid2f1_point, qid2onecand_point, f)
    f = open('webq_listwise_false_pointwise_true.txt', 'w', encoding='utf-8')
    false_true(qid2f1, qid2onecand, qid2f1_point, qid2onecand_point, f)
    import pdb; pdb.set_trace()
    # 获取qid与问句之间的对应关系
    # qa_list = load_compq()
    qa_list = load_webq()
    que2qid = {}
    for i, qa in enumerate(qa_list):
        que2qid[qa['utterance']] = i
    # f = open('/data/yhjia/cytan/train2.0.txt', 'r', encoding='utf-8')
    # f = open('/data2/yhjia/cytan/complex_test.txt', 'r', encoding='utf-8')
    f = open('/data2/yhjia/cytan/web_test.txt', 'r', encoding='utf-8')
    lines = f.readlines()
    i = 0
    sum_f1 = 0.0
    num = 0.0
    # for que in que_dic:
    for que in que2qid:
        qid = que2qid[que]
        # import pdb; pdb.set_trace()
        if(str(qid).zfill(4) in qid2f1):
            sum_f1 += qid2f1[str(qid).zfill(4)]
            num += 1
    print(sum_f1, num, sum_f1 / num)
    qid2diff = sort_error_instance(qid2maxf1, qid2maxcand, qid2f1, qid2onecand)
    fout = open('./查询图.txt', 'w', encoding='utf-8')
    for qid in qid2diff:
        fout.write(qid2diff[qid][0] + '\n')
        fout.write(qid2diff[qid][1] + '\n')
    fout.flush()
    import pdb; pdb.set_trace()


def compq_compare():
    file_name = './prediction_test_compq_top40.txt'
    file_name_pointwise = './prediction_test_compq_all.txt'
    scores = read_prediction_scores(file_name)
    scores_point = read_prediction_scores(file_name_pointwise)
    file_name = BASE_DIR + '/runnings/train_data/compq/compq_t_top40.txt'
    fileNameRank = BASE_DIR + '/runnings/train_data/compq/compq_rank1_f01_gradual_label_position_listwise_1_140_type_entity_time_ordinal_mainpath_test_all.txt'
    qid2cands, qid2pos, qid_list = read_cands(file_name, -2)
    qid2candsAll, qid2posAll, qid_listAll = read_cands(fileNameRank, -1)
    qid2f1, qid2onecand = get_qid2f1(qid2cands, qid2pos, qid_list, scores, -3)
    qid2f1_point, qid2onecand_point = get_qid2f1(qid2candsAll, qid2posAll, qid_listAll, scores_point, -2)
    qid2maxf1, qid2maxcand = get_qid2maxf1(qid2cands, qid2pos, qid_list)
    f = open('compq_rerank_true_rank_false.txt', 'w', encoding='utf-8')
    true_false(qid2f1, qid2onecand, qid2f1_point, qid2onecand_point, f)
    f = open('compq_listwise_false_pointwise_true.txt', 'w', encoding='utf-8')
    false_true(qid2f1, qid2onecand, qid2f1_point, qid2onecand_point, f)
    import pdb; pdb.set_trace()
    # 获取qid与问句之间的对应关系
    # qa_list = load_compq()
    qa_list = load_webq()
    que2qid = {}
    for i, qa in enumerate(qa_list):
        que2qid[qa['utterance']] = i
    # f = open('/data/yhjia/cytan/train2.0.txt', 'r', encoding='utf-8')
    # f = open('/data2/yhjia/cytan/complex_test.txt', 'r', encoding='utf-8')
    f = open('/data2/yhjia/cytan/web_test.txt', 'r', encoding='utf-8')
    lines = f.readlines()
    i = 0
    sum_f1 = 0.0
    num = 0.0
    # for que in que_dic:
    for que in que2qid:
        qid = que2qid[que]
        # import pdb; pdb.set_trace()
        if(str(qid).zfill(4) in qid2f1):
            sum_f1 += qid2f1[str(qid).zfill(4)]
            num += 1
    print(sum_f1, num, sum_f1 / num)
    qid2diff = sort_error_instance(qid2maxf1, qid2maxcand, qid2f1, qid2onecand)
    fout = open('./查询图.txt', 'w', encoding='utf-8')
    for qid in qid2diff:
        fout.write(qid2diff[qid][0] + '\n')
        fout.write(qid2diff[qid][1] + '\n')
    fout.flush()
    import pdb; pdb.set_trace()

def get_avg_f1(qid2f1):
    sumF1 = 0.0
    for que in qid2f1:
        sumF1 += qid2f1[que]
    print('avg f1:', sumF1 / len(qid2f1))
    
def get_qid2HopNum(qid2cands):
    qid2hopNum = {}
    pattern = re.compile(r'\<[ABCDEFGHIGK]\>')
    for que in qid2cands:
        qid2hopNum[que] = []
        for cand in qid2cands[que]:
            mainPath = cand[1]['mainPath']
            result = pattern.findall(mainPath)
            # print(mainPath, result)
            qid2hopNum[que].append(len(result) // 2 + 1)
        # import pdb; pdb.set_trace()
    return qid2hopNum

def scaleScores(qid2hopNum, scores, N):
    newScores = []
    i = 0
    for que in qid2hopNum:
        for hopNum in qid2hopNum[que]:
            if(hopNum == 1):
                scaleValue = 1.0
            else:
                scaleValue = 1.0 + 1.0 / N * (hopNum - 1)
            newScores.append(scores[i] / (scaleValue))
            i += 1
    return newScores

def selectionError():
    # file_name = '/data2/yhjia/kbqa/ckbqa/src/model_train/models/5641_listwise_bert_train65000examples_devneg50_seq150_CE_group_5_2_42_100/prediction'
    # file_name = '/data2/yhjia/kbqa/ckbqa/src/model_train/models/6481_listwise_bert_entitymainpath_trainexamples_devneg50_seq150_CE_group_5_1_42_128/prediction'
    # file_name = '/data2/yhjia/kbqa/ckbqa/src/model_train/models/6911_listwise_bert_entitymain_trainexamples_devneg50_seq150_CE_group_20_1_42_100/prediction'
    # file_name = '/home/chenwenliang/jiayonghui/ckbqa/prediction_max_sum_test128'
    # file_name = '/home/chenwenliang/jiayonghui/ckbqa/prediction_max_sum_test_nosoftmax128'
    # file_name = '/home/chenwenliang/jiayonghui/ckbqa/src/model_train/models/6860_8386_listwise_bert_notsoftmax_max_sum_word_sim_align_devneg50_seq150_CE_group_20_1_42_100/prediction_max_sum_test_nosoftmax128'
    # file_name = '/home/chenwenliang/jiayonghui/ckbqa/src/model_train/models/6939_8386_listwise_quesegment_base3799_nosofmax2dim_nochengfa_sim_align_devneg50_seq150_CE_group_10_1_42_100/prediction_valid'
    file_name = '/home/chenwenliang/jiayonghui/ckbqa/src/model_train/models/6766_8386_listwise_quesegment_base3799_nosofmax2dim_basedPath_sim_align_devneg50_seq150_CE_group_10_1_42_100/prediction_valid'
    scores = read_prediction_scores(file_name)
    # file_name = BASE_DIR + '/data/train_data/testset_model_data.txt'
    # file_name = BASE_DIR + '/data/train_data/3_1101/testset_model_data.txt'
    # file_name = BASE_DIR + '/data/train_data/y_6_4_1102/testset_model_data.txt'
    file_name = BASE_DIR + '/data/train_data/y_6_4_1102/dev_20%_neg50.txt'
    qid2cands, qid2pos, qid_list = read_cands(file_name, -2)
    qid2maxf1, qid2maxcand = get_qid2maxf1(qid2cands, qid2pos, qid_list)
    qid2f1, qid2onecand = get_qid2f1(qid2cands, qid2pos, qid_list, scores, qid2maxcand)
    # qid2hopNum = get_qid2HopNum(qid2cands)
    # for n in range(100, 200):
    #     newScores = scaleScores(qid2hopNum, scores, n)
    #     qid2f1, qid2onecand = get_qid2f1(qid2cands, qid2pos, qid_list, newScores, qid2maxcand)
    #     print(n, end='\t')
    #     get_avg_f1(qid2f1)
    # import pdb; pdb.set_trace()
    qid2diff = sort_error_instance(qid2maxf1, qid2maxcand, qid2f1, qid2onecand)
    fout = open('./8972_dev_selection_error.txt', 'w', encoding='utf-8')
    for qid in qid2diff:
        if(type(qid2diff[qid][0][1][1]) is not str):
            qid2diff[qid][0][1][1] = json.dumps(qid2diff[qid][0][1][1], ensure_ascii=False)
        if(type(qid2diff[qid][1][1][1]) is not str):
            qid2diff[qid][1][1][1] = json.dumps(qid2diff[qid][1][1][1], ensure_ascii=False)
        fout.write(str(qid2diff[qid][0]))
        fout.write('\n')
        fout.write(str(qid2diff[qid][1]) + '\n')
    fout.flush()
    # import pdb; pdb.set_trace()
    qid2true = sort_true_instance(qid2maxf1, qid2maxcand, qid2f1, qid2onecand)
    fout = open('./8972_dev_selection_true.txt', 'w', encoding='utf-8')
    for qid in qid2true:
        # import pdb; pdb.set_trace()
        if(type(qid2true[qid][0][1][1]) is not str):
            qid2true[qid][0][1][1] = json.dumps(qid2true[qid][0][1][1], ensure_ascii=False)
        if(type(qid2true[qid][1][1][1]) is not str):
            qid2true[qid][1][1][1] = json.dumps(qid2true[qid][1][1][1], ensure_ascii=False)
        fout.write(str(qid2true[qid][0]))
        fout.write('\n')
        fout.write(str(qid2true[qid][1]) + '\n')
    fout.flush()
    
def selectionErrorTest():
    '''todo '''
    # file_name = '/data2/yhjia/kbqa/ckbqa/src/model_train/models/5641_listwise_bert_train65000examples_devneg50_seq150_CE_group_5_2_42_100/prediction'
    # file_name = '/data2/yhjia/kbqa/ckbqa/src/model_train/models/6481_listwise_bert_entitymainpath_trainexamples_devneg50_seq150_CE_group_5_1_42_128/prediction'
    # file_name = '/data2/yhjia/kbqa/ckbqa/src/model_train/models/6911_listwise_bert_entitymain_trainexamples_devneg50_seq150_CE_group_20_1_42_100/prediction'
    # file_name = '/home/chenwenliang/jiayonghui/ckbqa/prediction_max_sum_test128'
    # file_name = '/home/chenwenliang/jiayonghui/ckbqa/prediction_max_sum_test_nosoftmax128'
    # file_name = '/home/chenwenliang/jiayonghui/ckbqa/src/model_train/models/6860_8386_listwise_bert_notsoftmax_max_sum_word_sim_align_devneg50_seq150_CE_group_20_1_42_100/prediction_max_sum_test_nosoftmax128'
    # file_name = '/home/chenwenliang/jiayonghui/ckbqa/src/model_train/models/6989_8386_listwise_quesegment_base3799_nosofmax2dim_nochengfa_sim_align_devneg50_seq150_CE_group_20_1_42_100/prediction_test128'
    dir_name = '/data4/cytan/re_ckbqa/ckbqa/src/model_train/models/0330ccks2019_comp/20_1_42_100'
    # file_name = dir_name + '/prediction_var_实体识别结果_加长串_baike_0411'    # 得分文件
    file_name = dir_name + '/prediction_var_noentity_2019_xgboost'
    '''neg 30'''
    dir_name = '/data4/cytan/re_ckbqa/ckbqa/src/model_train/models/0413_ccks2019_only_Ours/30_1_42_100'
    file_name = dir_name + '/0422_prediction_var_CCKS2019Only_实体识别结果_加长串_baike_19Order_0624reberta_top3_afterProcess_4hop_Top300'
    ''''''
    # dir_name = '/data4/cytan/re_ckbqa/ckbqa/src/model_train/models/0424_ccks2019_only_Ours_reuse3/10_1_42_100'
    # file_name = dir_name + '/0422_prediction_var_CCKS2019Only_pjzhang_baike_19Order_0624reberta_top3_afterProcess_4hop_Top300'
    scores = read_prediction_scores(file_name)
    # file_name = dir_name + '/prediction_test128\log'  # XXX 先注释了
    # scoreLog = readScoreLog(file_name)
    # file_name = BASE_DIR + '/data/train_data/multi_types_0105/testset_model_data.txt'
    # file_name = BASE_DIR + '/data/train_data/new3_0106_sparql_drop/testset_model_data.txt'  # 训练数据-test
    # file_name = BASE_DIR + '/data/train_data/0413_ccks2019_only_Ours/testset_model_data_实体识别结果_加长串_baike_0411.txt'
    file_name = BASE_DIR + '/data/train_data/0416_ccks2019Only_more_try/testset_model_data_CCKS2019Only_实体识别结果_加长串_baike_19Order_0624reberta_top3_afterProcess_4hop_Top300_V0422.txt'
    qid2cands, qid2pos, qid_list = read_cands(file_name, -2)
    qid2maxf1, qid2maxcand = get_qid2maxf1(qid2cands, qid2pos, qid_list)
    qid2f1, qid2onecand = get_qid2f1(qid2cands, qid2pos, qid_list, scores, qid2maxcand)
    # qid2scorelog = get_qid2scoreLog(qid2cands, qid2pos, qid_list, scores, qid2maxcand, scoreLog)
    # qid2hopNum = get_qid2HopNum(qid2cands)
    # for n in range(100, 200):
    #     newScores = scaleScores(qid2hopNum, scores, n)
    #     qid2f1, qid2onecand = get_qid2f1(qid2cands, qid2pos, qid_list, newScores, qid2maxcand)
    #     print(n, end='\t')
    #     get_avg_f1(qid2f1)
    # import pdb; pdb.set_trace()
    qid2diff = sort_error_instance(qid2maxf1, qid2maxcand, qid2f1, qid2onecand)
    fout = open('./0422_pjzhang_DSR300_ELTop3_alter_error.txt', 'w', encoding='utf-8')
    for qid in qid2diff:
        if(type(qid2diff[qid][0][1][1]) is not str):
            qid2diff[qid][0][1][1] = json.dumps(qid2diff[qid][0][1][1], ensure_ascii=False)
        if(type(qid2diff[qid][1][1][1]) is not str):
            qid2diff[qid][1][1][1] = json.dumps(qid2diff[qid][1][1][1], ensure_ascii=False)
        fout.write(str(qid2diff[qid][0]))
        fout.write('\n')
        # fout.write(str(qid2scorelog[qid][1]) + '\n')
        fout.write(str(qid2diff[qid][1]) + '\n')
        # fout.write(str(qid2scorelog[qid][0]) + '\n')
    fout.flush()
    # import pdb; pdb.set_trace()
    qid2true = sort_true_instance(qid2maxf1, qid2maxcand, qid2f1, qid2onecand)
    fout = open('./0422_pjzhang_DSR300_ELTop3_alter_true.txt', 'w', encoding='utf-8')
    for qid in qid2true:
        # import pdb; pdb.set_trace()
        if(type(qid2true[qid][0][1][1]) is not str):
            qid2true[qid][0][1][1] = json.dumps(qid2true[qid][0][1][1], ensure_ascii=False)
        if(type(qid2true[qid][1][1][1]) is not str):
            qid2true[qid][1][1][1] = json.dumps(qid2true[qid][1][1][1], ensure_ascii=False)
        fout.write(str(qid2true[qid][0]))
        fout.write('\n')
        # fout.write(str(qid2scorelog[qid][1]) + '\n')
        fout.write(str(qid2true[qid][1]) + '\n')
        # fout.write(str(qid2scorelog[qid][0]) + '\n')
    fout.flush()

def readSelectionFile(fileName):
    que2info = {}
    with open(fileName, 'r', encoding='utf-8') as fread:
        lines = fread.readlines()
        for line in lines:
            que = eval(line.strip())[1][0]
            if(que not in que2info):
                que2info[que] = []
            que2info[que].append(line)
            # import pdb; pdb.set_trace()
    return que2info

def compare():
    fileName2 = './8972_dev_selection_error.txt'    # base
    fileName1 = './8939_dev_selection_error.txt'
    fileName1 = './6989_test_selection_error.txt'    # base
    fileName2 = './7086_test_selection_error.txt'
    que2info1 = readSelectionFile(fileName1)
    que2info2 = readSelectionFile(fileName2)
    numSame = 0
    numDiff = 0
    for que in que2info2:
        if(que in que2info1):
            numSame += 1
            print(''.join(que2info2[que]))
        else:
            numDiff += 1
            # print(''.join(que2info2[que]))
    print('同时出错：', numSame, '不同的', numDiff, len(que2info2))
    
if __name__ == "__main__":
    
    # compare()

    # selectionError()
    selectionErrorTest()

    # webq_compare()
    # compq_compare()