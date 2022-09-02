from argparse import ArgumentParser
import yaml
import sys
import os
from tqdm import tqdm
from typing import List, Dict, Tuple
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.utils.data_processing import  getQuestionsWithComplex, readQuestionType
from src.eda.error_analysis import read_prediction_scores, read_cands, get_qid2maxf1, get_qid2f1


def main(question_file='', score_file='', candidate_file='', dataset_flag=''):
    #BASE_DIR + '/src/question_classification/build_data/0413_ccks2019_only/que_newType_dev.txt'
    #BASE_DIR + '/src/question_classification/build_data/CCKS2019_CompType/que_newType_test.txt'
    '''之前输入的问句 训练时的评分'''
    questions = getQuestionsWithComplex(question_file)
    que2type = readQuestionType(question_file)
    
    scores = read_prediction_scores(score_file)
    
    qid2cands, qid2pos, qid_list = read_cands(candidate_file, -2)
    qid2maxf1, qid2maxcand = get_qid2maxf1(qid2cands, qid2pos, qid_list)
    qid2f1, qid2onecand = get_qid2f1(qid2cands, qid2pos, qid_list, scores, qid2maxcand)
    quetype2true = {}
    quetype2all = {}
    for que in questions:
        # if que =='维力医疗哪个高管年龄最大?':
        #     import pdb;pdb.set_trace()
        queType = que2type[que]
        if(queType not in quetype2true):
            quetype2true[queType] = 0
            quetype2all[queType] = 0
        if(que in qid2f1):
            if qid2f1[que]==0:
                # print(que)
                pass
            quetype2true[queType] += qid2f1[que]
        quetype2all[queType] += 1
    F1 = 0.0
    count = 0
    for queType in quetype2true:
        F1+=quetype2true[queType] * 1.0
        
        count+=quetype2all[queType]
        print(queType, quetype2true[queType] * 1.0 / quetype2all[queType])
    print('all: %.4f'%(F1/count))
    if dataset_flag == 'Comp':
        print('固定数量: %.4f'%(F1/972))
    else:
        print('固定数量: %.4f'%(F1/766))
        # print('固定数量: %.4f'%(F1/len(questions)))
    print(quetype2all)

if __name__ == "__main__":
    parser = ArgumentParser(description = 'For KBQA')
    parser.add_argument("--config_file",default='',type=str)
    args = parser.parse_args()
    config = yaml.safe_load(open(BASE_DIR+args.config_file))

    question_file = BASE_DIR + config['question_file']
    score_file = BASE_DIR + config['score_file']
    candidate_file = BASE_DIR + config['candidate_file']
    dataset_flag = config['dataset_flag']
    
    main(question_file,score_file,candidate_file,dataset_flag)
