from typing import List
from pytorch_pretrained_bert import BertModel, BertForSequenceClassification, BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
import numpy as np
import random
import torch
from torch import nn,Tensor, tensor
from torch.nn.functional import cross_entropy
from torch.nn.modules import module
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
from tqdm import tqdm
from argparse import ArgumentParser
import json
import os
import sys
import pickle
import pdb
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# print(BASE_DIR)
sys.path.append(BASE_DIR)

class InputExample(object):
    """A single training/test example for simple sequence classification."""
    # 问句分类不需要test_b
    def __init__(self, que, rel, label):
        self.que = que
        self.rel = rel
        self.label = label
# A single set of features of data
class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, que_ids, rel_ids, label):
        self.que_ids = que_ids
        self.rel_ids = rel_ids
        self.label = label
class InputFeatures_Cross(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

        
# 加载训练数据
def load_data(pathi)-> List[InputExample]:
    # train_data = TensorDataset()
    examples = []
    with open(pathi,'r') as f:
        for line in f:
            if not line.strip():
                continue
            que, rel, label = line.strip().split('\t')
            # import pdb;pdb.set_trace();
            examples.append(InputExample(que=que, rel=rel, label=label))
            # if len(examples)==64:
            #     break
    return examples

def load_evalue_data(pathi) -> List[List]:
    '''
    json {que:[rel1, rel2, ..]}
    '''
    eval_que_examples = []
    eval_rels_examples = []
    data = json.load(open(pathi,'r'))
    eval_que_examples = data.keys()
    eval_rels_examples = [data[i] for i in eval_que_examples]
    return [eval_que_examples, eval_rels_examples]

def convert_eval_examples_to_features(examples: List, max_seq_length, tokenizer)->List:
    """Loads a data file into a list of `InputBatch`s.
    和上面没啥区别 不过是只处理rel
    """
    features = []
    for (ex_index, example) in enumerate(examples): # 遍历各行输入
        '''缺少对于超长问句的处理？'''
        example = tokenizer.tokenize(example)
        example_tokens = ["[CLS]"] + example + ["[SEP]"]

        example_ids = tokenizer.convert_tokens_to_ids(example_tokens)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(example_ids))
        example_ids += padding

        assert len(example_ids) == max_seq_length
        features.append(example_ids)
    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

class SimOfQueRel:
    def __init__(self, topk = 200):
        # # 加载bert的分词器    
        # XXX CCKS2019Only的结果
        '''0427DNS'''
        # self.vocab_path = BASE_DIR + '/data/models/que_rel_sim/0427ccks2019Only_5e-6_DNS_setup_Top300-0.9569_6/'
        # self.bert_path = BASE_DIR + '/data/models/que_rel_sim/0427ccks2019Only_5e-6_DNS_setup_Top300-0.9569_6/'
        # self.relEmbsFile = BASE_DIR + '/data/models/que_rel_sim/0427ccks2019Only_5e-6_DNS_setup_Top300-0.9569_6/rel_embed.pkl'
        '''0429DNS'''
        self.vocab_path = BASE_DIR + '/model/que_rel_sim/0429DNS_setup_Fulldata5e-6_0.9565_6/'
        self.bert_path = BASE_DIR + '/model/que_rel_sim/0429DNS_setup_Fulldata5e-6_0.9565_6/'
        self.relEmbsFile = BASE_DIR + '/model/que_rel_sim/0429DNS_setup_Fulldata5e-6_0.9565_6/rel_embed.pkl'
        # # '/data4/cytan/7_dense_retriver/models/new_ex_ccks_0319/train32_1-1_2e-5_Top100_DNS_setup/0.9277443776441772_7'
        # self.vocab_path = BASE_DIR + '/model/que_rel_sim/allhard/'
        # self.bert_path = BASE_DIR + '/model/que_rel_sim/allhard/'
        # self.relEmbsFile = BASE_DIR + '/model/que_rel_sim/allhard/rel_embed.pkl'
        self.relsFile = BASE_DIR + '/data/que_rel_sim/rel_dict.json'
        # self.que2relFile = BASE_DIR + '/build_DSR_data/que2rel_ccks2019.json'
        self.rels = self.readRels()
        self.K = topk
        with open(self.relEmbsFile, 'rb') as fread:
            self.relEmbs = pickle.load(fread)
        # import pdb; pdb.set_trace()
        self.device = torch.device("cuda", 0)
        self.max_seq_length = 65
        self.tokenizer = BertTokenizer.from_pretrained(self.vocab_path)
        # self.model = BertModel.from_pretrained(self.bert_path).to(self.device)
        self.model = BertModel.from_pretrained(self.bert_path)
        self.model.eval()
        print('init DSR with model: %s'%self.bert_path)
        # self.que2rel = json.load(open(self.que2relFile,'r'))
    
    def readRels(self):
        with open(self.relsFile, 'r') as fread:
            relsDict = json.load(fread)
            rels = [rel for rel in relsDict]
        return rels
    def cheat(self,que,topkRels, topkRelsList):
        '''
        根据 
        question
        topkRels: Topk relations 2 relID
        topkRelsList: Topk relations in order
        '''   
        triger2rel = {'毕业':'<毕业院校>','高管':'<公司高管>','学历':'<学历类别>','创始人':'<企业创始人>'}
        for key in triger2rel:
            if key in que:
                _rel = triger2rel[key]
                if _rel not in topkRelsList:    # 不在
                    relId = self.rels.index(_rel)
                    topkRels[_rel] = relId
                    pop_rel = topkRelsList[-1]  # 去掉最后一个
                    topkRels.pop(pop_rel)
                    topkRelsList = [_rel] + topkRelsList[:-1]
                else:   # 在 把位置提前
                    index = topkRelsList.index(_rel)
                    topkRelsList.pop(index)
                    topkRelsList.insert(0,_rel)
        return topkRels, topkRelsList

    def change_for_train(self, question, topkRels, topkRelsList):
        try:
            gold_rels = self.que2rel[question]
        except:
            return topkRels, topkRelsList
        for _ in gold_rels:
            _rel =  '<'+_+'>'
            if _rel not in topkRelsList:    # 不在
                relId = self.rels.index(_rel)
                topkRels[_rel] = relId
                pop_rel = topkRelsList[-1]  # 去掉最后一个
                topkRels.pop(pop_rel)
                topkRelsList = [_rel] + topkRelsList[:-1]
            else:   # 在 把位置提前
                index = topkRelsList.index(_rel)
                topkRelsList.pop(index)
                topkRelsList.insert(0,_rel)
        return topkRels, topkRelsList

    def predictionRelBasedThreshold(self, question,mode='train'):
        test_features = convert_eval_examples_to_features([question], self.max_seq_length, self.tokenizer) 
        # que_ids = torch.tensor([i for i in test_features], dtype=torch.long).to(self.device)
        que_ids = torch.tensor([i for i in test_features], dtype=torch.long)
        eval_que_embed = self.model(que_ids, output_all_encoded_layers = False)[0].mean(1)
        
        cosScores = torch.cosine_similarity(self.relEmbs, eval_que_embed.cpu())
        
        topk_rel_id = torch.topk(cosScores, self.K, dim=0,sorted=True)[1]   # 取topK [batch_size * K] # 转置是为了方便循环校验,放进cpu是预测完后一起计算R
        topkRels = {}
        topkRelsList = []
        for relId in topk_rel_id:
            topkRels[self.rels[relId]] = int(relId)
            topkRelsList.append(self.rels[relId])
        # import pdb;pdb.set_trace()
        
        if mode=='train':
            topkRels, topkRelsList = self.change_for_train(question, topkRels, topkRelsList)
        else:
            
            topkRels, topkRelsList = self.cheat(question, topkRels, topkRelsList)
            pass
        return topkRels, topkRelsList


class CrossSimOfQueRel:
    def __init__(self, topk = 200):
        # 加载bert的分词器    
        self.vocab_path = BASE_DIR + '/data/models/que_rel_sim/0321_32_DNS/'
        self.bert_path = '/data4/cytan/7_dense_retriver/models/question_classification/0.9841956010513211_0.9782827954190526_0'
        self.K = topk
        # import pdb; pdb.set_trace()
        self.device = torch.device("cuda", 0)
        self.max_seq_length = 65
        self.tokenizer = BertTokenizer.from_pretrained(self.vocab_path)
        self.model = BertForSequenceClassification.from_pretrained(self.bert_path, num_labels = 2).to(self.device)
        self.model.eval()

    def rank(self,que,RelsList):
        '''
        对这些做重排序
        :param que: question
        :param RelsList: seq to be sorted
        :return: 排序后的关系名称 无<> List[]
        '''
        # test_features = convert_eval_examples_to_features([que], self.max_seq_length, self.tokenizer) 
        test_features = self.convert_to_features(que, RelsList, self.max_seq_length)
        all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        eval_sampler = SequentialSampler(eval_data)
        dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=32)
        score_of_rels = []
        for input_ids, segment_ids, input_mask, label_id in dataloader:
            # pdb.set_trace()
            input_ids = input_ids.to(self.device)
            segment_ids = segment_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask,labels=None)
                logits = torch.sigmoid(logits)
            score = logits[:,1]
            score_of_rels.append(score)
        score_of_rels = torch.cat(score_of_rels,dim=0)
        values, indices = score_of_rels.topk(len(score_of_rels))    # 排序后的值，对应下标
        # pdb.set_trace()
        new_topkRelsList = [RelsList[i] for i in indices]
        return new_topkRelsList

    def convert_to_features(self, que, rel_list, max_seq_length=65):
        """Loads a data file into a list of `InputBatch`s."""
        # import pdb; pdb.set_trace()
        features_all = []
            
        for rel in rel_list:
            tokens_a = self.tokenizer.tokenize(que)
            tokens_b = self.tokenizer.tokenize(rel)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)  # 长度截断
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)
            if tokens_b:
                tokens += tokens_b + ["[SEP]"]
                segment_ids += [1] * (len(tokens_b) + 1)     
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            # import pdb; pdb.set_trace()
            input_mask = [1] * len(input_ids)
            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            label_id = 1
            # import pdb; pdb.set_trace()
            features_all.append(
                    InputFeatures_Cross(input_ids=input_ids,
                                input_mask=input_mask,
                                segment_ids=segment_ids,
                                label_id=label_id))
        # import pdb; pdb.set_trace()
        return features_all


class RelRanker:
    def __init__(self, topk) -> None:
        self.bi_ranker = SimOfQueRel(topk)
        self.cross_ranker = CrossSimOfQueRel(topk)
    def predictionRelBasedThreshold(self,que):
        relsDict, topkRelsList = self.bi_ranker.predictionRelBasedThreshold(que)    # 返回的关系 是有<>
        # pdb.set_trace()
        new_topkRelsList = self.cross_ranker.rank(que, topkRelsList)
        # pdb.set_trace()
        return relsDict, new_topkRelsList

    

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    que = '小说《红高粱》的作者毕业于哪所大学'
    ranker = SimOfQueRel(100)
    # import pdb;pdb.set_trace()
    ranker.predictionRelBasedThreshold(que)
    pass
    
    