from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys
import numpy as np
import torch
import math
from torch._C import dtype
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
from argparse import ArgumentParser
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import *
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
import torch.nn.functional as F
from torch.autograd import Variable

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
# from Model.Pairwise.Embedding import RelationEmbedding
from typing import List, Dict, Tuple

class positionBasedPooler(nn.Module):
    def __init__(self, inputDim, outputDim):
        super(positionBasedPooler, self).__init__()
        self.dense = nn.Linear(inputDim, outputDim)
        self.activation = nn.Tanh()

    def forward(self, tokens_tensor):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        # first_token_tensor = hidden_states[:, 0]
        # import pdb; pdb.set_trace()
        pooled_output = self.dense(tokens_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
    
class BertForSequenceClassificationCopy(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(BertForSequenceClassificationCopy, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        # first_token_tensor = _[:, 0]
        pooled_output = self.pooler(_)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

class BertForSequenceClassificationOneItem(BertPreTrainedModel):
   
    def __init__(self, config, num_labels):
        super(BertForSequenceClassificationOneItem, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size * 2, num_labels)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.pooler = positionBasedPooler(config.hidden_size, config.hidden_size)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, alignPositions=None):
        outputEmbs, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        # pooled_output = self.dropout(pooled_output)
        # logits = self.classifier(pooled_output)
        batch, alignNum = alignPositions.shape
        hidden_dim = outputEmbs.shape[-1]
        # import pdb; pdb.set_trace()
        extractEmb = torch.zeros((batch, hidden_dim)).to(alignPositions.device)
        for i, item in enumerate(alignPositions):
            extractEmb[i] = outputEmbs[i, item[1]]
        # extractFeature =  torch.cat((extractEmb[:,0,:], extractEmb[:, 1, :], extractEmb[:,0,:] - extractEmb[:,1,:]), 1)
        # extractEmbTensor = torch.Tensor(extractEmb)
        # import pdb; pdb.set_trace()
        extractFeature = self.pooler(extractEmb)
        extractFeature = self.dropout(extractFeature)
        logits = self.classifier(extractFeature)
        # import pdb; pdb.set_trace()
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

class BertForSequenceClassificationUnused(BertPreTrainedModel):
   
    def __init__(self, config, num_labels):
        super(BertForSequenceClassificationUnused, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size * 2, num_labels)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.pooler = positionBasedPooler(config.hidden_size, config.hidden_size)
        # self.pooler2 = positionBasedPooler(config.hidden_size, config.hidden_size)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, alignPositions=None):
        outputEmbs, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        # pooled_output = self.dropout(pooled_output)
        # logits = self.classifier(pooled_output)
        batch, alignNum = alignPositions.shape
        hidden_dim = outputEmbs.shape[-1]
        # import pdb; pdb.set_trace()
        extractEmb = torch.zeros((batch, hidden_dim)).to(alignPositions.device)
        for i, item in enumerate(alignPositions):
            extractEmb[i] = outputEmbs[i, item[1]]
        # extractFeature =  torch.cat((extractEmb[:,0,:], extractEmb[:, 1, :], extractEmb[:,0,:] - extractEmb[:,1,:]), 1)
        # extractEmbTensor = torch.Tensor(extractEmb)
        # import pdb; pdb.set_trace()
        extractFeature = self.pooler(extractEmb)
        extractFeature = self.dropout(extractFeature)
        logits = self.classifier(extractFeature)
        # import pdb; pdb.set_trace()
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits
        
class BertForSequenceClassificationSEP1SEP2(BertPreTrainedModel):
   
    def __init__(self, config, num_labels):
        super(BertForSequenceClassificationSEP1SEP2, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size * 2, num_labels)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.classifier2 = nn.Linear(config.hidden_size, num_labels)
        self.pooler1 = positionBasedPooler(config.hidden_size, config.hidden_size)
        self.pooler2 = positionBasedPooler(config.hidden_size, config.hidden_size)
        self.linearScore = nn.Linear(4, 2)
        self.linearScore1 = nn.Linear(num_labels, num_labels)
        self.linearScore2 = nn.Linear(num_labels, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, alignPositions=None):
        outputEmbs, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        # pooled_output = self.dropout(pooled_output)
        # logits = self.classifier(pooled_output)
        batch, alignNum = alignPositions.shape
        hidden_dim = outputEmbs.shape[-1]
        # import pdb; pdb.set_trace()
        # x = torch.index_select(outputEmbs, 1, alignPositions[:, 1])
        # outputEmbs[:, alignPositions[:, 1].view(-1, 1), :]
        # input_ids[:, alignPositions[:, 1].view(-1, 1)]
        # ########### sep1和sep2分类 ###############################
        # sep1Emb = torch.zeros((batch, hidden_dim)).to(alignPositions.device)
        # for i, item in enumerate(alignPositions):
        #     sep1Emb[i] = outputEmbs[i, item[1]]
        # sep2Emb = torch.zeros((batch, hidden_dim)).to(alignPositions.device)
        # for i, item in enumerate(alignPositions):
        #     sep2Emb[i] = outputEmbs[i, item[5] - 1]
        # extractFeature = self.pooler1(sep1Emb)
        # extractFeature = self.dropout(extractFeature)
        # logits1 = self.classifier(extractFeature)
        # extractFeature = self.pooler2(sep2Emb)
        # extractFeature = self.dropout(extractFeature)
        # logits2 = self.classifier(extractFeature)
        # logits = logits1 + logits2
        # ##############################################################
        # ########### cls和sep1分类 ###############################
        sep1Emb = torch.zeros((batch, hidden_dim)).to(alignPositions.device)
        for i, item in enumerate(alignPositions):
            sep1Emb[i] = outputEmbs[i, item[1]]
        sep2Emb = torch.zeros((batch, hidden_dim)).to(alignPositions.device)
        for i, item in enumerate(alignPositions):
            sep2Emb[i] = outputEmbs[i, item[0] - 1]
        extractFeature = self.pooler1(sep1Emb)
        extractFeature = self.dropout(extractFeature)
        logits1 = self.classifier(extractFeature)
        extractFeature = self.pooler2(sep2Emb)
        extractFeature = self.dropout(extractFeature)
        logits2 = self.classifier2(extractFeature)
        logits = (self.linearScore1(logits1) + self.linearScore2(logits2)) / 2.0
        
        # logits = torch.cat((logits1, logits2), 1)
        # logits = self.linearScore(logits)
        # logits = logits2
        # import pdb; pdb.set_trace()
        # logits = logits1 + logits2
        # ##############################################################
        # ########### cls和sep2分类 ###############################
        # sep1Emb = torch.zeros((batch, hidden_dim)).to(alignPositions.device)
        # for i, item in enumerate(alignPositions):
        #     sep1Emb[i] = outputEmbs[i, item[0] - 1]
        # sep2Emb = torch.zeros((batch, hidden_dim)).to(alignPositions.device)
        # for i, item in enumerate(alignPositions):
        #     sep2Emb[i] = outputEmbs[i, item[5] - 1]
        # extractFeature = self.pooler1(sep1Emb)
        # extractFeature = self.dropout(extractFeature)
        # logits1 = self.classifier(extractFeature)
        # extractFeature = self.pooler2(sep2Emb)
        # extractFeature = self.dropout(extractFeature)
        # logits2 = self.classifier(extractFeature)
        # logits = logits1 + logits2
        # ##############################################################
        ########### cls和unused1分类 ###############################
        # sep1Emb = torch.zeros((batch, hidden_dim)).to(alignPositions.device)
        # for i, item in enumerate(alignPositions):
        #     sep1Emb[i] = outputEmbs[i, item[0]]
        # sep2Emb = torch.zeros((batch, hidden_dim)).to(alignPositions.device)
        # for i, item in enumerate(alignPositions):
        #     sep2Emb[i] = outputEmbs[i, item[3] - 1]
        # extractFeature = self.pooler1(sep1Emb)
        # extractFeature = self.dropout(extractFeature)
        # logits1 = self.classifier(extractFeature)
        # extractFeature = self.pooler2(sep2Emb)
        # extractFeature = self.dropout(extractFeature)
        # logits2 = self.classifier(extractFeature)
        # logits = logits1 + logits2
        ##############################################################
        # import pdb; pdb.set_trace()
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits
        
class BertForSequenceClassificationCLS(BertPreTrainedModel):
   
    def __init__(self, config, num_labels):
        super(BertForSequenceClassificationCLS, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.classifier2 = nn.Linear(config.hidden_size, num_labels)
        # self.pooler1 = positionBasedPooler(config.hidden_size, config.hidden_size)
        self.pooler2 = positionBasedPooler(config.hidden_size, config.hidden_size)
        # self.linearScore = nn.Linear(4, 2)
        # self.linearScore1 = nn.Linear(num_labels, num_labels)
        # self.linearScore2 = nn.Linear(num_labels, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, alignPositions=None):
        outputEmbs, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        # pooled_output = self.dropout(pooled_output)
        # logits = self.classifier(pooled_output)
        batch, alignNum = alignPositions.shape
        hidden_dim = outputEmbs.shape[-1]
        # ########### cls和sep1分类 ###############################
        # sep1Emb = torch.zeros((batch, hidden_dim)).to(alignPositions.device)
        # for i, item in enumerate(alignPositions):
        #     sep1Emb[i] = outputEmbs[i, item[1]]
        # sep2Emb = torch.zeros((batch, hidden_dim)).to(alignPositions.device)
        # for i, item in enumerate(alignPositions):
        #     sep2Emb[i] = outputEmbs[i, item[0] - 1]
        sep2Emb = outputEmbs[:, 0]
        # extractFeature = self.pooler1(sep1Emb)
        # extractFeature = self.dropout(extractFeature)
        # logits1 = self.classifier(extractFeature)
        extractFeature = self.pooler2(sep2Emb)
        extractFeature = self.dropout(extractFeature)
        logits2 = self.classifier2(extractFeature)
        # logits = self.linearScore1(logits1) + self.linearScore2(logits2)
        
        # logits = torch.cat((logits1, logits2), 1)
        # logits = self.linearScore(logits)
        logits = logits2
        # import pdb; pdb.set_trace()
        # logits = logits1 + logits2
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits
        
class BertForSequenceClassificationSEP1CatSEP2(BertPreTrainedModel):
   
    def __init__(self, config, num_labels):
        super(BertForSequenceClassificationSEP1CatSEP2, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size * 2, num_labels)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        # self.classifier2 = nn.Linear(config.hidden_size, num_labels)
        self.pooler1 = positionBasedPooler(config.hidden_size * 2, config.hidden_size)
        # self.pooler2 = positionBasedPooler(config.hidden_size, config.hidden_size)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, alignPositions=None):
        outputEmbs, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        # pooled_output = self.dropout(pooled_output)
        # logits = self.classifier(pooled_output)
        batch, alignNum = alignPositions.shape
        hidden_dim = outputEmbs.shape[-1]
        # import pdb; pdb.set_trace()
        # x = torch.index_select(outputEmbs, 1, alignPositions[:, 1])
        # outputEmbs[:, alignPositions[:, 1].view(-1, 1), :]
        # input_ids[:, alignPositions[:, 1].view(-1, 1)]
        # ########### sep1和sep2分类 ###############################
        # sep1Emb = torch.zeros((batch, hidden_dim)).to(alignPositions.device)
        # for i, item in enumerate(alignPositions):
        #     sep1Emb[i] = outputEmbs[i, item[1]]
        # sep2Emb = torch.zeros((batch, hidden_dim)).to(alignPositions.device)
        # for i, item in enumerate(alignPositions):
        #     sep2Emb[i] = outputEmbs[i, item[5] - 1]
        # extractFeature = self.pooler1(sep1Emb)
        # extractFeature = self.dropout(extractFeature)
        # logits1 = self.classifier(extractFeature)
        # extractFeature = self.pooler2(sep2Emb)
        # extractFeature = self.dropout(extractFeature)
        # logits2 = self.classifier(extractFeature)
        # logits = logits1 + logits2
        # ##############################################################
        # ########### cls和sep1 cat 分类 ###############################
        sep1Emb = torch.zeros((batch, hidden_dim)).to(alignPositions.device)
        for i, item in enumerate(alignPositions):
            sep1Emb[i] = outputEmbs[i, item[1]]
        sep2Emb = torch.zeros((batch, hidden_dim)).to(alignPositions.device)
        for i, item in enumerate(alignPositions):
            sep2Emb[i] = outputEmbs[i, item[0] - 1]
        features = torch.cat((sep1Emb, sep2Emb), 1)
        # import pdb; pdb.set_trace()
        extractFeature = self.pooler1(features)
        extractFeature = self.dropout(extractFeature)
        logits = self.classifier(extractFeature)
        # ##############################################################
        # ########### cls和sep2分类 ###############################
        # sep1Emb = torch.zeros((batch, hidden_dim)).to(alignPositions.device)
        # for i, item in enumerate(alignPositions):
        #     sep1Emb[i] = outputEmbs[i, item[0] - 1]
        # sep2Emb = torch.zeros((batch, hidden_dim)).to(alignPositions.device)
        # for i, item in enumerate(alignPositions):
        #     sep2Emb[i] = outputEmbs[i, item[5] - 1]
        # extractFeature = self.pooler1(sep1Emb)
        # extractFeature = self.dropout(extractFeature)
        # logits1 = self.classifier(extractFeature)
        # extractFeature = self.pooler2(sep2Emb)
        # extractFeature = self.dropout(extractFeature)
        # logits2 = self.classifier(extractFeature)
        # logits = logits1 + logits2
        # ##############################################################
        ########### cls和unused1分类 ###############################
        # sep1Emb = torch.zeros((batch, hidden_dim)).to(alignPositions.device)
        # for i, item in enumerate(alignPositions):
        #     sep1Emb[i] = outputEmbs[i, item[0]]
        # sep2Emb = torch.zeros((batch, hidden_dim)).to(alignPositions.device)
        # for i, item in enumerate(alignPositions):
        #     sep2Emb[i] = outputEmbs[i, item[3] - 1]
        # extractFeature = self.pooler1(sep1Emb)
        # extractFeature = self.dropout(extractFeature)
        # logits1 = self.classifier(extractFeature)
        # extractFeature = self.pooler2(sep2Emb)
        # extractFeature = self.dropout(extractFeature)
        # logits2 = self.classifier(extractFeature)
        # logits = logits1 + logits2
        ##############################################################
        # import pdb; pdb.set_trace()
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits
        
        
class BertForSequenceClassificationSEP1SubSEP2(BertPreTrainedModel):
   
    def __init__(self, config, num_labels):
        super(BertForSequenceClassificationSEP1SubSEP2, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size * 2, num_labels)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.pooler1 = positionBasedPooler(config.hidden_size, config.hidden_size)
        # self.pooler2 = positionBasedPooler(config.hidden_size, config.hidden_size)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, alignPositions=None):
        outputEmbs, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        # pooled_output = self.dropout(pooled_output)
        # logits = self.classifier(pooled_output)
        batch, alignNum = alignPositions.shape
        hidden_dim = outputEmbs.shape[-1]
        ########### cls-sep2 ###############################
        sep1Emb = torch.zeros((batch, hidden_dim)).to(alignPositions.device)
        for i, item in enumerate(alignPositions):
            sep1Emb[i] = outputEmbs[i, item[0] - 1]
        sep2Emb = torch.zeros((batch, hidden_dim)).to(alignPositions.device)
        for i, item in enumerate(alignPositions):
            sep2Emb[i] = outputEmbs[i, item[5] - 1]
        # import pdb; pdb.set_trace()
        sepSub = sep1Emb - sep2Emb
        extractFeature = self.pooler1(sepSub)
        extractFeature = self.dropout(extractFeature)
        logits = self.classifier(extractFeature)
        ##############################################################
        # ########### question-query graph mean ###############################
        # sep1Emb = torch.zeros((batch, hidden_dim)).to(alignPositions.device)
        # for i, item in enumerate(alignPositions):
        #     sep1Emb[i] = torch.mean(outputEmbs[i, item[0]: item[1]], 0)
        # sep2Emb = torch.zeros((batch, hidden_dim)).to(alignPositions.device)
        # for i, item in enumerate(alignPositions):
        #     sep2Emb[i] = torch.mean(outputEmbs[i, item[2]: item[5]], 0)
        # # import pdb; pdb.set_trace()
        # sepSub = sep1Emb - sep2Emb
        # extractFeature = self.pooler1(sepSub)
        # extractFeature = self.dropout(extractFeature)
        # logits = self.classifier(extractFeature)
        # ##############################################################
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

class BertForSequenceClassificationAlignBased(BertPreTrainedModel):
   
    def __init__(self, config, num_labels):
        super(BertForSequenceClassificationAlignBased, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size * 2, num_labels)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.pooler = positionBasedPooler(config.hidden_size * 3, config.hidden_size)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, alignPositions=None):
        outputEmbs, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        # pooled_output = self.dropout(pooled_output)
        # logits = self.classifier(pooled_output)
        batch, alignNum = alignPositions.shape
        hidden_dim = outputEmbs.shape[-1]
        # import pdb; pdb.set_trace()
        extractEmb = torch.zeros((batch, alignNum, hidden_dim)).to(alignPositions.device)
        for i, item in enumerate(alignPositions):
            extractEmb[i] = outputEmbs[i, item]
        extractFeature =  torch.cat((extractEmb[:,0,:], extractEmb[:, 1, :], extractEmb[:,0,:] - extractEmb[:,1,:]), 1)
        # extractEmbTensor = torch.Tensor(extractEmb)
        # import pdb; pdb.set_trace()
        extractFeature = self.pooler(extractFeature)
        extractFeature = self.dropout(extractFeature)
        logits = self.classifier(extractFeature)
        # import pdb; pdb.set_trace()
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class BertForSequenceClassification3AlignBased(BertPreTrainedModel):
   
    def __init__(self, config, num_labels):
        super(BertForSequenceClassification3AlignBased, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size * 2, num_labels)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.pooler = positionBasedPooler(config.hidden_size * 3, config.hidden_size)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, alignPositions=None):
        outputEmbs, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        # pooled_output = self.dropout(pooled_output)
        # logits = self.classifier(pooled_output)
        batch, alignNum = alignPositions.shape
        hidden_dim = outputEmbs.shape[-1]
        # import pdb; pdb.set_trace()
        extractEmb = torch.zeros((batch, alignNum, hidden_dim)).to(alignPositions.device)
        for i, item in enumerate(alignPositions):
            extractEmb[i] = outputEmbs[i, item]
        extractFirstFeature =  torch.cat((extractEmb[:,0,:], extractEmb[:, 1, :], extractEmb[:,0,:] - extractEmb[:,1,:]), 1)
        # extractEmbTensor = torch.Tensor(extractEmb)
        # import pdb; pdb.set_trace()
        extractFirstFeature = self.pooler(extractFirstFeature)
        extractFirstFeature = self.dropout(extractFirstFeature)
        logitsFirst = self.classifier(extractFirstFeature)
        # extractSecondFeature =  torch.cat((extractEmb[:,0,:], extractEmb[:, 2, :], extractEmb[:,0,:] - extractEmb[:,2,:]), 1)
        # extractEmbTensor = torch.Tensor(extractEmb)
        queEmb = torch.max(extractEmb[:,0,:], 0)[0]
        # queEmb = torch.mean(extractEmb[:,0,:], 0)
        # import pdb; pdb.set_trace()
        queEmb = queEmb.unsqueeze(0).repeat(batch, 1)
        extractSecondFeature =  torch.cat((queEmb, extractEmb[:, 2, :], extractEmb[:,0,:] - extractEmb[:,2,:]), 1)
        # import pdb; pdb.set_trace()
        extractSecondFeature = self.pooler(extractSecondFeature)
        extractSecondFeature = self.dropout(extractSecondFeature)
        logitsSecond = self.classifier(extractSecondFeature)
        # logits = logitsFirst + logitsSecond
        # logits = logitsFirst
        logits = logitsSecond
        # import pdb; pdb.set_trace()
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

class BertForSequenceClassificationAttentionAlignBased(BertPreTrainedModel):
   
    def __init__(self, config, num_labels):
        super(BertForSequenceClassificationAttentionAlignBased, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size * 2, num_labels)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.pooler = positionBasedPooler(config.hidden_size * 3, config.hidden_size)
        self.scaleScore = nn.Linear(num_labels, num_labels)
        self.apply(self.init_bert_weights)

    def attention_net(self, queEmbs, contentEmbs):
        queEmbsPermute = queEmbs.permute(1, 0)
        # import pdb; pdb.set_trace()
        attentionWeights = torch.mm(contentEmbs, queEmbsPermute)
        attentionWeights /= math.sqrt(queEmbs.shape[1])
        # import pdb; pdb.set_trace()
        softmaxAttentionWeights = torch.nn.functional.softmax(attentionWeights, 1)
        context = torch.mm(softmaxAttentionWeights, queEmbs)
        # import pdb; pdb.set_trace()
        return context
        # import pdb; pdb.set_trace()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, alignPositions=None):
        outputEmbs, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        batch, alignNum = alignPositions.shape
        hidden_dim = outputEmbs.shape[-1]
        # extractQueEmbs = torch.mean(outputEmbs[:,alignPositions[0][0]:alignPositions[0][1],:], 0)
        extractQueEmbs = torch.max(outputEmbs[:,alignPositions[0][0]:alignPositions[0][1],:], 0)[0]
        extractFirstEmbs = torch.zeros((batch, hidden_dim)).to(alignPositions.device)
        for i, item in enumerate(alignPositions):
            extractFirstEmbs[i] = outputEmbs[i, item[3] - 1, :]
        extractSecondEmbs = torch.zeros((batch, hidden_dim)).to(alignPositions.device)
        for i, item in enumerate(alignPositions):
            extractSecondEmbs[i] = outputEmbs[i, item[5], :]
        firstQueContent = self.attention_net(extractQueEmbs, extractFirstEmbs)
        secondQueContent = self.attention_net(extractQueEmbs, extractSecondEmbs)
        # import pdb; pdb.set_trace()
        extractFirstFeature =  torch.cat((firstQueContent, extractFirstEmbs, firstQueContent - extractFirstEmbs), 1)
        extractFirstFeature = self.pooler(extractFirstFeature)
        extractFirstFeature = self.dropout(extractFirstFeature)
        logitsFirst = self.classifier(extractFirstFeature)
        extractSecondFeature =  torch.cat((secondQueContent, extractSecondEmbs, secondQueContent - extractSecondEmbs), 1)
        extractSecondFeature = self.pooler(extractSecondFeature)
        extractSecondFeature = self.dropout(extractSecondFeature)
        logitsSecond = self.classifier(extractSecondFeature)
        logits = logitsFirst + logitsSecond
        logits = self.scaleScore(logits)
        # logits = logitsSecond
        # import pdb; pdb.set_trace()
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits
        

class BertForSequenceClassificationAttentionAlignBasedQueNotSame(BertPreTrainedModel):
   
    def __init__(self, config, num_labels):
        super(BertForSequenceClassificationAttentionAlignBasedQueNotSame, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size * 2, num_labels)
        self.classifier = nn.Linear(config.hidden_size * 3, num_labels)
        self.pooler = positionBasedPooler(config.hidden_size * 3, config.hidden_size * 3)
        self.scaleScore = nn.Linear(num_labels, num_labels)
        self.apply(self.init_bert_weights)

    def attention_net(self, queEmbs, contentEmbs):
        # import pdb; pdb.set_trace()
        contentEmbs = contentEmbs.unsqueeze(1)
        queEmbsPermute = queEmbs.transpose(1, 2)
        # import pdb; pdb.set_trace()
        attentionWeights = torch.bmm(contentEmbs, queEmbsPermute)
        attentionWeights /= math.sqrt(queEmbs.shape[2])
        softmaxAttentionWeights = torch.nn.functional.softmax(attentionWeights, 2)
        context = torch.bmm(softmaxAttentionWeights, queEmbs)
        context = context.squeeze(1)
        # import pdb; pdb.set_trace()
        return context
        # import pdb; pdb.set_trace()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, alignPositions=None):
        outputEmbs, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        batch, alignNum = alignPositions.shape
        hidden_dim = outputEmbs.shape[-1]
        # extractQueEmbs = torch.mean(outputEmbs[:,alignPositions[0][0]:alignPositions[0][1],:], 0)
        # extractQueEmbs = torch.max(outputEmbs[:,alignPositions[0][0]:alignPositions[0][1],:], 0)[0]
        extractQueEmbs = outputEmbs[:,alignPositions[0][0]:alignPositions[0][1],:]
        # import pdb; pdb.set_trace()
        extractFirstEmbs = torch.zeros((batch, hidden_dim)).to(alignPositions.device)
        for i, item in enumerate(alignPositions):
            extractFirstEmbs[i] = outputEmbs[i, item[3] - 1, :]
        extractSecondEmbs = torch.zeros((batch, hidden_dim)).to(alignPositions.device)
        for i, item in enumerate(alignPositions):
            extractSecondEmbs[i] = outputEmbs[i, item[5], :]
        firstQueContent = self.attention_net(extractQueEmbs, extractFirstEmbs)
        secondQueContent = self.attention_net(extractQueEmbs, extractSecondEmbs)
        # import pdb; pdb.set_trace()
        extractFirstFeature =  torch.cat((firstQueContent, extractFirstEmbs, firstQueContent - extractFirstEmbs), 1)
        extractFirstFeature = self.pooler(extractFirstFeature)
        extractFirstFeature = self.dropout(extractFirstFeature)
        logitsFirst = self.classifier(extractFirstFeature)
        extractSecondFeature =  torch.cat((secondQueContent, extractSecondEmbs, secondQueContent - extractSecondEmbs), 1)
        extractSecondFeature = self.pooler(extractSecondFeature)
        extractSecondFeature = self.dropout(extractSecondFeature)
        logitsSecond = self.classifier(extractSecondFeature)
        logits = logitsFirst + logitsSecond
        logits = self.scaleScore(logits)
        # logits = logitsSecond
        # import pdb; pdb.set_trace()
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits
        
class BertForSequenceClassificationWordSim(BertPreTrainedModel):
   
    def __init__(self, config, num_labels):
        super(BertForSequenceClassificationWordSim, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size * 2, num_labels)
        self.classifier = nn.Linear(150, num_labels)
        # self.scaleScore = nn.Linear(num_labels, num_labels)
        self.apply(self.init_bert_weights)

    def attention_net(self, queEmbs, contentEmbs):
        # import pdb; pdb.set_trace()
        contentEmbs = contentEmbs.unsqueeze(1)
        queEmbsPermute = queEmbs.transpose(1, 2)
        # import pdb; pdb.set_trace()
        attentionWeights = torch.bmm(contentEmbs, queEmbsPermute)
        attentionWeights /= math.sqrt(queEmbs.shape[2])
        softmaxAttentionWeights = torch.nn.functional.softmax(attentionWeights, 2)
        context = torch.bmm(softmaxAttentionWeights, queEmbs)
        context = context.squeeze(1)
        # import pdb; pdb.set_trace()
        return context
        # import pdb; pdb.set_trace()
        
    def pathSum(self, softmaxAtt):
        # import pdb; pdb.set_trace()
        topkAtt, topkIds = softmaxAtt.topk(k = 10, dim = 2, largest=True)
        topkAtt = topkAtt.cpu()
        topkIds = topkIds.cpu()
        softmaxAtt = softmaxAtt.cpu()
        batches, rows, cols = softmaxAtt.shape
        scoresMatrix = torch.zeros(batches, rows, cols)
        weights = torch.ones(cols)
        weightsNew = torch.ones(cols)
        # import pdb; pdb.set_trace()
        for batchId in range(batches):
            # print(batchId)
            for row in range(rows):
                # for col in range(cols):
                if(row == 0):
                    scoresMatrix[batchId, row] = softmaxAtt[batchId, row]
                    continue
                for col in topkIds[batchId, row]:
                    currentMax = 0
                    weight = 1.0
                    # for colRecur in topkIds[batchId, row - 1]:
                    for colRecur in topkIds[batchId, row - 1]: # 遍历上一步所有的取值
                        # import pdb; pdb.set_trace()
                        # print('colRecur:', colRecur)
                        if(abs(colRecur - col) <= 1):
                            temp = scoresMatrix[batchId][row - 1][colRecur] + softmaxAtt[batchId][row - 1][col] * (weights[col] + 1)
                            if(temp > currentMax):
                                currentMax = temp
                                weight = weights[col] + 1
                        else:
                            temp = scoresMatrix[batchId][row - 1][colRecur] + softmaxAtt[batchId][row - 1][col]
                            if(temp > currentMax):
                                currentMax = temp
                                weight = 1.0
                    scoresMatrix[batchId, row, col] = currentMax
                    weightsNew[col] = weight
                weights = weightsNew
                weightsNew = torch.ones(cols)
        # import pdb; pdb.set_trace()
        return torch.max(scoresMatrix[:, -1, :], 1)[0]
        # return scoresMatrix
        # return 

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, alignPositions=None):
        outputEmbs, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        batch, alignNum = alignPositions.shape
        hidden_dim = outputEmbs.shape[-1]
        queEmbs = outputEmbs[:, alignPositions[0][0]: alignPositions[0][1]]
        pathEmbs = outputEmbs[:, alignPositions[0][2]:]
        pathEmbs = pathEmbs.transpose(1, 2)
        attentionWeights = torch.bmm(queEmbs, pathEmbs)
        attentionWeights /= math.sqrt(hidden_dim)
        softmaxAtt = torch.nn.functional.softmax(attentionWeights, 2)
        # logits = self.pathSum(softmaxAtt)
        maxAtt = torch.max(softmaxAtt, 2)[0]
        logits = torch.sum(maxAtt.view(batch, -1), 1)
        # import pdb; pdb.set_trace()
        return logits
        
        
class BertForSequenceClassificationWordSimMaxSum(BertPreTrainedModel):
   
    def __init__(self, config, num_labels):
        super(BertForSequenceClassificationWordSimMaxSum, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifierCLS = nn.Linear(config.hidden_size, 1)
        self.classifier = nn.Linear(1, 1)
        self.scaleWeight = nn.Parameter(torch.ones(1))
        # self.scaleScore = nn.Linear(num_labels, num_labels)
        self.apply(self.init_bert_weights)

    def attention_net(self, queEmbs, contentEmbs):
        # import pdb; pdb.set_trace()
        contentEmbs = contentEmbs.unsqueeze(1)
        queEmbsPermute = queEmbs.transpose(1, 2)
        # import pdb; pdb.set_trace()
        attentionWeights = torch.bmm(contentEmbs, queEmbsPermute)
        attentionWeights /= math.sqrt(queEmbs.shape[2])
        softmaxAttentionWeights = torch.nn.functional.softmax(attentionWeights, 2)
        context = torch.bmm(softmaxAttentionWeights, queEmbs)
        context = context.squeeze(1)
        # import pdb; pdb.set_trace()
        return context
        # import pdb; pdb.set_trace()
        

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, alignPositions=None):
        outputEmbs, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        batch, alignNum = alignPositions.shape
        hidden_dim = outputEmbs.shape[-1]
        queEmbs = outputEmbs[:, alignPositions[0][0]: alignPositions[0][1]]
        pathEmbs = outputEmbs[:, alignPositions[0][2]:]
        
        # *****************余弦相似度
        queEmbsCos = queEmbs.unsqueeze(2)
        pathEmbsCos = pathEmbs.unsqueeze(1)
        # import pdb; pdb.set_trace()
        # cosSim = torch.cosine_similarity(queEmbsCos, pathEmbsCos, dim = 3)
        # # cosSim = cosSim.unsqueeze(0)
        # # print('cosSim:',cosSim.shape, cosSim)
        # softmaxAtt = cosSim
        # ******************注意力dot*****************
        pathEmbs = pathEmbs.transpose(1, 2)
        attentionWeights = torch.bmm(queEmbs, pathEmbs)
        attentionWeights /= math.sqrt(hidden_dim)
        # softmaxAtt = torch.nn.functional.softmax(attentionWeights, 1)
        softmaxAtt = attentionWeights
        # ******************************
        # import pdb; pdb.set_trace()
        # self.pathSum(softmaxAtt)
        maxAtt = torch.max(softmaxAtt, 2)[0]
        logits = torch.sum(maxAtt.view(batch, -1), 1).view(-1, 1)
        # logits = self.scaleWeight.to(logits.device) * logits
        # # logits = self.classifier(logits)
        # pooled_output = self.dropout(pooled_output)
        # logitsCLS = self.classifierCLS(pooled_output)
        # # import pdb; pdb.set_trace()
        # # print('scale Weight:', self.scaleWeight)
        # return logits + logitsCLS
        return logits
    
class BertForSequenceClassificationWordSimMaxSumContinuous(BertPreTrainedModel):
   
    def __init__(self, config, num_labels):
        super(BertForSequenceClassificationWordSimMaxSumContinuous, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifierCLS = nn.Linear(config.hidden_size, 1)
        self.classifier = nn.Linear(1, 1)
        self.scaleWeight = nn.Parameter(torch.ones(1))
        # self.scaleScore = nn.Linear(num_labels, num_labels)
        self.apply(self.init_bert_weights)
    
    def isConflict(self, currentId, usedIds, conflict):
        for usedId in usedIds:
            if(currentId in conflict[usedId]):
                return True
        return False
    
    def recur(self, examples, conflict, i = 0, path = [], currentScores = [], maxPath = [0]):
        if(i >= len(examples)):
            # import pdb; pdb.set_trace()
            currentSum = sum(currentScores)
            if(currentSum > maxPath[0]):
                # print(currentScores)
                maxPath[0] = currentSum
            return 0
        for j, score in enumerate(examples[i]):
            if(not self.isConflict(j, path, conflict)):
                # used.append(j)
                path.append(j)
                currentScores.append(score)
                self.recur(examples, conflict, i + 1, path, currentScores, maxPath)
                path.pop()
                currentScores.pop()
        return maxPath[0]
    
    def getIndexs(self, casesTensor, spanPathNums):
        if(spanPathNums == 1):
            indexs = (casesTensor[:, 0])
        elif(spanPathNums == 2):
            indexs = (casesTensor[:, 0], casesTensor[:, 1])
        elif(spanPathNums == 3):
            indexs = (casesTensor[:, 0], casesTensor[:, 1], casesTensor[:, 2])
        else:
            indexs = (casesTensor[:, 0], casesTensor[:, 1], casesTensor[:, 2], casesTensor[:, 3])
        # elif(spanPathNums == 4):
        #     indexs = (casesTensor[:, 0], casesTensor[:, 1], casesTensor[:, 2], casesTensor[:, 3])
        # elif(spanPathNums == 5):
        #     indexs = (casesTensor[:, 0], casesTensor[:, 1], casesTensor[:, 2], casesTensor[:, 3], casesTensor[:, 4])
        # elif(spanPathNums == 6):
        #     indexs = (casesTensor[:, 0], casesTensor[:, 1], casesTensor[:, 2], casesTensor[:, 3], casesTensor[:, 4], casesTensor[:, 5])
        # else:
        #     indexs = (casesTensor[:, 0], casesTensor[:, 1], casesTensor[:, 2], casesTensor[:, 3], casesTensor[:, 4], casesTensor[:, 5], casesTensor[:, 6])
        return indexs
    
    def getCombineMarix(self, spanScores, spanPathNums):
        if(spanPathNums == 1):
            combineMatrix = spanScores[0].view(-1, 1)
            # matrix2 = torch.tensor(spanScores[1], dtype = torch.float).view(1, -1)
            # combineMatrix = combineMatrix + matrix2
        elif(spanPathNums == 2):
            combineMatrix = spanScores[0].view(-1, 1)
            matrix2 = spanScores[1].view(1, -1)
            combineMatrix = combineMatrix + matrix2
        elif(spanPathNums == 3):
            combineMatrix = spanScores[0].view(-1, 1, 1)
            matrix2 = spanScores[1].view(1, -1, 1)
            matrix3 = spanScores[2].view(1, 1, -1)
            combineMatrix = combineMatrix + matrix2 + matrix3
        else:
            combineMatrix = spanScores[0].view(-1, 1, 1, 1)
            matrix2 = spanScores[1].view(1, -1, 1, 1)
            matrix3 = spanScores[2].view(1, 1, -1, 1)
            matrix4 = spanScores[3].view(1, 1, 1, -1)
            combineMatrix = combineMatrix + matrix2 + matrix3 + matrix4
        # import pdb; pdb.set_trace()
        return combineMatrix
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, alignPositions=None, alignKeyWordsIndex = None):
        outputEmbs, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        batch, alignNum = alignPositions.shape
        maskValue = torch.tensor(0.0).to(input_ids.device)
        hidden_dim = outputEmbs.shape[-1]
        queEmbs = outputEmbs[:, alignPositions[0][0]: alignPositions[0][1]]
        queLen = queEmbs.shape[1]
        pathEmbs = outputEmbs[:, alignPositions[0][2]:]
        spans = [(begin, end) for begin in range(queLen) for end in range(begin + 1, min(begin + 5, queLen))][0: 100]
        
        # *****************余弦相似度
        # queEmbsCos = queEmbs.unsqueeze(2)
        # pathEmbsCos = pathEmbs.unsqueeze(1)
        # import pdb; pdb.set_trace()
        # cosSim = torch.cosine_similarity(queEmbsCos, pathEmbsCos, dim = 3)
        # # cosSim = cosSim.unsqueeze(0)
        # # print('cosSim:',cosSim.shape, cosSim)
        # softmaxAtt = cosSim
        # ******************注意力dot*****************
        pathEmbs = pathEmbs.transpose(1, 2)
        attentionWeights = torch.bmm(queEmbs, pathEmbs)
        attentionWeights /= math.sqrt(hidden_dim)
        softmaxAtt = torch.nn.functional.softmax(attentionWeights, 2)
        # softmaxAtt = softmaxAtt * self.scaleWeight
        # print(self.scaleWeight, self.scaleWeight.requires_grad, self.scaleWeight.grad)
        # softmaxAtt = attentionWeights
        # ******************************
        # alignKeyWordsIndex = alignKeyWordsIndex[0]
        # import pdb; pdb.set_trace()
        spanScoresTemp = []
        batchSpans = []
        newSpansBatch = []
        
        # print('alignKeyWords:', alignKeyWordsIndex.shape)
        for i, alignKeyWords in enumerate(alignKeyWordsIndex): # 遍历每个batch  待改进：批次化时可将路径中的片段个数固定，有待考量
            # temp = []
            temp2 = []
            spansFlag = [0] * len(spans)
            for alignKeyWord in alignKeyWords[0:alignKeyWords[-1][0]]:
                # temp2.append([torch.sum(softmaxAtt[i, span[0]: span[1], alignKeyWord[0]:alignKeyWord[1]]) / (span[1] - span[0] + 10e-7) \
                #                 for span in spans])
                # temp2.append([torch.sum(softmaxAtt[i, span[0]: span[1], alignKeyWord[0]:alignKeyWord[1]]) - 0.5 * (span[1] - span[0]) \
                #                 for span in spans]) # 带惩罚的评价标准
                if(alignKeyWord[1] - alignKeyWord[0] != 0):
                    # temp2.append([torch.sum(softmaxAtt[i, span[0]: span[1], alignKeyWord[0]:alignKeyWord[1]]) / (1.0 + (alignKeyWord[1] - alignKeyWord[0]) * (span[1] - span[0])) \
                    #                 for span in spans]) # 带惩罚的评价标准
                    temp2.append([torch.sum(torch.max(softmaxAtt[i, span[0]: span[1], alignKeyWord[0]:alignKeyWord[1]], 1)[0]) / ((span[1] - span[0])) \
                                        for span in spans]) #   
                else:
                    temp2.append([torch.sum(softmaxAtt[i, span[0]: span[1], alignKeyWord[0]:alignKeyWord[1]]) \
                                for span in spans]) # 带惩罚的评价标准
            temp2Tensor = torch.tensor(temp2, dtype=torch.float, requires_grad = True).to(input_ids.device)
            # print(temp2Tensor.shape)
            indices = torch.topk(temp2Tensor, 3, dim = 1)[1]
            # import pdb; pdb.set_trace()
            for indice in indices:
                for item in indice:
                    spansFlag[item] = 1
            newSpans = []
            for i, item in enumerate(spansFlag):
                if(item == 1):
                    newSpans.append(spans[i])
            newSpansBatch.append(newSpans)
            # print('alignKey:', alignKeyWords)
            # print('spans:', len(newSpans), newSpans)
            # import pdb; pdb.set_trace()
        conflictBatch = []
        for newSpans in newSpansBatch:
            conflict = {}
            for i_span, span1 in enumerate(newSpans):
                conflict[i_span] = []
                for j, span2 in enumerate(newSpans):
                    if(span2[1] - span2[0] == 0):
                        continue
                    if((span2[0] >= span1[0] and span2[0] < span1[1]) or (span2[1] > span1[0] and span2[1] <= span1[1])):
                        conflict[i_span].append(j)
                # print('conflict:', len(conflict[i_span]))
            conflictBatch.append(conflict)
        # print('冲突信息构建完成')
        scores = []
        for i, alignKeyWords in enumerate(alignKeyWordsIndex): # 遍历每个batch
            itemOfBatch = []
            for alignKeyWord in alignKeyWords[0:alignKeyWords[-1][0]]:
                itemOfBatch.extend([torch.sum(softmaxAtt[i, span[0]: span[1], alignKeyWord[0]:alignKeyWord[1]]) \
                                for span in newSpansBatch[i]])
            # import pdb; pdb.set_trace()
            itemOfBatch = torch.stack(itemOfBatch).view(-1, len(newSpansBatch[i]))
            spanPathNums = min(itemOfBatch.shape[0], 6)
            cases = []
            # import pdb; pdb.set_trace()
            conflict = conflictBatch[i]
            for key in conflict:
                if(len(conflict[key]) != 0):
                    for iPosition in range(spanPathNums):
                        temp = [[]]
                        for ith in range(spanPathNums - 1):
                            if(ith == iPosition):
                                temp = [tempItem + [key] for tempItem in temp]
                            temp = [tempItem + [item] for tempItem in temp for item in conflict[key]]
                        if(iPosition == spanPathNums - 1):
                            temp = [tempItem + [key] for tempItem in temp]
                        cases.extend(temp)
                        # print(len(conflict[key]), len(temp), len(cases))
            # import pdb; pdb.set_trace()
            # print('batch item 完成一个得分计算')
            casesTensor = torch.tensor(cases, dtype=torch.long, requires_grad = False)
            indexs = self.getIndexs(casesTensor, spanPathNums)
            itemOfBatch = itemOfBatch
            combineMatrix = self.getCombineMarix(itemOfBatch, spanPathNums).to(input_ids.device)
            # import pdb; pdb.set_trace()
            if(spanPathNums > 1):
                combineMatrix.index_put_(indexs, maskValue)
            score = torch.max(combineMatrix)
            scores.append(score)
        # logits = torch.tensor(scores, dtype = torch.float, requires_grad = True).to(input_ids.device) * self.scaleWeight
        logits = torch.stack(scores).view(-1, 1)
        # logits = self.classifier(logits)
        # import pdb; pdb.set_trace()
        return logits

    # def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, alignPositions=None, alignKeyWordsIndex = None):
    #     outputEmbs, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
    #     batch, alignNum = alignPositions.shape
    #     maskValue = torch.tensor(0.0).to(input_ids.device)
    #     hidden_dim = outputEmbs.shape[-1]
    #     queEmbs = outputEmbs[:, alignPositions[0][0]: alignPositions[0][1]]
    #     queLen = queEmbs.shape[1]
    #     pathEmbs = outputEmbs[:, alignPositions[0][2]:]
    #     spans = [(begin, end) for begin in range(queLen) for end in range(begin + 1, min(begin + 5, queLen))][0: 100]
        
    #     # *****************余弦相似度
    #     # queEmbsCos = queEmbs.unsqueeze(2)
    #     # pathEmbsCos = pathEmbs.unsqueeze(1)
    #     # # import pdb; pdb.set_trace()
    #     # cosSim = torch.cosine_similarity(queEmbsCos, pathEmbsCos, dim = 3)
    #     # # cosSim = cosSim.unsqueeze(0)
    #     # # print('cosSim:',cosSim.shape, cosSim)
    #     # softmaxAtt = cosSim
    #     # ******************注意力dot*****************
    #     pathEmbs = pathEmbs.transpose(1, 2)
    #     attentionWeights = torch.bmm(queEmbs, pathEmbs)
    #     attentionWeights /= math.sqrt(hidden_dim)
    #     softmaxAtt = torch.nn.functional.softmax(attentionWeights, 2)
    #     # softmaxAtt = attentionWeights
    #     # ******************************
    #     # alignKeyWordsIndex = alignKeyWordsIndex[0]
    #     # import pdb; pdb.set_trace()
    #     spanScoresTemp = []
    #     batchSpans = []
    #     newSpansBatch = []
    #     # print('alignKeyWords:', alignKeyWordsIndex.shape)
    #     for i, alignKeyWords in enumerate(alignKeyWordsIndex): # 遍历每个batch  待改进：批次化时可将路径中的片段个数固定，有待考量
    #         # temp = []
    #         temp2 = []
    #         spansFlag = [0] * len(spans)
    #         for alignKeyWord in alignKeyWords[0:alignKeyWords[-1][0]]:
    #             # temp2.append([torch.sum(softmaxAtt[i, span[0]: span[1], alignKeyWord[0]:alignKeyWord[1]]) / (span[1] - span[0] + 10e-7) \
    #             #                 for span in spans])
    #             temp2.append([torch.sum(softmaxAtt[i, span[0]: span[1], alignKeyWord[0]:alignKeyWord[1]]) - 0.5 * (span[1] - span[0]) \
    #                             for span in spans]) # 带惩罚的评价标准
    #         temp2Tensor = torch.tensor(temp2, dtype=torch.float).to(input_ids.device)
    #         # print(temp2Tensor.shape)
    #         indices = torch.topk(temp2Tensor, 3, dim = 1)[1]
    #         # import pdb; pdb.set_trace()
    #         for indice in indices:
    #             for item in indice:
    #                 spansFlag[item] = 1
    #         newSpans = []
    #         for i, item in enumerate(spansFlag):
    #             if(item == 1):
    #                 newSpans.append(spans[i])
    #         newSpansBatch.append(newSpans)
    #         # print('alignKey:', alignKeyWords)
    #         # print('spans:', len(newSpans), newSpans)
    #         # import pdb; pdb.set_trace()
    #     conflictBatch = []
    #     for newSpans in newSpansBatch:
    #         conflict = {}
    #         for i_span, span1 in enumerate(newSpans):
    #             conflict[i_span] = []
    #             for j, span2 in enumerate(newSpans):
    #                 if(span2[1] - span2[0] == 0):
    #                     continue
    #                 if((span2[0] >= span1[0] and span2[0] < span1[1]) or (span2[1] > span1[0] and span2[1] <= span1[1])):
    #                     conflict[i_span].append(j)
    #             # print('conflict:', len(conflict[i_span]))
    #         conflictBatch.append(conflict)
    #     # print('冲突信息构建完成')
    #     spanScores3 = []
    #     for i, alignKeyWords in enumerate(alignKeyWordsIndex): # 遍历每个batch
    #         temp = []
    #         for alignKeyWord in alignKeyWords[0:alignKeyWords[-1][0]]:
    #             temp.append([torch.sum(softmaxAtt[i, span[0]: span[1], alignKeyWord[0]:alignKeyWord[1]]) \
    #                             for span in newSpansBatch[i]])
    #         spanScores3.append(temp)
            
    #     # print('span 得分计算完成')
    #     import pdb; pdb.set_trace()
    #     scores = []
    #     for i, itemOfBatch in enumerate(spanScores3):
    #         spanPathNums = min(len(itemOfBatch), 6)
    #         cases = []
    #         # import pdb; pdb.set_trace()
    #         conflict = conflictBatch[i]
    #         for key in conflict:
    #             if(len(conflict[key]) != 0):
    #                 for iPosition in range(spanPathNums):
    #                     temp = [[]]
    #                     for ith in range(spanPathNums - 1):
    #                         if(ith == iPosition):
    #                             temp = [tempItem + [key] for tempItem in temp]
    #                         temp = [tempItem + [item] for tempItem in temp for item in conflict[key]]
    #                     if(iPosition == spanPathNums - 1):
    #                         temp = [tempItem + [key] for tempItem in temp]
    #                     cases.extend(temp)
    #                     # print(len(conflict[key]), len(temp), len(cases))
    #         # import pdb; pdb.set_trace()
    #         # print('batch item 完成一个得分计算')
    #         casesTensor = torch.tensor(cases, dtype=torch.long)
    #         indexs = self.getIndexs(casesTensor, spanPathNums)
    #         combineMatrix = self.getCombineMarix(itemOfBatch, spanPathNums).to(input_ids.device)
    #         # import pdb; pdb.set_trace()
    #         # print(indexs)
    #         if(spanPathNums > 1):
    #             combineMatrix.index_put_(indexs, maskValue)
    #         # else:
    #         #     import pdb; pdb.set_trace()
    #         score = torch.max(combineMatrix)
    #         scores.append(score)
    #     logits = torch.tensor(scores, dtype = torch.float).to(input_ids.device)
    #     # import pdb; pdb.set_trace()
    #     return logits
    
    
class BertForSequenceClassificationWordSimMaxSumContinuousQueSegCombine(BertPreTrainedModel):
   
    def __init__(self, config, num_labels):
        super(BertForSequenceClassificationWordSimMaxSumContinuousQueSegCombine, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifierCLS = nn.Linear(config.hidden_size, 1)
        self.classifier = nn.Linear(1, 1)
        self.scaleWeight = nn.Parameter(torch.ones(1))
        # self.scaleScore = nn.Linear(num_labels, num_labels)
        self.apply(self.init_bert_weights)
    
    def isConflict(self, currentId, usedIds, conflict):
        for usedId in usedIds:
            if(currentId in conflict[usedId]):
                return True
        return False
    
    def recur(self, examples, conflict, i = 0, path = [], currentScores = [], maxPath = [0]):
        if(i >= len(examples)):
            # import pdb; pdb.set_trace()
            currentSum = sum(currentScores)
            if(currentSum > maxPath[0]):
                # print(currentScores)
                maxPath[0] = currentSum
            return 0
        for j, score in enumerate(examples[i]):
            if(not self.isConflict(j, path, conflict)):
                # used.append(j)
                path.append(j)
                currentScores.append(score)
                self.recur(examples, conflict, i + 1, path, currentScores, maxPath)
                path.pop()
                currentScores.pop()
        return maxPath[0]
    
    def getIndexs(self, casesTensor, spanPathNums):
        if(spanPathNums == 1):
            indexs = (casesTensor[:, 0])
        elif(spanPathNums == 2):
            indexs = (casesTensor[:, 0], casesTensor[:, 1])
        elif(spanPathNums == 3):
            indexs = (casesTensor[:, 0], casesTensor[:, 1], casesTensor[:, 2])
        else:
            indexs = (casesTensor[:, 0], casesTensor[:, 1], casesTensor[:, 2], casesTensor[:, 3])
        # elif(spanPathNums == 4):
        #     indexs = (casesTensor[:, 0], casesTensor[:, 1], casesTensor[:, 2], casesTensor[:, 3])
        # elif(spanPathNums == 5):
        #     indexs = (casesTensor[:, 0], casesTensor[:, 1], casesTensor[:, 2], casesTensor[:, 3], casesTensor[:, 4])
        # elif(spanPathNums == 6):
        #     indexs = (casesTensor[:, 0], casesTensor[:, 1], casesTensor[:, 2], casesTensor[:, 3], casesTensor[:, 4], casesTensor[:, 5])
        # else:
        #     indexs = (casesTensor[:, 0], casesTensor[:, 1], casesTensor[:, 2], casesTensor[:, 3], casesTensor[:, 4], casesTensor[:, 5], casesTensor[:, 6])
        return indexs
    
    def getCombineMarix(self, spanScores, spanPathNums):
        if(spanPathNums == 1):
            combineMatrix = spanScores[0].view(-1, 1)
            # matrix2 = torch.tensor(spanScores[1], dtype = torch.float).view(1, -1)
            # combineMatrix = combineMatrix + matrix2
        elif(spanPathNums == 2):
            combineMatrix = spanScores[0].view(-1, 1)
            matrix2 = spanScores[1].view(1, -1)
            combineMatrix = combineMatrix + matrix2
        elif(spanPathNums == 3):
            combineMatrix = spanScores[0].view(-1, 1, 1)
            matrix2 = spanScores[1].view(1, -1, 1)
            matrix3 = spanScores[2].view(1, 1, -1)
            combineMatrix = combineMatrix + matrix2 + matrix3
        else:
            combineMatrix = spanScores[0].view(-1, 1, 1, 1)
            matrix2 = spanScores[1].view(1, -1, 1, 1)
            matrix3 = spanScores[2].view(1, 1, -1, 1)
            matrix4 = spanScores[3].view(1, 1, 1, -1)
            combineMatrix = combineMatrix + matrix2 + matrix3 + matrix4
        # import pdb; pdb.set_trace()
        return combineMatrix
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, alignPositions=None, \
                alignKeyWordsIndex = None, que_spans = None):
        outputEmbs, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        batch, alignNum = alignPositions.shape
        maskValue = torch.tensor(0.0).to(input_ids.device)
        hidden_dim = outputEmbs.shape[-1]
        queEmbs = outputEmbs[:, alignPositions[0][0]: alignPositions[0][1]]
        queLen = queEmbs.shape[1]
        pathEmbs = outputEmbs[:, alignPositions[0][2]:]
        spans = que_spans
        
        # *****************余弦相似度
        # queEmbsCos = queEmbs.unsqueeze(2)
        # pathEmbsCos = pathEmbs.unsqueeze(1)
        # import pdb; pdb.set_trace()
        # cosSim = torch.cosine_similarity(queEmbsCos, pathEmbsCos, dim = 3)
        # # cosSim = cosSim.unsqueeze(0)
        # # print('cosSim:',cosSim.shape, cosSim)
        # softmaxAtt = cosSim
        # ******************注意力dot*****************
        pathEmbs = pathEmbs.transpose(1, 2)
        attentionWeights = torch.bmm(queEmbs, pathEmbs)
        attentionWeights /= math.sqrt(hidden_dim)
        # softmaxAtt = torch.nn.functional.softmax(attentionWeights, 2)
        # softmaxAtt = softmaxAtt * self.scaleWeight
        # print(self.scaleWeight, self.scaleWeight.requires_grad, self.scaleWeight.grad)
        softmaxAtt = attentionWeights
        # ******************************
        # alignKeyWordsIndex = alignKeyWordsIndex[0]
        # import pdb; pdb.set_trace()
        spanScoresTemp = []
        batchSpans = []
        # print(spans.shape)
        newSpansBatch = spans
        # print('冲突信息构建完成')
        scores = []
        # print(len(alignKeyWordsIndex))
        # import pdb; pdb.set_trace()
        logData = []
        for i, alignKeyWords in enumerate(alignKeyWordsIndex): # 遍历每个batch
            itemOfBatch = []
            itemOfBatchPath = []
            for alignKeyWord in alignKeyWords[0:alignKeyWords[-1][0]]:
                itemOfBatch.extend([torch.sum(torch.max(softmaxAtt[i, span[0]: span[1], alignKeyWord[0]:alignKeyWord[1]], 1)[0]) / ((span[1] - span[0])) \
                                        for span in newSpansBatch[i][0: newSpansBatch[i][-1][0]]]) #   
                # itemOfBatch.extend([torch.sum(softmaxAtt[i, span[0]: span[1], alignKeyWord[0]:alignKeyWord[1]]) / ((span[1] - span[0])) \
                #                         for span in newSpansBatch[i][0: newSpansBatch[i][-1][0]]]) #   
                # itemOfBatch.extend([torch.sum(torch.mean(softmaxAtt[i, span[0]: span[1], alignKeyWord[0]:alignKeyWord[1]], 1)) / ((span[1] - span[0])) \
                #                         for span in newSpansBatch[i][0: newSpansBatch[i][-1][0]]]) #   
                itemOfBatchPath.extend([torch.sum(torch.max(softmaxAtt[i, span[0]: span[1], alignKeyWord[0]:alignKeyWord[1]], 0)[0]) / ((alignKeyWord[1] - alignKeyWord[0])) \
                                        for span in newSpansBatch[i][0: newSpansBatch[i][-1][0]]]) #  
                # import pdb; pdb.set_trace() 
                # itemOfBatch.extend([torch.sum(torch.max(softmaxAtt[i, span[0]: span[1], alignKeyWord[0]:alignKeyWord[1]], 1)[0]) \
                #                         for span in newSpansBatch[i][0: newSpansBatch[i][-1][0]]]) #    
            # itemOfBatch = torch.stack(itemOfBatch).view(-1, len(newSpansBatch[i]))
            ############################ 以查询图序列为核心 ############################
            itemOfBatchPath = torch.stack(itemOfBatchPath).view(-1, newSpansBatch[i][-1][0])
            # scoreOfItemBasedPath = torch.max(itemOfBatchPath, 1)[0]
            # logData.append(scoreOfItemBasedPath)
            # scoreOfItemBasedPathForQue = torch.max(itemOfBatchPath, 0)[0]
            # scores.append(torch.sum(scoreOfItemBasedPath) + torch.sum(scoreOfItemBasedPathForQue))
            # scores.append(torch.sum(scoreOfItemBasedPath))
            # import pdb; pdb.set_trace()
            ####################### 以问句为核心 ###########################
            itemOfBatch = torch.stack(itemOfBatch).view(-1, newSpansBatch[i][-1][0])
            itemOfBatch = itemOfBatch + itemOfBatchPath
            # import pdb; pdb.set_trace()
            # avgScore = torch.mean(itemOfBatch, 1).view(-1, 1)
            #### 以问句为基准计算得分 #########
            scoreOfItem = torch.max(itemOfBatch, 0)[0]
            logData.append(scoreOfItem)
            avgScore = torch.mean(scoreOfItem)
            # selfWeights = torch.relu(scoreOfItem - avgScore)
            selfWeights = torch.softmax(scoreOfItem - avgScore, 0) # 在上面归一化后这里的概率会很接近
            # import pdb; pdb.set_trace()
            # scoreOfItem = scoreOfItem * selfWeights * (1.0 - 1.0 / 100000.0 * (alignKeyWords[-1][0] - 1)) # 对三元组长度做惩罚，5.0是惩罚因子
            # scoreOfItem = scoreOfItem * (1.0 - 1.0 / 5.0 * (alignKeyWords[-1][0] - 1)) # 对三元组长度做惩罚，5.0是惩罚因子
            scoreOfItem = scoreOfItem * selfWeights
            # scoreOfItem = scoreOfItem
            # import pdb; pdb.set_trace()
            #### 以路径为基准计算得分 #########
            # scoreOfPathItem = torch.max(itemOfBatch, 1)[0]
            # score = torch.sum(scoreOfItem) + torch.sum(scoreOfPathItem)
            score = torch.sum(scoreOfItem)
            # score = torch.sum(scoreOfItem) + torch.sum(scoreOfItemBasedPath)
            scores.append(score)
            # scores.append(score * 10)
            ##################################
        logits = torch.stack(scores).view(-1, 1)
        # logits = self.classifier(logits)
        # import pdb; pdb.set_trace()
        return logits, logData
    
class BertForSequenceClassificationWordSimMaxSumContinuousQueSegSoftmax(BertPreTrainedModel):
   
    def __init__(self, config, num_labels):
        super(BertForSequenceClassificationWordSimMaxSumContinuousQueSegSoftmax, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifierCLS = nn.Linear(config.hidden_size, 1)
        self.classifier = nn.Linear(1, 1)
        self.scaleWeight = nn.Parameter(torch.ones(1))
        # self.scaleScore = nn.Linear(num_labels, num_labels)
        self.apply(self.init_bert_weights)
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, alignPositions=None, \
                alignKeyWordsIndex = None, que_spans = None):
        outputEmbs, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        batch, alignNum = alignPositions.shape
        maskValue = torch.tensor(0.0).to(input_ids.device)
        hidden_dim = outputEmbs.shape[-1]
        queEmbs = outputEmbs[:, alignPositions[0][0]: alignPositions[0][1]]
        queLen = queEmbs.shape[1]
        pathEmbs = outputEmbs[:, alignPositions[0][2]:]
        spans = que_spans
        # import pdb; pdb.set_trace()
        # *****************余弦相似度
        # queEmbsCos = queEmbs.unsqueeze(2)
        # pathEmbsCos = pathEmbs.unsqueeze(1)
        # import pdb; pdb.set_trace()
        # cosSim = torch.cosine_similarity(queEmbsCos, pathEmbsCos, dim = 3)
        # # cosSim = cosSim.unsqueeze(0)
        # # print('cosSim:',cosSim.shape, cosSim)
        # softmaxAtt = cosSim
        # ******************注意力dot*****************
        pathEmbs = pathEmbs.transpose(1, 2)
        attentionWeights = torch.bmm(queEmbs, pathEmbs)
        attentionWeights /= math.sqrt(hidden_dim)
        softmaxAtt = torch.nn.functional.softmax(attentionWeights, 2)
        # softmaxAtt = softmaxAtt * self.scaleWeight
        # print(self.scaleWeight, self.scaleWeight.requires_grad, self.scaleWeight.grad)
        # softmaxAtt = attentionWeights
        # ******************************
        # alignKeyWordsIndex = alignKeyWordsIndex[0]
        # import pdb; pdb.set_trace()
        newSpansBatch = spans
        # print('冲突信息构建完成')
        scores = []
        # print(len(alignKeyWordsIndex))
        # import pdb; pdb.set_trace()
        logData = []
        maskTensor = torch.ones(softmaxAtt.shape, requires_grad = False).to(outputEmbs.device)
        for i in range(batch):
            maskTensor[i, :, alignPositions[i][-1] - alignPositions[i][2]:] = 0.0
        softmaxAtt = softmaxAtt * maskTensor
        for i, alignKeyWords in enumerate(alignKeyWordsIndex): # 遍历每个batch
            # import pdb; pdb.set_trace()
            itemOfBatch = []
            itemOfBatchPath = []
            for alignKeyWord in alignKeyWords[0:alignKeyWords[-1][0]]:
                # itemOfBatch.extend([torch.sum(torch.max(softmaxAtt[i, span[0]: span[1], alignKeyWord[0]:alignKeyWord[1]], 1)[0]) / ((span[1] - span[0])) \
                #                         for span in newSpansBatch[i][0: newSpansBatch[i][-1][0]]]) #   
                itemOfBatch.extend([torch.sum(softmaxAtt[i, span[0]: span[1], alignKeyWord[0]:alignKeyWord[1]]) / ((span[1] - span[0])) \
                                        for span in newSpansBatch[i][0: newSpansBatch[i][-1][0]]]) #   
                # itemOfBatch.extend([torch.sum(torch.mean(softmaxAtt[i, span[0]: span[1], alignKeyWord[0]:alignKeyWord[1]], 1)) / ((span[1] - span[0])) \
                #                         for span in newSpansBatch[i][0: newSpansBatch[i][-1][0]]]) #   
                # itemOfBatchPath.extend([torch.sum(torch.max(softmaxAtt[i, span[0]: span[1], alignKeyWord[0]:alignKeyWord[1]], 0)[0]) / ((alignKeyWord[1] - alignKeyWord[0])) \
                #                         for span in newSpansBatch[i][0: newSpansBatch[i][-1][0]]]) #  
                # import pdb; pdb.set_trace() 
                # itemOfBatch.extend([torch.sum(torch.max(softmaxAtt[i, span[0]: span[1], alignKeyWord[0]:alignKeyWord[1]], 1)[0]) \
                #                         for span in newSpansBatch[i][0: newSpansBatch[i][-1][0]]]) #    
            # itemOfBatch = torch.stack(itemOfBatch).view(-1, len(newSpansBatch[i]))
            ############################ 以查询图序列为核心 ############################
            # itemOfBatchPath = torch.stack(itemOfBatchPath).view(-1, newSpansBatch[i][-1][0])
            # scoreOfItemBasedPath = torch.max(itemOfBatchPath, 1)[0]
            # logData.append(scoreOfItemBasedPath)
            # scoreOfItemBasedPathForQue = torch.max(itemOfBatchPath, 0)[0]
            # # scores.append(torch.sum(scoreOfItemBasedPath) + torch.sum(scoreOfItemBasedPathForQue))
            # scores.append(torch.sum(scoreOfItemBasedPath))
            # import pdb; pdb.set_trace()
            ####################### 以问句为核心 ###########################
            itemOfBatch = torch.stack(itemOfBatch).view(-1, newSpansBatch[i][-1][0])
            # import pdb; pdb.set_trace()
            # avgScore = torch.mean(itemOfBatch, 1).view(-1, 1)
            #### 以问句为基准计算得分 #########
            scoreOfItem = torch.max(itemOfBatch, 0)[0]
            logData.append(scoreOfItem)
            avgScore = torch.mean(scoreOfItem)
            # selfWeights = torch.relu(scoreOfItem - avgScore)
            selfWeights = torch.softmax(scoreOfItem - avgScore, 0) # 在上面归一化后这里的概率会很接近
            # import pdb; pdb.set_trace()
            # scoreOfItem = scoreOfItem * selfWeights * (1.0 - 1.0 / 100000.0 * (alignKeyWords[-1][0] - 1)) # 对三元组长度做惩罚，5.0是惩罚因子
            # scoreOfItem = scoreOfItem * (1.0 - 1.0 / 5.0 * (alignKeyWords[-1][0] - 1)) # 对三元组长度做惩罚，5.0是惩罚因子
            # scoreOfItem = scoreOfItem * selfWeights
            # scoreOfItem = scoreOfItem
            # import pdb; pdb.set_trace()
            #### 以路径为基准计算得分 #########
            # scoreOfPathItem = torch.max(itemOfBatch, 1)[0]
            # score = torch.sum(scoreOfItem) + torch.sum(scoreOfPathItem)
            score = torch.sum(scoreOfItem)
            scores.append(score * 50)
            # scores.append(score * 10)
            ##################################
        logits = torch.stack(scores).view(-1, 1)
        # logits = self.classifier(logits)
        # import pdb; pdb.set_trace()
        return logits, logData
    
class BertForSequenceClassificationWordSimMaxSumContinuousQueSegBasedPath(BertPreTrainedModel):
   
    def __init__(self, config, num_labels):
        super(BertForSequenceClassificationWordSimMaxSumContinuousQueSegBasedPath, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifierCLS = nn.Linear(config.hidden_size, 1)
        self.classifier = nn.Linear(1, 1)
        self.scaleWeight = nn.Parameter(torch.ones(1))
        # self.scaleScore = nn.Linear(num_labels, num_labels)
        self.apply(self.init_bert_weights)

    def score1v1(self, scores, positions):
        # import pdb; pdb.set_trace()
        newScores = []
        pos2scores = {}
        for i, item in enumerate(scores):
            pos = int(positions[i])
            if(pos not in pos2scores):
                pos2scores[pos] = []
            pos2scores[pos].append(item)
        for pos in pos2scores:
            newScores.append(max(pos2scores[pos]))
        newScores = torch.stack(newScores)
        # import pdb; pdb.set_trace()
        return newScores
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, alignPositions=None, \
                alignKeyWordsIndex = None, que_spans = None):
        outputEmbs, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        # outputEmbs, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True)
        # outputEmbs = outputEmbs[-12]
        # import pdb; pdb.set_trace()
        batch, alignNum = alignPositions.shape
        maskValue = torch.tensor(0.0).to(input_ids.device)
        hidden_dim = outputEmbs.shape[-1]
        queEmbs = outputEmbs[:, alignPositions[0][0]: alignPositions[0][1]]
        queLen = queEmbs.shape[1]
        pathEmbs = outputEmbs[:, alignPositions[0][2]:]
        spans = que_spans
        # import pdb; pdb.set_trace()
        # *****************余弦相似度
        # queEmbsCos = queEmbs.unsqueeze(2)
        # pathEmbsCos = pathEmbs.unsqueeze(1)
        # cosSim = torch.cosine_similarity(queEmbsCos, pathEmbsCos, dim = 3)
        # # cosSim = cosSim.unsqueeze(0)
        # # print('cosSim:',cosSim.shape, cosSim)
        # softmaxAtt = cosSim
        # import pdb; pdb.set_trace()
        # ******************注意力dot*****************
        pathEmbs = pathEmbs.transpose(1, 2)
        attentionWeights = torch.bmm(queEmbs, pathEmbs)
        # import pdb; pdb.set_trace()
        attentionWeights /= math.sqrt(hidden_dim)
        # softmaxAtt = torch.nn.functional.softmax(attentionWeights, 1)
        # softmaxAtt = softmaxAtt * self.scaleWeight
        # print(self.scaleWeight, self.scaleWeight.requires_grad, self.scaleWeight.grad)
        softmaxAtt = attentionWeights
        # ******************************
        # alignKeyWordsIndex = alignKeyWordsIndex[0]
        # import pdb; pdb.set_trace()
        newSpansBatch = spans
        # print('冲突信息构建完成')
        scores = []
        # print(len(alignKeyWordsIndex))
        # import pdb; pdb.set_trace()
        logData = []
        # maskTensor = torch.ones(softmaxAtt.shape, requires_grad = False).to(outputEmbs.device)
        # for i in range(batch):
        #     maskTensor[i, :, alignPositions[i][-1] - alignPositions[i][2]:] = 0.0
        # softmaxAtt = softmaxAtt * maskTensor
        for i, alignKeyWords in enumerate(alignKeyWordsIndex): # 遍历每个batch
            # import pdb; pdb.set_trace()
            itemOfBatch = []
            itemOfBatchPath = []
            itemOfBatchPathAlign = []
            for alignKeyWord in alignKeyWords[0:alignKeyWords[-1][0]]:
                # 基于问句计算得分a
                # itemOfBatchPathAlign.extend([torch.sum(torch.max(softmaxAtt[i, span[0]: span[1], alignKeyWord[0]:alignKeyWord[1]], 1)[0]) / ((span[1] - span[0])) \
                #                         for span in newSpansBatch[i][0: newSpansBatch[i][-1][0]]]) #
                # itemOfBatch.extend([torch.sum(torch.max(softmaxAtt[i, span[0]: span[1], alignKeyWord[0]:alignKeyWord[1]], 1)[0]) \
                #                         for span in newSpansBatch[i][0: newSpansBatch[i][-1][0]]]) #    
                # itemOfBatch.extend([torch.sum(softmaxAtt[i, span[0]: span[1], alignKeyWord[0]:alignKeyWord[1]]) / ((span[1] - span[0])) \
                #                         for span in newSpansBatch[i][0: newSpansBatch[i][-1][0]]]) #   
                # itemOfBatch.extend([torch.sum(torch.mean(softmaxAtt[i, span[0]: span[1], alignKeyWord[0]:alignKeyWord[1]], 1)) / ((span[1] - span[0])) \
                #                         for span in newSpansBatch[i][0: newSpansBatch[i][-1][0]]]) #   
                # 基于查询图序列计算得分
                itemOfBatchPath.extend([torch.sum(torch.max(softmaxAtt[i, span[0]: span[1], alignKeyWord[0]:alignKeyWord[1]], 0)[0]) / ((alignKeyWord[1] - alignKeyWord[0])) \
                                        for span in newSpansBatch[i][0: newSpansBatch[i][-1][0]]]) #  
                itemOfBatchPathAlign.extend([torch.sum(torch.max(softmaxAtt[i, span[0]: span[1], alignKeyWord[0]:alignKeyWord[1]], 0)[0]) \
                                        for span in newSpansBatch[i][0: newSpansBatch[i][-1][0]]]) #  
                # itemOfBatchPath.extend([torch.sum(torch.max(softmaxAtt[i, span[0]: span[1], alignKeyWord[0]:alignKeyWord[1]], 0)[0]) \
                #                         for span in newSpansBatch[i][0: newSpansBatch[i][-1][0]]]) #  
                # import pdb; pdb.set_trace() 
                # itemOfBatch.extend([torch.sum(torch.max(softmaxAtt[i, span[0]: span[1], alignKeyWord[0]:alignKeyWord[1]], 1)[0]) \
                #                         for span in newSpansBatch[i][0: newSpansBatch[i][-1][0]]]) #    
            # itemOfBatch = torch.stack(itemOfBatch).view(-1, len(newSpansBatch[i]))
            ############################ 以查询图序列为核心 ############################
            itemOfBatchPathAlign = torch.stack(itemOfBatchPathAlign).view(-1, newSpansBatch[i][-1][0])
            itemOfBatchPath = torch.stack(itemOfBatchPath).view(-1, newSpansBatch[i][-1][0])
            maxResultsBasedPath = torch.max(itemOfBatchPathAlign, 1)
            # import pdb; pdb.set_trace()
            # scoreOfItemBasedPath = maxResultsBasedPath[0]
            scoreOfItemBasedPath = torch.gather(itemOfBatchPath, 1, maxResultsBasedPath[1].view(-1,1)).view(-1)
            logData.append(torch.cat((scoreOfItemBasedPath, maxResultsBasedPath[1]), 0))
            scoreOfItemBasedPathForQue = torch.max(itemOfBatchPath, 0)[0]
            # scores.append(torch.sum(scoreOfItemBasedPath) + torch.sum(scoreOfItemBasedPathForQue))
            scoreOfItemBasedPath = self.score1v1(scoreOfItemBasedPath, maxResultsBasedPath[1])
            scores.append(torch.sum(scoreOfItemBasedPath))
            # import pdb; pdb.set_trace()
            ####################### 以问句为核心 ###########################
            # itemOfBatch = torch.stack(itemOfBatch).view(-1, newSpansBatch[i][-1][0])
            # # import pdb; pdb.set_trace()
            # # avgScore = torch.mean(itemOfBatch, 1).view(-1, 1)
            # #### 以问句为基准计算得分 #########
            # maxResults = torch.max(itemOfBatch, 0)
            # scoreOfItem = maxResults[0]
            # logData.append(torch.cat((maxResults[0], maxResults[1]), 0))
            # # avgScore = torch.mean(scoreOfItem)
            # # selfWeights = torch.relu(scoreOfItem - avgScore)
            # # selfWeights = torch.softmax(scoreOfItem - avgScore, 0) # 在上面归一化后这里的概率会很接近
            # # import pdb; pdb.set_trace()
            # # scoreOfItem = scoreOfItem * selfWeights * (1.0 - 1.0 / 100000.0 * (alignKeyWords[-1][0] - 1)) # 对三元组长度做惩罚，5.0是惩罚因子
            # # scoreOfItem = scoreOfItem * (1.0 - 1.0 / 5.0 * (alignKeyWords[-1][0] - 1)) # 对三元组长度做惩罚，5.0是惩罚因子
            # # scoreOfItem = scoreOfItem * selfWeights
            # scoreOfItem = scoreOfItem
            # # import pdb; pdb.set_trace()
            # #### 以路径为基准计算得分 #########
            # # scoreOfPathItem = torch.max(itemOfBatch, 1)[0]
            # # score = torch.sum(scoreOfItem) + torch.sum(scoreOfPathItem)
            # score = torch.sum(scoreOfItem)
            # scores.append(score)
            # # scores.append(score * 10)
            ##################################
        logits = torch.stack(scores).view(-1, 1)
        # logits = self.classifier(logits)
        # import pdb; pdb.set_trace()
        return logits, logData
    
class BertForSequenceClassificationWordSimMaxSumContinuousQueSeg(BertPreTrainedModel):
   
    def __init__(self, config, num_labels):
        super(BertForSequenceClassificationWordSimMaxSumContinuousQueSeg, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifierCLS = nn.Linear(config.hidden_size, 1)
        self.classifier = nn.Linear(1, 1)
        self.scaleWeight = nn.Parameter(torch.ones(1))
        # self.scaleScore = nn.Linear(num_labels, num_labels)
        self.apply(self.init_bert_weights)
    
    def isConflict(self, currentId, usedIds, conflict):
        for usedId in usedIds:
            if(currentId in conflict[usedId]):
                return True
        return False
    
    def recur(self, examples, conflict, i = 0, path = [], currentScores = [], maxPath = [0]):
        if(i >= len(examples)):
            # import pdb; pdb.set_trace()
            currentSum = sum(currentScores)
            if(currentSum > maxPath[0]):
                # print(currentScores)
                maxPath[0] = currentSum
            return 0
        for j, score in enumerate(examples[i]):
            if(not self.isConflict(j, path, conflict)):
                # used.append(j)
                path.append(j)
                currentScores.append(score)
                self.recur(examples, conflict, i + 1, path, currentScores, maxPath)
                path.pop()
                currentScores.pop()
        return maxPath[0]
    
    def getIndexs(self, casesTensor, spanPathNums):
        if(spanPathNums == 1):
            indexs = (casesTensor[:, 0])
        elif(spanPathNums == 2):
            indexs = (casesTensor[:, 0], casesTensor[:, 1])
        elif(spanPathNums == 3):
            indexs = (casesTensor[:, 0], casesTensor[:, 1], casesTensor[:, 2])
        else:
            indexs = (casesTensor[:, 0], casesTensor[:, 1], casesTensor[:, 2], casesTensor[:, 3])
        # elif(spanPathNums == 4):
        #     indexs = (casesTensor[:, 0], casesTensor[:, 1], casesTensor[:, 2], casesTensor[:, 3])
        # elif(spanPathNums == 5):
        #     indexs = (casesTensor[:, 0], casesTensor[:, 1], casesTensor[:, 2], casesTensor[:, 3], casesTensor[:, 4])
        # elif(spanPathNums == 6):
        #     indexs = (casesTensor[:, 0], casesTensor[:, 1], casesTensor[:, 2], casesTensor[:, 3], casesTensor[:, 4], casesTensor[:, 5])
        # else:
        #     indexs = (casesTensor[:, 0], casesTensor[:, 1], casesTensor[:, 2], casesTensor[:, 3], casesTensor[:, 4], casesTensor[:, 5], casesTensor[:, 6])
        return indexs
    
    def getCombineMarix(self, spanScores, spanPathNums):
        if(spanPathNums == 1):
            combineMatrix = spanScores[0].view(-1, 1)
            # matrix2 = torch.tensor(spanScores[1], dtype = torch.float).view(1, -1)
            # combineMatrix = combineMatrix + matrix2
        elif(spanPathNums == 2):
            combineMatrix = spanScores[0].view(-1, 1)
            matrix2 = spanScores[1].view(1, -1)
            combineMatrix = combineMatrix + matrix2
        elif(spanPathNums == 3):
            combineMatrix = spanScores[0].view(-1, 1, 1)
            matrix2 = spanScores[1].view(1, -1, 1)
            matrix3 = spanScores[2].view(1, 1, -1)
            combineMatrix = combineMatrix + matrix2 + matrix3
        else:
            combineMatrix = spanScores[0].view(-1, 1, 1, 1)
            matrix2 = spanScores[1].view(1, -1, 1, 1)
            matrix3 = spanScores[2].view(1, 1, -1, 1)
            matrix4 = spanScores[3].view(1, 1, 1, -1)
            combineMatrix = combineMatrix + matrix2 + matrix3 + matrix4
        # import pdb; pdb.set_trace()
        return combineMatrix

    
    def score1v1(self, scores, positions):
        # import pdb; pdb.set_trace()
        newScores = []
        pos2scores = {}
        for i, item in enumerate(scores):
            pos = int(positions[i])
            if(pos not in pos2scores):
                pos2scores[pos] = []
            pos2scores[pos].append(item)
        for pos in pos2scores:
            newScores.append(max(pos2scores[pos]))
        newScores = torch.stack(newScores)
        # import pdb; pdb.set_trace()
        return newScores

    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, alignPositions=None, \
                alignKeyWordsIndex = None, que_spans = None):
        outputEmbs, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        batch, alignNum = alignPositions.shape
        maskValue = torch.tensor(0.0).to(input_ids.device)
        hidden_dim = outputEmbs.shape[-1]
        queEmbs = outputEmbs[:, alignPositions[0][0]: alignPositions[0][1]]
        queLen = queEmbs.shape[1]
        pathEmbs = outputEmbs[:, alignPositions[0][2]:]
        spans = que_spans
        # import pdb; pdb.set_trace()
        # *****************余弦相似度
        # queEmbsCos = queEmbs.unsqueeze(2)
        # pathEmbsCos = pathEmbs.unsqueeze(1)
        # # import pdb; pdb.set_trace()
        # cosSim = torch.cosine_similarity(queEmbsCos, pathEmbsCos, dim = 3)
        # # cosSim = cosSim.unsqueeze(0)
        # # print('cosSim:',cosSim.shape, cosSim)
        # softmaxAtt = cosSim
        # ******************注意力dot*****************
        pathEmbs = pathEmbs.transpose(1, 2)
        attentionWeights = torch.bmm(queEmbs, pathEmbs)
        attentionWeights /= math.sqrt(hidden_dim)
        # softmaxAtt = torch.nn.functional.softmax(attentionWeights, 2)
        # softmaxAtt = softmaxAtt * self.scaleWeight
        # print(self.scaleWeight, self.scaleWeight.requires_grad, self.scaleWeight.grad)
        softmaxAtt = attentionWeights
        # ******************************
        # alignKeyWordsIndex = alignKeyWordsIndex[0]
        # import pdb; pdb.set_trace()
        newSpansBatch = spans
        # print('冲突信息构建完成')
        scores = []
        # print(len(alignKeyWordsIndex))
        # import pdb; pdb.set_trace()
        logData = []
        # maskTensor = torch.ones(softmaxAtt.shape, requires_grad = False).to(outputEmbs.device)
        # for i in range(batch):
        #     maskTensor[i, :, alignPositions[i][-1] - alignPositions[i][2]:] = 0.0
        # softmaxAtt = softmaxAtt * maskTensor
        for i, alignKeyWords in enumerate(alignKeyWordsIndex): # 遍历每个batch
            # import pdb; pdb.set_trace()
            itemOfBatch = []
            itemOfBatchPath = []
            for alignKeyWord in alignKeyWords[0:alignKeyWords[-1][0]]:
                ##### 词与词之间的相似度用字相似度加权得到
                # temp = []
                # for span in newSpansBatch[i][0: newSpansBatch[i][-1][0]]:
                #     wordSoftmax = torch.softmax(softmaxAtt[i, span[0]: span[1], alignKeyWord[0]:alignKeyWord[1]], 1)
                #     wordsim = torch.sum(torch.sum(wordSoftmax * softmaxAtt[i, span[0]: span[1], alignKeyWord[0]:alignKeyWord[1]], 1)) / ((span[1] - span[0]))
                #     # import pdb; pdb.set_trace()
                #     itemOfBatch.append(wordsim)
                #     # import pdb; pdb.set_trace()
                itemOfBatch.extend([torch.sum(torch.max(softmaxAtt[i, span[0]: span[1], alignKeyWord[0]:alignKeyWord[1]], 1)[0]) / ((span[1] - span[0])) \
                                        for span in newSpansBatch[i][0: newSpansBatch[i][-1][0]]]) #  
                # itemOfBatch.extend([torch.sum(torch.max(softmaxAtt[i, span[0]: span[1], alignKeyWord[0]:alignKeyWord[1]], 1)[0]) \
                #                         for span in newSpansBatch[i][0: newSpansBatch[i][-1][0]]]) #    
                # itemOfBatch.extend([torch.sum(softmaxAtt[i, span[0]: span[1], alignKeyWord[0]:alignKeyWord[1]]) / ((span[1] - span[0])) \
                #                         for span in newSpansBatch[i][0: newSpansBatch[i][-1][0]]]) #   
                # itemOfBatch.extend([torch.sum(torch.mean(softmaxAtt[i, span[0]: span[1], alignKeyWord[0]:alignKeyWord[1]], 1)) / ((span[1] - span[0])) \
                #                         for span in newSpansBatch[i][0: newSpansBatch[i][-1][0]]]) #   
                # itemOfBatchPath.extend([torch.sum(torch.max(softmaxAtt[i, span[0]: span[1], alignKeyWord[0]:alignKeyWord[1]], 0)[0]) / ((alignKeyWord[1] - alignKeyWord[0])) \
                #                         for span in newSpansBatch[i][0: newSpansBatch[i][-1][0]]]) #  
                # import pdb; pdb.set_trace() 
                # itemOfBatch.extend([torch.sum(torch.max(softmaxAtt[i, span[0]: span[1], alignKeyWord[0]:alignKeyWord[1]], 1)[0]) \
                #                         for span in newSpansBatch[i][0: newSpansBatch[i][-1][0]]]) #    
            # itemOfBatch = torch.stack(itemOfBatch).view(-1, len(newSpansBatch[i]))
            ############################ 以查询图序列为核心 ############################
            # itemOfBatchPath = torch.stack(itemOfBatchPath).view(-1, newSpansBatch[i][-1][0])
            # scoreOfItemBasedPath = torch.max(itemOfBatchPath, 1)[0]
            # logData.append(scoreOfItemBasedPath)
            # scoreOfItemBasedPathForQue = torch.max(itemOfBatchPath, 0)[0]
            # # scores.append(torch.sum(scoreOfItemBasedPath) + torch.sum(scoreOfItemBasedPathForQue))
            # scores.append(torch.sum(scoreOfItemBasedPath))
            # import pdb; pdb.set_trace()
            ####################### 以问句为核心 ###########################
            itemOfBatch = torch.stack(itemOfBatch).view(-1, newSpansBatch[i][-1][0])
            # seqItemWeights = torch.softmax(itemOfBatch, 0) # 问句中每个词的表示由加权查询图序列得到
            # scoreOfItemWeight = torch.sum(seqItemWeights * itemOfBatch, 0) # 加权后的得分
            # import pdb; pdb.set_trace()
            # avgScore = torch.mean(itemOfBatch, 1).view(-1, 1)
            #### 以问句为基准计算得分 #########
            maxResults = torch.max(itemOfBatch, 0)
            minResults = torch.min(itemOfBatch, 0)[0]
            # scoreOfItem = maxResults[0] - minResults
            scoreOfItem = maxResults[0]
            scoreOfItem1v1 = self.score1v1(maxResults[0], maxResults[1])
            # mask2 = (torch.min(torch.softmax(itemOfBatch, 0), 0)[0] < 0.01).long()
            # mask2 = torch.softmax(maxResults[0], 0) * newSpansBatch[i][-1][0]
            # mask2 = torch.softmax(maxResults[0], 0)
            mask2 = maxResults[0] / torch.sum(maxResults[0]) # 线性归一化权值
            # threshold = 0.1
            # mask3 = torch.relu(mask2 - threshold) / (mask2 - threshold + 0.000001) # 掩码掉得分较低的
            # scoreOfItem = scoreOfItemWeight
            # scoreOfItem = scoreOfItemWeight * mask2
            scoreOfItem = scoreOfItem
            # import pdb; pdb.set_trace()
            # scoreOfItem = (((scoreOfItem - minResults) - 2) > 0).long() * scoreOfItem
            logData.append(torch.cat((maxResults[0], maxResults[1], alignKeyWords[-1][0].view(-1)), 0))
            # avgScore = torch.mean(scoreOfItem)
            # selfWeights = torch.relu(scoreOfItem - avgScore)
            # selfWeights = torch.softmax(scoreOfItem - avgScore, 0) # 在上面归一化后这里的概率会很接近
            
            # import pdb; pdb.set_trace()
            # scoreOfItem = scoreOfItem * selfWeights * (1.0 - 1.0 / 100000.0 * (alignKeyWords[-1][0] - 1)) # 对三元组长度做惩罚，5.0是惩罚因子
            # scoreOfItem = scoreOfItem * (1.0 - 1.0 / 5.0 * (alignKeyWords[-1][0] - 1)) # 对三元组长度做惩罚，5.0是惩罚因子
            # scoreOfItem = scoreOfItem * selfWeights
            scoreOfItem = scoreOfItem
            # import pdb; pdb.set_trace()
            #### 以路径为基准计算得分 #########
            # scoreOfPathItem = torch.max(itemOfBatch, 1)[0]
            # score = torch.sum(scoreOfItem) + torch.sum(scoreOfPathItem)
            # score = torch.sum(scoreOfItem) * len(scoreOfItem1v1)
            score = torch.sum(scoreOfItem1v1) * (len(scoreOfItem1v1) / (alignKeyWords[-1][0] * 1.0))
            # if((len(scoreOfItem1v1) * 1.0 / alignKeyWords[-1][0]) < 1):
            #     import pdb; pdb.set_trace()
            scores.append(score)
            # scores.append(score * 10)
            ##################################
        logits = torch.stack(scores).view(-1, 1)
        # logits = self.classifier(logits)
        # import pdb; pdb.set_trace()
        return logits, logData
    
    # 对应的关系不能冲突
    # def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, alignPositions=None, \
    #             alignKeyWordsIndex = None, que_spans = None):
    #     outputEmbs, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
    #     batch, alignNum = alignPositions.shape
    #     maskValue = torch.tensor(0.0).to(input_ids.device)
    #     hidden_dim = outputEmbs.shape[-1]
    #     queEmbs = outputEmbs[:, alignPositions[0][0]: alignPositions[0][1]]
    #     queLen = queEmbs.shape[1]
    #     pathEmbs = outputEmbs[:, alignPositions[0][2]:]
    #     spans = que_spans
        
    #     # *****************余弦相似度
    #     # queEmbsCos = queEmbs.unsqueeze(2)
    #     # pathEmbsCos = pathEmbs.unsqueeze(1)
    #     # import pdb; pdb.set_trace()
    #     # cosSim = torch.cosine_similarity(queEmbsCos, pathEmbsCos, dim = 3)
    #     # # cosSim = cosSim.unsqueeze(0)
    #     # # print('cosSim:',cosSim.shape, cosSim)
    #     # softmaxAtt = cosSim
    #     # ******************注意力dot*****************
    #     pathEmbs = pathEmbs.transpose(1, 2)
    #     attentionWeights = torch.bmm(queEmbs, pathEmbs)
    #     attentionWeights /= math.sqrt(hidden_dim)
    #     softmaxAtt = torch.nn.functional.softmax(attentionWeights, 2)
    #     # softmaxAtt = softmaxAtt * self.scaleWeight
    #     # print(self.scaleWeight, self.scaleWeight.requires_grad, self.scaleWeight.grad)
    #     # softmaxAtt = attentionWeights
    #     # ******************************
    #     # alignKeyWordsIndex = alignKeyWordsIndex[0]
    #     # import pdb; pdb.set_trace()
    #     spanScoresTemp = []
    #     batchSpans = []
    #     # print(spans.shape)
    #     newSpansBatch = spans
    #     conflictBatch = []
    #     for newSpans in newSpansBatch:
    #         conflict = {}
    #         for i_span, span1 in enumerate(newSpans):
    #             conflict[i_span] = []
    #             for j, span2 in enumerate(newSpans):
    #                 if(span2[1] - span2[0] == 0):
    #                     continue
    #                 if((span2[0] >= span1[0] and span2[0] < span1[1]) or (span2[1] > span1[0] and span2[1] <= span1[1])):
    #                     conflict[i_span].append(j)
    #             # print('conflict:', len(conflict[i_span]))
    #         conflictBatch.append(conflict)
    #     # print('冲突信息构建完成')
    #     scores = []
    #     # print(len(alignKeyWordsIndex))
    #     # import pdb; pdb.set_trace()
    #     for i, alignKeyWords in enumerate(alignKeyWordsIndex): # 遍历每个batch
    #         itemOfBatch = []
    #         for alignKeyWord in alignKeyWords[0:alignKeyWords[-1][0]]:
    #             itemOfBatch.extend([torch.sum(softmaxAtt[i, span[0]: span[1], alignKeyWord[0]:alignKeyWord[1]]) \
    #                             for span in newSpansBatch[i]])
    #         # import pdb; pdb.set_trace()
    #         itemOfBatch = torch.stack(itemOfBatch).view(-1, len(newSpansBatch[i]))
    #         spanPathNums = min(itemOfBatch.shape[0], 6)
    #         cases = []
    #         # import pdb; pdb.set_trace()
    #         conflict = conflictBatch[i]
    #         for key in conflict:
    #             if(len(conflict[key]) != 0):
    #                 for iPosition in range(spanPathNums):
    #                     temp = [[]]
    #                     for ith in range(spanPathNums - 1):
    #                         if(ith == iPosition):
    #                             temp = [tempItem + [key] for tempItem in temp]
    #                         temp = [tempItem + [item] for tempItem in temp for item in conflict[key]]
    #                     if(iPosition == spanPathNums - 1):
    #                         temp = [tempItem + [key] for tempItem in temp]
    #                     cases.extend(temp)
    #                     # print(len(conflict[key]), len(temp), len(cases))
    #         # import pdb; pdb.set_trace()
    #         # print('batch item 完成一个得分计算')
    #         casesTensor = torch.tensor(cases, dtype=torch.long, requires_grad = False)
    #         indexs = self.getIndexs(casesTensor, spanPathNums)
    #         itemOfBatch = itemOfBatch
    #         combineMatrix = self.getCombineMarix(itemOfBatch, spanPathNums).to(input_ids.device)
    #         # import pdb; pdb.set_trace()
    #         if(spanPathNums > 1):
    #             combineMatrix.index_put_(indexs, maskValue)
    #         score = torch.max(combineMatrix)
    #         scores.append(score)
    #     # logits = torch.tensor(scores, dtype = torch.float, requires_grad = True).to(input_ids.device) * self.scaleWeight
    #     logits = torch.stack(scores).view(-1, 1)
    #     # logits = self.classifier(logits)
    #     # import pdb; pdb.set_trace()
    #     return logits    

    
class BertForSequenceClassificationWordSimMaxSpan(BertPreTrainedModel):
   
    def __init__(self, config, num_labels):
        super(BertForSequenceClassificationWordSimMaxSpan, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifierCLS = nn.Linear(config.hidden_size, 1)
        # self.classifier = nn.Linear(1, 1)
        self.scaleWeight = nn.Parameter(torch.ones(1))
        # self.scaleScore = nn.Linear(num_labels, num_labels)
        self.apply(self.init_bert_weights)
        

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, alignPositions=None, alignKeyWordsIndex = None):
        outputEmbs, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        batch, alignNum = alignPositions.shape
        maskValue = torch.tensor(0.0).to(input_ids.device)
        hidden_dim = outputEmbs.shape[-1]
        queEmbs = outputEmbs[:, alignPositions[0][0]: alignPositions[0][1]]
        queLen = queEmbs.shape[1]
        pathEmbs = outputEmbs[:, alignPositions[0][2]:]
        # spans = [(begin, end) for begin in range(queLen) for end in range(begin + 1, min(begin + 5, queLen))][0: 100]
        
        # *****************余弦相似度
        # queEmbsCos = queEmbs.unsqueeze(2)
        # pathEmbsCos = pathEmbs.unsqueeze(1)
        # # import pdb; pdb.set_trace()
        # cosSim = torch.cosine_similarity(queEmbsCos, pathEmbsCos, dim = 3)
        # # cosSim = cosSim.unsqueeze(0)
        # # print('cosSim:',cosSim.shape, cosSim)
        # softmaxAtt = cosSim
        # ******************注意力dot*****************
        pathEmbs = pathEmbs.transpose(1, 2)
        attentionWeights = torch.bmm(queEmbs, pathEmbs)
        attentionWeights /= math.sqrt(hidden_dim)
        # softmaxAtt = torch.nn.functional.softmax(attentionWeights, 2)
        softmaxAtt = attentionWeights
        # ******************************
        # alignKeyWordsIndex = alignKeyWordsIndex[0]
        # print(outputEmbs.shape)
        # print(len(alignKeyWordsIndex), len(alignKeyWordsIndex[0]), len(alignKeyWordsIndex[0][0]))
        # softmaxAtt2 = Variable(softmaxAtt, requires_grad=False)
        # softmaxAtt2 = softmaxAtt
        maskTensor = torch.ones(softmaxAtt.shape, requires_grad = False).to(outputEmbs.device)
        for i, alignKeyWordsItem in enumerate(alignKeyWordsIndex):
            maskTensor[i, :, 0:alignKeyWordsItem[0][0]] = maskValue
            length = alignKeyWordsItem[-1][0]
            # print(alignKeyWordsItem.shape, length, alignKeyWordsItem)
            try:
                for j in range(length - 1):
                    maskTensor[i, :, alignKeyWordsItem[j][1]: alignKeyWordsItem[j + 1][0]] = maskValue
                maskTensor[i, :, alignKeyWordsItem[length - 1][1]:] = maskValue
            except:
                import pdb; pdb.set_trace()
        softmaxAtt = softmaxAtt * maskTensor
            
        # import pdb; pdb.set_trace()
        maxAtt = torch.max(softmaxAtt, 2)[0]
        logits = torch.sum(maxAtt.view(batch, -1), 1).view(-1, 1)
        # logits = self.classifier(logits)
        # logits = logits * self.scaleWeight
        # import pdb; pdb.set_trace()
        return logits
    
class BertForSequenceClassificationWordSimMaxSumGlobal(BertPreTrainedModel):
   
    def __init__(self, config, num_labels):
        super(BertForSequenceClassificationWordSimMaxSumGlobal, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size * 2, num_labels)
        self.classifier = nn.Linear(1, 1)
        # self.scaleScore = nn.Linear(num_labels, num_labels)
        self.apply(self.init_bert_weights)  

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, alignPositions=None):
        outputEmbs, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        batch, alignNum = alignPositions.shape
        batch, queLen, hidden_dim = outputEmbs.shape
        queEmbs = outputEmbs
        pathEmbs = outputEmbs
        # import pdb; pdb.set_trace()
        # *****************余弦相似度
        # queEmbsCos = queEmbs.unsqueeze(2)
        # pathEmbsCos = pathEmbs.unsqueeze(1)
        # # import pdb; pdb.set_trace()
        # cosSim = torch.cosine_similarity(queEmbsCos, pathEmbsCos, dim = 3)
        # # cosSim = cosSim.unsqueeze(0)
        # # print('cosSim:',cosSim.shape, cosSim)
        # softmaxAtt = cosSim
        # ******************注意力dot*****************
        pathEmbs = pathEmbs.transpose(1, 2)
        attentionWeights = torch.bmm(queEmbs, pathEmbs)
        # import pdb; pdb.set_trace()
        attentionWeights[:, range(queLen), range(queLen)] = -10e7
        attentionWeights /= math.sqrt(hidden_dim)
        # softmaxAtt = torch.nn.functional.softmax(attentionWeights, 2)
        softmaxAtt = attentionWeights
        softmaxAtt = softmaxAtt[:, 0: alignPositions[0][1], alignPositions[0][2]:]
        # ******************************
        # import pdb; pdb.set_trace()
        # self.pathSum(softmaxAtt)
        maxAtt = torch.max(softmaxAtt, 2)[0]
        logits = torch.sum(maxAtt.view(batch, -1), 1).view(-1, 1)
        # logits = self.classifier(logits)
        # import pdb; pdb.set_trace()
        return logits
    
class BertForSequenceClassificationWordSimMatrixLSTM(BertPreTrainedModel):
   
    def __init__(self, config, num_labels):
        super(BertForSequenceClassificationWordSimMatrixLSTM, self).__init__(config)
        self.maxPathLen = 74
        self.num_labels = num_labels
        self.bert = BertModel(config)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size * 2, num_labels)
        self.classifier = nn.Linear(self.maxPathLen, num_labels)
        # self.scaleScore = nn.Linear(num_labels, num_labels)
        self.lstm = torch.nn.LSTM(input_size = self.maxPathLen, hidden_size = self.maxPathLen, num_layers = 2, batch_first = True)
        self.apply(self.init_bert_weights)

    def attention_net(self, queEmbs, contentEmbs):
        # import pdb; pdb.set_trace()
        contentEmbs = contentEmbs.unsqueeze(1)
        queEmbsPermute = queEmbs.transpose(1, 2)
        # import pdb; pdb.set_trace()
        attentionWeights = torch.bmm(contentEmbs, queEmbsPermute)
        attentionWeights /= math.sqrt(queEmbs.shape[2])
        softmaxAttentionWeights = torch.nn.functional.softmax(attentionWeights, 2)
        context = torch.bmm(softmaxAttentionWeights, queEmbs)
        context = context.squeeze(1)
        # import pdb; pdb.set_trace()
        return context
        # import pdb; pdb.set_trace()
        

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, alignPositions=None):
        outputEmbs, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        outputEmbs = outputEmbs.detach()
        batch, alignNum = alignPositions.shape
        hidden_dim = outputEmbs.shape[-1]
        # queLen = torch.max(alignPositions[0][1], 49)
        queEmbs = outputEmbs[:, alignPositions[0][0]: alignPositions[0][1]]
        pathEmbs = outputEmbs[:, alignPositions[0][2]:]
        pathEmbs = pathEmbs[:, 0:self.maxPathLen, :]
        # *****************余弦相似度
        # queEmbsCos = queEmbs.unsqueeze(2)
        # pathEmbsCos = pathEmbs.unsqueeze(1)
        # # import pdb; pdb.set_trace()
        # cosSim = torch.cosine_similarity(queEmbsCos, pathEmbsCos, dim = 3)
        # # cosSim = cosSim.unsqueeze(0)
        # # print('cosSim:',cosSim.shape, cosSim)
        # softmaxAtt = cosSim
        # ******************注意力dot*****************
        pathEmbs = pathEmbs.transpose(1, 2)
        attentionWeights = torch.bmm(queEmbs, pathEmbs)
        attentionWeights /= math.sqrt(hidden_dim)
        
        ############ lstm 得分计算 #####################
        output, (h_n, c_n) = self.lstm(attentionWeights)
        features = h_n.transpose(0, 1)
        features = features[:, 1, :]
        logits = self.classifier(features).view(-1)
        # import pdb; pdb.set_trace()
        ############### 自定义得分计算 #############################
        # # softmaxAtt = torch.nn.functional.softmax(attentionWeights, 1)
        # softmaxAtt = attentionWeights
        # # ******************************
        # # import pdb; pdb.set_trace()
        # # self.pathSum(softmaxAtt)
        # maxAtt = torch.max(softmaxAtt, 2)[0]
        # logits = torch.sum(maxAtt.view(batch, -1), 1)
        # # import pdb; pdb.set_trace()
        return logits
    

class BertForSequenceClassificationWordSimMatrixCNN(BertPreTrainedModel):
   
    def __init__(self, config, num_labels):
        super(BertForSequenceClassificationWordSimMatrixCNN, self).__init__(config)
        self.maxPathLen = 74
        self.num_labels = num_labels
        self.bert = BertModel(config)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size * 2, num_labels)
        self.classifier = nn.Linear(150, num_labels)
        # self.classifierCNN = nn.Linear(self.maxPathLen, num_labels)
        # self.scaleScore = nn.Linear(num_labels, num_labels)
        ######## textcnn ####################
        # self.convs = torch.nn.ModuleList([nn.Conv2d(1, 100, (k, self.maxPathLen)) for k in [1, 2, 3]])
        # self.dropout = nn.Dropout(0.1)
        # self.fc = nn.Linear(3 * 100, 1)
        #####################################
        ##################cnn ######################
        self.convs = torch.nn.ModuleList([nn.Conv2d(1, 1, (k, j))  for k in [1, 2, 3, 4] for j in range(1, k + 3)])
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(266, 1)
        ##########################################
        # self.lstm = torch.nn.LSTM(input_size = self.maxPathLen, hidden_size = self.maxPathLen, num_layers = 2, batch_first = True)
        
        self.apply(self.init_bert_weights)
        

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, alignPositions=None):
        outputEmbs, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        # outputEmbs = outputEmbs.detach()
        batch, alignNum = alignPositions.shape
        hidden_dim = outputEmbs.shape[-1]
        # queLen = torch.max(alignPositions[0][1], 49)
        queEmbs = outputEmbs[:, alignPositions[0][0]: alignPositions[0][1]]
        pathEmbs = outputEmbs[:, alignPositions[0][2]:]
        pathEmbs = pathEmbs[:, 0:self.maxPathLen, :]
        # *****************余弦相似度
        # queEmbsCos = queEmbs.unsqueeze(2)
        # pathEmbsCos = pathEmbs.unsqueeze(1)
        # # import pdb; pdb.set_trace()
        # cosSim = torch.cosine_similarity(queEmbsCos, pathEmbsCos, dim = 3)
        # # cosSim = cosSim.unsqueeze(0)
        # # print('cosSim:',cosSim.shape, cosSim)
        # softmaxAtt = cosSim
        # ******************注意力dot*****************
        pathEmbs = pathEmbs.transpose(1, 2)
        attentionWeights = torch.bmm(queEmbs, pathEmbs)
        attentionWeights /= math.sqrt(hidden_dim)
        # ############ textcnn 得分计算 #####################
        # attentionWeights = torch.nn.functional.softmax(attentionWeights, 2)
        # attentionWeights = attentionWeights.unsqueeze(1)
        # x = [F.relu(conv(attentionWeights)).squeeze(3) for conv in self.convs] # len(Ks)*(N,Knum,W)
        # x = [F.max_pool1d(line,line.size(2)).squeeze(2) for line in x]
        # x = torch.cat(x,1) #(N,Knum*len(Ks))
        # x = self.dropout(x)
        # logits = self.fc(x).view(-1)
        ############ cnn 得分计算 #####################
        attentionWeights = torch.nn.functional.softmax(attentionWeights, 2)
        attentionWeights = attentionWeights.unsqueeze(1)
        x = [F.relu(conv(attentionWeights)).squeeze(3) for conv in self.convs]
        # import pdb; pdb.set_trace()
        x = [torch.max(line, 3)[0].squeeze(1) for line in x]
        
        # import pdb; pdb.set_trace()
        x = torch.cat(x,1) #(N,Knum*len(Ks))
        logits = torch.sum(x, 1).view(-1, 1)
        # logits = self.fc(x).view(-1)
        ########################################
        ############### 自定义得分计算 #############################
        # ############ lstm 得分计算 #####################
        # output, (h_n, c_n) = self.lstm(attentionWeights)
        # features = h_n.transpose(0, 1)
        # features = features[:, 1, :]
        # logits = self.classifier(features).view(-1)
        # # import pdb; pdb.set_trace()
        # ############### 自定义得分计算 #############################
        # # softmaxAtt = torch.nn.functional.softmax(attentionWeights, 1)
        # softmaxAtt = attentionWeights
        # # ******************************
        # # import pdb; pdb.set_trace()
        # # self.pathSum(softmaxAtt)
        # maxAtt = torch.max(softmaxAtt, 2)[0]
        # logits = torch.sum(maxAtt.view(batch, -1), 1)
        # # import pdb; pdb.set_trace()
        return logits
    
class BertForSequenceClassificationWordSimMatrixCNNTune(BertPreTrainedModel):
   
    def __init__(self, config, num_labels):
        super(BertForSequenceClassificationWordSimMatrixCNNTune, self).__init__(config)
        self.maxPathLen = 74
        self.num_labels = num_labels
        self.bert = BertModel(config)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size * 2, num_labels)
        self.classifier = nn.Linear(self.maxPathLen, num_labels)
        # self.scaleScore = nn.Linear(num_labels, num_labels)
        self.convs = torch.nn.ModuleList([nn.Conv2d(1, 100, (k, self.maxPathLen)) for k in [1, 2, 3]])
        # self.lstm = torch.nn.LSTM(input_size = self.maxPathLen, hidden_size = self.maxPathLen, num_layers = 2, batch_first = True)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(3 * 100, 1)
        self.apply(self.init_bert_weights)
        

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, alignPositions=None):
        outputEmbs, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        # outputEmbs = outputEmbs.detach()
        batch, alignNum = alignPositions.shape
        hidden_dim = outputEmbs.shape[-1]
        # queLen = torch.max(alignPositions[0][1], 49)
        queEmbs = outputEmbs[:, alignPositions[0][0]: alignPositions[0][1]]
        pathEmbs = outputEmbs[:, alignPositions[0][2]:]
        pathEmbs = pathEmbs[:, 0:self.maxPathLen, :]
        # *****************余弦相似度
        # queEmbsCos = queEmbs.unsqueeze(2)
        # pathEmbsCos = pathEmbs.unsqueeze(1)
        # # import pdb; pdb.set_trace()
        # cosSim = torch.cosine_similarity(queEmbsCos, pathEmbsCos, dim = 3)
        # # cosSim = cosSim.unsqueeze(0)
        # # print('cosSim:',cosSim.shape, cosSim)
        # softmaxAtt = cosSim
        # ******************注意力dot*****************
        pathEmbs = pathEmbs.transpose(1, 2)
        attentionWeights = torch.bmm(queEmbs, pathEmbs)
        attentionWeights /= math.sqrt(hidden_dim)
        ############ cnn 得分计算 #####################
        attentionWeights = torch.nn.functional.softmax(attentionWeights, 2)
        attentionWeights = attentionWeights.unsqueeze(1)
        
        x = [F.relu(conv(attentionWeights)).squeeze(3) for conv in self.convs] # len(Ks)*(N,Knum,W)
        
        x = [F.max_pool1d(line,line.size(2)).squeeze(2) for line in x]
        x = torch.cat(x,1) #(N,Knum*len(Ks))
        x = self.dropout(x)
        logits = self.fc(x).view(-1)
        # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        ############### 自定义得分计算 #############################
        # ############ lstm 得分计算 #####################
        # output, (h_n, c_n) = self.lstm(attentionWeights)
        # features = h_n.transpose(0, 1)
        # features = features[:, 1, :]
        # logits = self.classifier(features).view(-1)
        # # import pdb; pdb.set_trace()
        # ############### 自定义得分计算 #############################
        # # softmaxAtt = torch.nn.functional.softmax(attentionWeights, 1)
        # softmaxAtt = attentionWeights
        # # ******************************
        # # import pdb; pdb.set_trace()
        # # self.pathSum(softmaxAtt)
        # maxAtt = torch.max(softmaxAtt, 2)[0]
        # logits = torch.sum(maxAtt.view(batch, -1), 1)
        # # import pdb; pdb.set_trace()
        return logits
    
        
class BertForSequenceClassificationAttentionAlignBasedAvgGraphQueNotSame(BertPreTrainedModel):
   
    def __init__(self, config, num_labels):
        super(BertForSequenceClassificationAttentionAlignBasedAvgGraphQueNotSame, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size * 2, num_labels)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.pooler = positionBasedPooler(config.hidden_size * 3, config.hidden_size)
        self.scaleScore = nn.Linear(num_labels, num_labels)
        self.apply(self.init_bert_weights)

    def attention_net(self, queEmbs, contentEmbs):
        # import pdb; pdb.set_trace()
        contentEmbs = contentEmbs.unsqueeze(1)
        queEmbsPermute = queEmbs.transpose(1, 2)
        # import pdb; pdb.set_trace()
        attentionWeights = torch.bmm(contentEmbs, queEmbsPermute)
        attentionWeights /= math.sqrt(queEmbs.shape[2])
        softmaxAttentionWeights = torch.nn.functional.softmax(attentionWeights, 2)
        context = torch.bmm(softmaxAttentionWeights, queEmbs)
        context = context.squeeze(1)
        # import pdb; pdb.set_trace()
        return context
        # import pdb; pdb.set_trace()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, alignPositions=None):
        outputEmbs, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        batch, alignNum = alignPositions.shape
        hidden_dim = outputEmbs.shape[-1]
        # extractQueEmbs = torch.mean(outputEmbs[:,alignPositions[0][0]:alignPositions[0][1],:], 0)
        # extractQueEmbs = torch.max(outputEmbs[:,alignPositions[0][0]:alignPositions[0][1],:], 0)[0]
        extractQueEmbs = outputEmbs[:,alignPositions[0][0]:alignPositions[0][1],:]
        # import pdb; pdb.set_trace()  
        extractFirstEmbs = torch.zeros((batch, hidden_dim)).to(alignPositions.device)
        for i, item in enumerate(alignPositions):
            firstEmbs = outputEmbs[i, item[2]: item[3], :]
            # import pdb; pdb.set_trace()
            # extractFirstEmbs[i] = torch.mean(firstEmbs, 0)
            extractFirstEmbs[i] = torch.max(firstEmbs, 0)[0]
        extractSecondEmbs = torch.zeros((batch, hidden_dim)).to(alignPositions.device)
        for i, item in enumerate(alignPositions):
            secondEmbs = outputEmbs[i, item[4]: item[5], :]
            # extractSecondEmbs[i] = torch.mean(secondEmbs, 0)
            extractSecondEmbs[i] = torch.max(secondEmbs, 0)[0]
        firstQueContent = self.attention_net(extractQueEmbs, extractFirstEmbs)
        secondQueContent = self.attention_net(extractQueEmbs, extractSecondEmbs)
        # import pdb; pdb.set_trace()
        extractFirstFeature =  torch.cat((firstQueContent, extractFirstEmbs, firstQueContent - extractFirstEmbs), 1)
        extractFirstFeature = self.pooler(extractFirstFeature)
        extractFirstFeature = self.dropout(extractFirstFeature)
        logitsFirst = self.classifier(extractFirstFeature)
        extractSecondFeature =  torch.cat((secondQueContent, extractSecondEmbs, secondQueContent - extractSecondEmbs), 1)
        extractSecondFeature = self.pooler(extractSecondFeature)
        extractSecondFeature = self.dropout(extractSecondFeature)
        logitsSecond = self.classifier(extractSecondFeature)
        logits = logitsFirst + logitsSecond
        logits = self.scaleScore(logits)
        # logits = logitsSecond
        # import pdb; pdb.set_trace()
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits
        
class BertForSequenceClassificationAttentionAlignBasedAvgGraphQueNotSameDot(BertPreTrainedModel):
   
    def __init__(self, config, num_labels):
        super(BertForSequenceClassificationAttentionAlignBasedAvgGraphQueNotSameDot, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size * 2, num_labels)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.poolerQue = positionBasedPooler(config.hidden_size, config.hidden_size)
        self.poolerGraph = positionBasedPooler(config.hidden_size, config.hidden_size)
        self.scaleScore = nn.Linear(num_labels, num_labels)
        self.apply(self.init_bert_weights)

    def attention_net(self, queEmbs, contentEmbs):
        # import pdb; pdb.set_trace()
        contentEmbs = contentEmbs.unsqueeze(1)
        queEmbsPermute = queEmbs.transpose(1, 2)
        # import pdb; pdb.set_trace()
        attentionWeights = torch.bmm(contentEmbs, queEmbsPermute)
        attentionWeights /= math.sqrt(queEmbs.shape[2])
        softmaxAttentionWeights = torch.nn.functional.softmax(attentionWeights, 2)
        context = torch.bmm(softmaxAttentionWeights, queEmbs)
        context = context.squeeze(1)
        # import pdb; pdb.set_trace()
        return context
        # import pdb; pdb.set_trace()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, alignPositions=None):
        outputEmbs, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        batch, alignNum = alignPositions.shape
        hidden_dim = outputEmbs.shape[-1]
        # extractQueEmbs = torch.mean(outputEmbs[:,alignPositions[0][0]:alignPositions[0][1],:], 0)
        # extractQueEmbs = torch.max(outputEmbs[:,alignPositions[0][0]:alignPositions[0][1],:], 0)[0]
        extractQueEmbs = outputEmbs[:,alignPositions[0][0]:alignPositions[0][1],:]
        # import pdb; pdb.set_trace()  
        extractFirstEmbs = torch.zeros((batch, hidden_dim)).to(alignPositions.device)
        for i, item in enumerate(alignPositions):
            firstEmbs = outputEmbs[i, item[2]: item[3], :]
            # import pdb; pdb.set_trace()
            extractFirstEmbs[i] = torch.mean(firstEmbs, 0)
        extractSecondEmbs = torch.zeros((batch, hidden_dim)).to(alignPositions.device)
        for i, item in enumerate(alignPositions):
            secondEmbs = outputEmbs[i, item[4]: item[5], :]
            extractSecondEmbs[i] = torch.mean(secondEmbs, 0)
        firstQueContent = self.attention_net(extractQueEmbs, extractFirstEmbs)
        secondQueContent = self.attention_net(extractQueEmbs, extractSecondEmbs)
        # import pdb; pdb.set_trace()
        scoresFirst = torch.zeros((batch)).to(alignPositions.device)
        for i, item in enumerate(firstQueContent):
            scoresFirst[i] = torch.dot(self.poolerQue(item), self.poolerGraph(extractFirstEmbs[i])) / math.sqrt(hidden_dim)
            
        scoresSecond = torch.zeros((batch)).to(alignPositions.device)
        for i, item in enumerate(secondQueContent):
            scoresSecond[i] = torch.dot(self.poolerQue(item), self.poolerGraph(extractSecondEmbs[i])) / math.sqrt(hidden_dim)
        # import pdb; pdb.set_trace()
        logits = (scoresFirst + scoresSecond) / 2.0
        logits = self.scaleScore(logits.view(-1, 1))
        # logits = 0.2 * scoresFirst + 0.8 * scoresSecond
        # import pdb; pdb.set_trace()
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits
        
class FixBertForSequenceClassificationAttentionAlignBasedAvgGraphQueNotSame(BertPreTrainedModel):
   
    def __init__(self, config, num_labels):
        super(FixBertForSequenceClassificationAttentionAlignBasedAvgGraphQueNotSame, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size * 2, num_labels)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.pooler = positionBasedPooler(config.hidden_size * 3, config.hidden_size)
        self.scaleScore = nn.Linear(num_labels, num_labels)
        self.apply(self.init_bert_weights)

    def attention_net(self, queEmbs, contentEmbs):
        # import pdb; pdb.set_trace()
        contentEmbs = contentEmbs.unsqueeze(1)
        queEmbsPermute = queEmbs.transpose(1, 2)
        # import pdb; pdb.set_trace()
        attentionWeights = torch.bmm(contentEmbs, queEmbsPermute)
        attentionWeights /= math.sqrt(queEmbs.shape[2])
        softmaxAttentionWeights = torch.nn.functional.softmax(attentionWeights, 2)
        context = torch.bmm(softmaxAttentionWeights, queEmbs)
        context = context.squeeze(1)
        # import pdb; pdb.set_trace()
        return context
        # import pdb; pdb.set_trace()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, alignPositions=None):
        outputEmbs, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        # outputEmbs.detach_()
        # import pdb; pdb.set_trace()
        outputEmbs = outputEmbs.detach()
        batch, alignNum = alignPositions.shape
        hidden_dim = outputEmbs.shape[-1]
        # extractQueEmbs = torch.mean(outputEmbs[:,alignPositions[0][0]:alignPositions[0][1],:], 0)
        # extractQueEmbs = torch.max(outputEmbs[:,alignPositions[0][0]:alignPositions[0][1],:], 0)[0]
        extractQueEmbs = outputEmbs[:,alignPositions[0][0]:alignPositions[0][1],:]
        # import pdb; pdb.set_trace()  
        extractFirstEmbs = torch.zeros((batch, hidden_dim)).to(alignPositions.device)
        for i, item in enumerate(alignPositions):
            firstEmbs = outputEmbs[i, item[2]: item[3], :]
            # import pdb; pdb.set_trace()
            extractFirstEmbs[i] = torch.mean(firstEmbs, 0)
        extractSecondEmbs = torch.zeros((batch, hidden_dim)).to(alignPositions.device)
        for i, item in enumerate(alignPositions):
            secondEmbs = outputEmbs[i, item[4]: item[5], :]
            extractSecondEmbs[i] = torch.mean(secondEmbs, 0)
        firstQueContent = self.attention_net(extractQueEmbs, extractFirstEmbs)
        secondQueContent = self.attention_net(extractQueEmbs, extractSecondEmbs)
        # import pdb; pdb.set_trace()
        extractFirstFeature =  torch.cat((firstQueContent, extractFirstEmbs, firstQueContent - extractFirstEmbs), 1)
        extractFirstFeature = self.pooler(extractFirstFeature)
        extractFirstFeature = self.dropout(extractFirstFeature)
        logitsFirst = self.classifier(extractFirstFeature)
        extractSecondFeature =  torch.cat((secondQueContent, extractSecondEmbs, secondQueContent - extractSecondEmbs), 1)
        extractSecondFeature = self.pooler(extractSecondFeature)
        extractSecondFeature = self.dropout(extractSecondFeature)
        logitsSecond = self.classifier(extractSecondFeature)
        logits = logitsFirst + logitsSecond
        logits = self.scaleScore(logits)
        # logits = logitsSecond
        # import pdb; pdb.set_trace()
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class BertForSequenceClassificationParaAttentionAlignBased(BertPreTrainedModel):
   
    def __init__(self, config, num_labels):
        super(BertForSequenceClassificationParaAttentionAlignBased, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size * 2, num_labels)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.scoreWeight = nn.Linear(config.hidden_size * 2, 1)
        self.pooler = positionBasedPooler(config.hidden_size * 3, config.hidden_size)
        self.apply(self.init_bert_weights)

    def attention_net(self, queEmbs, contentEmbs):
        queNum = queEmbs.shape[0]
        contentNum = contentEmbs.shape[0]
        queEmbsPermute = queEmbs.permute(1, 0)
        queEmbsExpand = queEmbs.unsqueeze(0)
        queEmbsExpandRepeat = queEmbsExpand.repeat(contentNum, 1, 1)
        contentEmbsExpand = contentEmbs.unsqueeze(1)
        contentEmbsRepeat = contentEmbsExpand.repeat(1, queNum, 1)
        queContent = torch.cat((queEmbsExpandRepeat, contentEmbsRepeat), 2).view(queNum * contentNum, -1)
        attentionWeights = self.scoreWeight(queContent).view(contentNum, -1)
        # import pdb; pdb.set_trace()
        # attentionWeights = torch.mm(contentEmbs, queEmbsPermute)
        # attentionWeights /= math.sqrt(queEmbs.shape[1])
        # import pdb; pdb.set_trace()
        softmaxAttentionWeights = torch.nn.functional.softmax(attentionWeights, 1)
        context = torch.mm(softmaxAttentionWeights, queEmbs)
        return context
        # import pdb; pdb.set_trace()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, alignPositions=None):
        outputEmbs, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        outputEmbs.detach()
        batch, alignNum = alignPositions.shape
        hidden_dim = outputEmbs.shape[-1]
        extractQueEmbs = outputEmbs[0,alignPositions[0][0]:alignPositions[0][1],:]
        extractFirstEmbs = torch.zeros((batch, hidden_dim)).to(alignPositions.device)
        for i, item in enumerate(alignPositions):
            extractFirstEmbs[i] = outputEmbs[i, item[3] - 1, :]
        extractSecondEmbs = torch.zeros((batch, hidden_dim)).to(alignPositions.device)
        for i, item in enumerate(alignPositions):
            extractSecondEmbs[i] = outputEmbs[i, item[5], :]
        firstQueContent = self.attention_net(extractQueEmbs, extractFirstEmbs)
        secondQueContent = self.attention_net(extractQueEmbs, extractSecondEmbs)
        # import pdb; pdb.set_trace()
        extractFirstFeature =  torch.cat((firstQueContent, extractFirstEmbs, firstQueContent - extractFirstEmbs), 1)
        # extractEmbTensor = torch.Tensor(extractEmb)
        # import pdb; pdb.set_trace()
        extractFirstFeature = self.pooler(extractFirstFeature)
        extractFirstFeature = self.dropout(extractFirstFeature)
        logitsFirst = self.classifier(extractFirstFeature)
        extractSecondFeature =  torch.cat((secondQueContent, extractSecondEmbs, secondQueContent - extractSecondEmbs), 1)
        # extractEmbTensor = torch.Tensor(extractEmb)
        # import pdb; pdb.set_trace()
        extractSecondFeature = self.pooler(extractSecondFeature)
        extractSecondFeature = self.dropout(extractSecondFeature)
        logitsSecond = self.classifier(extractSecondFeature) 
        logits = logitsFirst + logitsSecond
        # import pdb; pdb.set_trace()
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class BertForSequenceClassificationAvgEmbeddingAttentionAlignBased(BertPreTrainedModel):
   
    def __init__(self, config, num_labels):
        super(BertForSequenceClassificationAvgEmbeddingAttentionAlignBased, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size * 2, num_labels)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.pooler = positionBasedPooler(config.hidden_size * 3, config.hidden_size)
        self.apply(self.init_bert_weights)

    def attention_net(self, queEmbs, contentEmbs):
        queEmbsPermute = queEmbs.permute(1, 0)
        # import pdb; pdb.set_trace()
        attentionWeights = torch.mm(contentEmbs, queEmbsPermute)
        attentionWeights /= math.sqrt(queEmbs.shape[1])
        # import pdb; pdb.set_trace()
        softmaxAttentionWeights = torch.nn.functional.softmax(attentionWeights, 1)
        context = torch.mm(softmaxAttentionWeights, queEmbs)
        return context
        # import pdb; pdb.set_trace()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, alignPositions=None):
        outputEmbs, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        batch, alignNum = alignPositions.shape
        hidden_dim = outputEmbs.shape[-1]
        # extractQueEmbs = outputEmbs[0,alignPositions[0][0]:alignPositions[0][1],:]
        extractQueEmbs = torch.max(outputEmbs[:,alignPositions[0][0]:alignPositions[0][1],:], 0)[0]
        # import pdb; pdb.set_trace()
        extractFirstEmbs = torch.zeros((batch, hidden_dim)).to(alignPositions.device)
        for i, item in enumerate(alignPositions):
            firstEmbs = outputEmbs[i, item[2]: item[3], :]
            # import pdb; pdb.set_trace()
            extractFirstEmbs[i] = torch.mean(firstEmbs, 0)
        extractSecondEmbs = torch.zeros((batch, hidden_dim)).to(alignPositions.device)
        for i, item in enumerate(alignPositions):
            secondEmbs = outputEmbs[i, item[4]: item[5], :]
            extractSecondEmbs[i] = torch.mean(secondEmbs, 0)
        firstQueContent = self.attention_net(extractQueEmbs, extractFirstEmbs)
        secondQueContent = self.attention_net(extractQueEmbs, extractSecondEmbs)
        # import pdb; pdb.set_trace()
        extractFirstFeature =  torch.cat((firstQueContent, extractFirstEmbs, firstQueContent - extractFirstEmbs), 1)
        # extractEmbTensor = torch.Tensor(extractEmb)
        # import pdb; pdb.set_trace()
        extractFirstFeature = self.pooler(extractFirstFeature)
        extractFirstFeature = self.dropout(extractFirstFeature)
        logitsFirst = self.classifier(extractFirstFeature)
        extractSecondFeature =  torch.cat((secondQueContent, extractSecondEmbs, secondQueContent - extractSecondEmbs), 1)
        # extractEmbTensor = torch.Tensor(extractEmb)
        # import pdb; pdb.set_trace()
        extractSecondFeature = self.pooler(extractSecondFeature)
        extractSecondFeature = self.dropout(extractSecondFeature)
        logitsSecond = self.classifier(extractSecondFeature)
        logits = logitsFirst + logitsSecond
        # import pdb; pdb.set_trace()
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

#### 1
# class BertForSequenceClassificationAvgEmbeddingParaAttentionAlignBased(BertPreTrainedModel):
   
#     def __init__(self, config, num_labels):
#         super(BertForSequenceClassificationAvgEmbeddingParaAttentionAlignBased, self).__init__(config)
#         self.num_labels = num_labels
#         self.bert = BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         # self.classifier = nn.Linear(config.hidden_size * 2, num_labels)
#         self.classifier = nn.Linear(config.hidden_size, num_labels)
#         self.scoreWeight = nn.Linear(config.hidden_size * 2, 1)
#         self.pooler = positionBasedPooler(config.hidden_size * 3, config.hidden_size)
#         self.quePoller = positionBasedPooler(config.hidden_size, config.hidden_size)
#         self.apply(self.init_bert_weights)

#     def attention_net(self, queEmbs, contentEmbs):
#         queEmbs = self.quePoller(queEmbs)
#         queNum = queEmbs.shape[0]
#         contentNum = contentEmbs.shape[0]
#         queEmbsExpand = queEmbs.unsqueeze(0)
#         queEmbsExpandRepeat = queEmbsExpand.repeat(contentNum, 1, 1)
#         contentEmbsExpand = contentEmbs.unsqueeze(1)
#         contentEmbsRepeat = contentEmbsExpand.repeat(1, queNum, 1)
#         queContent = torch.cat((queEmbsExpandRepeat, contentEmbsRepeat), 2).view(queNum * contentNum, -1)
#         attentionWeights = self.scoreWeight(queContent).view(contentNum, -1)
#         softmaxAttentionWeights = torch.nn.functional.softmax(attentionWeights, 1)
#         context = torch.mm(softmaxAttentionWeights, queEmbs)
#         # import pdb; pdb.set_trace()
#         return context
#         # import pdb; pdb.set_trace()

#     def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, alignPositions=None):
#         outputEmbs, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
#         outputEmbs.detach()
#         batch, alignNum = alignPositions.shape
#         hidden_dim = outputEmbs.shape[-1]
#         extractQueEmbs = outputEmbs[0,alignPositions[0][0]:alignPositions[0][1],:]
#         # import pdb; pdb.set_trace()
#         extractFirstEmbs = torch.zeros((batch, hidden_dim)).to(alignPositions.device)
#         for i, item in enumerate(alignPositions):
#             firstEmbs = outputEmbs[i, item[2]: item[3], :]
#             # import pdb; pdb.set_trace()
#             extractFirstEmbs[i] = torch.mean(firstEmbs, 0)
#         extractSecondEmbs = torch.zeros((batch, hidden_dim)).to(alignPositions.device)
#         for i, item in enumerate(alignPositions):
#             secondEmbs = outputEmbs[i, item[4]: item[5], :]
#             extractSecondEmbs[i] = torch.mean(secondEmbs, 0)
#         firstQueContent = self.attention_net(extractQueEmbs, extractFirstEmbs)
#         secondQueContent = self.attention_net(extractQueEmbs, extractSecondEmbs)
#         # import pdb; pdb.set_trace()
#         extractFirstFeature =  torch.cat((firstQueContent, extractFirstEmbs, firstQueContent - extractFirstEmbs), 1)
#         # extractEmbTensor = torch.Tensor(extractEmb)
#         # import pdb; pdb.set_trace()
#         extractFirstFeature = self.pooler(extractFirstFeature)
#         extractFirstFeature = self.dropout(extractFirstFeature)
#         logitsFirst = self.classifier(extractFirstFeature)
#         extractSecondFeature =  torch.cat((secondQueContent, extractSecondEmbs, secondQueContent - extractSecondEmbs), 1)
#         # extractEmbTensor = torch.Tensor(extractEmb)
#         # import pdb; pdb.set_trace()
#         extractSecondFeature = self.pooler(extractSecondFeature)
#         extractSecondFeature = self.dropout(extractSecondFeature)
#         logitsSecond = self.classifier(extractSecondFeature)
#         logits = logitsFirst + logitsSecond
#         # import pdb; pdb.set_trace()
#         if labels is not None:
#             loss_fct = CrossEntropyLoss()
#             loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#             return loss
#         else:
#             return logits

####### 2
class BertForSequenceClassificationAvgEmbeddingParaAttentionAlignBased(BertPreTrainedModel):
   
    def __init__(self, config, num_labels):
        super(BertForSequenceClassificationAvgEmbeddingParaAttentionAlignBased, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size * 2, num_labels)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.scoreWeight = nn.Linear(config.hidden_size * 2, 1)
        self.pooler = positionBasedPooler(config.hidden_size * 3, config.hidden_size)
        self.quePoller = positionBasedPooler(config.hidden_size, config.hidden_size)
        self.contentPoller = positionBasedPooler(config.hidden_size, config.hidden_size) 
        self.apply(self.init_bert_weights)

    def attention_net(self, queEmbsQ, contentEmbs):
        queEmbs = self.quePoller(queEmbsQ)
        contentEmbs = self.contentPoller(contentEmbs)
        queNum = queEmbs.shape[0]
        contentNum = contentEmbs.shape[0]
        queEmbsExpand = queEmbs.unsqueeze(0)
        queEmbsExpandRepeat = queEmbsExpand.repeat(contentNum, 1, 1)
        contentEmbsExpand = contentEmbs.unsqueeze(1)
        contentEmbsRepeat = contentEmbsExpand.repeat(1, queNum, 1)
        queContent = torch.cat((queEmbsExpandRepeat, contentEmbsRepeat), 2).view(queNum * contentNum, -1)
        attentionWeights = self.scoreWeight(queContent).view(contentNum, -1)
        softmaxAttentionWeights = torch.nn.functional.softmax(attentionWeights, 1)
        context = torch.mm(softmaxAttentionWeights, queEmbsQ)
        # import pdb; pdb.set_trace()
        return context
        # import pdb; pdb.set_trace()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, alignPositions=None):
        outputEmbs, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        batch, alignNum = alignPositions.shape
        hidden_dim = outputEmbs.shape[-1]
        extractQueEmbs = outputEmbs[0,alignPositions[0][0]:alignPositions[0][1],:]
        # import pdb; pdb.set_trace()
        extractFirstEmbs = torch.zeros((batch, hidden_dim)).to(alignPositions.device)
        for i, item in enumerate(alignPositions):
            firstEmbs = outputEmbs[i, item[2]: item[3], :]
            # import pdb; pdb.set_trace()
            extractFirstEmbs[i] = torch.mean(firstEmbs, 0)
        extractSecondEmbs = torch.zeros((batch, hidden_dim)).to(alignPositions.device)
        for i, item in enumerate(alignPositions):
            secondEmbs = outputEmbs[i, item[4]: item[5], :]
            extractSecondEmbs[i] = torch.mean(secondEmbs, 0)
        firstQueContent = self.attention_net(extractQueEmbs, extractFirstEmbs)
        secondQueContent = self.attention_net(extractQueEmbs, extractSecondEmbs)
        # scoresFirst = torch.zeros((batch)).to(alignPositions.device)
        # for i, item in enumerate(firstQueContent):
        #     scoresFirst[i] = torch.dot(item, extractFirstEmbs[i])
            
        # scoresSecond = torch.zeros((batch)).to(alignPositions.device)
        # for i, item in enumerate(secondQueContent):
        #     scoresSecond[i] = torch.dot(item, extractSecondEmbs[i])
        # # import pdb; pdb.set_trace()
        # logits = scoresFirst + scoresSecond
        extractFirstFeature =  torch.cat((firstQueContent, extractFirstEmbs, firstQueContent - extractFirstEmbs), 1)
        # extractEmbTensor = torch.Tensor(extractEmb)
        # import pdb; pdb.set_trace()
        extractFirstFeature = self.pooler(extractFirstFeature)
        extractFirstFeature = self.dropout(extractFirstFeature)
        logitsFirst = self.classifier(extractFirstFeature)
        extractSecondFeature =  torch.cat((secondQueContent, extractSecondEmbs, secondQueContent - extractSecondEmbs), 1)
        # extractEmbTensor = torch.Tensor(extractEmb)
        # import pdb; pdb.set_trace()
        extractSecondFeature = self.pooler(extractSecondFeature)
        extractSecondFeature = self.dropout(extractSecondFeature)
        logitsSecond = self.classifier(extractSecondFeature)
        logits = logitsFirst + logitsSecond
        # import pdb; pdb.set_trace()
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits
####### 3
class BertForSequenceClassificationAvgEmbeddingParaDotAttentionAlignBased(BertPreTrainedModel):
       
    def __init__(self, config, num_labels):
        super(BertForSequenceClassificationAvgEmbeddingParaDotAttentionAlignBased, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size * 2, num_labels)
        # self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.scoreWeight = nn.Linear(config.hidden_size * 2, 1)
        self.pooler = positionBasedPooler(config.hidden_size * 3, config.hidden_size)
        self.quePoller = positionBasedPooler(config.hidden_size, config.hidden_size)
        self.contentPoller = positionBasedPooler(config.hidden_size, config.hidden_size) 
        self.scoreTrans = nn.Linear(1, 1)
        self.apply(self.init_bert_weights)

    def attention_net(self, queEmbsQ, contentEmbs):
        queEmbs = self.quePoller(queEmbsQ)
        contentEmbs = self.contentPoller(contentEmbs)
        queNum = queEmbs.shape[0]
        contentNum = contentEmbs.shape[0]
        queEmbsExpand = queEmbs.unsqueeze(0)
        queEmbsExpandRepeat = queEmbsExpand.repeat(contentNum, 1, 1)
        contentEmbsExpand = contentEmbs.unsqueeze(1)
        contentEmbsRepeat = contentEmbsExpand.repeat(1, queNum, 1)
        queContent = torch.cat((queEmbsExpandRepeat, contentEmbsRepeat), 2).view(queNum * contentNum, -1)
        attentionWeights = self.scoreWeight(queContent).view(contentNum, -1)
        softmaxAttentionWeights = torch.nn.functional.softmax(attentionWeights, 1)
        context = torch.mm(softmaxAttentionWeights, queEmbsQ)
        # import pdb; pdb.set_trace()
        return context
        # import pdb; pdb.set_trace()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, alignPositions=None):
        outputEmbs, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        batch, alignNum = alignPositions.shape
        hidden_dim = outputEmbs.shape[-1]
        extractQueEmbs = outputEmbs[0,alignPositions[0][0]:alignPositions[0][1],:]
        # import pdb; pdb.set_trace()
        extractFirstEmbs = torch.zeros((batch, hidden_dim)).to(alignPositions.device)
        for i, item in enumerate(alignPositions):
            firstEmbs = outputEmbs[i, item[2]: item[3], :]
            # import pdb; pdb.set_trace()
            extractFirstEmbs[i] = torch.mean(firstEmbs, 0)
        extractSecondEmbs = torch.zeros((batch, hidden_dim)).to(alignPositions.device)
        for i, item in enumerate(alignPositions):
            secondEmbs = outputEmbs[i, item[4]: item[5], :]
            extractSecondEmbs[i] = torch.mean(secondEmbs, 0)
        firstQueContent = self.attention_net(extractQueEmbs, extractFirstEmbs)
        secondQueContent = self.attention_net(extractQueEmbs, extractSecondEmbs)
        scoresFirst = torch.zeros((batch)).to(alignPositions.device)
        for i, item in enumerate(firstQueContent):
            scoresFirst[i] = torch.dot(item, extractFirstEmbs[i])
            
        scoresSecond = torch.zeros((batch)).to(alignPositions.device)
        for i, item in enumerate(secondQueContent):
            scoresSecond[i] = torch.dot(item, extractSecondEmbs[i])
        # import pdb; pdb.set_trace()
        logits = scoresFirst + scoresSecond
        logits = self.scoreTrans(logits.view(-1, 1))
        # extractFirstFeature =  torch.cat((firstQueContent, extractFirstEmbs, firstQueContent - extractFirstEmbs), 1)
        # # extractEmbTensor = torch.Tensor(extractEmb)
        # # import pdb; pdb.set_trace()
        # extractFirstFeature = self.pooler(extractFirstFeature)
        # extractFirstFeature = self.dropout(extractFirstFeature)
        # logitsFirst = self.classifier(extractFirstFeature)
        # extractSecondFeature =  torch.cat((secondQueContent, extractSecondEmbs, secondQueContent - extractSecondEmbs), 1)
        # # extractEmbTensor = torch.Tensor(extractEmb)
        # # import pdb; pdb.set_trace()
        # extractSecondFeature = self.pooler(extractSecondFeature)
        # extractSecondFeature = self.dropout(extractSecondFeature)
        # logitsSecond = self.classifier(extractSecondFeature)
        # logits = logitsFirst + logitsSecond
        # import pdb; pdb.set_trace()
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits
        
class BertForSequenceClassificationAvgEmbedding3ParaDotAttentionAlignBased(BertPreTrainedModel):
       
    def __init__(self, config, num_labels):
        super(BertForSequenceClassificationAvgEmbedding3ParaDotAttentionAlignBased, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size * 2, num_labels)
        # self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.scoreWeight = nn.Linear(config.hidden_size * 2, 1)
        self.pooler = positionBasedPooler(config.hidden_size * 3, config.hidden_size)
        # self.quePoller = positionBasedPooler(config.hidden_size, config.hidden_size)
        # self.contentPoller = positionBasedPooler(config.hidden_size, config.hidden_size) 
        self.queQ = nn.Linear(config.hidden_size, config.hidden_size)
        self.queV = nn.Linear(config.hidden_size, config.hidden_size)
        
        # self.vPoller = positionBasedPooler(config.hidden_size, config.hidden_size)
        self.contentK = nn.Linear(config.hidden_size, config.hidden_size)
        self.scoreTrans = nn.Linear(1, 1)
        self.apply(self.init_bert_weights)

    def attention_net(self, queEmbsQ, contentEmbs):
        queEmbs = self.queQ(queEmbsQ)
        v = self.queV(queEmbsQ)
        # contentEmbs = self.contentPoller(contentEmbs)
        queNum = queEmbs.shape[0]
        contentNum = contentEmbs.shape[0]
        queEmbsExpand = queEmbs.unsqueeze(0)
        queEmbsExpandRepeat = queEmbsExpand.repeat(contentNum, 1, 1)
        contentEmbsExpand = contentEmbs.unsqueeze(1)
        contentEmbsRepeat = contentEmbsExpand.repeat(1, queNum, 1)
        queContent = torch.cat((queEmbsExpandRepeat, contentEmbsRepeat), 2).view(queNum * contentNum, -1)
        attentionWeights = self.scoreWeight(queContent).view(contentNum, -1)
        softmaxAttentionWeights = torch.nn.functional.softmax(attentionWeights, 1)
        context = torch.mm(softmaxAttentionWeights, v)
        # import pdb; pdb.set_trace()
        return context
        # import pdb; pdb.set_trace()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, alignPositions=None):
        outputEmbs, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        batch, alignNum = alignPositions.shape
        hidden_dim = outputEmbs.shape[-1]
        extractQueEmbs = outputEmbs[0,alignPositions[0][0]:alignPositions[0][1],:]
        # import pdb; pdb.set_trace()
        extractFirstEmbs = torch.zeros((batch, hidden_dim)).to(alignPositions.device)
        for i, item in enumerate(alignPositions):
            firstEmbs = outputEmbs[i, item[2]: item[3], :]
            # import pdb; pdb.set_trace()
            extractFirstEmbs[i] = torch.mean(firstEmbs, 0)
        extractSecondEmbs = torch.zeros((batch, hidden_dim)).to(alignPositions.device)
        for i, item in enumerate(alignPositions):
            secondEmbs = outputEmbs[i, item[4]: item[5], :]
            extractSecondEmbs[i] = torch.mean(secondEmbs, 0)
        extractFirstEmbs = self.contentK(extractFirstEmbs)
        firstQueContent = self.attention_net(extractQueEmbs, extractFirstEmbs)
        extractSecondEmbs = self.contentK(extractSecondEmbs)
        secondQueContent = self.attention_net(extractQueEmbs, extractSecondEmbs)
        scoresFirst = torch.zeros((batch)).to(alignPositions.device)
        for i, item in enumerate(firstQueContent):
            scoresFirst[i] = torch.dot(item, extractFirstEmbs[i])
            
        scoresSecond = torch.zeros((batch)).to(alignPositions.device)
        for i, item in enumerate(secondQueContent):
            scoresSecond[i] = torch.dot(item, extractSecondEmbs[i])
        # import pdb; pdb.set_trace()
        logits = scoresFirst + scoresSecond
        logits = self.scoreTrans(logits.view(-1, 1))
        # extractFirstFeature =  torch.cat((firstQueContent, extractFirstEmbs, firstQueContent - extractFirstEmbs), 1)
        # # extractEmbTensor = torch.Tensor(extractEmb)
        # # import pdb; pdb.set_trace()
        # extractFirstFeature = self.pooler(extractFirstFeature)
        # extractFirstFeature = self.dropout(extractFirstFeature)
        # logitsFirst = self.classifier(extractFirstFeature)
        # extractSecondFeature =  torch.cat((secondQueContent, extractSecondEmbs, secondQueContent - extractSecondEmbs), 1)
        # # extractEmbTensor = torch.Tensor(extractEmb)
        # # import pdb; pdb.set_trace()
        # extractSecondFeature = self.pooler(extractSecondFeature)
        # extractSecondFeature = self.dropout(extractSecondFeature)
        # logitsSecond = self.classifier(extractSecondFeature)
        # logits = logitsFirst + logitsSecond
        # import pdb; pdb.set_trace()
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class BertForSequenceClassificationAvgEmbeddingQKV(BertPreTrainedModel):
       
    def __init__(self, config, num_labels):
        super(BertForSequenceClassificationAvgEmbeddingQKV, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size * 2, num_labels)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.scoreWeight = nn.Linear(config.hidden_size * 2, 1)
        self.pooler = positionBasedPooler(config.hidden_size * 3, config.hidden_size)
        # self.quePoller = positionBasedPooler(config.hidden_size, config.hidden_size)
        # self.contentPoller = positionBasedPooler(config.hidden_size, config.hidden_size) 
        self.queQ = nn.Linear(config.hidden_size, config.hidden_size)
        self.queV = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        # self.vPoller = positionBasedPooler(config.hidden_size, config.hidden_size)
        self.contentK = nn.Linear(config.hidden_size, config.hidden_size)
        # self.scoreTrans = nn.Linear(1, 1)
        self.apply(self.init_bert_weights)

    def attention_net(self, queEmbsQ, contentEmbs):
        # import pdb; pdb.set_trace()
        queEmbs = self.queQ(queEmbsQ)
        v = self.queV(queEmbsQ)
        # contentEmbs = self.contentPoller(contentEmbs)
        queNum = queEmbs.shape[1]
        contentNum = contentEmbs.shape[0]
        # queEmbsExpand = queEmbs.unsqueeze(0)
        # queEmbsExpandRepeat = queEmbsExpand.repeat(contentNum, 1, 1)
        contentEmbsExpand = contentEmbs.unsqueeze(1)
        contentEmbsRepeat = contentEmbsExpand.repeat(1, queNum, 1)
        
        queContent = torch.cat((queEmbs, contentEmbsRepeat), 2).view(queNum * contentNum, -1)
        attentionWeights = self.scoreWeight(queContent).view(contentNum, -1)
        # import pdb; pdb.set_trace()
        softmaxAttentionWeights = torch.nn.functional.softmax(attentionWeights, 1)
        context = torch.zeros((queEmbs.shape[0], queEmbs.shape[2])).to(queEmbs.device)
        for i in range(queEmbs.shape[0]):
            context[i] = torch.mm(softmaxAttentionWeights[i:i + 1, :], v[i])
        # import pdb; pdb.set_trace()
        return context


    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, alignPositions=None, eval = False):
        outputEmbs, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        batch, alignNum = alignPositions.shape
        hidden_dim = outputEmbs.shape[-1]
        
        # extractQueEmbs = torch.max(outputEmbs[:,alignPositions[0][0]:alignPositions[0][1],:], 0)[0]
        # extractQueEmbs = outputEmbs[0,alignPositions[0][0]:alignPositions[0][1],:]
        # import pdb; pdb.set_trace()
        extractQueEmbs = torch.zeros(batch, alignPositions[0][1] - alignPositions[0][0], hidden_dim).to(alignPositions.device)
        extractFirstEmbs = torch.zeros((batch, hidden_dim)).to(alignPositions.device)
        for i, item in enumerate(alignPositions):
            firstEmbs = outputEmbs[i, item[2]: item[3], :]
            extractFirstEmbs[i] = torch.mean(firstEmbs, 0)
            extractQueEmbs[i] = outputEmbs[i, alignPositions[0][0]:alignPositions[0][1], :]
        # import pdb; pdb.set_trace()
        extractSecondEmbs = torch.zeros((batch, hidden_dim)).to(alignPositions.device)
        for i, item in enumerate(alignPositions):
            secondEmbs = outputEmbs[i, item[4]: item[5], :]
            extractSecondEmbs[i] = torch.mean(secondEmbs, 0)
        extractFirstEmbs = self.contentK(extractFirstEmbs)
        firstQueContent = self.attention_net(extractQueEmbs, extractFirstEmbs)
        extractSecondEmbs = self.contentK(extractSecondEmbs)
        secondQueContent = self.attention_net(extractQueEmbs, extractSecondEmbs)
        # scoresFirst = torch.zeros((batch)).to(alignPositions.device)
        # for i, item in enumerate(firstQueContent):
        #     scoresFirst[i] = torch.dot(item, extractFirstEmbs[i])
            
        # scoresSecond = torch.zeros((batch)).to(alignPositions.device)
        # for i, item in enumerate(secondQueContent):
        #     scoresSecond[i] = torch.dot(item, extractSecondEmbs[i])
        # # import pdb; pdb.set_trace()
        # logits = scoresFirst + scoresSecond
        # logits = self.scoreTrans(logits.view(-1, 1))
        extractFirstFeature =  torch.cat((firstQueContent, extractFirstEmbs, firstQueContent - extractFirstEmbs), 1)
        # extractEmbTensor = torch.Tensor(extractEmb)
        # import pdb; pdb.set_trace()
        extractFirstFeature = self.pooler(extractFirstFeature)
        # extractFirstFeature = self.LayerNorm(extractFirstFeature)
        # import pdb; pdb.set_trace()
        extractFirstFeature = self.dropout(extractFirstFeature)
        logitsFirst = self.classifier(extractFirstFeature)
        extractSecondFeature =  torch.cat((secondQueContent, extractSecondEmbs, secondQueContent - extractSecondEmbs), 1)
        # extractEmbTensor = torch.Tensor(extractEmb)
        # import pdb; pdb.set_trace()
        extractSecondFeature = self.pooler(extractSecondFeature)
        # extractSecondFeature = self.LayerNorm(extractSecondFeature)
        extractSecondFeature = self.dropout(extractSecondFeature)
        logitsSecond = self.classifier(extractSecondFeature)
        logits = logitsFirst + logitsSecond
        # import pdb; pdb.set_trace()
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits
        
class BertForSequenceClassificationAvgEmbeddingQKVQueSame(BertPreTrainedModel):
       
    def __init__(self, config, num_labels):
        super(BertForSequenceClassificationAvgEmbeddingQKVQueSame, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size * 2, num_labels)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.scoreWeight = nn.Linear(config.hidden_size * 2, 1)
        self.pooler = positionBasedPooler(config.hidden_size * 3, config.hidden_size)
        # self.quePoller = positionBasedPooler(config.hidden_size, config.hidden_size)
        # self.contentPoller = positionBasedPooler(config.hidden_size, config.hidden_size) 
        self.queQ = nn.Linear(config.hidden_size, config.hidden_size)
        self.queV = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        # self.vPoller = positionBasedPooler(config.hidden_size, config.hidden_size)
        self.contentK = nn.Linear(config.hidden_size, config.hidden_size)
        # self.scoreTrans = nn.Linear(1, 1)
        self.apply(self.init_bert_weights)

    def attention_net(self, queEmbsQ, contentEmbs):
        queEmbs = self.queQ(queEmbsQ)
        v = self.queV(queEmbsQ)
        # contentEmbs = self.contentPoller(contentEmbs)
        queNum = queEmbs.shape[0]
        contentNum = contentEmbs.shape[0]
        queEmbsExpand = queEmbs.unsqueeze(0)
        queEmbsExpandRepeat = queEmbsExpand.repeat(contentNum, 1, 1)
        contentEmbsExpand = contentEmbs.unsqueeze(1)
        contentEmbsRepeat = contentEmbsExpand.repeat(1, queNum, 1)
        queContent = torch.cat((queEmbsExpandRepeat, contentEmbsRepeat), 2).view(queNum * contentNum, -1)
        attentionWeights = self.scoreWeight(queContent).view(contentNum, -1)
        softmaxAttentionWeights = torch.nn.functional.softmax(attentionWeights, 1)
        context = torch.mm(softmaxAttentionWeights, v)
        # import pdb; pdb.set_trace()
        return context
        # import pdb; pdb.set_trace()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, alignPositions=None, eval = False):
        outputEmbs, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        batch, alignNum = alignPositions.shape
        hidden_dim = outputEmbs.shape[-1]
        
        extractQueEmbs = torch.max(outputEmbs[:,alignPositions[0][0]:alignPositions[0][1],:], 0)[0]
        # extractQueEmbs = outputEmbs[0,alignPositions[0][0]:alignPositions[0][1],:]
        # import pdb; pdb.set_trace()
        # extractQueEmbs = torch.zeros(batch, alignPositions[0][1] - alignPositions[0][0], hidden_dim)
        extractFirstEmbs = torch.zeros((batch, hidden_dim)).to(alignPositions.device)
        for i, item in enumerate(alignPositions):
            firstEmbs = outputEmbs[i, item[2]: item[3], :]
            # import pdb; pdb.set_trace()
            extractFirstEmbs[i] = torch.mean(firstEmbs, 0)
            # extractQueEmbs = outputEmbs[i, alignPositions[0][0]:alignPositions[0][1], :]
        extractSecondEmbs = torch.zeros((batch, hidden_dim)).to(alignPositions.device)
        for i, item in enumerate(alignPositions):
            secondEmbs = outputEmbs[i, item[4]: item[5], :]
            extractSecondEmbs[i] = torch.mean(secondEmbs, 0)
        extractFirstEmbs = self.contentK(extractFirstEmbs)
        firstQueContent = self.attention_net(extractQueEmbs, extractFirstEmbs)
        extractSecondEmbs = self.contentK(extractSecondEmbs)
        secondQueContent = self.attention_net(extractQueEmbs, extractSecondEmbs)
        # scoresFirst = torch.zeros((batch)).to(alignPositions.device)
        # for i, item in enumerate(firstQueContent):
        #     scoresFirst[i] = torch.dot(item, extractFirstEmbs[i])
            
        # scoresSecond = torch.zeros((batch)).to(alignPositions.device)
        # for i, item in enumerate(secondQueContent):
        #     scoresSecond[i] = torch.dot(item, extractSecondEmbs[i])
        # # import pdb; pdb.set_trace()
        # logits = scoresFirst + scoresSecond
        # logits = self.scoreTrans(logits.view(-1, 1))
        extractFirstFeature =  torch.cat((firstQueContent, extractFirstEmbs, firstQueContent - extractFirstEmbs), 1)
        # extractEmbTensor = torch.Tensor(extractEmb)
        # import pdb; pdb.set_trace()
        extractFirstFeature = self.pooler(extractFirstFeature)
        extractFirstFeature = self.LayerNorm(extractFirstFeature)
        # import pdb; pdb.set_trace()
        extractFirstFeature = self.dropout(extractFirstFeature)
        logitsFirst = self.classifier(extractFirstFeature)
        extractSecondFeature =  torch.cat((secondQueContent, extractSecondEmbs, secondQueContent - extractSecondEmbs), 1)
        # extractEmbTensor = torch.Tensor(extractEmb)
        # import pdb; pdb.set_trace()
        extractSecondFeature = self.pooler(extractSecondFeature)
        extractSecondFeature = self.LayerNorm(extractSecondFeature)
        extractSecondFeature = self.dropout(extractSecondFeature)
        logitsSecond = self.classifier(extractSecondFeature)
        logits = logitsFirst + logitsSecond
        # import pdb; pdb.set_trace()
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits
        
class BertForSequenceClassificationAvgEmbeddingQKVScale(BertPreTrainedModel):
       
    def __init__(self, config, num_labels):
        super(BertForSequenceClassificationAvgEmbeddingQKVScale, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size * 2, num_labels)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.scoreWeight = nn.Linear(config.hidden_size * 2, 1)
        self.pooler = positionBasedPooler(config.hidden_size * 3, config.hidden_size)
        # self.quePoller = positionBasedPooler(config.hidden_size, config.hidden_size)
        # self.contentPoller = positionBasedPooler(config.hidden_size, config.hidden_size) 
        self.queQ = nn.Linear(config.hidden_size, config.hidden_size)
        self.queV = nn.Linear(config.hidden_size, config.hidden_size)
        
        # self.vPoller = positionBasedPooler(config.hidden_size, config.hidden_size)
        self.contentK = nn.Linear(config.hidden_size, config.hidden_size)
        self.scoreTrans = nn.Linear(2, 2)
        self.apply(self.init_bert_weights)

    def attention_net(self, queEmbsQ, contentEmbs):
        queEmbs = self.queQ(queEmbsQ)
        v = self.queV(queEmbsQ)
        # contentEmbs = self.contentPoller(contentEmbs)
        queNum = queEmbs.shape[0]
        contentNum = contentEmbs.shape[0]
        queEmbsExpand = queEmbs.unsqueeze(0)
        queEmbsExpandRepeat = queEmbsExpand.repeat(contentNum, 1, 1)
        contentEmbsExpand = contentEmbs.unsqueeze(1)
        contentEmbsRepeat = contentEmbsExpand.repeat(1, queNum, 1)
        queContent = torch.cat((queEmbsExpandRepeat, contentEmbsRepeat), 2).view(queNum * contentNum, -1)
        attentionWeights = self.scoreWeight(queContent).view(contentNum, -1)
        softmaxAttentionWeights = torch.nn.functional.softmax(attentionWeights, 1)
        context = torch.mm(softmaxAttentionWeights, v)
        # import pdb; pdb.set_trace()
        return context
        # import pdb; pdb.set_trace()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, alignPositions=None):
        outputEmbs, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        batch, alignNum = alignPositions.shape
        hidden_dim = outputEmbs.shape[-1]
        extractQueEmbs = outputEmbs[0,alignPositions[0][0]:alignPositions[0][1],:]
        # import pdb; pdb.set_trace()
        extractFirstEmbs = torch.zeros((batch, hidden_dim)).to(alignPositions.device)
        for i, item in enumerate(alignPositions):
            firstEmbs = outputEmbs[i, item[2]: item[3], :]
            # import pdb; pdb.set_trace()
            extractFirstEmbs[i] = torch.mean(firstEmbs, 0)
        extractSecondEmbs = torch.zeros((batch, hidden_dim)).to(alignPositions.device)
        for i, item in enumerate(alignPositions):
            secondEmbs = outputEmbs[i, item[4]: item[5], :]
            extractSecondEmbs[i] = torch.mean(secondEmbs, 0)
        extractFirstEmbs = self.contentK(extractFirstEmbs)
        firstQueContent = self.attention_net(extractQueEmbs, extractFirstEmbs)
        extractSecondEmbs = self.contentK(extractSecondEmbs)
        secondQueContent = self.attention_net(extractQueEmbs, extractSecondEmbs)
        # scoresFirst = torch.zeros((batch)).to(alignPositions.device)
        # for i, item in enumerate(firstQueContent):
        #     scoresFirst[i] = torch.dot(item, extractFirstEmbs[i])
            
        # scoresSecond = torch.zeros((batch)).to(alignPositions.device)
        # for i, item in enumerate(secondQueContent):
        #     scoresSecond[i] = torch.dot(item, extractSecondEmbs[i])
        # # import pdb; pdb.set_trace()
        # logits = scoresFirst + scoresSecond
        # logits = self.scoreTrans(logits.view(-1, 1))
        extractFirstFeature =  torch.cat((firstQueContent, extractFirstEmbs, firstQueContent - extractFirstEmbs), 1)
        # extractEmbTensor = torch.Tensor(extractEmb)
        # import pdb; pdb.set_trace()
        extractFirstFeature = self.pooler(extractFirstFeature)
        extractFirstFeature = self.dropout(extractFirstFeature)
        logitsFirst = self.classifier(extractFirstFeature)
        extractSecondFeature =  torch.cat((secondQueContent, extractSecondEmbs, secondQueContent - extractSecondEmbs), 1)
        # extractEmbTensor = torch.Tensor(extractEmb)
        # import pdb; pdb.set_trace()
        extractSecondFeature = self.pooler(extractSecondFeature)
        extractSecondFeature = self.dropout(extractSecondFeature)
        logitsSecond = self.classifier(extractSecondFeature)
        logits = logitsFirst + logitsSecond
        logits = self.scoreTrans(logits)
        # import pdb; pdb.set_trace()
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

class BertForSequenceWithRels(BertPreTrainedModel):

    def __init__(self, config, num_labels):
        super(BertForSequenceWithRels, self).__init__(config)
        self.num_labels = num_labels
        # import pdb; pdb.set_trace()
        self.relEmbedding = torch.from_numpy(RelationEmbedding().embedding.astype(np.float32))
        self.relEmbeddingMatrix = torch.nn.Embedding.from_pretrained(self.relEmbedding, freeze=False)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.classifier_2 = nn.Linear(config.hidden_size * 2, num_labels)
        self.classifier_3 = nn.Linear(config.hidden_size * 3, num_labels)
        self.classifier_transe = nn.Linear(config.hidden_size * 2 + 50, num_labels)
        self.classifier_transe2 = nn.Linear(config.hidden_size + 50, 300)
        self.classifier_base_transe = nn.Linear(config.hidden_size + 300, num_labels)
        self.activation = nn.Tanh()
        self.apply(self.init_bert_weights)


    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, rels_ids = None):
        # import pdb; pdb.set_trace()
        #***********************使用transe直接拼接***********************
        rels_ids = rels_ids.view(-1,2,2)[:,0].view(-1,2)
        rels_emb = self.relEmbeddingMatrix(rels_ids)
        rels_emb = rels_emb.permute(0, 2, 1)
        rels_emb = torch.nn.functional.avg_pool1d(rels_emb, kernel_size=rels_emb.shape[-1]).squeeze(-1)
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = pooled_output.view(-1, 2 * 768)
        pooled_output = torch.cat((pooled_output, rels_emb), 1)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier_transe(pooled_output)
        #***********************使用transe先映射为特征再拼接***********************
        # rels_ids = rels_ids.view(-1,2,2)[:,0].view(-1,2)
        # rels_emb = self.relEmbeddingMatrix(rels_ids)
        # rels_emb = rels_emb.permute(0, 2, 1)
        # rels_emb = torch.nn.functional.avg_pool1d(rels_emb, kernel_size=rels_emb.shape[-1]).squeeze(-1)
        # _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        # pooled_output1 = pooled_output.view(-1, 2, 768)[:,0,:].view(-1, 768)
        # pooled_output2 = pooled_output.view(-1, 2, 768)[:,1,:].view(-1, 768)
        # pooled_output_transe = torch.cat((pooled_output2, rels_emb), 1)
        # pooled_output_transe = self.classifier_transe2(pooled_output_transe)
        # pooled_output_transe = self.activation(pooled_output_transe)
        # baseCatTranse = torch.cat((pooled_output1, pooled_output_transe), 1)
        # pooled_output = self.dropout(baseCatTranse)
        # logits = self.classifier_base_transe(pooled_output)
        ######################不使用transe###############################
        # _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        # pooled_output = pooled_output.view(-1, 2, 768)[:,0,:].view(-1, 768)
        # pooled_output = self.dropout(pooled_output)
        # logits = self.classifier(pooled_output)
        # import pdb; pdb.set_trace()
        return logits

    
class BertForSequenceWithAnswerType(BertPreTrainedModel):

    def __init__(self, config, num_labels, mid_dim = 768):
        super(BertForSequenceWithAnswerType, self).__init__(config)
        self.num_labels = num_labels
        self.mid_dim = mid_dim
        # import pdb; pdb.set_trace()
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.denseCat = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()
        self.apply(self.init_bert_weights)


    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, rels_ids = None):
        ######################问句与答案相似度和语义相似度拼接###############################
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        denseCat = self.denseCat(pooled_output.view(-1, 2 * 768)) 
        denseCat = self.activation(denseCat)
        pooled_output = self.dropout(denseCat)
        logits = self.classifier(pooled_output)
        ###############问句与答案相似度和语义相似度得分相加########################
        # _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        # pooled_output = self.dropout(pooled_output)
        # logits = self.classifier(pooled_output)
        # logits = torch.sum(logits.view(-1, 2, 2),1)
        # import pdb; pdb.set_trace()
        ##########不使用answer信息#############################
        # input_ids = input_ids.view(-1, 2, 100)
        # token_type_ids = token_type_ids.view(-1, 2, 100)
        # attention_mask = attention_mask.view(-1, 2, 100)
        # input_ids1 = input_ids[:, 0, :]
        # token_type_ids1 = token_type_ids[:, 0, :]
        # attention_mask1 = attention_mask[:, 0, :]
        # _, pooled_output = self.bert(input_ids1, token_type_ids1, attention_mask1)
        # pooled_output = self.dropout(pooled_output)
        # logits = self.classifier(pooled_output)
        return logits


class BertFor2PairSequenceWithAnswerType(BertPreTrainedModel):
    
    def __init__(self, config, num_labels):
        super(BertFor2PairSequenceWithAnswerType, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.bert2 = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.denseCat = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, rels_ids = None):
        ##############问句与答案字符串的编码和语义相似度编码不采用同一个bert##############
        input_ids = input_ids.view(-1, 2, 100)
        token_type_ids = token_type_ids.view(-1, 2, 100)
        attention_mask = attention_mask.view(-1, 2, 100)
        input_ids1 = input_ids[:, 0, :]
        input_ids2 = input_ids[:, 1, :]
        token_type_ids1 = token_type_ids[:, 0, :]
        token_type_ids2 = token_type_ids[:, 1, :]
        attention_mask1 = attention_mask[:, 0, :]
        attention_mask2 = attention_mask[:, 1, :]
        _, pooled_output1 = self.bert(input_ids1, token_type_ids1, attention_mask1)
        _, pooled_output2 = self.bert2(input_ids2, token_type_ids2, attention_mask2)
        pooled_output = torch.cat((pooled_output1,pooled_output2),1)
        denseCat = self.denseCat(pooled_output) 
        denseCat = self.activation(denseCat)
        pooled_output = self.dropout(denseCat)
        logits = self.classifier(pooled_output)
        return logits

