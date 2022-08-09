# [代码和数据整理中，请稍候](https://github.com/cytan17726/KBQA_QueryGraphGeneration/tree/cytan)

## 基本目录结构

config 参数

data 相关数据
    train_data
        graph_ranker: 相关排序模型的训练数据

model 训练的模型

排序模型-当时训练的

CCKS2019: Luo Ours Yhi

CCKS2019_Comp

__Luo__ __Ours__ __Yhi__

## 依赖

- `python`: 3.6.7
- `torch`: 1.4.0
- `pytorch_pretrained_bert`: 0.6.2
- `tqdm`
- `yaml`
- `fuzzywuzzy`: 0.18.0
- `elasticsearch`: 7.13.1
- `scikit-learn`: 0.20.4
- `xgboost`: 1.4.1
- `pymysql`: 1.0.2
- `jieba`: 0.42.1

数据下载地址(百度网盘):
[链接](https://pan.baidu.com/s/1UzczuOdBNAwjP9h8Sf0cjA), 提取码: vbab

## 快速复现实验结果

我们提供了生成的查询图候选，经排序即可得到

## 逐步

### 数据库构建

Mysql pkuorder和pkubase

### 预处理

包括 实体链接、关系预测，同时也直接提供我们生成的结果

### 查询图生成

cd src/build_query_graph
Yih
Luo
Our

### 查询图排序

序列化: cd src/querygraph2seq

构建训练数据: cd src/build_model_data

排序: cd 

### 结果评价

cd src/eda
