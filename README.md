# [代码和数据整理中，请稍候](https://github.com/cytan17726/KBQA_QueryGraphGeneration/tree/cytan)

## 基本目录结构

config 若干参数

data 相关数据
    train_data
        graph_ranker: 相关排序模型的训练数据

model 训练的模型

src 相关代码

__Luo__ __Ours__ __Yhi__

## Setups

代码在以下环境中测试:

- Ubuntu 16.04
- python 3.6.7
- Pytorch 1.4.0

其余依赖:

- `pytorch_pretrained_bert`: 0.6.2
- `scikit-learn`: 0.20.4
- `xgboost`: 1.4.1
- `pymysql`: 1.0.2
- `jieba`: 0.42.1
- `tqdm`
- `yaml`

训练的模型和部分中间结果:
[下载地址(百度网盘)](https://pan.baidu.com/s/1UzczuOdBNAwjP9h8Sf0cjA), 提取码: vbab

## 快速复现实验结果

我们在网盘中提供了各阶段性结果，包括：

- 数据
- 预处理数据（关系预测提供训练后的模型）
- 查询图生成
- 训练数据
- 排序模型

我们在本地评估性能结果如下:

|查询图生成性能|CCKS2019|CCKS2019-Comp|
|:---:|:---:|:---:|
|Yih等|85.40|71.07|
|Luo等|86.49|71.93|
|Ours|89.47|86.91|

|KBQA性能|CCKS2019|CCKS2019-Comp|
|:---:|:---:|:---:|
|Yih等|72.43|59.50|
|Luo等|73.55|60.66|
|Ours|73.86|73.39|

## 逐步

### 数据库构建(TODO)

Mysql pkuorder和pkubase

### 预处理

包括 实体链接、关系预测，这里提供我们生成和训练的结果,[见网盘]

### 查询图生成

cd src/build_query_graph

Yih

- `nohup python main_Yhi_for_train.py > log/ccks2019_Yhi_train.log&`

Luo

- `nohup python main_STAGG_for_test_0308.py >log/Luo_test.log&`

Our

- `nohup python main_based_filter_rel_for_test_ccksOnly.py >log/CCKS2019_Ours_test.log&`

### 查询图排序

序列化:
```
cd src/querygraph2seq
python querygraph_to_seq.py
```

构建训练数据: `cd src/build_model_data`

排序: `cd src/model_train`
`python train_listwise_multi_types_1.py`

### 结果评价[完成整理]

需要自行修改config文件, 位置在/config/eda/eval_test.yaml

```
cd src/eda
bash eval_test.sh
```

若有更多问题可联系 cytan17726@stu.suda.edu.cn