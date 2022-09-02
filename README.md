# [面向知识图谱问答的查询图生成方法](https://github.com/cytan17726/KBQA_QueryGraphGeneration/tree/master)

代码和数据整理中，请稍候(在重新测试...在肝了在肝了)

## 基本目录结构

config 若干参数

data 相关数据
    train_data
        graph_ranker: 相关排序模型的训练数据

model 训练的模型

src 相关代码

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

- 新构建KBQA数据(ch2,后续开放)
- 预处理数据与模型(ch3.1, 关系预测提供训练后的模型)
- 查询图生成-阶段结果(ch3.2)
- 查询图排序-训练数据(ch3.3)
- 查询图排序-已训练模型(ch3.3)
- 查询图排序-打分结果(ch3.3)

我们在本地评估性能结果如下,可作为您的参考:

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


### 0 数据库构建(TODO)

您可以基于[CCKS2019 中文知识图谱问答](https://www.biendata.xyz/competition/ccks_2019_6/data/)提供的数据, 构建KB

本系统使用Mysql数据库进行KB的存储与检索

更多详细信息后续添加完善

### 1 预处理

包括 节点识别、关系预测

    节点识别, 提供实体词节点的识别结果, 其余内嵌在代码中
    关系预测提供已训练模型

### 2 查询图生成(todo 修改传参方式)
修改文件内的参数(对应 CCKS2019和CCKS2019-Comp两个数据集上的表现)
cd src/build_query_graph

- Yih

```
nohup python main_Yih_for_test.py > log/CCKS2019_Yih_test.log&
nohup python main_Yih_for_test.py > log/CCKS2019_Comp_Yih_test.log&
```

- Luo

```
nohup python main_Luo_for_test.py > log/CCKS2019_Luo_test.log&
nohup python main_Luo_for_test.py > log/CCKS2019_Comp_Luo_test.log&
```

- Our

```
nohup python main_based_filter_rel_for_test.py > log/CCKS2019_Ours_test.log&
nohup python main_based_filter_rel_for_test.py > log/CCKS2019_Comp_Ours_test.log&
```


### 3 查询图排序

1. 序列化[TODO-test和valid已完成]
```
cd src/querygraph2seq
python querygraph_to_seq.py
```

2. 构建训练数据[TODO]
```
cd src/build_model_data
python build_train_data_for_analysis.py
python build_test_data.py[sure]
```

3. 训练排序模型[TODO-整理各部分的对应参数]
```
cd src/model_train
python train_listwise_multi_types_1.py
```

### 结果评价[完成整理]

需要自行修改config文件, 位置在/config/eda/eval_test.yaml

```
cd src/eda
bash eval_test.sh
```

若有更多问题可联系 cytan17726@stu.suda.edu.cn