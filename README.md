# [面向知识图谱问答的查询图生成方法](https://github.com/cytan17726/KBQA_QueryGraphGeneration/tree/master)


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

我们在网盘中提供了各阶段结果，包括：

- 新构建KBQA数据(ch2,后续开放结构更清晰的独立版本)
- 预处理数据与模型(关系预测提供训练后的模型)
    - 输入数据 (/data/dataset)
    - 关系预测模型 (/model/que_rel_sim)
- 查询图生成-阶段结果(/data/candidates)
- 查询图排序-训练数据(/data/train_data/graph_ranker)
- 查询图排序-已训练模型(/model/rank_model)
- 查询图排序-打分结果(/data/scores)

我们在本地评估性能结果如下,可作为您的参考:

|查询图生成性能|CCKS2019|CCKS2019-Comp|
|:---:|:---:|:---:|
|Yih等|85.13|71.07|
|Luo等|86.23|71.93|
|Ours|89.47|86.91|

|KBQA性能|CCKS2019|CCKS2019-Comp|
|:---:|:---:|:---:|
|Yih等|72.17|59.50|
|Luo等|73.29|60.66|
|Ours|73.86|73.39|


### 0 数据库构建(后续完善)

您可以基于[CCKS2019 中文知识图谱问答](https://www.biendata.xyz/competition/ccks_2019_6/data/)提供的数据, 构建KB

本系统使用Mysql数据库进行KB的存储与检索

更多详细信息后续添加完善

### 1 预处理(已完成)

包括 节点识别、关系预测

    节点识别, 提供实体词节点的识别结果, 其余内嵌在代码中
    关系预测提供已训练模型

### 2 查询图生成(已完成)
修改文件内的参数(对应 CCKS2019和CCKS2019-Comp两个数据集上的表现)

- Yih

```
cd src/build_query_graph
nohup python main_Yih_for_test.py > log/CCKS2019_Yih_test.log&
nohup python main_Yih_for_test.py > log/CCKS2019_Comp_Yih_test.log&
```

- Luo

```
cd src/build_query_graph
nohup python main_Luo_for_test.py > log/CCKS2019_Luo_test.log&
nohup python main_Luo_for_test.py > log/CCKS2019_Comp_Luo_test.log&
```

- Our

```
cd src/build_query_graph
nohup python main_based_filter_rel_for_test.py > log/CCKS2019_Ours_test.log&
nohup python main_based_filter_rel_for_test.py > log/CCKS2019_Comp_Ours_test.log&
```


### 3 查询图排序(已完成)

1. 序列化
需要调整文件中的对应参数(参考注释)
```
cd src/querygraph2seq
python querygraph_to_seq.py
```

2. 转化为模型输入数据

```
cd src/build_model_data
python build_test_data.py[将test候选查询图转化为模型输入格式]
python build_train_data_for_analysis.py [非必要,已经提供训练数据]
```

3. 训练排序模型

- 非必要步骤,我们于网盘中提供了已训练模型. 当时使用的训练配置参考对应目录下的文件
```
cd src/model_train
python train_listwise_multi_types_1.py
```

4. 候选打分

对应脚本为`src/model_train/infer.sh`
可按需修改config中参数(/config/eda)
```
gpu_id  显卡编号
infer_data  构建的训练数据
best_model_dir_name 预测使用的model目录
score_file 输出文件
```

### 4 结果评价

查询图生成

```
cd /src/build_query_graph
python cal_recall_with_multi_types.py
需要修改candsFile (line25起)，即需要评价的查询图候选文件
```

查询图排序

```
cd src/eda
bash eval_test.sh
需要对应修改config文件, 位置在/config/eda/eval_test.yaml
```

若有更多问题可联系 cytan17726@stu.suda.edu.cn