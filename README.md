# 面向知识图谱问答的查询图生成方法

hi! 你好!

本项目是关于一种查询图生成方法的实现代码，用于知识图谱问答。其中还附有我们构建的一份含有多种复杂句的中文知识图谱问答数据集(/CCKS2019_Comp)

更多细节可参考我们的工作: __谈川源, 贾永辉, 陈跃鹤, 陈文亮. 面向知识图谱问答的查询图生成方法. CCKS2022.__

转投期刊中, 待接收后更新最终稿获取方式. 目前您可以email我获取会议提交版.

## ⚙️Setups

代码在以下环境中测试:

- python 3.6.7
  - `torch`==1.4.0
  - `pytorch_pretrained_bert`==0.6.2
  - `scikit-learn`==0.20.4
  - `xgboost`==1.4.1
  - `pymysql`==1.0.2
  - `jieba`==0.42.1
  - `tqdm`
  - `yaml`

训练的模型和部分中间结果:
[下载地址(百度网盘)](https://pan.baidu.com/s/1UzczuOdBNAwjP9h8Sf0cjA), 提取码: vbab

## 🚀快速复现实验结果

我们在网盘中提供了各阶段结果, 包括:

- 预处理数据与模型(关系预测提供训练后的模型)
  - 输入数据 (/data/dataset)
  - 关系预测模型 (/model/que_rel_sim)
- 查询图生成-阶段结果(/data/candidates)
- 查询图排序-训练数据(/data/train_data/graph_ranker)
- 查询图排序-已训练模型(/model/rank_model)
- 查询图排序-打分结果(/data/scores)

我们在本地评估性能结果如下, 可作为您的参考:

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

### 0️⃣ 数据库构建(后续完善)

您可以基于[CCKS2019 中文知识图谱问答](https://www.biendata.xyz/competition/ccks_2019_6/data/)提供的数据, 构建KB

本系统使用Mysql数据库进行KB的存储与检索

更多详细信息后续添加完善

### 1️⃣ 预处理(已完成)

包括 节点识别、关系预测

    节点识别, 提供实体词节点的识别结果, 其余内嵌在代码中
    关系预测提供已训练模型

### 2️⃣ 查询图生成(已完成)

修改文件内的参数(对应 CCKS2019和CCKS2019-Comp两个数据集上的表现)

- Yih

```bash
cd src/build_query_graph

# CCKS2019 line29-32
nohup python main_Yih_for_test.py > log/CCKS2019_Yih_test.log&
# CCKS2019-Comp line33-36
nohup python main_Yih_for_test.py > log/CCKS2019_Comp_Yih_test.log&
```

- Luo

```bash
cd src/build_query_graph

# CCKS2019 line26-29
nohup python main_Luo_for_test.py > log/CCKS2019_Luo_test.log&
# CCKS2019-Comp line30-33
nohup python main_Luo_for_test.py > log/CCKS2019_Comp_Luo_test.log&
```

- Our

```bash
cd src/build_query_graph
# CCKS2019 line30-33
nohup python main_based_filter_rel_for_test.py > log/CCKS2019_Ours_test.log&
# CCKS2019-Comp line34-37
nohup python main_based_filter_rel_for_test.py > log/CCKS2019_Comp_Ours_test.log&
```

### 3️⃣ 查询图排序(已完成)

3.1 序列化

需要调整文件中的对应参数(参考注释)

```bash
cd src/querygraph2seq
python querygraph_to_seq.py
```

3.2 转化为模型输入数据

``` bash
cd src/build_model_data
# 将test候选查询图转化为模型输入格式
python build_test_data.py   

# 非必要,已经提供训练数据
python build_train_data_for_analysis.py 
```

3.3 训练排序模型

非必要步骤,我们于网盘中提供了已训练模型. 当时使用的训练配置参考对应目录下的文件

```bash
cd src/model_train
python train_listwise_multi_types_1.py
```

3.4 候选打分

可按需修改config中参数(/config/eda)

```yaml
gpu_id: '显卡编号'
infer_data: '构建的训练数据-from step3.2'
best_model_dir_name: '预测使用的model目录'
score_file: '输出文件'
```

预测脚本为`src/model_train/infer.sh`

```bash
cd src/model_train/
bash infer.sh
```

### 4️⃣ 结果评价

4.1 查询图生成

```bash
cd /src/build_query_graph

# 需要修改candsFile (line25起)，即需要评价的查询图候选文件
python cal_recall_with_multi_types.py
```

4.2 查询图排序

```bash
cd src/eda

# 需要对应修改config文件, 位置在/config/eda/eval_test.yaml
bash eval_test.sh
```

## 后续工作、问题与建议

这是我初次进行相关工作, 上述内容可能不甚完善。请各位老师多多批评指正, 我会及时调整。未尽事宜可联系: cytan17726@stu.suda.edu.cn