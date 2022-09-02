# 批量删除查询图生成进程

ps -au|grep "python main_based_filter_rel_for_train"|grep -v grep|awk '{print "kill -9 " $2}'|sh

ps -au|grep "python main_based_filter_rel_for_test"|grep -v grep|awk '{print "kill -9 " $2}'|sh

# 师兄的test 查询图性能

false: 163 true: 1067
1071.25399309431 1390 0.7706863259671296