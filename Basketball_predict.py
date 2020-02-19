'''
实验流程：
1.获取比赛统计数据
2.比赛数据分析，得到代表每场比赛每支队伍 状态 的特征表达
3.利用机器学习方法学习每场比赛与胜利队伍的关系，并对2016-2017的比赛进行预测
'''
''' 
获取数据：2015-2016NBA Season Summary 【NBA赛季总结】 参考：Basketball Reference.com
    三个部分：
     T表       1）Team Per Game Stats【每支队伍平均每场比赛的表现统计】         ｝
     O表       2）Opponent Per Game Stats【所遇到的对手平均每场比赛的统计信息】 ｝  评估球队 过去 的战力
     M表       3）Miscellaneous Stats【综合统计数据】                         ｝ 
    
            4）【2015~2016 年的 NBA 常规赛及季后赛的每场比赛的比赛数据】       ｝  评估Elo score
    
            5）【2016~2017 年的 NBA 的常规赛比赛安排数据】                    ｝  进行预测
            
数据分析：
    Elo score 等级分制度  【Elo 的最初用于国际象棋中更好地对不同的选手进行等级划分。在现在
                           很多的竞技运动或者游戏中都会采取 Elo 等级分制度对选手或玩家进行
                           等级划分，如足球、篮球、棒球比赛或 LOL、DOTA 等游戏。】
    介绍参考：https://blog.csdn.net/BruceBorgia/article/details/103827360
'''
'''
    1.pandas库   ---官方文档：https://pandas.pydata.org/pandas-docs/stable/index.html
                 ---中文文档：https://www.jianshu.com/p/a77b0bc736f2
                【Pandas是作为Python数据分析著名的工具包，提供了多种数据选取的方法，方便实用】                
                相关BIF：
                    1) df[]         ---这种情况一次只能选取行或者列，即一次选取中，只能为行或
                                        者列设置筛选条件（只能为一个维度设置筛选条件）
                    2) df.loc[]
                       df.iloc[]    ---可以同时为多个维度设置筛选条件。
                       df.ix[]
                       
                    3) df.at[]      ---准确定位一个单元格
                       df.iat[]
                                        
                    4) drop         ---它不改变原有的df中的数据，而是返回另一个dataframe来存
                                        放删除后的数据。drop函数默认删除行，列需要加axis = 1
                                        
                    5) merge        ---作为DataFrame对象之间所有标准数据库连接操作的入口.
                                    参数：https://www.yiibai.com/pandas/python_pandas_merging_joining.html
                                        a.left  -- 一个DataFrame对象
                                        b.right -- 另一个DataFrame对象
                                        c.on    -- 列(名称)连接，必须在左和右DataFrame对象中存在(找到)
                                        d.how   -- 它是left, right, outer以及inner之中的一个，默认为内inner
                    
                    6) set_index    ---可以设置单索引和复合索引
                                    参数：https://blog.csdn.net/liu_liuqiu/article/details/99473357
                                        a.drop      --默认True
                                        b.inplace   --默认False
                    
                    7） iterrows    ---是在数据框中的行进行迭代的一个生成器，它返回每行的索引及一个包含行本身的对象
                                    参数：https://blog.csdn.net/Softdiamonds/article/details/80218777
                                       【Pandas的基础结构可以分为两种：数据框和序列。数据框是拥有轴标签的二维链表，
                                         换言之数据框是拥有标签的行和列组成的矩阵 - 列标签为列名，行标签为索引。
                                         Pandas中的行和列是Pandas序列 - 拥有轴标签的一维链表。】
                    
                    8） loc         ---以类似字典的方式来获取某一列的值
                                    参数：https://blog.csdn.net/xihuanzhi1854/article/details/89843272
                    
                    9)  items       ---将一个字典以dict_items的形式返回，因为字典是无序的，所以返回的列表也是无序的
                                        以（key，value）返回--为字典的遍历函数
                                    参数：https://www.runoob.com/python/att-dictionary-items.html
                            
'''
'''
    2.numpy库    ---
                相关BIF：
                    1）nan和inf     ---参考：https://www.jianshu.com/p/d9caa4ab46e1
                
                    2) nan_to_num(x)---使用numpy数组的过程中时常会出现nan或者inf的元素，可能会造成数值计算时的一些错误
                                        这里提供一个numpy库函数的用法，使nan和inf能够最简单地转换成相应的数值。
                                    参数：https://blog.csdn.net/u010158659/article/details/50814706/
    
'''
'''
    3.math库     ---中文文档：https://blog.csdn.net/zkzbhh/article/details/78384180
                相关BIF：
                    1) pow      ---计算幂次方
                    
                    
    
    4.csv库      ---官方文档：https://docs.python.org/2/library/csv.html
                相关BIF：
                    1) writer   ---返回一个writer对象，该对象负责将用户数据转换为给定文件状对象上的定界字符串
                                    参考：https://docs.python.org/2/library/csv.html
                    
                    2) writerow ---参考上面
                
    5.random库   ---中文文档：https://www.cnblogs.com/randysun/p/11202474.html
                相关BIF：
                    1) random   ---random() 方法返回随机生成的一个实数，它在[0,1)范围内。
'''
'''
    6.sklearn库  ---官方文档：https://scikit-learn.org/stable/
                相关BIF：
                    1) LogisticRegression   ---
                                参考：https://blog.csdn.net/mrxjh/article/details/78499801
                    
                    2) fit      ---：训练模型。
    
                    3) cross_val_score交叉验证   ---【过拟合：为了达到某种目的而过度严格】
                                参考：https://blog.csdn.net/qq_36523839/article/details/80707678
                                
                                    相关参数：https://www.cnblogs.com/lzhc/p/9175707.html
                                    
                    4) predict_proba---predict是训练后返回预测结果，是标签值。
                                       predict_proba返回的是一个 n 行 k 列的数组， 第 i 行 第 j 列上的数值是模型预测 第 
                                       i 个预测样本为某个标签的概率，并且每一行的概率和为1
                                       参考：https://www.cnblogs.com/mrtop/p/10309083.html                    
        
'''
import pandas as pd
import numpy as np
import math
import csv
import random
from sklearn import linear_model    # 线性模型
from sklearn.model_selection import cross_val_score     # 选型 cross_val_score --通过交叉验证评估分数

# 当每支队伍没有elo等级分时，赋予其基础elo等级分1600
base_elo = 1600
team_elos = {}
team_stats = {}
X = []
y = []
# 存放数据的目录
folder = 'data'

# 从 T、O 和 M 表格中读入数据，去除一些无关数据并将这三个表格通过Team属性列进行连接 ---
def initialize_data(Mstat, Ostat, Tstat):
    new_Mstat = Mstat.drop(['Rk', 'Arena'], axis=1)
    new_Ostat = Ostat.drop(['Rk', 'G', 'MP'], axis=1)       # 删除excel中带'...'的列
    new_Tstat = Tstat.drop(['Rk', 'G', 'MP'], axis=1)

    team_stats1 = pd.merge(new_Mstat, new_Ostat, how='left', on='Team')     # 合并两个数据帧，含有Team开头的列
    team_stats1 = pd.merge(team_stats1, new_Tstat, how='left', on='Team')   # 将按照左数据帧来合并，右数据帧抛弃
    return team_stats1.set_index('Team', inplace=False, drop=True)          # 将Team列数据设为行的单索引

# 获取每支队伍的Elo Score等级分函数，当在开始没有等级分时，将其赋予初始base_elo值   ---返回  字典｛key，value｝
def get_elo(team):
    try:
        return team_elos[team]      # { team1:elo , team2:elo... } 依照team里面的元素创建key和value
    except:
        # 当最初没有elo时，给每个队伍最初赋base_elo
        team_elos[team] = base_elo
        return team_elos[team]

# 计算每个球队的elo值
def calc_elo(win_team, lose_team):      # 根据笔记里面的elo算法
    winner_rank = get_elo(win_team)     # winner_rank等于RA
    loser_rank = get_elo(lose_team)     # loser_rank等于RB

    rank_diff = winner_rank - loser_rank    # RA-RB
    exp = (rank_diff  * -1) / 400
    odds = 1 / (1 + math.pow(10, exp))      # pow计算幂次方
    # 根据rank级别修改K值
    if winner_rank < 2100:
        k = 32
    elif winner_rank >= 2100 and winner_rank < 2400:
        k = 24
    else:
        k = 16
    # 更新 rank 数值
    new_winner_rank = round(winner_rank + (k * (1 - odds)))     # round() 方法返回浮点数x的四舍五入值
    new_loser_rank = round(loser_rank + (k * (0 - odds)))
    return new_winner_rank, new_loser_rank

# 基于我们初始好的统计数据，及每支队伍的 Elo score 计算结果，建立对应 2015~2016 年常
# 规赛和季后赛中每场比赛的数据集（在主客场比赛时，我们认为主场作战的队伍更加有优势一点，
# 因此会给主场作战队伍相应加上 100 等级分）
def  build_dataSet(all_data):
    print("Building data set..")
    X = []
    skip = 0
    for index, row in all_data.iterrows():

        Wteam = row['WTeam']        # 寻找row中包含 'WTeam' 的 序列 赋值给Wteam
        Lteam = row['LTeam']

        # 获取最初的elo或是每个队伍最初的elo值
        team1_elo = get_elo(Wteam)  # 得到一个 ｛Wteam1：elo1....｝的字典
        team2_elo = get_elo(Lteam)

        # 给主场比赛的队伍加上100的elo值
        if row['WLoc'] == 'H':
            team1_elo += 100
        else:
            team2_elo += 100

        # 把elo当为评价每个队伍的第一个特征值
        team1_features = [team1_elo]
        team2_features = [team2_elo]

        # 添加我们从basketball reference.com获得的每个队伍的统计信息
        for key, value in team_stats.loc[Wteam].items():       #将 elo 和每个队伍的统计信息合并在一起
            team1_features.append(value)
        for key, value in team_stats.loc[Lteam].items():
            team2_features.append(value)

        # 将两支队伍的特征值随机的分配在每场比赛数据的左右两侧
        # 并将对应的0/1赋给y值
        if random.random() > 0.5:
            X.append(team1_features + team2_features)
            y.append(0)
        else:
            X.append(team2_features + team1_features)
            y.append(1)

        if skip == 0:
            print('X',X)            #看不懂【打印X数据集的内容】*********************************************************************************
            skip = 1

        # 根据这场比赛的数据更新队伍的elo值
        new_winner_rank, new_loser_rank = calc_elo(Wteam, Lteam)
        team_elos[Wteam] = new_winner_rank
        team_elos[Lteam] = new_loser_rank

    return np.nan_to_num(X), y

# 最终在 main 函数中调用这些数据处理函数，使用 sklearn 的Logistic Regression方法建立回归模型
if __name__ == '__main__':             # __name__是内置的变量，就是判断下面是不是mian脚本，如果是别的脚本名称，就不执行
    Mstat = pd.read_csv(folder + '/15-16Miscellaneous_Stat.csv')
    Ostat = pd.read_csv(folder + '/15-16Opponent_Per_Game_Stat.csv')    #读取excel文件，带 绝对路径
    Tstat = pd.read_csv(folder + '/15-16Team_Per_Game_Stat.csv')

    team_stats = initialize_data(Mstat, Ostat, Tstat)

    result_data = pd.read_csv(folder + '/2015-2016_result.csv')
    X, y = build_dataSet(result_data)
    print(X)
    print()
    print(y)
    # 训练网络模型
    print("Fitting on %d game samples.." % len(X))

    model = linear_model.LogisticRegression()
    model.fit(X, y)

    # 利用10折交叉验证计算训练正确率 【 K折交叉验证 】原理：https://blog.csdn.net/qq_36523839/article/details/80707678
    print("Doing cross-validation..")  # 进行 交叉验证    【要知道 交叉验证 原理】
    print(cross_val_score(model, X, y, cv = 10, scoring='accuracy', n_jobs=-1).mean()) # mean干什么用的？#看不懂*********************************************************************************

# 利用模型对一场新的比赛进行胜负判断，并返回其胜利的概率   ---返回一个 标签 的 对应概率 列表
def predict_winner(team_1, team_2, model):
    features = []

    # team 1，客场队伍
    features.append(get_elo(team_1))
    for key, value in team_stats.loc[team_1].items():
        features.append(value)

    # team 2，主场队伍
    features.append(get_elo(team_2) + 100)
    for key, value in team_stats.loc[team_2].items():
        features.append(value)

    features = np.nan_to_num(features)
    return model.predict_proba([features])  #可以考虑使用predict 获取 预测胜利队伍

# 利用训练好的model在16-17年的比赛中进行预测
print('Predicting on new schedule..')
schedule1617 = pd.read_csv(folder + '/16-17Schedule.csv')
result = []
for index, row in schedule1617.iterrows():
    team1 = row['Vteam']
    team2 = row['Hteam']
    pred = predict_winner(team1, team2, model)
    prob = pred[0][0]       # 列表元素读取   获取每行 team1 team2 的胜率
    if prob > 0.5:
        winner = team1
        loser = team2
        result.append([winner, loser, prob])
    else:
        winner = team2
        loser = team1
        result.append([winner, loser, 1 - prob])

with open('16-17Result.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['win', 'lose', 'probability'])
    writer.writerows(result)
    print('done.')
print(pd.read_csv('16-17Result.csv',header=0))

