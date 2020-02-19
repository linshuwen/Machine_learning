import  pandas as pd
import  numpy  as np
import  matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler # 导入标准化函数，用于数据处理【标准化，降维，特征选择等】
from sklearn import linear_model
from sklearn.linear_model import  LogisticRegression  # 线性模型
from sklearn.model_selection import cross_val_score    # 选型 cross_val_score --通过交叉验证评估分数
from sklearn.model_selection import  KFold

file_train='paper_train_data.csv'
file_text='paper_text_data.csv'

pre_train=pd.read_csv(file_train,encoding='gbk')     # print(result.info()) 缺失值
pre_text=pd.read_csv(file_text,encoding='gbk')

train_data=pre_train.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
text_data=pre_text.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)

print('train_data:')
print(train_data)
print("***************************************************************************")
print('text_data:')
print(text_data)
print("***************************************************************************")
# train_data.hist()     【图表显示】
# plt.show()
# ********************************************************************************
train_array=train_data.values
train_X=train_array[:,4:7]
train_y=train_array[:,7]

text_array=text_data.values
text_X=text_array[:,4:7]
text_y=text_array[:,7]

print('transform train_X,train_y:')
scaler_train=StandardScaler().fit(train_X)  # 计算数据转换的方式 https://blog.csdn.net/qq_32806793/article/details/83037014
rescale_X_train=scaler_train.transform(train_X)   # 得到标准化处理后的数据   【特征缩放；标准化/归一化】
np.set_printoptions(precision=3) # 打印全部数组 ，precision设置float输出精度为3
print(rescale_X_train)
print(train_y)

print('transform text_X,text_y:')
scaler_text=StandardScaler().fit(text_X)
rescale_X_text=scaler_text.transform(text_X)
print(rescale_X_text)
print(text_y)
print("***************************************************************************")
# kfold=KFold(n_splits=10,shuffle=True,random_state=7)
# model = linear_model.LogisticRegression(solver='liblinear') # model.fit(X,y.astype('int'))

model=LogisticRegression()
model.fit(train_X,train_y.astype('int'))

train_X = np.nan_to_num(train_X)
preds=model.predict([train_X])   # 输入数据进行预测得到结果


# results_train=cross_val_score(model,train_X,train_y,cv=kfold,scoring='accuracy')

# print('results_train:')
# print(results_train.mean())
print("***************************************************************************")


