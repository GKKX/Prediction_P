# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 20:28:29 2023

@author: 21039
"""

from datetime import datetime
start_time = datetime.now()

from xgboost import XGBRegressor
#import xgboost as xgb
from sklearn import metrics
import numpy as np
from sklearn.model_selection import KFold,cross_validate as CVS,cross_val_predict as CVP,train_test_split as TTS
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import random
import joblib

# 载入数据集
## Load the dataset
data=pd.read_excel(r'xlsx/MMOFP.xlsx')
data=data.dropna(axis=0) ## Missing value handling

##Divide the data set
X = data.iloc[:,np.r_[1:10]].values
X = X.astype(np.float64)
Y = data.iloc[:,11].values
Y = Y.astype(np.float64)
X_train, X_test, y_train, y_test = TTS(X, Y, test_size=0.3, random_state=42)
#data2 = pd.read_excel(io='F:/class/ST/NEW_ya/MMOFP.xlsx')
#data1=data2.dropna(axis=0)

##划分数据集
##X1=pd.read_excel(io=r'C:/Users/Lenovo/Desktop/Ya-Guan/MMOFP.xlsx',usecols=[1,2,3,4,5,6,7,8,9],names=None)#列从0开始计算
##Y1=pd.read_excel(io=r'C:/Users/Lenovo/Desktop/Ya-Guan/MMOFP.xlsx',usecols=[11],names=None)

# 2、缺失值处理
#X=X.dropna(axis=0)
#Y=Y.dropna(axis=0)

#划分数据集
X_train, X_test, y_train, y_test = TTS(X, Y, test_size=0.3)
				
            
# 拟合XGBoost模型
model = XGBRegressor(  booster='gbtree',#gblinear 
                       #objective ='reg;squarederror',
                       gamma=0.1,
                       max_depth=6,
                       reg_alpha=0.8, 
                       reg_lambda=1,
                       subsample=0.8,#对于每棵树，随机采样的比例
                       min_child_weight=0.7,
                       nthread=4,
                       learning_rate=0.1, 
                       n_estimators=500, 
                       colsample_bytree=1,#每棵树每次节点分裂的时候列采样的比例
                       n_jobs=-1,             
                       seed=1314).fit(X_train, y_train)

## save model
joblib.dump(model, "model/xgboost.pt")
##预测
Y_predict1 = model.predict(X_train)
Y_predict2 = model.predict(X_test)


##模型评估
#训练集
MSE=metrics.mean_squared_error(y_train,Y_predict1)
print('MSE_train={}'.format(MSE))
MAE=metrics.mean_absolute_error(y_train,Y_predict1)
print('MAE_train={}'.format(MAE))
RMSE=np.sqrt(metrics.mean_squared_error(y_train,Y_predict1))  # RMSE
print('RMSE_train={}'.format(RMSE))
R2=metrics.r2_score(y_train,Y_predict1)
print('R2_train={}'.format(R2))

#测试集
MSE=metrics.mean_squared_error(y_test,Y_predict2)
print('MSE_test={}'.format(MSE))
MAE=metrics.mean_absolute_error(y_test,Y_predict2)
print('MAE_test={}'.format(MAE))
RMSE=np.sqrt(metrics.mean_squared_error(y_test,Y_predict2))  # RMSE
print('RMSE_test={}'.format(RMSE))
R2=metrics.r2_score(y_test,Y_predict2)
print('R2_test={}'.format(R2))

end_time = datetime.now()
#交叉验证
#交叉验证定义
cv = KFold(n_splits=10 ,shuffle=True,random_state=1402)
#交叉验证的结果
scores= CVS(model,X,Y,cv=cv,return_train_score=True
                   ,scoring=('r2', 'neg_mean_squared_error','neg_mean_absolute_error')
                   ,verbose=True,n_jobs=-1)
train_CVS_=(scores['train_neg_mean_squared_error'])
test_CVS_=(scores['test_neg_mean_squared_error'])
train_CVS_r2=scores['train_r2']
test_CVS_r2=scores['test_r2']
train_CVS_RMSE=(abs(train_CVS_)**0.5)
test_CVS_RMSE=(abs(test_CVS_)**0.5)
train_CVS_MAE = -scores['train_neg_mean_absolute_error']
test_CVS_MAE = -scores['test_neg_mean_absolute_error']

#绘图
plt.scatter(Y_predict1,y_train)
plt.scatter(Y_predict2,y_test)
print('Duration: {}'.format(end_time - start_time))