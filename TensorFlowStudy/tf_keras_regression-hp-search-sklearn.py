'''
    超参数搜索，执行有问题，待解决
'''

import  matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import sklearn
import pandas as pd
import sys
import time
import  os
import tensorflow as tf
from tensorflow import keras



#    数据集
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
print("housing.DESCR"+housing.DESCR)
print("data"+str(housing.data.shape))
print("hausing.target"+str(housing.target.shape))

from sklearn.model_selection import train_test_split

'''
train_test_split是交叉验证中常用的函数，功能是从样本中随机的按比例选取train data和testdata
X_train,X_test, y_train, y_test = train_test_split(train_data,train_target,test_size=0.4, random_state=0)

train_data：所要划分的样本特征集
train_target：所要划分的样本结果
test_size：样本占比，如果是整数的话就是样本的数量
random_state：是随机数的种子。
'''
x_train_all,x_test,y_train_all,y_test = train_test_split(
    housing.data,housing.target,random_state=7,test_size=0.5
)
x_train,x_valid,y_train,y_valid = train_test_split(
    x_train_all,y_train_all,random_state=11,test_size=0.8
)
print("x_train",x_train.shape,y_train.shape)
print("x_valid",x_valid.shape,y_valid.shape)
print("x_test",x_test.shape,y_test.shape)

# 归一化训练，验证，测试数据 x=(x-u)/std(u:均值，std:方差)
from  sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_valid_scaled = scaler.fit_transform(x_valid)
x_test_scaled = scaler.fit_transform(x_test)

# RandomizedSearchCV
# 1,转换为sklearn 的model
# 2,定义参数集合
# 3，搜索参数

def build_model(hidden_layers=1, layer_size=30, learning_rate=3e-3):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(layer_size,activation='relu',input_shape=x_train.shape[1:]))
    for _ in range(hidden_layers-1):
        model.add(keras.layers.Dense(layer_size,activation='relu'))
    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.SGD(learning_rate)
    model.compile(loss='mse',optimizer=optimizer)
    return model

callbacks = [
    keras.callbacks.EarlyStopping(patience=5, min_delta=1e-2),
]


from scipy.stats import reciprocal
param_distribution={
    "hidden_layers": [1, 2, 3, 4],
    "layer_size": np.arange(1, 100),
    "learning_rate": reciprocal(1e-4, 1e-2),
}

'''
为什么查找超参数只训练7k多？
    cross_validation:训练集分成n份，n-1训练，最后一份验证
'''
from sklearn.model_selection import RandomizedSearchCV

sklearn_model = keras.wrappers.scikit_learn.KerasRegressor(build_fn=build_model)

# n_iter 随机寻找参数组合的数量，默认值为10。
# n_jobs 并行计算时使用的计算机核心数量，默认值为1。当n_jobs的值设为-1时，则使用所有的处理器。
# cv = 5,  # 交叉验证 份数
random_search_cv = RandomizedSearchCV(sklearn_model,param_distribution,n_iter=10,n_jobs=1)

random_search_cv.fit(x_train_scaled, y_train, epochs=10, validation_data=(x_valid_scaled, y_valid), callbacks=callbacks)

print("random_search_cv.best_params_="+str(random_search_cv.best_params_))
print("random_search_cv.best_score_="+str(random_search_cv.best_score_))
print("random_search_cv.best_estimator_"+str(random_search_cv.best_estimator_))

model = random_search_cv.best_estimator_.__module__
model_evaluate = model.evluate(x_test_scaled,y_test)
print(model_evaluate)

def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.title("sklearn_model")
    plt.gca().set_ylim(0, 1)
    plt.show()
# plot_learning_curves(history)