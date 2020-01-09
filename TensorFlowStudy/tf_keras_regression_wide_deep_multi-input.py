'''
wide&deep模型：
    关于：
        1，16年发布，用于分类和回归
        2，应用到了google play中的应用推荐
        3，原始论文：https://arxiv.org/pdf/1606.07792v1.pdf
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
# 多输入
input_wide = keras.layers.Input(shape=[5])
input_deep = keras.layers.Input(shape=[6])
hidden1 = keras.layers.Dense(30,activation='relu')(input_deep)
hidden2 = keras.layers.Dense(30,activation='relu')(hidden1)
concat = keras.layers.concatenate([input_wide,hidden2])
output = keras.layers.Dense(1)(concat)
model = keras.models.Model(inputs = [input_wide,input_deep],outputs=[output])

print(model.summary)

# loss 损失函数：均方误差
# optimizer 优化器：批处理梯度下降
model.compile(loss='mean_squared_error',optimizer='sgd')


# Tensorboard,earlystopping,ModelCheckpoint
logdir = './wide&deep_model-callbacks'
if not os.path.exists(logdir):
    os.mkdir(logdir)

output_model_file = os.path.join(logdir,"fashion_mnist_model.h5")

callbacks =[
    # keras.callbacks.TensorBoard(logdir),
    # keras.callbacks.ModelCheckpoint(output_model_file,save_best_only=True),
    keras.callbacks.EarlyStopping(patience=5, min_delta=1e-2),
]

x_train_scaled_wide = x_train_scaled[:,:5]
x_train_scaled_deep = x_train_scaled[:,2:]
x_valid_scaled_wide = x_valid_scaled[:,:5]
x_valid_scaled_deep = x_valid_scaled[:,2:]
x_test_scaled_wide = x_test_scaled[:,:5]
x_test_scaled_deep = x_test_scaled[:,2:]


history = model.fit([x_train_scaled_wide,x_train_scaled_deep], y_train,
            validation_data=([x_valid_scaled_wide,x_valid_scaled_deep],y_valid),
            epochs=50,
            callbacks = callbacks)

def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.title(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))
    plt.gca().set_ylim(0,2)
    plt.show()

plot_learning_curves(history)

model_evaluate =model.evaluate([x_test_scaled_wide,x_test_scaled_deep],y_test)
print("model_evaluate"+str(model_evaluate))
