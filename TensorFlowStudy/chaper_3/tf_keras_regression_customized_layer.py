'''
自定义larer
'''

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import sklearn
import pandas as pd
import sys
import time
import os
import tensorflow as tf
from tensorflow import keras

#    数据集
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
print("housing.DESCR" + housing.DESCR)
print("data" + str(housing.data.shape))
print("hausing.target" + str(housing.target.shape))

from sklearn.model_selection import train_test_split

'''
train_test_split是交叉验证中常用的函数，功能是从样本中随机的按比例选取train data和testdata
X_train,X_test, y_train, y_test = train_test_split(train_data,train_target,test_size=0.4, random_state=0)

train_data：所要划分的样本特征集
train_target：所要划分的样本结果
test_size：样本占比，如果是整数的话就是样本的数量
random_state：是随机数的种子。
'''
x_train_all, x_test, y_train_all, y_test = train_test_split(
    housing.data, housing.target, random_state=7, test_size=0.5
)
x_train, x_valid, y_train, y_valid = train_test_split(
    x_train_all, y_train_all, random_state=11, test_size=0.8
)
print("x_train", x_train.shape, y_train.shape)
print("x_valid", x_valid.shape, y_valid.shape)
print("x_test", x_test.shape, y_test.shape)

# 归一化训练，验证，测试数据 x=(x-u)/std(u:均值，std:方差)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_valid_scaled = scaler.fit_transform(x_valid)
x_test_scaled = scaler.fit_transform(x_test)


def customized_mse(y_true, y_pred):
    # tf.reduce_mean 求均值
    return tf.reduce_mean(tf.square(y_pred - y_true))

# Lambda
# tf.nn.softplus:log(1+e^x)
customized_softplus = keras.layers.Lambda(lambda x:tf.nn.softplus(x))
print(customized_softplus([-10.,-5.,0.,5.,10.]))


# 自定义层次模型
class CustomizedDenseLayer(keras.layers.Layer):
    def __init__(self,units,activation=None,**kwargs):
        self.units = units
        self.activation = keras.layers.Activation(activation)
        super(CustomizedDenseLayer, self).__init__(**kwargs)
    def build(self,input_shape):
        '''构建所需要的参数'''
        # x*w+b.input_shape:[None,a] w:[a,b]output_shape:[None]
        self.kernel = self.add_weight(name = 'kernel',
                                      shape=(input_shape[1],self.units),
                                      initializer = 'uniform',
                                      trainable =True)
        self.bias = self.add_weight(name = 'bias',
                                    shape = (self.units,),
                                    initializer = 'zeros',
                                    trainable =True)
        super(CustomizedDenseLayer,self).build(input_shape)
    def call(self,x):
        '''完整正向计算'''
        return self.activation(x @ self.kernel+self.bias)

# learning_rate:[le-4,3e-4,le-3,3e-3,le-2,3e-2]
# w=w+grad*learning_rate
model = keras.models.Sequential([
    CustomizedDenseLayer(30, activation='relu', input_shape=x_train.shape[1:]),
    CustomizedDenseLayer(1),

    # 等价于：
        # keras.layers.Dense(1,activation="softplus")
        # 或
        # keras.layers.Dense(1),keras.layers.Dense("softplus")
    customized_softplus
])

# loss 损失函数：均方误差
# optimizer 优化器：批处理梯度下降
model.compile(loss=customized_mse, optimizer='sgd',
              metrics=["mean_squared_error"])

callbacks = [
    keras.callbacks.EarlyStopping(patience=50, min_delta=1e-2),
]
history = model.fit(x_train_scaled, y_train, epochs=10,
                    validation_data=(x_valid_scaled, y_valid),
                    callbacks=callbacks)


def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.title(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    plt.gca().set_ylim(0, 4)
    plt.show()


plot_learning_curves(history)

model_evaluate = model.evaluate(x_test_scaled, y_test)
print("model_evaluate" + str(model_evaluate))
