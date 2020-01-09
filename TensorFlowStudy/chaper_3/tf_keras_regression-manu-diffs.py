'''

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

def customized_mse(y_true,y_pred):
    # tf.reduce_mean 求均值
    return tf.reduce_mean(tf.square(y_pred-y_true))




# learning_rate:[le-4,3e-4,le-3,3e-3,le-2,3e-2]
# w=w+grad*learning_rate
model = keras.models.Sequential([
    keras.layers.Dense(30,activation='relu',input_shape=x_train.shape[1:]),
    keras.layers.Dense(1),
])


# loss 损失函数：均方误差
# optimizer 优化器：批处理梯度下降
model.compile(loss=customized_mse,optimizer='sgd',
              metrics=["meaan_squared_error"])

callbacks =[
    keras.callbacks.EarlyStopping(patience=50, min_delta=1e-2),
]

# metric 使用
# metric 评价函数  MeanSquareError均方误差
# metric可以累加数据
metric = keras.metrics.MeanSquareError()
# 输出结果（5-2）**2=9
print(metric([5.],[2.]))
# 输出结果 （9+（1-0）**2）/2=5
print(metric([0.],[1.]))
# 输出 5
print(metric.result())

# 如果不想累加 之前的记录会清空
metric.reset_states()
# 与之前的记录无关；等于 （1-3）**2=4
print(metric([1.],[3.]))

# 1,batch 遍历训练集 metric
#   1.1 自动求导
# 2,epoch 结束 验证集 ，metric
epochs = 100
batch_size =32
steps_per_epoch = len(x_train_scaled)
optimizer = keras.optimizers.SGD()
metric = keras.metrics.MeanSquareError()

def random_batch(x,y,batch_size = 32):
    idx = np.random.randint(0,len(x),size=batch_size)
    return x[idx],y[idx]
model = keras.models.Sequential([
    keras.layers.Dense(30,activation = 'relu',input_shape=x_train.shape[1:]),
    keras.layers.Dense(1),
])

for epoch in range(epochs):
    metric.reset_states()
    for step in range(steps_per_epoch):
        x_batch,y_batch = random_batch(x_train_scaled,y_train,batch_size)

        with tf.GradientTape() as tape:
            y_pred = model(x_batch)
            loss = tf.reduce_mean(
                keras.losses.mean_squared_error(y_batch,y_pred))
            metric(y_batch,y_pred)

        grads = tape.gradient(loss,model.variables)
        grads_and_vars = zip(grads,model.variables)
        optimizer.apply_gradients(grads_and_vars)
        print("\rEpoch",epoch," train mse:",metric.result().numpy(),end="")

    y_valid_pred = model(x_valid_scaled)
    valid_loss = tf.reduce_mean(
        keras.losses.mean_squared_error(y_valid_pred,y_valid))
    print("\t","valid mse:",valid_loss.numpy())



def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.title(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))
    plt.gca().set_ylim(0,4)
    plt.show()

# plot_learning_curves(history)


model_evaluate =model.evaluate(x_test_scaled,y_test)
print("model_evaluate"+str(model_evaluate))
