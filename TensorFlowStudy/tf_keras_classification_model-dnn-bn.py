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
# 本数据集包含60,000个28x28灰度图像，共10个时尚分类作为训练集。测试集包含10,000张图片
fashion_mnist = keras.datasets.fashion_mnist
# X_train和X_test：是形如（nb_samples, 28, 28）的灰度图片数据，数据类型是无符号8位整形（uint8）
# y_train和y_test：是形如（nb_samples,）标签数据，标签的范围是0~9
(x_train_all,y_train_all),(x_test,y_test) = fashion_mnist.load_data()
x_valid,x_train = x_train_all[:5000],x_train_all[5000:]
y_valid,y_train = y_train_all[:5000],y_train_all[5000:]

# x = (x-u)/std 归一化
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(
    x_train.astype(np.float32).reshape(-1,1)).reshape(-1,28,28)
x_valid_scaled = scaler.transform(
    x_valid.astype(np.float32).reshape(-1,1)).reshape(-1,28,28)
x_test_scaled = scaler.transform(
    x_test.astype(np.float32).reshape(-1,1)).reshape(-1,28,28)


model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28,28]))
for _ in range(20):
    model.add(keras.layers.Dense(100,activation="relu"))
    # 实现批归一化 （归一化放在之后）
    # 批归一化可以加快网络的收敛速度。避免梯度消失；
    model.add(keras.layers.BatchNormalization())
    # 批归一化放在之前的操作
    """
    model.add(keras.layers.Dense(100))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.laryers.Activation('relu'))
    
    """
model.add(keras.layers.Dense(10,activation ='softmax'))
# loss 损失函数：交叉熵
# optimizer 优化器：批处理梯度下降
# metrics 指标：精度
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])


# Tensorboard,earlystopping,ModelCheckpoint
logdir = './dnn-bn-callbacks'
if not os.path.exists(logdir):
    os.mkdir(logdir)

output_model_file = os.path.join(logdir,"fashion_mnist_model.h5")

callbacks =[
    keras.callbacks.TensorBoard(logdir),
    keras.callbacks.ModelCheckpoint(output_model_file,save_best_only=True),
    keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3),

]

history = model.fit(x_train_scaled,y_train, epochs=10,
            validation_data=(x_valid_scaled,y_valid),
            callbacks = callbacks)


def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,2)
    plt.show()

plot_learning_curves(history)