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

print(tf.__version__)
print(sys.version_info)
for module in mpl,np,pd,sklearn,tf,keras:
    print(module.__name__,module.__version__)



#    数据集
# 本数据集包含60,000个28x28灰度图像，共10个时尚分类作为训练集。测试集包含10,000张图片
fashion_mnist = keras.datasets.fashion_mnist
# X_train和X_test：是形如（nb_samples, 28, 28）的灰度图片数据，数据类型是无符号8位整形（uint8）
# y_train和y_test：是形如（nb_samples,）标签数据，标签的范围是0~9
(x_train_all,y_train_all),(x_test,y_test) = fashion_mnist.load_data()
x_valid,x_train = x_train_all[:5000],x_train_all[5000:]
y_valid,y_train = y_train_all[:5000],y_train_all[5000:]

# print(x_valid.shape,y_valid.shape)
# print(x_train.shape,y_train.shape)
# print(x_test.shape,y_test.shape)
# def show_single_image(img_arr):
#     plt.imshow(img_arr,cmap='binary')
#     plt.show()
#
# show_single_image(x_train[0])

# 显示n 张图片
# def show_imgs(n_rows,n_cols,x_data,y_data,class_names):
#     assert len(x_data)==len(y_data)
#     assert n_rows*n_cols<len(x_data)
#     plt.figure(figsize=(n_cols*1.4,n_rows*1.6))
#     for row in range(n_rows):
#         for col in range(n_cols):
#             index = n_cols*row+col
#             plt.subplot(n_rows,n_cols,index+1)
#             plt.imshow(x_data[index],cmap='binary',interpolation='nearest')
#             plt.axis('off')
#             plt.title(class_names[y_data[index]])
#     plt.show()
#
# class_names =['T-shirt','Trouser','Pullover','Dress','Coat',
#               'Sandal','Shirt','Sneaker','Bag','Ankle boot']
# show_imgs(3,5,x_train,y_train,class_names)
#

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28,28]))
model.add(keras.layers.Dense(300,activation='sigmoid'))
model.add(keras.layers.Dense(200,activation='sigmoid'))
model.add(keras.layers.Dense(10,activation='softmax'))
# 损失函数：交叉熵
# 优化器：批处理梯度下降
# 指标：精度
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

# model.layers
# model.summary()


history = model.fit(x_train,y_train,epochs=10,
          validation_data=(x_valid,y_valid))


def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()

plot_learning_curves(history)