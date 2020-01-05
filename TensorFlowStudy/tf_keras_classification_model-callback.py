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
model.add(keras.layers.Dense(300,activation='sigmoid'))
model.add(keras.layers.Dense(200,activation='sigmoid'))
model.add(keras.layers.Dense(10,activation='softmax'))
# 损失函数：交叉熵
# 优化器：批处理梯度下降
# 指标：精度
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])


# EarlyStopping
  # keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
    # monitor: 被监测的数据。
    # min_delta: 在被监测的数据中被认为是提升的最小变化， 例如，小于 min_delta 的绝对变化会被认为没有提升。
    # patience: 没有进步的训练轮数，在这之后训练就会被停止。
    # verbose: 详细信息模式。
    # mode: {auto, min, max} 其中之一。 在 min 模式中， 当被监测的数据停止下降，训练就会停止；在 max 模式中，当被监测的数据停止上升，训练就会停止；在 auto 模式中，方向会自动从被监测的数据的名字中判断出来。
    # baseline: 要监控的数量的基准值。 如果模型没有显示基准的改善，训练将停止。
    # restore_best_weights: 是否从具有监测数量的最佳值的时期恢复模型权重。 如果为 False，则使用在训练的最后一步获得的模型权重。

# ModelCheckpoint
  # keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    # filepath: 字符串，保存模型的路径。
    # monitor: 被监测的数据。
    # verbose: 详细信息模式，0 或者 1 。
    # save_best_only: 如果 save_best_only=True， 被监测数据的最佳模型就不会被覆盖。
    # mode: {auto, min, max} 的其中之一。 如果 save_best_only=True，那么是否覆盖保存文件的决定就取决于被监测数据的最大或者最小值。 对于 val_acc，模式就会是 max，而对于 val_loss，模式就需要是 min，等等。 在 auto 模式中，方向会自动从被监测的数据的名字中判断出来。
    # save_weights_only: 如果 True，那么只有模型的权重会被保存 (model.save_weights(filepath))， 否则的话，整个模型会被保存 (model.save(filepath))。
    # period: 每个检查点之间的间隔（训练轮数）。

# TensorBoard
  # keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
    # log_dir: 用来保存被 TensorBoard 分析的日志文件的文件名。
    # histogram_freq: 对于模型中各个层计算激活值和模型权重直方图的频率（训练轮数中）。 如果设置成 0 ，直方图不会被计算。对于直方图可视化的验证数据（或分离数据）一定要明确的指出。
    # write_graph: 是否在 TensorBoard 中可视化图像。 如果 write_graph 被设置为 True，日志文件会变得非常大。
    # write_grads: 是否在 TensorBoard 中可视化梯度值直方图。 histogram_freq 必须要大于 0 。
    # batch_size: 用以直方图计算的传入神经元网络输入批的大小。
    # write_images: 是否在 TensorBoard 中将模型权重以图片可视化。
    # embeddings_freq: 被选中的嵌入层会被保存的频率（在训练轮中）。
    # embeddings_layer_names: 一个列表，会被监测层的名字。 如果是 None 或空列表，那么所有的嵌入层都会被监测。
    # embeddings_metadata: 一个字典，对应层的名字到保存有这个嵌入层元数据文件的名字。 查看 详情 关于元数据的数据格式。 以防同样的元数据被用于所用的嵌入层，字符串可以被传入。
    # embeddings_data: 要嵌入在 embeddings_layer_names 指定的层的数据。 Numpy 数组（如果模型有单个输入）或 Numpy 数组列表（如果模型有多个输入）。 Learn ore about embeddings。
    # update_freq: 'batch' 或 'epoch' 或 整数。当使用 'batch' 时，在每个 batch 之后将损失和评估值写入到 TensorBoard 中。同样的情况应用到 'epoch' 中。如果使用整数，例如 10000，这个回调会在每 10000 个样本之后将损失和评估值写入到 TensorBoard 中。注意，频繁地写入到 TensorBoard 会减缓你的训练。
# tensorboard --logdir=callbacks 查看生成TensorBoard 。注意要在当前目录下执行，不能跑到callbacks。这个问题坑了一个小时！！！



# Tensorboard,earlystopping,ModelCheckpoint
logdir = './callbacks'
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