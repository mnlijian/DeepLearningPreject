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

t = tf.constant([[1.,2.,3.],[4.,5.,6.]])

# index
print(t)
# 取除第0列的所有列（2，3列）
print(t[:,1:])
# 取最后一列
print(t[...,1])

# 加
print(t+10)
# 平方
print(tf.square(t))
# 乘自己转置
print(t @ tf.transpose(t))

# numpy conversion
print(t.numpy())
# 用np 计算 t 的 平方
print(np.square(t))
np_t = np.array([[1.,2.,3.],[4.,5.,6.]])
# np 转换成tf
print(tf.constant(np_t))

# Scalars  标量 0 纬
t = tf.constant(2.812)
print(t.numpy())
print(t.shape)

t = tf.constant("cafe")
print(t)
print(tf.strings.length(t))
print(tf.strings.length(t,unit='UTF8_CHAR'))
print(tf.strings.unicode_decode(t,"UTF8"))

t = tf.constant(["cafe","coffee","咖啡"])
print(tf.strings.length(t, unit="UTF8_CHAR"))
r = tf.strings.unicode_decode(t, "UTF8")
# tf.RaggedTensor
print(r)

# RaggedTensor  tensorflow2.0
r = tf.ragged.constant([[11,12],[21,22,23],[],[41]])
# index op
print(r)
print(r[1])
print(r[1:2])

# ops on ragged tensor
r2 = tf.ragged.constant([[51,52],[],[71]])

# 按照行的方式拼接起来
print(tf.concat([r,r2],axis = 0))

# 如果要列向拼接会报错，因为按照列的元素对应个数不同
# print(tf.concat([r,r2],axis = 1))

# 变成普通的tensor函数  用 0 补位，0跟在数字的后面位
print(r.to_tensor())


# sparse tensor 稀疏矩阵
# indices 代表哪位有数字【0，1】代表第零行第一列有数字  (必须是排好序) 不能是【0，2】【0，1】
# values indices对应位值的数组
# dense_shape 矩阵的形状
s = tf.SparseTensor(indices = [[0,1],[1,0],[2,3]],
                    values=[1.,2.,3.],
                    dense_shape = [3,4])
print(s)
print(tf.sparse.to_dense(s))

# ops on spars tensors
s2 = s*2.0
print(s2)
try:
    s3=s+1
except TypeError as ex:
    print(ex)

s4 = tf.constant([[10.,20.],
                  [30.,40.],
                  [50.,60.],
                  [70.,80.]])
print(tf.sparse.sparse_dense_matmul(s,s4))


s5 = tf.SparseTensor(indices = [[0,2],[0,1],[2,3]],
                    values=[1.,2.,3.],
                    dense_shape = [3,4])
print(s5)
# 如果 SparseTensor 没有排好序 直接调用to_dense会报错，需要调用以下方法进行转化
s6 = tf.sparse.reorder(s5)
print(tf.sparse.to_dense(s6))


# Variables
v= tf.Variable([[1.,2.,3.],[4.,5.,6.]])
print(v)
print(v.value)
print(v.numpy)

# assign value 可以被重新赋值;
v.assign(2*v)
print(v.numpy())
v[0,1].assign(42)
print(v.numpy)
v[1].assign([7.,8.,9.])
print(v.numpy())


# 赋值只能用assign 不能用 =
# TypeError: 'ResourceVariable' object does not support item assignment
try:
    v[1] = [7.,8.,9.]
except TypeError as ex:
    print(ex)