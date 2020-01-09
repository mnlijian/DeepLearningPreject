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

def f(x):
    return 3.* x ** 2+2. * x-1

def approximate_derivative(f,x,eps=1e-3):
    return (f(x+eps)-f(x-eps))/(2. * eps)


print(approximate_derivative(f,1.))




def g(x1,x2):
     return (x1+5)*(x2**2)

def approximate_gradient(g,x1,x2,eps=1e-3):
    dg_x1 = approximate_derivative(lambda x:g(x,x2),x1,eps)
    dg_x2 = approximate_derivative(lambda x:g(x1,x),x2,eps)
    return dg_x1,dg_x2

print(approximate_gradient(g,2.,3.))

x1 = tf.Variable(2.0)
x2 = tf.Variable(3.0)

# tape 只能用一次
with tf.GradientTape() as tape:
    z = g(x1,x2)
#     执行完后tape就会被消解
dz_x1 = tape.gradient(z,x1)
print(dz_x1)

try:
    # 不会被执行 抛出 GradientTape.gradient can only be called once on non-persistent tapes.
    dz_x2 = tape.gradient(z,x2)
    print(dz_x2)
except RuntimeError as ex:
    print(ex)


x1 = tf.Variable(2.0)
x2 = tf.Variable(3.0)

# persistent= True tape系统不会释放，需要手动释放
with tf.GradientTape(persistent= True) as tape:
    z = g(x1,x2)

dz_x1 = tape.gradient(z,x1)
dz_x2 = tape.gradient(z,x2)
print(dz_x1,dz_x2)
# 释放资源
del tape


x1 = tf.Variable(2.0)
x2 = tf.Variable(3.0)

# persistent= True tape系统不会释放，需要手动释放
with tf.GradientTape() as tape:
    z = g(x1,x2)
# 求得是链表
dz_x1x2 = tape.gradient(z,[x1,x2])
print(dz_x1x2)



x1 = tf.Variable(2.0)
x2 = tf.Variable(3.0)

# persistent= True tape系统不会释放，需要手动释放
with tf.GradientTape() as tape:
    tape.watch(x1)
    tape.watch(x2)
    z = g(x1,x2)
# 求得是链表
dz_x1x2 = tape.gradient(z,[x1,x2])
print(dz_x1x2)


# 两个函数对同一个自变量求导
x = tf.Variable(5.0)
with tf.GradientTape() as tape:
    z1 = 3*x
    z2 = x**2
tape.gradient([z1,z2],x)

# 二阶求导
x1 = tf.Variable(2.0)
x2 = tf.Variable(3.0)
with tf.GradientTape(persistent=True) as outer_tape:
    with tf.GradientTape(persistent=True) as inner_tape:
        z = g(x1,x2)
    inner_grads = inner_tape.gradient(z,[x1,x2])
outer_grads = [outer_tape.gradient(inner_grad,[x1,x2])
               for inner_grad in inner_grads]

print(outer_grads)
del inner_tape
del outer_tape


# 求梯度
learning_rate = 0.1
x = tf.Variable(0.0)

for _ in range(100):
    with tf.GradientTape() as tape:
        z = f(x)
    dz_dx = tape.gradient(z,x)
    x.assign_sub(learning_rate*dz_dx)
print(x)



learning_rate = 0.1
x = tf.Variable(0.0)

optimizer = keras.optimizers.SGD(lr=learning_rate)
for _ in range(100):
    with tf.GradientTape() as tape:
        z = f(x)
    dz_dx = tape.gradient(z,x)
    optimizer.apply_gradients([(dz_dx,x)])
print(x)

