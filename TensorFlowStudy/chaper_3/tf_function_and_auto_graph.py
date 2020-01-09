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

# tf.function and auto-graph.
def scaled_elu(z,scale=1.0,alpha=1.0):
    #z>=0?scale*z:scale*alpha*tf.nn.elu(z)
    is_positive = tf.greater_equal(z,0.0)
    return scale*tf.where(is_positive,z,alpha*tf.nn.elu(z))

print(scaled_elu(tf.constant(-3.)))
print(scaled_elu(tf.constant([-3,-2.5])))

scaled_elu_tf = tf.function(scaled_elu)

print(scaled_elu_tf(tf.constant(-3.)))
print(scaled_elu_tf(tf.constant([-3,-2.5])))

print(scaled_elu_tf.python_function is scaled_elu)

# 转化后的优势就是快


def converge_to_2(n_iters):
    total = tf.constant(0.)
    increment = tf.constant(1.)
    for _ in range(n_iters):
        total+=increment
        increment /=2.0
    return total

print(converge_to_2(20))


var = tf.Variable(0.)

# 静态的，所有的参数需要提前初始化好；
@tf.function
def add_21():
    return var.assign_add(21)
print(add_21())

@tf.function(input_signature=[tf.TensorSpec([None],tf.int32,name='x')])
def cube(z):
    return tf.pow(z,3)

try:
    print(cube(tf.constant(1., 2., 3.)))
except ValueError as ex:
    print(ex)


print(cube(tf.constant([1,2,3])))

cube_func_int32 = cube.get_concrete_function(
    tf.TensorSpec([None],tf.int32)
)
print(cube_func_int32)


print(cube_func_int32 is cube.get_concrete_function(
    tf.TensorSpec([5],tf.int32)
))
print(cube_func_int32 is cube.get_concrete_function(tf.constant([1,2,3])))

cube_func_int32.graph