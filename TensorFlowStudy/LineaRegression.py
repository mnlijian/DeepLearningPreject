
import tensorflow as tf



def myregression():
    '''
    自实现一个线性回归预测
    :return: None
    '''
    # 1，准备数据，x 特征值【100，1】y目标值[100]
    x= tf.random_normal([100,1],mean=1.75,stddev=0.5,name='x_data')
#     矩阵相乘必须是二维的
    y_true = tf.matmul(x,[[0.7]]) + 0.8
#     2,建立线性回归模型 1个特征，1个权重，一个偏置 y=xw+b
# 随机给一个权重和偏置的值，让他去计算损失，然后再当前状态下优化
    weight = tf.Variable(tf.random_normal([1,1],mean=0.0,stddev=1.0),name='w')
    bias = tf.Variable(0.0,name = 'b')

    y_predict = tf.matmul(x,weight)+bias

#     3,建立损失函数，均方误差
    loss = tf.reduce_mean(tf.square(y_true-y_predict))

#     4,梯度下降优化损失 leaning_rate：
    train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)



if __name__ =="__main__":
    myregression()