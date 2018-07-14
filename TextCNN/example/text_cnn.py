#! /usr/bin/env python
#!__utf-8__
import tensorflow as tf
import numpy as np
class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)
        # Embedding layer
        # embedding层在cpu中进行计算！
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            # 随机初始化embeddings：大小为：行：词表大小——列：词向量的维度——也就是说开将所有的词向量保存在W中
            self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),name="W")
            # 根据input（id）在W中查找vector：应该是对每一句话扫一遍——根据vocabulary生成id——然后在W中——生成此句子的vector矩阵！
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            # 在每一句话生成的vector矩阵后面增加一维：填充1！
            # 为什么要扩展一维?？？？？？？？？？？？？？？？
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
        # Create a convolution + maxpool layer for each filter size
        # 因为卷积核有多种，所以得写个for循环——每一次使用一种卷积——所以需要最后将各自的池化层的结果保存，拼接在一起
        #然而在AlexNet等cnn结构中由于在相同的卷积层使用的是一种卷积，所以池化层结果不用拼接！
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer:——[filter_height, filter_width, in_channels, 这种卷积个数]
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")#随机初始化！
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")#有几个卷积核就有几个bias
                #tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu：默认是使用gpu, name)
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,#输入
                    W,#卷积weight
                    strides=[1, 1, 1, 1],#在4个维度上的步长！
                    padding="VALID",#same：取边——valid：不到边！
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,#卷积层的输出
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],#池化窗口大小[1, height, width, 1]：height原文中是对卷积输出总体max，所以是sequence_length - filter_size + 1
                    strides=[1, 1, 1, 1],#4个维度的步长
                    padding='VALID',#不取边
                    name="pool")
                #将池化层的输出拼接在一块
                pooled_outputs.append(pooled)
        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)#卷积的总个数！
        self.h_pool = tf.concat(pooled_outputs, 3)#拼接池化层输出
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])#有几个卷积就有几维的输出！
        # Add dropout_：池化层后面接drop_out!
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable("W",shape=[num_filters_total, num_classes],initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)#在loss中加入w的l2正则
            l2_loss += tf.nn.l2_loss(b)#在loss中加入b的l2正则
            #对drop_out输出的结果进行softmax计算
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)#交叉熵损失！
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss#损失函数最终形式！
        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")