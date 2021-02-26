#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: chenzhen
@Date: 2020-04-10 17:04:46
@LastEditTime: 2020-04-30 09:25:28
@LastEditors: chenzhen
@Description:
'''

import sys
sys.path.append('../../')

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
import matrixslow as ms
from matrixslow.trainer import SimpleTrainer
# from matrixslow_serving.exporter import Exporter


# 加载MNIST数据集，取一部分样本并归一化

# 输入图像尺寸
img_shape = (28, 28)

X, y = fetch_openml('mnist_784', version=1, return_X_y=True, cache=True)
X, y = X[:100] / 255, y.astype(np.int)[:100]
X = np.reshape(np.array(X), (100, *img_shape))

# 将整数形式的标签转换成One-Hot编码
oh = OneHotEncoder(sparse=False)
one_hot_label = oh.fit_transform(y.values.reshape(-1, 1))


# 输入图像
x = ms.core.Variable(img_shape, init=False, trainable=False, name='img_input')

# One-Hot标签
one_hot = ms.core.Variable(dim=(10, 1), init=False, trainable=False)

# 第一卷积层
conv1 = ms.layer.conv([x], img_shape, 3, (5, 5), "ReLU")

# 第一池化层
pooling1 = ms.layer.pooling(conv1, (3, 3), (2, 2))

# 第二卷积层
conv2 = ms.layer.conv(pooling1, (14, 14), 3, (3, 3), "ReLU")

# 第二池化层
pooling2 = ms.layer.pooling(conv2, (3, 3), (2, 2))

# 全连接层
fc1 = ms.layer.fc(ms.ops.Concat(*pooling2), 147, 120, "ReLU")

# 输出层
output = ms.layer.fc(fc1, 120, 10, None)

# 分类概率
predict = ms.ops.SoftMax(output, name='softmax_output')

# 交叉熵损失
loss = ms.ops.loss.CrossEntropyWithSoftMax(output, one_hot)

# 学习率
learning_rate = 0.05

# 优化器
optimizer = ms.optimizer.RMSProp(ms.default_graph, loss, learning_rate)

accuracy = ms.ops.metrics.Accuracy(output, one_hot)


trainer = SimpleTrainer(
    [x], one_hot, loss, optimizer, epoches=10, batch_size=32,
    eval_on_train=True, metrics_ops=[accuracy])

trainer.train_and_eval({x.name: X}, one_hot_label, {x.name: X}, one_hot_label)

# exporter = Exporter()
# sig = exporter.signature('img_input', 'softmax_output')

saver = ms.trainer.Saver('./epoches10')
saver.save(model_file_name='my_model.json',
           weights_file_name='my_weights.npz', service_signature=None)
