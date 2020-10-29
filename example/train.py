# -*- coding: utf-8 -*-
import argparse
import random
import sys
sys.path.append('.')
sys.path.append('../')
import time

import matplotlib
import numpy as np

import matrixslow as ms
from matrixslow.dist.ps import ps
from matrixslow.layer import *
from matrixslow.ops import Add, Logistic, MatMul, ReLU, SoftMax, Reshape
from matrixslow.ops.loss import CrossEntropyWithSoftMax, LogLoss
from matrixslow.ops.metrics import Accuracy, Metrics
from matrixslow.optimizer import *
from matrixslow.trainer import (DistTrainerParameterServer,
                                DistTrainerRingAllReduce, Saver, SimpleTrainer)
from matrixslow.util import *
from matrixslow.util import ClassMining, vis

from matrixslow.model import *

from matrixslow_serving.exporter import Exporter


def plot_data(data_x, data_y, weights=None, bias=None):
    '''
    绘制数据节点和线性模型，只绘制2维或3维
    如果特征维度>3,默认使用前3个特征绘制
    '''
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    assert len(data_x) == len(data_y)
    data_dim = data_x.shape[1]
    plot_3d = False if data_dim < 3 else True

    xcord1 = []
    ycord1 = []
    zcord1 = []
    xcord2 = []
    ycord2 = []
    zcord2 = []
    for i in range(data_x.shape[0]):
        if int(data_y[i]) == 1:
            xcord1.append(data_x[i, 0])
            ycord1.append(data_x[i, 1])
            if plot_3d:
                zcord1.append(data_x[i, 2])
        else:
            xcord2.append(data_x[i, 0])
            ycord2.append(data_x[i, 1])
            if plot_3d:
                zcord2.append(data_x[i, 2])
    fig = plt.figure()
    if plot_3d:
        ax = Axes3D(fig)
        ax.scatter(xcord1, ycord1, zcord1, s=30, c='red', marker='s')
        ax.scatter(xcord2, ycord2, zcord2, s=30, c='green')
    else:
        ax = fig.add_subplot(111)
        ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
        ax.scatter(xcord2, ycord2, s=30, c='green')

    if weights is not None and bias is not None:
        x1 = np.arange(-1.0, 1.0, 0.1)
        if plot_3d:
            x2 = np.arange(-1.0, 1.0, 0.1)
            x1, x2 = np.meshgrid(x1, x2)
        weights = np.array(weights)
        bias = np.array(bias)
        if plot_3d:
            y = (-weights[0][0] * x1 -
                 weights[0][1] * x2 - bias[0][0]) / weights[0][2]
            ax.plot_surface(x1, x2, y)
        else:
            y = (-weights[0][0] * x1 - bias[0][0]) / weights[0][1]
            ax.plot(x1, y)
    plt.show()


def random_gen_dateset(feature_num, sample_num, test_radio=0.3, seed=41):
    '''
    生成二分类样本
    '''
    random.seed(seed)
    rand_bias = np.mat(np.random.uniform(-0.1, 0.1, (sample_num, 1)))
    rand_weights = np.mat(np.random.uniform(-1, 1, (feature_num, 1)))
    data_x = np.mat(np.random.uniform(-1, 1, (sample_num, feature_num)))
    data_y = (data_x * rand_weights) + rand_bias
    data_y = np.where(data_y > 0, 1, 0)
    train_size = int(sample_num * (1 - test_radio))

    return (data_x[:train_size, :],
            data_y[:train_size, :],
            data_x[train_size:, :],
            data_y[train_size:, :])


def build_simple_model(feature_num):
    with ms.name_scope('Input'):
        x = ms.Variable((feature_num, 1), init=False,
                        trainable=False, name='img')

    with ms.name_scope('Hidden'):
        w1 = ms.Variable((HIDDEN1_SIZE, feature_num), init=True,
                         trainable=True, name='weights_w1')
        b1 = ms.Variable((HIDDEN1_SIZE, 1), init=True,
                         trainable=True, name='bias_b1')
        hidden1 = Add(MatMul(w1, x), b1)
        w2 = ms.Variable((HIDDEN2_SIZE, HIDDEN1_SIZE),
                         init=True, trainable=True, name='weights_w2')
        b2 = ms.Variable((HIDDEN2_SIZE, 1), init=True,
                         trainable=True, name='bias_b2')
        hidden2 = Add(MatMul(w2, hidden1), b2)

    with ms.name_scope('Logits'):
        w3 = ms.Variable((CLASSES, HIDDEN2_SIZE),
                         init=True, trainable=True)
        b3 = ms.Variable((CLASSES, 1), init=True, trainable=True)
        logits = Add(MatMul(w3, hidden2), b3, name='logits')
    return x, logits


def build_model(feature_num):
    '''
    构建DNN计算图网络
    '''
    with ms.name_scope('Conv'):
        x = Variable((feature_num, 1), init=False,
                     trainable=False, name="img")  # 占位符，28x28 的图像
        img = Reshape(x, (28, 28))
        conv1 = conv([img], (28, 28), 6, (3, 3), "ReLU")  # 第一卷积层
        pooling1 = pooling(conv1, (3, 3), (2, 2))  # 第一池化层

        conv2 = conv(pooling1, (14, 14), 6, (3, 3), "ReLU")  # 第二卷积层
        pooling2 = pooling(conv2, (3, 3), (2, 2))  # 第二池化层

    with ms.name_scope('Hidden'):
        fc1 = fc(Concat(*pooling2), 294, 100, "ReLU")  # 第一全连接层

    with ms.name_scope('Logits'):
        logits = fc(fc1, 100, 10, "None")  # 第二全连接层

    return x, logits, None, None


def build_metrics(logits, y, metrics_names=None):
    metrics_ops = []
    for m_name in metrics_names:
        metrics_ops.append(ClassMining.get_instance_by_subclass_name(
            Metrics, m_name)(logits, y, need_save=True))

    return metrics_ops


def save():
    exporter = Exporter()
    sig = exporter.signature('Input/img', 'Logits/logits')
    saver = Saver('./export')
    saver.save(model_file_name='my_model.json',
               weights_file_name='my_weights.npz', service_signature=sig)


def train(train_x, train_y, test_x, test_y, epoches, batch_size, mode, worker_index=None):

    # x, logits, w, b = build_model(FEATURE_DIM)
    x, logits = build_simple_model(FEATURE_DIM)
    # x, logits = multilayer_perception(FEATURE_DIM, CLASSES, [100, 100, ], "ReLU")

    y = ms.Variable((CLASSES, 1), init=False,
                    trainable=False, name='placeholder_y')
    loss_op = CrossEntropyWithSoftMax(logits, y, name='loss')
    optimizer_op = optimizer.Adam(ms.default_graph, loss_op, learning_rate=0.005)

    if mode == 'local':
        trainer = SimpleTrainer(x, y, logits, loss_op, optimizer_op,
                                epoches=epoches, batch_size=batch_size,
                                eval_on_train=True,
                                metrics_ops=build_metrics(
                                    logits, y, ['Accuracy', 'Recall', 'F1Score', 'Precision']))
    elif mode == 'ps':

        trainer = DistTrainerParameterServer(x, y, logits, loss_op, optimizer_op,
                                             epoches=epoches, batch_size=batch_size,
                                             eval_on_train=True, cluster_conf=cluster_conf,
                                             metrics_ops=build_metrics(
                                                 logits, y, ['Accuracy', 'Recall', 'F1Score', 'Precision']))
    elif mode == 'allreduce':
        trainer = DistTrainerRingAllReduce(x, y, logits, loss_op, optimizer_op,
                                           epoches=epoches, batch_size=batch_size,
                                           eval_on_train=True, cluster_conf=cluster_conf, worker_index=worker_index,
                                           metrics_ops=build_metrics(
                                               logits, y, ['Accuracy']))
    trainer.train(train_x, train_y, test_x, test_y)

    # return w, b


def inference_after_building_model(test_x, test_y):
    '''
    提前构建计算图，再把保存的权值恢复到新构建的计算图中
    要求构建的计算图必须与原计算图保持完全一致
    '''
    # 重新构建计算图
    x, logits = build_simple_model(FEATURE_DIM)
    y = ms.Variable((CLASSES, 1), init=False,
                    trainable=False, name='placeholder_y')

    # 从文件恢复模型
    saver = Saver('./export')
    saver.load(model_file_name='my_model.json',
               weights_file_name='my_weights.npz')

    accuracy = Accuracy(logits, y)

    for index in range(len(test_x)):
        features = test_x[index]
        label_onehot = test_y[index]
        x.set_value(np.mat(features).T)
        y.set_value(np.mat(label_onehot).T)

        logits.forward()
        accuracy.forward()

        pred = np.argmax(logits.value)
        gt = np.argmax(y.value)
        if pred != gt:
            print('prediction: {} and groudtruch: {} '.format(pred, gt))
    print('accuracy: {}'.format(accuracy.value))


def inference_without_building_model(test_x, test_y):
    '''
    不需要构建计算图，完全从保存的文件中把计算图和相应的权值恢复
    如果要使用计算图，需要通过节点名称，调用get_node_from_graph获取相应的节点引用
    '''
    saver = Saver('./export')
    saver.load(model_file_name='my_model.json',
               weights_file_name='my_weights.npz')

    x = ms.get_node_from_graph('Input/img')
    y = ms.get_node_from_graph('placeholder_y')
    logits = ms.get_node_from_graph('logits', name_scope='Logits')
    accuracy = Accuracy(logits, y)

    for index in range(len(test_x)):
        features = test_x[index]
        label_onehot = test_y[index]
        x.set_value(np.mat(features).T)
        y.set_value(np.mat(label_onehot).T)

        logits.forward()
        accuracy.forward()

        pred = np.argmax(logits.value)
        gt = np.argmax(y.value)
        if pred != gt:
            print('False prediction: {} and groudtruch: {} '.format(pred, gt))
    print('accuracy: {}'.format(accuracy.value))


FEATURE_DIM = 784
TOTAL_EPOCHES = 1
BATCH_SIZE = 32
HIDDEN1_SIZE = 20
HIDDEN2_SIZE = 10
CLASSES = 10


cluster_conf = {
    # "ps": [
    #     "localhost:5001"
    # ],
    # "workers": [
    #     "localhost:6000",
    #     "localhost:6002",
    #     "localhost:6004"
    # ]
    "ps": [
        # "k0625v.add.lycc.qihoo.net:5000"
        "localhost:5000"
    ],
    "workers": [
        "localhost:6000",
        "localhost:6002",
        "localhost:6004"
        # "k0110v.add.lycc.qihoo.net:5000",
        # "k0629v.add.lycc.qihoo.net:5000"
        # "p30217v.hulk.shbt.qihoo.net:5000",
        # "k0631v.add.lycc.qihoo.net:5000",
        # "k7791v.add.bjyt.qihoo.net:5000"
    ]
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str)
    parser.add_argument('--role', type=str)
    parser.add_argument('--worker_index', type=int)
    parser.add_argument('--phase', type=str)

    args = parser.parse_args()

    role = args.role
    if role == 'ps':
        server = ps.ParameterServiceServer(cluster_conf, sync=True)
        server.serve()

    else:
        train_x, train_y, test_x, test_y = util.mnist('D:/develop/project/MatrixSlow/dataset/MNIST')
        mode = args.mode
        phase = args.phase
        worker_index = args.worker_index
        if phase == 'train':
            start = time.time()
            train(train_x[:], train_y[:], test_x[:],
                  test_y[:], TOTAL_EPOCHES, BATCH_SIZE, mode, worker_index)
            # w, b = train(train_x, train_y, test_x,
            #              test_y, TOTAL_EPOCHES, BATCH_SIZE)
            end = time.time()
            print('Train time cost: ', end-start)
            save()

        elif phase == 'eval':
            # inference_after_building_model(test_x, test_y)
            inference_without_building_model(test_x, test_y)
        else:
            print('Usage: ./{} train|eval'.format(sys.argv[0]))
