"""
神经网络层
"""
import numpy as np

from ..core import Variable
from ..ops import (Convolve, Add, ScalarMultiply, ReLU,
                   Logistic, MaxPooling, MatMul)


def conv(feature_maps, input_shape, kernels, kernel_shape, activation):
    """
    A convolutional layer.

    Args:
        feature_maps: list, 包含多个输入特征图.
        input_shape: tuple, 包含输入特征图的形状（宽和高）.
        kernels: int, 卷积层的卷积核数量.
        kernel_shape: tuple, 卷积核的形状（宽和高）.
        activation: str, 激活函数类型.

    Returns:
        list, 包含多个输出特征图.
    """
    ones = Variable(input_shape, init=False, trainable=False)
    ones.set_value(np.mat(np.ones(input_shape)))

    outputs = []
    for i in range(kernels):
        channels = []
        for fm in feature_maps:
            kernel = Variable(kernel_shape, init=True, trainable=True)
            conv = Convolve(fm, kernel)
            channels.append(conv)

        channels = Add(*channels)
        bias = ScalarMultiply(Variable((1, 1), init=True, trainable=True),
                              ones)
        affine = Add(channels, bias)

        if activation == 'ReLU':
            outputs.append(ReLU(affine))
        elif activation == 'Logistic':
            outputs.append(Logistic(affine))
        elif activation == 'None':
            outputs.append(affine)
        else:
            raise ValueError(f'Activation {activation} is not defined.')

    assert len(outputs) == kernels
    return outputs


def pooling(feature_maps, kernel_shape, stride):
    """
    A max pooling layer.

    Args:
        feature_maps: list，包含多个输入特征图，它们应该是形状相同的矩阵节点.
        kernel_shape: tuple，池化核的形状（宽和高).
        stride: tuple, 包含横向和纵向步幅.

    return:
        list，包含多个输出特征图.
    """
    outputs = []
    for fm in feature_maps:
        outputs.append(MaxPooling(fm, size=kernel_shape, stride=stride))
    return outputs


def fc(input, input_size, size, activation):
    """
    A full-connected neural network layer.

    Args:
        input: 输入向量
        input_size: 输入向量的维度
        size: 神经元个数，即输出向量的维度
        activation: 激活函数类型

    Returns：
        输出向量
    """
    weights = Variable((size, input_size), init=True, trainable=True)
    bias = Variable((size, 1), init=True, trainable=True)
    affine = Add(MatMul(weights, input), bias)

    if activation == 'ReLU':
        return ReLU(affine)
    elif activation == 'Logistic':
        return Logistic(affine)
    elif activation == 'None':
        return affine
    else:
        raise ValueError(f'Activation {activation} is not defined.')
