"""
优化器类
"""

import abc

import numpy as np

from ..core import Node, Variable, get_node_from_graph
from ..core.graph import Graph


class Optimizer(abc.ABC):
    """
    优化器抽象类
    """
    def __init__(self, graph, target, learning_rate=0.01):
        """
        优化器的构造函数接受计算图对象，目标节点对象和学习率
        """
        assert isinstance(target, Node) and isinstance(graph, Graph)
        self.graph = graph
        self.target = target
        self.learning_rate = learning_rate

        # 为每个参与训练的节点累加一个mini batch的全部样本的梯度, 用于各种梯度下降法的变体
        self.acc_gradient = dict()
        self.acc_no = 0

    def one_step(self):
        """
        计算并累加样本的梯度
        """
        self.forward_backward()
        self.acc_no += 1

    def get_gradient(self, node):
        """
        返回样本的平均梯度
        """
        assert node in self.acc_gradient
        return self.acc_gradient[node] / self.acc_no

    @abc.abstractmethod
    def _update(self):
        """
        抽象方法，执行具体的梯度更新算法，由子类实现
        """

    def apply_gradient(self, node_gradients_dict,
                       summarize=False, acc_no=None):
        """
        手动赋值梯度
        """
        for node, gradient in node_gradients_dict.items():
            if isinstance(node, Node):
                pass
            else:
                target_node = get_node_from_graph(node)
                assert target_node is not None
                assert self.acc_gradient[target_node].shape == gradient.shape
                if summarize:
                    self.acc_gradient[target_node] += gradient
                else:
                    self.acc_gradient[target_node] = gradient

        if summarize:
            self.acc_no += acc_no
        else:
            if acc_no is None:
                # 传入的是平均梯度，强制让acc_no变为1，避免梯度更新时重复平均
                self.acc_no = 1
            else:
                self.acc_no = acc_no

    def update(self, var_gradients=None):
        """
        使用雅可比矩阵，根据梯度下降法更新参数
        """
        if var_gradients is not None:
            self.apply_gradient(var_gradients)

        # 执行更新
        self._update()

        # 清楚累加梯度
        self.acc_gradient.clear()
        self.acc_no = 0

    def forward_backward(self):
        """
        前向传播计算结果节点的值
        并反向传播计算结果节点对各节点的雅可比矩阵
        """

        # 清除计算图中所有节点的雅可比矩阵
        self.graph.clear_jacobi()

        # 前向传播计算结果节点
        self.target.forward()

        # 反向传播计算雅可比矩阵
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                node.backward(self.target)

                # 最终结果（标量）对节点值得雅可比矩阵是一个行向量，其转置是梯度（列向量）
                # 这里将梯度reshape成与节点值相同的形状，好对节点值进行更新
                gradient = node.jacobi.T.reshape(node.shape())
                if node not in self.acc_gradient:
                    self.acc_gradient[node] = gradient
                else:
                    self.acc_gradient[node] += gradient


class GradientDescent(Optimizer):
    """
    梯度下降优化器
    """

    def __init__(self, graph, target, learning_rate=0.01):
        super().__init__(self, graph, target)
        self.learning_rate = learning_rate

    def _update(self):
        """
        朴素梯度下降法
        """
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                # 取得该节点在当前批的平均梯度
                gradient = self.get_gradient(node)

                # 用朴素梯度下降法更新变量节点的值
                node.set_value(node.value - self.learning_rate * gradient)
