"""基本节点模块。
节点有以下几个类型：
    1. Variable，变量节点。其没有父节点，作为模型权重（可训练）或输入数据（不可训练）。
    2. Operator，操作节点。其表示某种运算操作，通常为计算图的中间变量。
    3. LossFunction，损失函数节点。
    4. Metrics，评估节点。其仅进行前向计算，用于评估模型。
"""

import abc
import numpy as np

from .graph import default_graph


class Node(abc.ABC):
    """
    计算图节点类基类
    """
    def __init__(self, *parents, **kargs):
        # 计算图对象，默认为全局对象 default_graph
        self.kargs = kargs
        self.graph = kargs.get('graph', default_graph)
        self.need_save = kargs.get('need_save', True)
        self.gen_node_name(**kargs)

        self.parents = list(parents)  # 父节点列表
        self.children = []  # 子节点列表
        self.value = None  # 本节点的值
        self.jacobi = None  # 结果节点对本节点的雅可比矩阵（梯度），使用numpy.matrix实现

        # 将本节点添加到父节点的子节点列表中
        for parent in self.parents:
            parent.children.append(self)

        # 将本节点添加到计算图中
        self.graph.add_node(self)

    def get_parents(self):
        """
        获取本节点的父节点
        """
        return self.parents

    def get_children(self):
        """
        获取本节点的子节点
        """
        return self.children

    def gen_node_name(self, **kargs):
        """
        生成节点名称，如果用户不指定，则根据节点类型生成类似于 MatMul:3 的节点名，
        如果指定了 name_scope，则生成类似 Hidden/MatMul:3 的节点名
        """
        self.name = kargs.get('name', f'{self.__class__.__name__}:'
                              f'{self.graph.node_count()}')
        if self.graph.name_scope:
            self.name = f'{self.graph.name_scope}/{self.name}'

    def forward(self):
        """
        前向传播计算本节点的值，若父节点的值未被计算，则递归调用父节点的forward
        方法
        """
        for parent in self.parents:
            if parent.value is None:
                parent.forward()

        self.compute()

    @abc.abstractmethod
    def compute(self):
        """
        抽象方法，根据父节点的值计算本节点的值
        """

    @abc.abstractmethod
    def get_jacobi(self, parent):
        """
        抽象方法，计算结果节点对某个父节点的雅可比矩阵
        """

    def backward(self, result):
        """
        反向传播，计算结果节点对本节点的雅可比矩阵
        """
        if self.jacobi is None:
            if self is result:
                self.jacobi = np.mat(np.eye(self.dimension()))
            else:
                self.jacobi = np.mat(
                    np.zeros((result.dimension(), self.dimension()))
                )
                for child in self.get_children():
                    if child.value is not None:
                        self.jacobi += child.backward(result) \
                            * child.get_jacobi(self)

        return self.jacobi

    def clear_jacobi(self):
        """
        清空结果节点对本节点的雅可比矩阵
        """
        self.jacobi = None

    def dimension(self):
        """
        返回本节点的值展开成向量后的维数
        """
        return self.value.shape[0] * self.value.shape[1]

    def shape(self):
        """
        返回本节点的值作为矩阵的形状：（行数，列数）
        """
        return self.value.shape

    def reset_value(self, recursive=True):
        """
        重置本节点的值，并递归重置本节点的下游节点的值
        """
        self.value = None
        if recursive:
            for child in self.children:
                child.reset_value()


class Variable(Node):
    """
    变量节点，没有父节点
    """
    def __init__(self, dim, init=False, trainable=True, **kargs):
        """
        变量节点没有父节点，其构造函数接受变量节点的形状、是否初始化以及是否参与训练的标识
        """
        super().__init__(**kargs)
        self.dim = dim

        # 如果需要初始化，则以正态分布随机初始化变量的值
        if init:
            self.value = np.mat(np.random.normal(0, 0.1, self.dim))

        # 变量节点是否参与训练
        self.trainable = trainable

    def set_value(self, value):
        """
        为变量赋值
        """
        assert isinstance(value, np.matrix) and value.shape == self.dim

        # 本节点的值被改变，重置所有下游节点的值
        self.reset_value()
        self.value = value

    def compute(self):
        pass

    def get_jacobi(self, parent):
        pass
