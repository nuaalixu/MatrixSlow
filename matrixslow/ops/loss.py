"""
损失函数模块
"""

import numpy as np

from ..core import Node
from ..ops import SoftMax


class LossFunction(Node):
    """
    定义损失函数抽象类
    """
    pass


class PerceptionLoss(LossFunction):
    """
    感知机损失，输入为正时为0，输入为负时为输入的相反数
    """

    def compute(self):
        self.value = np.mat(np.where(
            self.parents[0].value >= 0.0, 0.0, -self.parents[0].value
        ))

    def get_jacobi(self, parent):
        """
        雅克比矩阵为对角阵，每个对角线元素对应一个父节点元素。若父节点元素大于0，则
        相应对角线元素（偏导数）为0，否则为-1。
        """
        diag = np.where(parent.value >= 0.0, 0.0, -1)
        return np.diag(diag.ravel())


class CrossEntropyWithSoftMax(LossFunction):
    """
    对第一个父节点施加SoftMax之后，再以第二个父节点为标签One-Hot向量计算交叉熵
    """

    def compute(self):
        self.prob = SoftMax.softmax(self.parents[0].value)
        self.value = np.mat(-np.sum(np.multiply(self.parents[1].value,
                                                np.log(self.prob + 1e-10))))

    def get_jacobi(self, parent):
        if parent is self.parens[0]:
            return (self.prob - self.parents[1].value).T
        else:
            return (-np.log(self.prob + 1e-10)).T
