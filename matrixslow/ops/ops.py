"""
运算节点模块
"""

import numpy as np

from ..core import Node


def fill_diagonal(to_be_filled, filler):
    """
    将 filler 矩阵填充在 to_be_filled 的对角线上
    """
    assert to_be_filled.shape[0] / \
        filler.shape[0] == to_be_filled.shape[1] / filler.shape[1]
    n = int(to_be_filled.shape[0] / filler.shape[0])

    r, c = filler.shape
    for i in range(n):
        # np.matrix类支持[x, y]索引
        to_be_filled[i * r:(i + 1) * r, i * c:(i + 1) * c] = filler

    return to_be_filled


class Operator(Node):
    """
    定义操作符抽象类
    """
    pass


class Add(Operator):
    """
    （多个）矩阵加法
    """

    def compute(self):
        self.value = np.mat(np.zeros(self.parents[0].shape()))

        for parent in self.parents:
            self.value += parent.value

    def get_jacobi(self, parent):
        # 矩阵之和对其中任一个矩阵的雅可比矩阵是单位矩阵
        return np.mat(np.eye(self.dimension()))


class MatMul(Operator):
    """
    矩阵乘法
    """

    def compute(self):
        assert len(self.parents) == 2 and \
            self.parents[0].shape()[1] == self.parents[1].shape()[0]
        self.value = self.parents[0].value * self.parents[1].value

    def get_jacobi(self, parent):
        """
        将矩阵乘法视作映射，求映射对参与计算的矩阵的雅可比矩阵
        """
        zeros = np.mat(np.zeros((self.dimension(), parent.dimension())))
        if parent is self.parents[0]:
            return fill_diagonal(zeros, self.parents[1].value.T)
        else:
            jacobi = fill_diagonal(zeros, self.parents[0].value)
            row_sort = np.arange(self.dimension()).reshape(
                self.shape()[::-1]).T.ravel()
            col_sort = np.arange(parent.dimension()).reshape(
                parent.shape()[::-1]).T.ravel()
            return jacobi[row_sort, :][:, col_sort]


class SoftMax(Operator):
    """
    SoftMax函数
    """
    @staticmethod
    def softmax(a):
        a[a > 1e2] = 1e2  # 防止指数过大
        ep = np.power(np.e, a)
        return ep/np.sum(ep)

    def compute(self):
        self.value = SoftMax.softmax(self.parents[0].value)

    def get_jacobi(self, parent):
        """
        我们不实现SoftMax节点的get_jacobi函数，
        训练时使用CrossEntropyWithSoftMax节点
        """
        raise NotImplementedError("Don't use SoftMax's get_jacobi")


class Step(Operator):
    """
    阶跃函数
    """

    def compute(self):
        self.value = np.mat(np.where(self.parents[0].value >= 0.0, 1.0, 0.0))

    def get_jacobi(self, parent):
        assert parent is self.parents[0]
        return np.mat(np.zeros((self.dimension, self.dimension)))


class logistic(Operator):
    """
    对向量的分量施加Logistic函数
    """

    def compute(self):
        x = self.parents[0].value
        # 对父节点的每个分量施加Logistic
        self.value = np.mat(
            1.0 / (1.0 + np.power(np.e, np.where(-x > 1e2, 1e2, -x))))

    def get_jacobi(self, parent):
        assert parent is self.parents[0]
        return np.diag(np.mat(np.multiply(self.value, 1 - self.value)).A1)


class ReLU(Operator):
    """
    对矩阵的元素施加ReLU函数
    """

    nslope = 0.1  # 负半轴的斜率

    def compute(self):
        self.value = np.mat(np.where(
            self.parents[0].value > 0.0,
            self.parents[0].value,
            self.nslope * self.parents[0].value
        ))

    def get_jacobi(self, parent):
        assert parent is self.parents[0]
        return np.diag(
            np.where(self.parents[0].value.A1 > 0.0, 1.0, self.nslope))


class Reshape(Operator):
    """
    改变父节点的值（矩阵）的形状
    """
    def __init__(self, *parent, **kargs):
        super().__init__(*parent, **kargs)
        self.to_shape = kargs.get('shape')
        assert isinstance(self.to_shape, tuple) and len(self.to_shape) == 2

    def compute(self):
        self.value = self.parents[0].value.reshape(self.to_shape)

    def get_jacobi(self, parent):
        assert parent is self.parents[0]
        return np.mat(np.eye(self.dimension()))


class Multiply(Operator):
    """
    求两个父节点的值的element-wise乘积
    """

    def compute(self):
        self.value = np.multiply(self.parents[0].value, self.parents[1].value)

    def get_jacobi(self, parent):
        if parent is self.parents[0]:
            return np.diag(self.parents[1].value.A1)
        else:
            return np.diag(self.parents[0].value.A1)


class Convolve(Operator):
    """
    以第二个父节点的值为滤波器，对第一个父节点的值做二维离散卷积
    """
    def __init__(self, *parents, **kargs):
        assert len(parents) == 2
        super().__init__(*parents, **kargs)
        self.padded = None

    def compute(self):
        data = self.parents[0].value  # 图像
        kernel = self.parents[1].value  # 滤波器

        w, h = data.shape  # 图像的宽和高
        kw, kh = kernel.shape  # 滤波器尺寸
        hkw, hkh = int(kw / 2), int(kh / 2)  # 滤波器尺寸的一半,为了padding

        # 补齐数据边缘
        pw, ph = tuple(np.add(data.shape, np.multiply((hkw, hkh), 2)))
        self.padded = np.mat(np.zeros((pw, ph)))
        self.padded[hkw:hkw + w, hkh:hkh + h] = data
        self.value = np.mat(np.zeros(w, h))

        # 二维离线卷积
        for i in np.arange(hkw, hkw + w):
            for j in np.arange(hkh, hkh + h):
                self.value[i - hkw, j - hkh] = np.sum(
                    np.multiply(
                        self.padded[i - hkw:i - hkw + kw,
                                    j - hkh:j - hkh + kh],
                        kernel)
                )

    def get_jacobi(self, parent):
        data = self.parents[0].value
        kernel = self.parents[1].value

        w, h = data.shape
        kw, kh = kernel.to_shape
        hkw, hkh = int(kw / 2), int(kh / 2)

        pw, ph = tuple(np.add(data.shape, np.multiply((hkw, hkh), 2)))

        jacobi = []
        if parent is self.parents[0]:
            for i in np.arange(hkw, hkw + w):
                for j in np.arange(hkh, hkh + h):
                    mask = np.mat(np.zeros((pw, ph)))
                    mask[i - hkw:i - hkw + kw, j - hkh:j - hkh + kh] = kernel
                    jacobi.append(mask[hkw:hkw + w, hkh:hkh + h].A1)
        elif parent is self.parents[1]:
            for i in np.arange(hkw, hkw + w):
                for j in np.arange(hkh, hkh + h):
                    jacobi.append(
                        self.padded[i - hkw:i - hkw + kw,
                                    j - hkh:j - hkh + kh].A1
                    )
        else:
            raise Exception("You aren't my father")

        return np.mat(jacobi)


class MaxPooling(Operator):
    """
    最大值池化
    """

    def __init__(self, *parents, **kargs):
        super().__init__(*parents, **kargs)
        self.stride = kargs.get('stride')
        assert self.stride is not None
        self.stride = tuple(self.stride)
        assert isinstance(self.stride, tuple) and len(self.stride) == 2

        self.size = kargs.get('size')
        assert self.size is not None
        self.size = tuple(self.size)
        assert isinstance(self.size, tuple) and len(self.size) == 2

        self.flag = None

    def compute(self):
        data = self.parents[0].value  # 输入特征图
        w, h = data.shape  # 输入特征图的宽和高
        dim = w * h
        sw, sh = self.stride  # 步长
        kw, kh = self.size  # 池化核尺寸
        hkw, hkh = int(kw / 2), int(kh / 2)  # 池化核长宽的一半

        result = []
        flag = []

        for i in np.arange(0, w, sw):
            row = []
            for j in np.arange(0, h, sh):
                # 取池化窗口中的最大值
                top, bottom = max(0, i - hkw), min(w, i + hkw + 1)
                left, right = max(0, i - hkh), min(h, j + hkh + 1)
                window = data[top:bottom, left:right]
                row.append(np.max(window))

                # 记录最大值在原特征图中的位置
                pos = np.argmax(window)
                w_width = right - left
                # 将池化窗口的坐标转为特征图坐标
                offset_w, offset_h = top + pos // w_width, left + pos % w_width
                # 将二维坐标转为flatten一维坐标
                offset = offset_w * w + offset_h
                tmp = np.zeros(dim)
                tmp[offset] = 1
                flag.append(tmp)

            result.append(row)

        self.flag = np.mat(flag)
        self.value = np.mat(result)

    def get_jacobi(self, parent):
        assert parent is self.parents[0] and self.jacobi is not None
        return self.flag
