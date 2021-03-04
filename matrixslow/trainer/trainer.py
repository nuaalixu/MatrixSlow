"""训练器基类
"""

import abc
import time

import numpy as np


class Trainer(abc.ABC):
    """训练器基类
    """

    def __init__(self, input_x, input_y,
                 loss_op, optimizer, epoches, 
                 batch_size=8, eval_on_train=False, metrics_ops=None,
                 *args, **kargs):

        # 计算图的输入节点，可以有多个，类型是list
        self.inputs = input_x

        # 计算图的标签节点
        self.input_y = input_y

        # 损失函数
        self.loss_op = loss_op

        # 优化器
        self.optimizer = optimizer

        # 训练执行的迭代轮数
        self.epoches = epoches
        self.epoch = 0

        # 批大小
        self.batch_size = batch_size

        # 是否在训练迭代中进行评估
        self.eval_on_train = eval_on_train

        # 评估指标列表
        self.metrics_ops = metrics_ops

        self.print_iteration_interval = kargs.get('print_iteration_interval',
                                                  100)

    def train_and_eval(self, train_x, train_y, test_x=None, test_y=None):
        """开始训练（评估）流程
        """

        assert len(train_x) == len(self.inputs)

        if test_x is not None and test_y is not None:
            assert len(test_x) == len(self.inputs)

        # 初始化权值变量
        self._variable_weight_init()
        print('[INIT] Variable weights init finished')

        # 传入数据，开始主循环
        self.main_loop(train_x, train_y, test_x, test_y)

    def main_loop(self, train_x, train_y, test_x, test_y):
        """训练（评估）的主循环
        """

        # 第一层循环，迭代epoches轮
        for self.epoch in range(self.epoches):
            # 模型训练
            self.train(train_x, train_y)

            # 如果需要,对模型进行评估
            if self.eval_on_train and (test_x is not None
                                       and test_y is not None):
                self.eval(test_x, test_y)

    def train(self, train_x, train_y):
        """使用训练集进行模型训练（one epoch）。
        """

        print(f'- Epoch [{self.epoch + 1}] train start, '
              f'batch size: {self.batch_size}, '
              f'train data size: {len(train_x)}')
        start_time = time.time()
        last_batch_start_time = time.time()
        last_iter_start_time = time.time()

        # 遍历训练数据集
        for i in range(len(list(train_x.values())[0])):
            # 使用一个样本，执行一次前向传播和反向传播
            self.one_step(self._get_input_values(train_x, i), train_y[i])

            if (i+1) % self.print_iteration_interval == 0:
                print(f'-- iteration [{i+1}] finished, '
                      f'time cost: {time.time() - last_iter_start_time:.2f} '
                      f'and loss value: {float(self.loss_op.value):4f}')
                last_iter_start_time = time.time()

            if (i+1) % self.batch_size == 0:
                last_batch_end_time = time.time()
                last_update_start_time = time.time()
                self._optimizer_update()
                computing_cost = last_batch_end_time - last_batch_start_time
                gra_update_cost = time.time() - last_update_start_time
                print(f'---- Batch [{int(i+1)/self.batch_size}] finished, '
                      f'computing cost: {computing_cost:.2f}, ',
                      f'gradients update cost {gra_update_cost:.2f} '
                      f'and total cost: {computing_cost+gra_update_cost:.2f}')
                last_batch_start_time = time.time()
        print(f'- Epoch [{self.epoch+1}] train finished, '
              f'time cost: {time.time()-start_time:.2f}')

    def eval(self, test_x, test_y):
        """使用测试集进行评估
        """
        eval_start_time = time.time()
        for metrics_op in self.metrics_ops:
            metrics_op.reset()

        for i in range(len(list(test_x.values())[0])):
            self.one_step(self._get_input_values(test_x, i),
                          test_y[i], is_eval=True)

            for metrics_op in self.metrics_ops:
                metrics_op.forward()

        metrics_str = f'Epoch [{self.epoch+1}] evaluation metrics'
        for metrics_op in self.metrics_ops:
            metrics_str += (metrics_op.value_str() + ',')

        eval_end_time = time.time()
        metrics_str += f' eval cost: {eval_end_time - eval_start_time:.2f}'
        print(metrics_str)

    def _get_input_values(self, x, index):
        """按索引从数据集中取值。

        Args：
            x: dict类型的数据集
            index：int类型的索引

        Returns:
            包含数据的dict
        """

        input_values = dict()
        for input_node_name in x.keys():
            input_values[input_node_name] = x[input_node_name][index]

        return input_values

    def one_step(self, data_x, data_y, is_eval=False):
        """执行一次前向计算和后向计算（可能）
        """

        for i in range(len(self.inputs)):
            # 根据输入节点的名称，从输入数据dict中找到对应数据
            input_value = data_x.get(self.inputs[i].name)
            self.inputs[i].set_value(np.mat(input_value).T)

        self.input_y.set_value(np.mat(data_y).T)

        # 只有在训练阶段在执行优化器
        if not is_eval:
            self.optimizer.one_step()

    @abc.abstractmethod
    def _variable_weight_init(self):
        """权值变量初始化，具体的初始化由子类完成
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _optimizer_update(self):
        """调用优化器执行参数更新
        """
        raise NotImplementedError()
