# -*- coding: utf-8 -*-
"""分布式训练器
"""

from ..core import (Variable, update_node_value_in_graph)
from ..core.graph import default_graph
from ..dist import ps
from .trainer import Trainer


class DistTrainerParameterServer(Trainer):

    def __init__(self, *args, **kargs):
        Trainer.__init__(self, *args, **kargs)
        cluster_conf = kargs['cluster_conf']
        ps_host = cluster_conf['ps'][0]
        self.ps_client = ps.ParameterServiceClient(ps_host)

    def _variable_weights_init(self):
        '''
        多个worker通过ps保证变量节点初始化一致
        '''

        var_weights_dict = dict()
        for node in default_graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                var_weights_dict[node.name] = node.value

        # 把自己的初始值发送给ps，由ps决定使用哪个Worker并返回
        duplicated_var_weights_dict = self.ps_client.variable_weights_init(
            var_weights_dict)

        # 使用ps返回的初始值，重新初始化本地
        for var_name, weights in duplicated_var_weights_dict.items():
            update_node_value_in_graph(var_name, weights)

        print('[INIT] Worker variable weights initialized')

    def _optimizer_update(self):

        # 把当前梯度push到ps上。此操作可能被block，直到所有节点都pull完成
        acc_gradient = self.optimizer.acc_gradient
        self.ps_client.push_gradients(
            acc_gradient, self.optimizer.acc_no)

        # 从ps把所有节点的平均梯度pull回来。此操作可能被block直到所有节点都push完成
        node_gradients_dict = self.ps_client.pull_gradients()

        # 使用平均梯度，利用优化器的优化算法，更新本地变量
        self.optimizer.update(node_gradients_dict)
