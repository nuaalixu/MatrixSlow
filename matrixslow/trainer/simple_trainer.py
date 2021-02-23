"""一个简单的训练器
"""

from .trainer import Trainer


class SimpleTrainer(Trainer):
    """简单训练器，使用默认方法。
    """
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    def _variable_weight_init(self):
        """不做统一的初始化操作，即使用节点本身的初始化方法。
        """

        pass

    def _optimizer_update(self):
        self.optimizer.update()
