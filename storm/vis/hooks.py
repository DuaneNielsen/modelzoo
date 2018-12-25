from storm.util import RemovableHandle
from collections import OrderedDict


class HookPoint:
    def __init__(self):
        self.local_context = {}
        self.hooks = OrderedDict()

    def __getstate__(self):
        state = self.__dict__.copy()
        state['hooks'] = OrderedDict()
        state['local_context'] = {}
        return state

    def clear(self):
        self.hooks.clear()

    def register(self, func):
        handle = RemovableHandle(self.hooks)
        self.hooks[handle.id] = func
        return handle

    def execute(self, *args, **kwargs):
        for closure in self.hooks.values():
            closure(*args, **kwargs)


def singleton(cls):
    obj = cls()
    # Always return the same object
    cls.__new__ = staticmethod(lambda cls: obj)
    # Disable __init__
    try:
        del cls.__init__
    except AttributeError:
        pass
    return cls


@singleton
class HookAPI:
    def __init__(self):
        self.global_context = {}
        self.epoch_end = HookPoint()
        self.train_end = HookPoint()
        self.test_end = HookPoint()

    def clear_hooks(self):
        self.epoch_end.clear()
        self.train_end.clear()
        self.test_end.clear()

    def execute_epoch_end(self, epoch, total_epochs, **kwargs):
        """
        :param epoch: epoch number
        :param total_epochs: total number of epochs in run
        :return:
        """
        self.epoch_end.execute(epoch, total_epochs, **kwargs)

    def execute_train_end(self, current_batch, batch_total, loss, **kwargs):
        """
        :param current_batch:
        :param batch_total: total number of batches in run
        :param loss: the loss as a torch.Tensor
        :return:
        """
        self.train_end.execute(current_batch, batch_total, loss, **kwargs)

    def execute_test_end(self, current_batch, batch_total, loss, **kwargs):
        """
        :param current_batch:
        :param batch_total: total number of batches in run
        :param loss: the loss as a torch.Tensor
        :return:
        """
        self.test_end.execute(current_batch, batch_total, loss, **kwargs)


hooks = HookAPI()
