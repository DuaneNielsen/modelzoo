import collections
import weakref


class Handles:
    def __init__(self):
        self.handles = []

    def __iadd__(self, removable_handle):
        self.handles.append(removable_handle)
        return self

    def remove(self):
        for handle in self.handles:
            handle.remove()


class RemovableHandle(object):
    """A handle which provides the capability to remove a hook."""

    next_id = 0

    def __init__(self, hooks_dict):
        self.hooks_dict_ref = weakref.ref(hooks_dict)
        self.id = RemovableHandle.next_id
        RemovableHandle.next_id += 1

    def remove(self):
        hooks_dict = self.hooks_dict_ref()
        if hooks_dict is not None and self.id in hooks_dict:
            del hooks_dict[self.id]

    def __getstate__(self):
        return (self.hooks_dict_ref(), self.id)

    def __setstate__(self, state):
        if state[0] is None:
            # create a dead reference
            self.hooks_dict_ref = weakref.ref(collections.OrderedDict())
        else:
            self.hooks_dict_ref = weakref.ref(state[0])
        self.id = state[1]
        RemovableHandle.next_id = max(RemovableHandle.next_id, self.id + 1)

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.remove()


class Hookable:
    """
    Add entry points for removable hooks.
    """
    def __init__(self):
        self.context = {}
        self.init_hooks = collections.OrderedDict()
        self.before_hooks = collections.OrderedDict()
        self.after_hooks = collections.OrderedDict()

    def register_init_hook(self, func):
        """ Adds a closure to be executed before minibatch step, use trainer.context['key'] to store context to be
        transmitted to the after_hook, or over the lifetime of the batch

        variables to persist over the run can be stored in run.metadata['key']

        :param func: closure, arguments are 'trainer, payload, input_data, target_data, model, optimizer, lossfunc,
        dataloader, selector, run'
        :return: a handle to remove the hook
        """

        handle = RemovableHandle(self.init_hooks)
        self.init_hooks[handle.id] = func
        return handle

    def register_before_hook(self, func):
        """ Adds a closure to be executed before minibatch step, use trainer.context['key'] to store context to be
        transmitted to the after_hook, or over the lifetime of the batch

        variables to persist over the run can be stored in run.metadata['key']

        :param func: closure, arguments are 'trainer, payload, input_data, target_data, model, optimizer, lossfunc,
        dataloader, selector, run'
        :return: a handle to remove the hook
        """

        handle = RemovableHandle(self.before_hooks)
        self.before_hooks[handle.id] = func
        return handle

    def register_after_hook(self, func):
        """ Adds a closure to be executed after minibatch step, use trainer.context['key'] to store context to be
        transmitted to the after_hook, or over the lifetime of the batch

        variables to persist over the run can be stored in run.metadata['key']

        :param func: closure, arguments are 'trainer, payload, input_data, target_data, model, optimizer, lossfunc,
        dataloader, selector, run, output, loss'
        :return: a handle to remove the hook
        """
        handle = RemovableHandle(self.after_hooks)
        self.after_hooks[handle.id] = func
        return handle

    def execute_init(self, init_args):

        for closure in self.init_hooks.values():
            closure(self, init_args)

    def execute_before(self, before_args):

        for closure in self.before_hooks.values():
            closure(self, before_args)

    def execute_after(self, after_args):
        for closure in self.after_hooks.values():
            closure(self, after_args)
