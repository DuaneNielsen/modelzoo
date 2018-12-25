from storm.zoo.registry import Init, Params, Loader
from storm.config import config
from storm.vis.hooks import hooks

from collections import namedtuple
from pathlib import Path

import torch


class SimpleTrainer:

    def train(self, model, optimizer, lossfunc, dataloader, selector, run, epoch):
        device = config.device()
        model.to(device)
        model.train()
        model.epoch = epoch

        for i, payload in enumerate(dataloader):

            input_data = selector.get_input(payload, device)
            target_data = selector.get_target(payload, device)

            optimizer.zero_grad()
            output_data = model(*input_data)
            if type(output_data) == tuple:
                loss = lossfunc(*output_data, *target_data)
            else:
                loss = lossfunc(output_data, *target_data)
            loss.backward()
            optimizer.step()

            hooks.execute_train_end(i, len(dataloader), loss)

            run.step += 1


class SimpleTester:

    def test(self, model, lossfunc, dataloader, selector, run, epoch):
        device = config.device()
        model.to(device)
        model.eval()
        model.epoch = epoch

        for i, payload in enumerate(dataloader):

            input_data = selector.get_input(payload, device)
            target_data = selector.get_target(payload, device)

            output_data = model(*input_data)
            if type(output_data) == tuple:
                loss = lossfunc(*output_data, *target_data)
            else:
                loss = lossfunc(output_data, *target_data)

            hooks.execute_test_end(i, len(dataloader), loss)

            run.step += 1


class SimpleInference:
    def infer(self, model, lossfunc, dataloader, selector, run, epoch):
        device = config.device()
        model.to(device)
        model.eval()
        model.epoch = epoch

        for payload in dataloader:
            input_data = selector.get_input(payload, device)

            output_data = model(*input_data)

            run.step += 1


class Epoch:
    def __init__(self, index, run):
        super(Epoch, self).__init__()
        self.ix = index
        self.run = run


class EpochIter:
    def __init__(self, num_epochs, run):
        self.num_epochs = num_epochs
        self.run = run
        self.last_epoch = run.epochs + num_epochs

    def __iter__(self):
        return self

    def __next__(self):
        if self.run.epochs == self.last_epoch:
            raise StopIteration
        epoch = Epoch(self.run.epochs, self.run)
        self.run.epoch = epoch
        self.run.epochs += 1
        return epoch


class Run:
    def __init__(self, model_init, opt, loss_fn, data_package, trainer=None, tester=None, run_name=None,
                 weights_init_func=None):
        """
        :param model_init: the model to train
        :param opt: the optimizer for the model, should already be initialized with model params
        :param loss_fn: the loss function to use for training
        :param data_package: specifies what to load, how to split the dataset, and what is inputs and targets
        :param trainer: the trainer to use
        :param tester: the tester to use
        :param run_name name of the run
        :param tensorboard: to use tensorboard or not
        :param weights_init_func: to initialize the model weights
        """

        if opt is not None and not isinstance(opt, Init):
            raise Exception('optimizer should be a Initializer, dont put a naked optimizer in!')

        self.loader = Loader(model_init, weights_init_func)

        self.opt_i = opt
        self.opt = None

        self.loss_fn_i = loss_fn
        self.loss = None

        self.data_package = data_package

        self.trainer = trainer if trainer is not None else SimpleTrainer()
        self.trainer.run = self

        self.tester = tester if tester is not None else SimpleTester()
        self.tester.run = self

        self.run_name = run_name
        self.run_id = None
        self.epochs = 0
        self.step = 0
        self.epoch = None
        self.total_epochs = 0

        self.context = {}

        self.weights_init_func = weights_init_func

    def construct_model_and_optimizer(self):
        self.construct_model()
        if self.opt_i is not None:
            self.opt = self.opt_i.construct(self.model)
            self.opt.run = self
        return self.model, self.opt

    def construct_model(self):
        self.model = self.loader.construct()
        return self.model

    def construct_loss(self):
        print(repr(self.loss_fn_i.__class__), repr(Init))
        if callable(self.loss_fn_i):
            self.loss = self.loss_fn_i
        else:
            self.loss = self.loss_fn_i.construct()
        return self.loss

    def init_run_dir(self, model, increment_run=True, tensorboard=True):
        if increment_run:
            config.increment_run_id()
        if self.run_name is None:
            self.run_id = 'runs/' + config.rolling_run_number() + '/' + config.slug(model)
        else:
            self.run_id = 'runs/' + config.rolling_run_number() + '/' + self.run_name

    def construct(self, increment_run=True, tensorboard=True, data_package=None):
        self.data_package = data_package if self.data_package is None else self.data_package
        self.construct_model_and_optimizer()
        self.construct_loss()
        self.init_run_dir(self.model, increment_run, tensorboard)
        return self.model, self.opt, self.loss, self.data_package, self.trainer, self.tester, self

    def for_epochs(self, num_epochs):
        self.total_epochs = num_epochs
        return EpochIter(num_epochs, self)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['model'] = None
        state['opt'] = None
        state['loss'] = None
        #todo deleting too many stuffs, need to unhook the hooks instead
        state['tb'] = None
        state['epoch'] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state

    def save(self, filename=None):
        if filename is None:
            file = Path(self.run_id + '/epoch' + '%04d' % self.epochs + '.run')
        else:
            file = Path(filename)
        import pickle
        with file.open('wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(file):
        run = Run.load(file)
        return run.construct_model()

    @staticmethod
    def load(file):
        import pickle
        with open(file, 'rb') as f:
            return pickle.load(f)

    def export_model_params(self, filename):
        torch.save(self.model.state_dict(), filename)

    @staticmethod
    def resume(file, data_package, increment_run=False):
        run = Run.load(file)
        return run.construct(increment_run=increment_run, data_package=data_package)


class SimpleRunFac:
    def __init__(self, increment_run=True):
        self.run_list = []
        self.data_package = None
        self.increment_run = increment_run

    def __iter__(self):
        self.run_id = 0
        if self.increment_run:
            config.increment_run_id()
        return self

    def __next__(self):
        if self.run_id == len(self.run_list):
            raise StopIteration
        run = self.run_list[self.run_id].construct(increment_run=False, data_package=self.data_package)
        self.run_id += 1
        return run

    @staticmethod
    def resume(run_dir, data_package, increment_run=False):
        run_fac = SimpleRunFac()
        run_fac.increment_run = increment_run
        run_fac.data_package = data_package
        dir = Path(run_dir)
        for subdir in dir.glob('*'):
            if subdir.is_dir():
                # super pythonic (and completely unreadable) way to get the last epoch in the run
                last_epoch = sorted([f for f in subdir.glob('*.run')], reverse=True)[0]
                run_fac.run_list.append(Run.load(last_epoch.absolute()))
        return run_fac

    # todo this needs a re-write, should support re-use of RUNS not RunFacs
    @staticmethod
    def reuse(run_dir, data_package):
        """
        Reuses an existing run on a new datasset
        :param run_dir:
        :param data_package:
        :return:
        """
        return SimpleRunFac.resume(run_dir, data_package, True)


ModelOpt = namedtuple('ModelOpt', 'model, opt')

RunParams = namedtuple('RunParams', 'model, opt, loss_fn, data_package, trainer, tester, run_name, tensorboard')


class RunFac:
    def __init__(self, model=None, opt=None, loss_fn=None, data_package=None, trainer=None, tester=None, run_name=None,
                 tensorboard=True):
        self.model = model
        self.opt = opt
        self.loss_fn = loss_fn
        self.data_package = data_package
        self.trainer = trainer
        self.tester = tester
        self.run_name = run_name
        self.tensorboard = tensorboard

        self._model_opts = []
        self.data_packages = []
        self.loss_fns = []
        self.run_id = 0
        self.run_list = []

    def add_model_opt(self, model, opt):
        self._model_opts.append(ModelOpt(model, opt))

    def all_empty(self):
        return len(self._model_opts) + len(self.data_packages) + len(self.loss_fns) == 0

    def build_run(self):
        if self.all_empty():
            self.run_list.append(RunParams(model=self.model,
                                           opt=self.opt,
                                           loss_fn=self.loss_fn,
                                           data_package=self.data_package,
                                           trainer=self.trainer,
                                           tester=self.tester,
                                           run_name=self.run_name,
                                           tensorboard=self.tensorboard))

        for loss_fn in self.loss_fns:
            if isinstance(loss_fn, tuple):
                self.run_list.append(RunParams(model=self.model,
                                               opt=self.opt,
                                               loss_fn=loss_fn[1],
                                               data_package=self.data_package,
                                               trainer=self.trainer,
                                               tester=self.tester,
                                               run_name=loss_fn[0],
                                               tensorboard=self.tensorboard))
            else:
                self.run_list.append(RunParams(model=self.model,
                                               opt=self.opt,
                                               loss_fn=loss_fn,
                                               data_package=self.data_package,
                                               trainer=self.trainer,
                                               tester=self.tester,
                                               run_name=None,
                                               tensorboard=self.tensorboard))

    def __iter__(self):
        self.run_id = 0
        self.build_run()
        return self

    def __next__(self):
        if self.run_id == len(self.run_list):
            raise StopIteration
        run = Run(*self.run_list[self.run_id])
        self.run_id += 1
        return run.model, run.opt, run.loss_fn, run.data_package, run.trainer, run.tester, run

    def __getitem__(self, item):
        self.build_run()
        return self.run_list[item]
