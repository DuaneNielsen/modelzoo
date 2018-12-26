from pkg_resources import resource_filename
from pathlib import Path
import torch
import pickle
from storm.config import config, slug
from storm.zoo.mdnrnn import MDNRNN as mdnrnn


def test_visuals_v1():
    filepath = Path(resource_filename(__name__, 'files/str-convvae4fixed_1_loss_00465.lod'))
    return Loader.load(str(filepath))


def MDNRNN(i_size, z_size, hidden_size, num_layers, n_gaussians):
    return Loader(Params(mdnrnn, i_size=i_size, z_size=z_size, hidden_size=hidden_size,
                         num_layers=num_layers, n_gaussians=n_gaussians)).construct()


class Loader:
    def __init__(self, model_init, weights_init_func=None):
        """
        :param model: the model to train
        :param weights_init_func: to initialize the model weights
        """
        if not isinstance(model_init, Init):
            raise Exception('model should be a Initializer, dont put a naked model in!')

        self.model_init = model_init
        self.weights_init_func = weights_init_func

        self.model = None
        self.model_params = None

    def construct(self):
        self.model = self.model_init.construct()
        self.inject_modules(self.model)

        if self.model_params is not None:
            self.model.load_state_dict(self.model_params)
        elif self.weights_init_func is not None:
            self.model.apply(self.weights_init_func)

        return self.model

    def inject_modules(self, model):
        if model is not None:
            for model in model.modules():
                model.loader = self

    def __getstate__(self):
        state = self.__dict__.copy()
        if state['model'] is not None:
            state['model_params'] = self.model.state_dict()
        state['model'] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state

    def save(self, epoch=0.0, test_loss=0.0, filename=None, dirpath=None):
        name = slug(f'{type(self.model).__name__}_{epoch}_loss_{test_loss:.4f}') + '.lod'
        if filename is not None:
            file = Path(filename)
        elif dirpath is not None:
            file = Path(dirpath) / name
        else:
            file = Path(config.run_path()) / name

        file.parent.mkdir(parents=True, exist_ok=True)
        with file.open('wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file):
        with open(file, 'rb') as f:
            constructor = pickle.load(f)
        return constructor.construct()

    def export_model_params(self, filename):
        torch.save(self.model.state_dict(), filename)


class Init:
    def construct(self, model=None):
        return NotImplementedError


class Params(Init):
    def __init__(self, clazz, *args, **kwargs):
        self.clazz = clazz
        self.args = args
        self.kwargs = kwargs

    def construct(self, model=None):
        if model is None:
            return self.clazz(*self.args, **self.kwargs)
        else:
            return self.clazz(model.parameters(), *self.args, **self.kwargs)