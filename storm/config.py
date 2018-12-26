import os
from urllib.parse import quote
import torch
from pathlib import Path
import json
import logging
import unicodedata
import re


def slugify(value, allow_unicode=False):
    """
    Convert to ASCII if 'allow_unicode' is False. Convert spaces to hyphens.
    Remove characters that aren't alphanumerics, underscores, or hyphens.
    Convert to lowercase. Also strip leading and trailing whitespace.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    return re.sub(r'[-\s]+', '-', value)


def slug(model):
    repr_string = model.repr_string if hasattr(model, 'repr_string') else repr(model)
    return slugify(type(model).__name__ + '-' + repr_string)


class Singleton(type):
    """
    Define an Instance operation that lets clients access its unique
    instance.
    """

    def __init__(cls, name, bases, attrs, **kwargs):
        super().__init__(name, bases, attrs)
        cls._instance = None

    def __call__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__call__(*args, **kwargs)
        return cls._instance


class Config(metaclass=Singleton):
    def __init__(self, DATA_PATH=None):
        # environment variables
        self.BUILD_TAG = os.environ.get('BUILD_TAG', 'build_tag').replace('"', '')
        self.GIT_COMMIT = os.environ.get('GIT_COMMIT', 'git_commit').replace('"', '')
        self.TORCH_DEVICE = os.environ.get('TORCH_DEVICE', 'no_env_set').replace('"', '')

        if DATA_PATH is not None:
            self.DATA_PATH = DATA_PATH
        else:
            self.DATA_PATH = os.environ.get('DATA_PATH', 'c:\data').replace('"', '')

        self.configpath = Path('run_config.json')

        if self.configpath.exists():
            self.config = Config.load(str(self.configpath))
        else:
            self.config = {}
            self.config['run_id'] = 0
            self.save(str(self.configpath))

        logfile = self.getLogPath('most_improved.log')
        logging.basicConfig(filename=logfile.absolute())

        self.globaldata = {}

    def rolling_run_number(self):
        return "{0:0=3d}".format(self.config['run_id'] % 1000)

    def run_id_string(self, model):
        return 'runs/' + self.rolling_run_number() + '/' + slug(model)

    def run_path(self):
        return 'runs/' + self.rolling_run_number()

    def convert_to_url(self, run, host=None, port='6006'):
        if host is None:
            import socket
            host = socket.gethostname()
        url = run.replace('\\', '\\\\')
        url = run.replace('/', '\\\\')
        url = quote(url)
        url = 'http://' + host + ':' + port + '/#scalars&regexInput=' + url
        return url

    def run_url_link(self, model):
        run = self.run_id_string(model)
        url = self.convert_to_url(run)
        return url

    def tb_run_dir(self, param):
        if isinstance(param, torch.nn.Module):
            return self.run_id_string(param)
        else:
            return param

    def device(self):
        if str(self.TORCH_DEVICE) is not 'no_env_set':
            device = torch.device(str(self.TORCH_DEVICE))
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device

    def __str__(self):
        return 'DATA_PATH ' + str(self.DATA_PATH) + \
               ' GIT_COMMIT ' + str(self.GIT_COMMIT) + \
               ' TORCH_DEVICE ' + str(self.TORCH_DEVICE)

    def basepath(self):
        return Path(self.DATA_PATH)

    def datapath(self, datapath):
        datadir = Path(self.DATA_PATH).joinpath(datapath)
        return datadir.absolute()

    def modelpath(self):
        return self.basepath() / 'storm'

    def getLogPath(self, name):
        logfile = Path(self.DATA_PATH) / 'logs' / name
        logfile.parent.mkdir(parents=True, exist_ok=True)
        return logfile

    def update(self, key, value):
        self.config[key] = value
        self.save(self.configpath)

    def increment(self, key):
        self.config[key] += 1
        self.save(self.configpath)

    def increment_run_id(self):
        self.increment('run_id')
        return str(self.config['run_id'])

    def save(self, filename):
        with open(filename, 'w') as configfile:
            json.dump(self.config, fp=configfile, indent=2)

    @staticmethod
    def load(filename):
        with open(filename, 'r') as configfile:
            return json.load(fp=configfile)

    def model_fn(self, model):
        if 'epoch' not in model.metadata:
            model.metadata['epoch'] = 1
        else:
            model.metadata['epoch'] += 1
        file = Path(self.tb_run_dir(model) + '_' + str(model.metadata['epoch']) + '.md')
        return file.absolute()


config = Config()
