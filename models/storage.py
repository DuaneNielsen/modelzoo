import pickle
from pathlib import Path
import inspect
import hashlib
import logging
import unicodedata
import re
import random
import string

log = logging.getLogger('Storage')


def slugify(value, allow_unicode=False):
    """
    Turn a string into something use-able as a filename.

    Convert to ASCII if 'allow_unicode' is False. Convert spaces to hyphens.
    Remove characters that aren't alphanumerics, underscores, or hyphens.
    Convert to lowercase. Also strip leading and trailing whitespace.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value)\
            .encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    return re.sub(r'[-\s]+', '-', value)


def slug(model):
    """Generate a model filename."""
    repr_string = model.repr_string if hasattr(model, 'repr_string') \
        else repr(model)
    return slugify(type(model).__name__ + '-' + repr_string)


class Storeable:
    """
    Stores the object params for initialization.

    Storable MUST be the first in the inheritance chain
    So put it as the first class in the inheritance
    ie: class MyModel(Storable, nn.Module)
    the init method must also be called as the LAST one in the sequence..
    ie: nn.Module.__init__(self)
        Storable.__init(self, arg1, arg2, etc)
    fixing to make less fragile is on todo, but not trivial...
    """

    def __init__(self):
        self.classname = type(self)

        # snag the args from the child class during initialization
        stack = inspect.stack()
        child_callable = stack[1][0]
        argname, _, _, argvalues = inspect.getargvalues(child_callable)

        self.repr_string = ""
        arglist = []
        for key in argname:
            if key != 'self':
                self.repr_string += ' (' + key + '): ' + str(argvalues[key])
                arglist.append(argvalues[key])

        self.args = tuple(arglist)
        self.metadata = {}
        self.metadata['guid'] = self.guid()
        self.metadata['class_nguid'] = self.class_guid()
        self.metadata['classame'] = type(self).__name__
        self.metadata['args'] = self.repr_string
        self.metadata['repr'] = repr(self)
        self.metadata['slug'] = slug(self)

    def extra_repr(self):
        """Return a string that represents the model."""
        return self.repr_string

    def guid(self):
        """Compute a unique GUID for each model/args instance."""
        return ''.join(
            random.choices(string.ascii_uppercase + string.digits,
                           k=16))

    def class_guid(self):
        """Compute a unique GUID for each model/args pair."""
        md5 = hashlib.md5()
        md5.update(self.repr_string.encode('utf8'))
        return md5.digest().hex()

    def __getstate__(self):
        """We only save the init params and weights to disk."""
        return [self.metadata, self.args, self.state_dict()]

    def __setstate__(self, state):
        """Initialize a fresh model from disk with weights."""
        log.debug(state)
        self.__init__(*state[1])
        self.metadata = state[0]
        self.load_state_dict(state[2])

    def save(self, filename=None):
        """
        Save the model to disk.

        :param filename: name of the file to save to
        :return: the name
        """
        path = Path(filename)
        self.metadata['filename'] = path.name
        from datetime import datetime
        self.metadata['timestamp'] = datetime.utcnow()
        self.metadata['parameters'] = \
            sum(p.numel() for p in self.parameters() if p.requires_grad)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('wb') as f:
            metadata, args, state_dict = self.__getstate__()
            pickle.dump(metadata, f)
            pickle.dump(self, f)
        return path.name

    @staticmethod
    def load(filename):
        """
        Load the model from disk.

        :param filename: the filename to load
        :return: the model
        """
        with open(filename, 'rb') as f:
            try:
                _ = pickle.load(f)
                model = pickle.load(f)
            except Exception as e:
                message = "got exception when loading {}".format(filename)
                log.error(message)
                log.error(e)
                raise
            return model

    @staticmethod
    def load_metadata(filename, data_dir=None):
        """Load metadata only."""
        with Storeable.fn(filename, data_dir).open('rb') as f:
            return pickle.load(f)

    @staticmethod
    def update_metadata(filename, metadata_dict, data_dir=None):
        """Load model from disk and flag it as reloaded."""
        assert type(metadata_dict) is dict
        model = Storeable.load(filename, data_dir)
        model.metadata = metadata_dict
        model.save(filename, data_dir)
