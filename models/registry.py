from models.train import Run
from pkg_resources import resource_filename
from pathlib import Path


def test_visuals_v1():
    filepath = Path(resource_filename(__name__, 'runs/epoch0002.run'))
    print(str(filepath))

    return Run.load_model(str(filepath))
