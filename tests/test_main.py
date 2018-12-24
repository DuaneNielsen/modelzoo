from unittest import TestCase
from zoo.mdnrnn import MDNRNN
from zoo.registry import test_visuals_v1


class Environments(TestCase):
    def test_mdn(self):
        i_size = 16 + 6
        z_size = 16
        model = MDNRNN(i_size=i_size,
                       z_size=z_size,
                       hidden_size=32,
                       num_layers=3,
                       n_gaussians=3)
        assert model is not None

    def test_registry(self):
        model = test_visuals_v1()