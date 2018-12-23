import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from .util import RemovableHandle
from collections import OrderedDict
from statistics import mean


class Lossable(_Loss):
    def __init__(self):
        super().__init__()
        self.hooks = OrderedDict()

    def register_hook(self, closure):
        handle = RemovableHandle(self.hooks)
        self.hooks[handle.id] = closure
        return handle

    def execute_hooks(self, **loss_terms):
        for closure in self.hooks.values():
            for key, value in loss_terms.items():
                closure(self, key, value)

    def forward(self, *input):
        raise NotImplemented


class BceKldLoss(Lossable):
    """
    :param transform applies a transform to the ground truth
    """
    def __init__(self, beta=1.0, transform=None):
        super().__init__()
        self.beta = beta
        self.transform = transform

    # Reconstruction + KL divergence losses summed over all elements and batch
    def forward(self, recon_x, mu, logvar, x):
        if self.transform is not None:
            x = self.transform(x)

        BCE = F.binary_cross_entropy(recon_x, x, reduction='elementwise_mean')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.numel()

        self.execute_hooks(kld_loss=KLD, bce_loss=BCE)
        return BCE + KLD


class MseKldLoss(Lossable):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    # Reconstruction + KL divergence losses summed over all elements and batch
    def forward(self, recon_x, mu, logvar, x):

        MSE = F.mse_loss(recon_x, x, reduction='elementwise_mean')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework
        # https: // openreview.net / forum?id = Sy2fzU9gl
        KLD = KLD * self.beta / mu.numel()

        self.execute_hooks(kld_loss=KLD, mse_loss=MSE)
        return MSE + KLD


from torch.distributions import Uniform, kl_divergence


class MSEKLDUniformLoss(Lossable):
    def __init__(self):
        super().__init__()

    def forward(self, recon_x, low, high, x):
        MSE = F.mse_loss(recon_x, x, reduction='elementwise_mean')

        uniform = Uniform(torch.zeros_like(low), torch.ones_like(high))

        KLD = kl_divergence(uniform, Uniform(low, high)).mean() / low.numel()

        self.execute_hooks(kld_loss=KLD, mse_loss=MSE)

        return KLD + MSE

class GECOMseKldLoss(Lossable):
    def __init__(self, beta=1.0, cappa=0.99):
        super().__init__()
        self.beta = beta
        self.running_mse = None

    # Reconstruction + KL divergence losses summed over all elements and batch
    def forward(self, recon_x, mu, logvar, x):

        MSE = F.mse_loss(recon_x, x, reduction='elementwise_mean')

        if self.running_mse is None:
            self.running_mse = MSE.data.mean().item()
        else:
            self.running_mse = self.running_mse * self.cappa + MSE.data.mean().item() * (1 - self.cappa)


        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework
        # https: // openreview.net / forum?id = Sy2fzU9gl
        KLD = KLD * self.beta / mu.numel()

        self.execute_hooks(kld_loss=KLD, mse_loss=MSE)
        return MSE + KLD


class BcelKldLoss(Lossable):
    # Reconstruction + KL divergence losses summed over all elements and batch
    def forward(self, recon_x, mu, logvar, x):
        BCE = F.binary_cross_entropy_with_logits(recon_x, x, reduction='elementwise_mean')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD


class BceLoss(Lossable):
    # Reconstruction + KL divergence losses summed over all elements and batch
    def forward(self, recon_x, mu, logvar, x):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='elementwise_mean')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        #KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE #+ KLD


class MSELoss(Lossable):
    def forward(self, recon_x, x):
        MSE = F.mse_loss(recon_x, x)
        self.execute_hooks(mse_loss=MSE)
        return MSE


class TestMSELoss(Lossable):
    def forward(self, y, x):
        loss = F.mse_loss(y, x)
        self.execute_hooks(mse_loss=loss)
        return loss