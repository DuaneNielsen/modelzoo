import torch
from torch import nn as nn


class BaseAE(nn.Module):
    """Base class for AutoEncoder."""

    def __init__(self, encoder, decoder):
        nn.Module.__init__(self)

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        """Forward pass."""
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x):
        """Encode to latent space."""
        return self.encoder(x)


class MultiChannelAE(nn.Module):
    """Base class for AutoEncoder with multiple channels for segmented input."""

    def __init__(self):
        nn.Module.__init__(self)
        self.ae_l = []
        self.ch_l = []
        self.decode_ch_l = []

    def add_ae(self, auto_encoder, channels, decoder_channels=None):
        """
        Add an AE to the stack for the channels.

        :param auto_encoder: the BaseAE to use
        :param channels: LongTensor of channels to use for input
        """
        self.ae_l.append(auto_encoder)
        self.ch_l.append(channels)
        self.decode_ch_l.append(decoder_channels)

    def forward(self, x):
        """Forward pass."""
        z_l = []
        for channels, ae in zip(self.ch_l, self.ae_l):
            z = ae.encode(x[:, channels, :, :])
            z_l.append(z)
        result = torch.cat(z_l, dim=1)
        return result

    def decode(self, z):
        """Decode the latents space."""
        y = []
        for ae, channels in zip(self.ae_l, self.decode_ch_l):
            if channels is not None:
                y.append(ae.decoder(z[:, channels, :, :]))
        return torch.cat(y, dim=1)


class DummyCoder(nn.Module):
    """Dummy module for testing."""

    def forward(self, x):
        """Forward pass."""
        return x


class DummyAE(BaseAE):
    """Dummy AE for testing."""

    def __init__(self):
        BaseAE.__init__(self, DummyCoder(), DummyCoder())


class BaseVAE(nn.Module):
    """Base class for variational auto-encoder."""

    def __init__(self, encoder, decoder, variational=False):
        nn.Module.__init__(self)

        self.encoder = encoder
        self.decoder = decoder
        self.variational = variational

    def forward(self, x):
        """Forward pass."""
        indices = None
        logvar = None

        encoded = self.encoder(x)
        mu = encoded[0]
        logvar = encoded[1]
        if len(encoded) > 2:
            indices = encoded[2]

        z = self.reparameterize(mu, logvar)

        if indices is not None:
            decoded = self.decoder(z, indices)
        else:
            decoded = self.decoder(z)

        return decoded, mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparametization trick."""
        if self.training and self.variational:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def sample(self, eps):
        """Sample from the model."""
        images = self.decoder(eps)
        return images
