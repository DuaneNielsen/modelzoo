from zoo.base import BaseVAE
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of convolutions.

    :param tuple of (h,w)
    :returns tuple of (h,w)
    """
    from math import floor
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)

    if type(pad) is not tuple:
        pad = (pad, pad)

    h = floor(((h_w[0] + (2 * pad[0]) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1)
    w = floor(((h_w[1] + (2 * pad[1]) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1)
    return h, w


def conv_transpose_output_shape(h_w, kernel_size=1, stride=1, pad=0, output_padding=0):
    """Compute the output shape of a transposed convolution."""
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = (h_w[0] - 1) * stride - (2 * pad) + kernel_size[0] + output_padding
    w = (h_w[1] - 1) * stride - (2 * pad) + kernel_size[1] + output_padding
    return h, w


def default_maxunpool_indices(output_shape, kernel_size, batch_size, channels, device):
    """
    Generate a default index map for nn.MaxUnpool2D operation.

    :param output_shape: the shape that was put into the nn.MaxPool2D operation
    in terms of nn.MaxUnpool2D this will be the output_shape
    :param pool_size: the kernel size of the MaxPool2D
    """
    ph = kernel_size[0]
    pw = kernel_size[1]
    h = output_shape[0]
    w = output_shape[1]
    ih = output_shape[0] // 2
    iw = output_shape[1] // 2
    h_v = torch.arange(ih, dtype=torch.int64, device=device) * pw * ph * iw
    w_v = torch.arange(iw, dtype=torch.int64, device=device) * pw
    h_v = torch.transpose(h_v.unsqueeze(0), 1, 0)
    return (h_v + w_v).expand(batch_size, channels, -1, -1)


class ConvVAE4Fixed(BaseVAE):
    """Convolutional VAE."""

    def __init__(self, input_shape, z_size, variational=True, first_kernel=5, first_stride=2, second_kernel=5,
                 second_stride=2):
        self.input_shape = input_shape
        self.z_size = z_size
        encoder = self.Encoder(input_shape, z_size, first_kernel, first_stride, second_kernel, second_stride)
        decoder = self.Decoder(z_size, encoder.z_shape, first_kernel, first_stride, second_kernel, second_stride)
        BaseVAE.__init__(self, encoder, decoder, variational)

    class Encoder(nn.Module):
        """Encoder."""

        def __init__(self, input_shape, z_size, first_kernel=5, first_stride=2, second_kernel=5, second_stride=2):
            nn.Module.__init__(self)
            # batchnorm in autoencoding is a thing
            # https://arxiv.org/pdf/1602.02282.pdf

            # encoder
            self.e_conv1 = nn.Conv2d(3, 32, kernel_size=first_kernel, stride=first_stride)
            self.e_bn1 = nn.BatchNorm2d(32)
            output_shape = conv_output_shape(input_shape, kernel_size=first_kernel, stride=first_stride)

            self.e_conv2 = nn.Conv2d(32, 128, kernel_size=second_kernel, stride=second_stride)
            self.e_bn2 = nn.BatchNorm2d(128)
            output_shape = conv_output_shape(output_shape, kernel_size=second_kernel, stride=second_stride)

            self.e_conv3 = nn.Conv2d(128, 128, kernel_size=second_kernel, stride=second_stride)
            self.e_bn3 = nn.BatchNorm2d(128)
            self.z_shape = conv_output_shape(output_shape, kernel_size=second_kernel, stride=second_stride)

            self.e_mean = nn.Conv2d(128, z_size, kernel_size=self.z_shape, stride=1)
            self.e_logvar = nn.Conv2d(128, z_size, kernel_size=self.z_shape, stride=1)

        def forward(self, x):
            """Forward pass."""
            encoded = F.relu(self.e_bn1(self.e_conv1(x)))
            encoded = F.relu(self.e_bn2(self.e_conv2(encoded)))
            encoded = F.relu(self.e_bn3(self.e_conv3(encoded)))
            mean = self.e_mean(encoded)
            logvar = self.e_logvar(encoded)
            return mean, logvar

    class Decoder(nn.Module):
        """Decoder."""

        def __init__(self, z_size, z_shape, first_kernel=5, first_stride=2, second_kernel=5, second_stride=2):
            nn.Module.__init__(self)

            # decoder
            self.d_conv1 = nn.ConvTranspose2d(z_size, 128, kernel_size=z_shape, stride=1)
            self.d_bn1 = nn.BatchNorm2d(128)

            self.d_conv2 = nn.ConvTranspose2d(128, 128, kernel_size=second_kernel, stride=second_stride,
                                              output_padding=(1, 0))
            self.d_bn2 = nn.BatchNorm2d(128)

            self.d_conv3 = nn.ConvTranspose2d(128, 32, kernel_size=second_kernel, stride=second_stride,
                                              output_padding=(0, 1))
            self.d_bn3 = nn.BatchNorm2d(32)

            self.d_conv4 = nn.ConvTranspose2d(32, 3, kernel_size=first_kernel, stride=first_stride, output_padding=1)

        def forward(self, z):
            """Forward pass."""
            decoded = F.relu(self.d_bn1(self.d_conv1(z)))
            decoded = F.relu(self.d_bn2(self.d_conv2(decoded)))
            decoded = F.relu(self.d_bn3(self.d_conv3(decoded)))
            decoded = self.d_conv4(decoded)
            return torch.sigmoid(decoded)
