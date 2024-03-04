import torch
from math import ceil, log2
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

def fftconv(x: Tensor, k: Tensor) -> Tensor:
    # Perform convolution using FFT.
    # x: Input tensor of shape (B, H, L).
    # k: Kernel tensor of shape (H, L).
    L = x.shape[-1]
    x_fft = torch.fft.rfft(x, n=2 * L)
    k_fft = torch.fft.rfft(k, n=2 * L)
    result_fft = x_fft * k_fft
    result = torch.fft.irfft(result_fft)[..., :L]
    return result


class Kernel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(self.config.p_dropout)
        self._kern = None

    def forward(self, x):
        k = self.smooth(self._kern, self.config.smooth_size)
        k = self.squash(k)
        k = self.dropout(k)
        x = fftconv(x, k)
        return x

    def squash(self, k):
        if self.config.kern_squash:
            return F.relu(torch.abs(k) - self.config.kern_lam) * torch.sign(k)
        return k

    def smooth(self, k, p):
        if self.config.kern_smooth:
            # Assuming 'p' is the padding size for the smoothing window
            window_size = 2 * p + 1
            # Create a smoothing kernel
            smoothing_kernel = (
                torch.ones((1, 1, window_size)).to(k.device) / window_size
            )
            # Pad the input_tensor along the last dimension
            input_padded = F.pad(k.unsqueeze(1), (p, p), mode="reflect")
            # Apply the smoothing kernel
            smoothed_tensor = F.conv1d(input_padded, smoothing_kernel)
            return smoothed_tensor.squeeze(1)
        else:
            return k


class RandomKernel(Kernel):
    def __init__(self, config):
        super().__init__(config)
        self._kern = nn.Parameter(torch.randn(self.config.H, self.config.L) * 0.002)


class SGConvKernel(Kernel):
    """
    I'm going to follow the description in the original paper:

    For one channel, each kernel (H,L) is constructed from a set of sub-kernels(H,kernel_dim).
    These sub-kernels are weighted by a decay factor so kernels farther to the back have lower values.
    This creates an inductive bias for prioritizing recent information. Furthermore, fewer parameters
    are used for the parts of the kernel farther back. The size they upscale is a power of two based
    on how far back the sub-kernel is.
    """

    def __init__(self, config):
        super().__init__(config)
        self.subkernel_dim = self.config.sg_kern_dim
        self.decay_rate = self.config.sg_decay_rate
        self.num_subkernels = int(
            ceil(log2(self.config.L / self.config.sg_kern_dim)) + 1
        )

        self.interp_factors = [2 ** max(0, i - 1) for i in range(self.num_subkernels)]
        self.decay_factors = [self.decay_rate**i for i in range(self.num_subkernels)]
        self.subkernels = nn.ParameterList()

        for i, _ in enumerate(range(self.num_subkernels)):
            subkernel = nn.Parameter(
                torch.randn(self.config.H, self.config.sg_kern_dim)
            )
            self.subkernels.append(subkernel)

        self.kernel_norm = None

    def forward(self, x):
        scaled_subkernels = [
            F.interpolate(
                subkernel.unsqueeze(0), scale_factor=interp_factor, mode="linear"
            ).squeeze(0)
            * decay_coef
            for decay_coef, interp_factor, subkernel in zip(
                self.decay_factors, self.interp_factors, self.subkernels
            )
        ]
        k = torch.cat(scaled_subkernels, dim=-1)

        if self.kernel_norm is None:
            self.kernel_norm = k.norm(dim=-1, keepdim=True).detach()

        k = k / self.kernel_norm

        self._kern = k

        x = super().forward(x)

        return x


class LongConv(nn.Module):
    def __init__(self, config, init_type="rand"):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(self.config.p_dropout)
        self.act = nn.GELU()
        self.lin_out = nn.Sequential(
            nn.Linear(self.config.H, 2 * self.config.H, bias=True), nn.GLU(dim=-1)
        )
        self.skip = nn.Linear(self.config.H, self.config.H)
        if init_type == "rand":
            self.kern = RandomKernel(self.config)
        if init_type == "sgconv":
            self.kern = SGConvKernel(self.config)

    def forward(self, x):
        res = x
        x = self.kern(x)
        x = self.dropout(self.act(x))
        x = self.lin_out(x.transpose(-1, -2)).transpose(-1, -2)
        x = x + self.skip(res.transpose(-1, -2)).transpose(-1, -2)
        return x


class LongConvDiscriminative(nn.Module):
    def __init__(self, config, init_type="rand"):
        super().__init__()
        self.config = config
        self.encoder = nn.Linear(self.config.d_in, self.config.H)
        self.conv_layers = nn.ModuleList(
            [LongConv(self.config, init_type) for _ in range(self.config.num_layers)]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(self.config.H) for _ in range(self.config.num_layers)]
        )
        self.dropouts = nn.ModuleList(
            [
                nn.Dropout1d(self.config.p_main_dropout)
                for _ in range(self.config.num_layers)
            ]
        )
        self.decoder = nn.Linear(self.config.H, self.config.d_out)

    def forward(self, x):
        x_type = x.dtype
        x = self.encoder(x.transpose(-1, -2)).transpose(-1, -2)
        for layer, norm, dropout in zip(self.conv_layers, self.norms, self.dropouts):
            z = norm(x.transpose(-1, -2)).transpose(-1, -2).to(x_type)
            z = layer(z)
            z = dropout(z)
            x = z + x
        x = torch.mean(x, dim=-1)
        x = self.decoder(x)
        return x


class ConvConfig:
    def __init__(
        self,
        d_in,
        H,
        L,
        p_main_dropout,
        d_out=10,
        p_dropout=0.1,
        p_kern_dropout=0.1,
        kern_squash=True,
        kern_lam=0.1,
        num_layers=2,
        sg_kern_dim=64,
        sg_decay_rate=0.5,
        smooth_size=3,
        kern_smooth=False,
    ):
        self.d_in = d_in
        self.H = H
        self.L = L
        self.d_out = d_out
        self.p_main_dropout = p_main_dropout
        self.p_dropout = p_dropout
        self.p_kern_dropout = p_kern_dropout
        self.kern_squash = kern_squash
        self.kern_lam = kern_lam
        self.num_layers = num_layers
        self.sg_kern_dim = sg_kern_dim
        self.sg_decay_rate = sg_decay_rate
        self.smooth_size = smooth_size
        self.kern_smooth = kern_smooth
