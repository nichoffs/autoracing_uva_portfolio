import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        activation_fn=True,
    ):
        super(ConvBlock, self).__init__(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm1d(out_channels),
            nn.GELU() if activation_fn else nn.Identity(),
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self[0].weight)
        bn = self[1]
        nn.init.ones_(bn.weight)
        nn.init.zeros_(bn.bias)


class Downsample(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Downsample, self).__init__(
            ConvBlock(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        )


class LongConv(nn.Module):
    def __init__(
        self, kernel_dim, seq_len, in_chan, chan, out_chan, interp_factor, decay_coef
    ):
        super().__init__()

        self.kernel_dim = kernel_dim
        self.seq_len = seq_len
        self.in_chan = in_chan
        self.chan = chan
        self.interp_factor = interp_factor
        self.decay_coef = decay_coef

        self.num_kernels = int(
            1 + torch.ceil(torch.log2(torch.Tensor([self.seq_len / self.kernel_dim])))
        )
        self.kernel_interp_dims = [
            kernel_dim * interp_factor**i for i in range(self.num_kernels)
        ]

        # Register decay coefficients as buffers
        self.decay_coefs = [decay_coef ** (i) for i in range(self.num_kernels)]
        for i, decay in enumerate(self.decay_coefs):
            self.register_buffer(f"decay_coef_{i}", torch.tensor(decay))

        # Initialize kernel list as parameters
        self.kernel_list = nn.ParameterList(
            [
                nn.Parameter(torch.randn(chan, in_chan, kernel_dim))
                for _ in range(self.num_kernels)
            ]
        )

        # Initialize upscaled kernels
        self.upscaled_kernels = [
            F.interpolate(
                self.kernel_list[i] * getattr(self, f"decay_coef_{i}"), interp_dim
            )
            for i, interp_dim in enumerate(self.kernel_interp_dims)
        ]

        self.kernel = torch.cat(self.upscaled_kernels, dim=-1)
        self.kernel = self.kernel[..., :seq_len]

        self.Z = torch.norm(self.kernel, p="fro")  # Frobenius norm

        self.kernel = self.kernel / self.Z

        self.channel_mix = nn.Conv1d(chan * in_chan, out_chan, kernel_size=1)

    def forward(self, x):
        with torch.no_grad():
            k_f = torch.fft.rfft(self.kernel, n=2 * self.seq_len)
            x_f = torch.fft.rfft(x, n=2 * self.seq_len)

        y_f = torch.einsum("bhl,chl->bchl", x_f, k_f)
        with torch.no_grad():
            y = torch.fft.irfft(y_f, n=2 * self.seq_len)[..., : self.seq_len]
        y = y.flatten(1, 2)
        y = self.channel_mix(y)

        return y


class Wavenet(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len, kernel_size):
        super(Wavenet, self).__init__()

        self.kernel_size = kernel_size
        self.num_layers = self.calculate_max_layers(seq_len, kernel_size)
        self.dilation_factors = [2**i for i in range(self.num_layers)]

        self.dim_matcher = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.filter_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Tanh(),
                    nn.Conv1d(
                        out_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        dilation=dilation_factor,
                    ),
                )
                for dilation_factor in self.dilation_factors
            ]
        )
        self.gate_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Sigmoid(),
                    nn.Conv1d(
                        out_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        dilation=dilation_factor,
                    ),
                )
                for dilation_factor in self.dilation_factors
            ]
        )
        self.final_conv = nn.Conv1d(out_channels, out_channels, kernel_size=1, stride=1)

    def calculate_max_layers(self, seq_len, kernel_size):
        max_layers = int(torch.log2(torch.tensor(seq_len)).item())
        max_receptive_field = (2**max_layers) * (kernel_size - 1)

        if max_receptive_field > seq_len:
            max_layers -= (
                1  # Decrease layers to keep receptive field within sequence length
            )

        return max_layers

    def calculate_receptive_field(self):
        return (2**self.num_layers) * (self.kernel_size - 1)

    def forward(self, x):
        x = self.dim_matcher(x)
        res = x
        for i in range(self.num_layers):
            dilation_factor = 2**i
            padding = (self.kernel_size - 1) * dilation_factor

            # Apply manual padding
            x_padded = F.pad(x, (padding, 0))

            # Apply convolution and slice the output
            filter_out = self.filter_convs[i](x_padded)
            gate_out = self.gate_convs[i](x_padded)
            x = (filter_out * gate_out)[
                :, :, : x.shape[-1]
            ]  # Slice to match input length

            res = res + x
        res = F.relu(res)
        res = self.final_conv(x)
        res = F.relu(res)
        probs = F.softmax(res, dim=1)
        return res
