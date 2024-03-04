import torch
import torch.nn as nn

from helpers import Downsample, LongConv

kernel_dim = 32
seq_len = 16384
in_chan = 1
chan = 3
channels_list = [8, 16, 64, 128, 256]
latent_channels = 128
batch_size = 16
interp_factor = 2
decay_coef = 0.5


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels,
        channels_list,
        latent_channels,
        kernel_dim,
        seq_len,
        chan,
        interp_factor,
        decay_coef,
    ):
        super(Encoder, self).__init__()

        self.layers = (
            nn.ModuleList()
        )  # Change to ModuleList for proper module registration
        current_in_channels = in_channels

        # Iterate over the channels_list and create LongConv -> Downsample blocks
        for out_channels in channels_list:
            self.layers.append(
                LongConv(
                    kernel_dim,
                    seq_len,  # seq_len needs to be adjusted if LongConv changes it
                    current_in_channels,
                    chan,
                    out_channels,
                    interp_factor,
                    decay_coef,
                )
            )
            self.layers.append(Downsample(out_channels, out_channels))
            seq_len = (
                seq_len // 2
            )  # Adjust seq_len for next LongConv, assuming Downsample halves it

            # Update in_channels for the next iteration
            current_in_channels = out_channels

        self.out_latent = nn.Conv1d(channels_list[-1], latent_channels, kernel_size=1)

    def forward(self, x):
        for block in self.layers:
            x = block(x)
        x = self.out_latent(x)
        return x


if __name__ == "__main__":
    # Example usage
    in_chan = 1
    channels_list = [8, 16]
    latent_channels = 128
    kernel_dim = 32
    seq_len = 16384
    chan = 3
    interp_factor = 2
    decay_coef = 0.5
    batch_size = 16

    encoder = Encoder(
        in_chan,
        channels_list,
        latent_channels,
        kernel_dim,
        seq_len,
        chan,
        interp_factor,
        decay_coef,
    )
    x = torch.randn((batch_size, in_chan, seq_len))
    print(f"{encoder(x).shape=}")
