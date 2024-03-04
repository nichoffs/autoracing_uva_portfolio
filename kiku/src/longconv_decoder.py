import torch
import torch.nn as nn

from helpers import LongConv

kernel_dim = 32
seq_len = 16384
in_chan = 1
chan = 3
channels_list = [8, 16, 64, 128, 256]
latent_channels = 128
batch_size = 16
interp_factor = 2
decay_coef = 0.5


class Decoder(nn.Module):
    def __init__(
        self,
        out_channels,
        channels_list,
        in_channels,
        out_channel,
        kernel_dim,
        seq_len,
        chan,
        interp_factor,
        decay_coef,
    ):
        super(Decoder, self).__init__()

        self.layers = nn.ModuleList()
        current_out_channels = out_channels

        # Reverse the channels_list for the decoder
        reversed_channels_list = list(reversed(channels_list))

        for in_channels in reversed_channels_list:
            # Upsample followed by LongConv
            self.layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
            self.layers.append(
                LongConv(
                    kernel_dim,
                    seq_len,
                    current_out_channels,
                    chan,
                    in_channels,
                    interp_factor,
                    decay_coef,
                )
            )
            # Update the sequence length and out_channels for the next iteration
            seq_len = seq_len * 2
            current_out_channels = in_channels

        # Final layer to match the original input channels
        self.final_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        for block in self.layers:
            x = block(x)
        x = self.final_conv(x)
        return x


if __name__ == "__main__":
    # Example usage
    latent_channels = 128
    channels_list = [8, 16]
    in_chan = 1
    kernel_dim = 32
    seq_len = 16384 // (
        2 ** len(channels_list)
    )  # Adjusted for the initial seq_len of the decoder
    chan = 3
    interp_factor = 2
    decay_coef = 0.5
    batch_size = 16

    decoder = Decoder(
        latent_channels,
        channels_list,
        in_chan,
        kernel_dim,
        seq_len,
        chan,
        interp_factor,
        decay_coef,
    )
    x = torch.randn((batch_size, latent_channels, seq_len))
    print(f"{decoder(x).shape=}")
