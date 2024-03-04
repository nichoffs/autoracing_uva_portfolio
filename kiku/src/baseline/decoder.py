import torch
import torch.nn as nn
import torch.nn.functional as F

from src.helpers import ConvBlock, Wavenet


class UpBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, seq_len, kernel_size):
        super(UpBlock, self).__init__(
            nn.Upsample(scale_factor=2, mode="nearest"),
            Wavenet(in_channels, out_channels, seq_len, kernel_size),
        )


class Decoder(nn.Module):
    def __init__(self, in_channels, channels_list, out_channels, seq_len, kernel_size):
        super(Decoder, self).__init__()

        # Calculate padding for 'same' effect
        padding = kernel_size // 2

        # Initial convolution layer
        self.initial_conv = ConvBlock(
            in_channels, channels_list[0], kernel_size=1, stride=1, padding=0
        )

        # Create a list of UpBlocks
        self.up_blocks = nn.ModuleList()
        for i in range(len(channels_list) - 1):
            self.up_blocks.append(
                UpBlock(channels_list[i], channels_list[i + 1], seq_len, kernel_size)
            )

        # Final upsampling
        self.final_upsample = UpBlock(
            channels_list[-1], out_channels, seq_len, kernel_size
        )

    def forward(self, x):
        x = self.initial_conv(x)
        for up_block in self.up_blocks:
            x = up_block(x)
        x = self.final_upsample(x)
        return x
