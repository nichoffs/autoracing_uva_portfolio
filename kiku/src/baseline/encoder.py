import torch.nn as nn
from src.helpers import ConvBlock, Downsample


class Encoder(nn.Module):
    def __init__(self, in_channels, channels_list, latent_channels):
        super().__init__()

        # Define the stem convolution block
        self.stem = ConvBlock(in_channels, channels_list[0], 3)

        # Sequentially stack Downsample blocks
        layers = []
        for in_ch, out_ch in zip(channels_list, channels_list[1:]):
            layers.append(Downsample(in_ch, out_ch))
            layers.append(nn.GELU())
        self.blocks = nn.Sequential(*layers)

        # Define the final downsample layer to latent space
        self.to_latent = Downsample(channels_list[-1], latent_channels)

    def forward(self, x):
        x = self.stem(x)  # Apply the stem block
        x = self.blocks(x)  # Pass through the sequence of Downsample blocks
        x = self.to_latent(x)  # Convert to latent representation
        return x
