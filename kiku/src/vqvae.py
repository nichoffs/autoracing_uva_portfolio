import torch
import torch.nn as nn
import torch.nn.functional as F

from src.baseline.decoder import Decoder
from src.baseline.encoder import Encoder
from src.vq import VectorQuantizer


class VQVAE(nn.Module):
    def __init__(self, encoder, decoder, vq):
        super().__init__()

        self.encoder = encoder
        self.vq = vq
        self.decoder = decoder

    def forward(self, x):
        z = self.encoder(x)
        z_q = self.vq(z.permute(0, 2, 1)).permute(0, 2, 1)
        recon_x = self.decoder(z_q)
        return recon_x, z_q, z
