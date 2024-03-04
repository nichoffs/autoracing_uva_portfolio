import os

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from torch.optim import NAdam
from torch.utils.data import DataLoader, Dataset
from torchaudio.datasets import LIBRISPEECH

from src.baseline.decoder import Decoder
from src.baseline.encoder import Encoder
from src.vq import VectorQuantizer
from src.vqvae import VQVAE
from src.dataops.ljspeech import LJSpeechDataset


# Usage example

# Parameters
batch_size = 16
in_channels = 1
out_channels = 256
latent_channels = 32
seq_len = 16384
kernel_size = 2
num_embeddings = 1200
decoder_channel_list = [4, 2]
encoder_channel_list = decoder_channel_list.reverse()
learning_rate = 0.001
num_epochs = 5
device = "cpu"


ljspeech_dataset = LJSpeechDataset("./data/ljspeech/data/wavs", seq_len=seq_len)
ljspeech_loader = DataLoader(ljspeech_dataset, batch_size=batch_size, shuffle=True)

# Initialize the components of VQ-VAE
encoder = Encoder(in_channels, decoder_channel_list, latent_channels)
decoder = Decoder(
    latent_channels, decoder_channel_list, out_channels, seq_len, kernel_size
)
vq = VectorQuantizer(num_embeddings, latent_channels)

# Initialize VQ-VAE model
model = VQVAE(encoder, decoder, vq).to(device)

# Define optimizer
optimizer = NAdam(model.parameters(), lr=learning_rate)


class VQLoss(nn.Module):
    def __init__(self, commitment_cost=0.25):
        super(VQLoss, self).__init__()
        self.commitment_cost = commitment_cost
        # self.mse_loss = nn.MSELoss()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, x, recon_x, z, z_q):
        # Reconstruction loss
        # recon_loss = self.mse_loss(recon_x, x)
        target = x.long()  # Removing channel dimension and converting to long
        recon_loss = self.cross_entropy(recon_x, target.squeeze())

        # Vector quantization loss
        codebook_loss = torch.mean((z_q.detach() - z) ** 2)
        commitment_loss = torch.mean((z_q - z.detach()) ** 2)

        # Combine the losses
        loss = recon_loss + codebook_loss + self.commitment_cost * commitment_loss
        return loss


# Initialize the VQ-VAE loss
loss_function = VQLoss(commitment_cost=0.25).to(device)
model_save_path = "./saved_models"  # Define the directory to save models
os.makedirs(model_save_path, exist_ok=True)  # Create the directory if it doesn't exist

fixed_sample, _ = next(iter(ljspeech_loader))
fixed_sample = fixed_sample.to(device)

for epoch in range(num_epochs):
    model.train()
    for ix, (x, sr) in enumerate(ljspeech_loader):
        x = x.to(device)
        optimizer.zero_grad()

        # Forward pass
        recon_x, z_q, z = model(x.float())
        # Compute loss
        loss = loss_function(x, recon_x, z, z_q)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {loss:.4f}")

        # Save reconstructions every 10 epochs
        # if (ix + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            recon_fixed_sample, _, _ = model(fixed_sample.float())

        # Assuming you have a mu-law decoding function
        decoded_recon_sample = torchaudio.transforms.MuLawDecoding()(
            recon_fixed_sample.squeeze(0)
        ).cpu()
        decoded_fixed_sample = torchaudio.transforms.MuLawDecoding()(
            fixed_sample.squeeze(0)
        ).cpu()
        # Create a directory for the current epoch
        epoch_dir = f"epoch_{epoch+1}"
        os.makedirs(epoch_dir, exist_ok=True)

        # Save the reconstructed and original audio
        torchaudio.save(
            os.path.join(epoch_dir, "reconstruction.wav"),
            decoded_recon_sample,
            ljspeech_dataset.sample_rate,
        )
        torchaudio.save(
            os.path.join(epoch_dir, "original.wav"),
            decoded_fixed_sample,
            ljspeech_dataset.sample_rate,
        )

    # Save the model
    torch.save(
        model.state_dict(), os.path.join(model_save_path, f"vqvae_epoch_{epoch+1}.pth")
    )
