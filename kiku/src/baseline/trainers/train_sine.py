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
import imageio


class SineWaveDataset(Dataset):
    def __init__(
        self, seq_len, sample_rate=44100, frequency=440.0, quantization_channels=256
    ):
        super(SineWaveDataset, self).__init__()
        self.seq_len = seq_len
        self.sample_rate = sample_rate
        self.frequency = frequency
        self.timesteps = (torch.arange(0, seq_len) / sample_rate).float()
        self.mu_law_encoding = torchaudio.transforms.MuLawEncoding(
            quantization_channels
        )

    def __len__(self):
        return self.seq_len

    def __getitem__(self, idx):
        waveform = torch.sin(2 * np.pi * self.frequency * self.timesteps)
        waveform = self.mu_law_encoding(waveform)
        return waveform.unsqueeze(0)


# Usage example

# Parameters
batch_size = 16
in_channels = 1
out_channels = 256
latent_channels = 32
seq_len = 1024
kernel_size = 2
num_embeddings = 1200
decoder_channel_list = [32, 12]
encoder_channel_list = decoder_channel_list.reverse()
learning_rate = 0.001
num_epochs = 5
device = "mps"


sine_dataset = SineWaveDataset(seq_len=seq_len)
sine_loader = DataLoader(sine_dataset, batch_size=16, shuffle=True)

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
        target = x.squeeze(
            1
        ).long()  # Removing channel dimension and converting to long

        recon_loss = self.cross_entropy(recon_x, target)

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

for epoch in range(num_epochs):
    model.train()
    for batch in sine_loader:
        x = batch.to(device)

        optimizer.zero_grad()

        # Forward pass
        recon_x, z_q, z = model(x.float())

        # Compute loss
        loss = loss_function(x, recon_x, z, z_q)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {loss:.4f}")

    # Save the model
    torch.save(
        model.state_dict(), os.path.join(model_save_path, f"vqvae_epoch_{epoch+1}.pth")
    )


def generate_sample(model, start_sample, seq_len, device):
    model.eval()
    samples = start_sample.to(device).float()  # Ensure the input is Float type

    with torch.no_grad():
        for _ in range(seq_len):
            output = model(samples)
            next_sample = output[:, :, -1].argmax(dim=1).float()  # Convert to Float
            next_sample = next_sample.unsqueeze(0).unsqueeze(0)  # Adjust dimensions

            samples = torch.cat((samples, next_sample), dim=2)

            if samples.size(2) > seq_len:
                samples = samples[:, :, -seq_len:]

    # Decode the mu-law encoded samples
    decoded_samples = mu_law_decoding(samples.squeeze(0)).cpu()
    return decoded_samples


# Generate a new audio sample
start_sample = torch.randn(1, 1, 100)  # Random noise as the starting point
generated_seq_len = 1024  # Length of the sequence to generate
generated_audio = generate_sample(model, start_sample, generated_seq_len, device)

# Convert the generated audio to a numpy array and save as a WAV file
generated_audio_np = generated_audio.numpy()
torchaudio.save(
    "generated_audio.wav",
    torch.FloatTensor(generated_audio_np),
    sine_dataset.sample_rate,
)
