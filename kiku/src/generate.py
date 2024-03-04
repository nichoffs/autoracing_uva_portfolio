import os

import torch
import torchaudio

from baseline.decoder import Decoder
from baseline.encoder import Encoder
from vq import VectorQuantizer
from vqvae import VQVAE

# Parameters
in_channels = 1
latent_channels = 128
seq_len = 16384  # Updated to 16384 as per your latest script
kernel_size = 3
num_embeddings = 1200
encoder_channel_list = [1, 4, 8, 16, 64, 128]
decoder_channel_list = [1, 4, 8, 16, 64, 128]
device = "cpu"  # Make sure this matches your hardware configuration

# Initialize the components of VQ-VAE
encoder = Encoder(in_channels, encoder_channel_list, latent_channels)
decoder = Decoder(latent_channels, decoder_channel_list, in_channels, kernel_size)
vq = VectorQuantizer(num_embeddings, latent_channels)

# Initialize VQ-VAE model
model = VQVAE(encoder, decoder, vq).to(device)

# Load the model
model_path = "./saved_models/vqvae_epoch_1.pth"  # Update with the desired epoch
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()


# Function to generate audio sample
def generate_audio_sample(model, input_tensor):
    with torch.no_grad():
        reconstructed, _, _ = model(input_tensor)
    return reconstructed


import os

print(f"{os.getcwd()=}")
# Path to an audio file for testing
audio_path = (
    "./data/audiocaps/sound_tensors/-8ysWJSrITc.mp3"  # Replace with a valid file path
)
waveform, sample_rate = torchaudio.load(audio_path)

# Process the audio to match the input requirements of your model
if waveform.size(1) > seq_len:
    waveform = waveform[:, -seq_len:]

# Generate audio sample
waveform = waveform.to(device)
reconstructed_waveform = generate_audio_sample(model, waveform.unsqueeze(0))

# Convert the waveform to a format that can be played
reconstructed_waveform = reconstructed_waveform.squeeze().cpu().numpy()

# Save the reconstructed waveform
reconstructed_audio_dir = "./data/audiocaps/generated/"  # Define the directory path
os.makedirs(
    reconstructed_audio_dir, exist_ok=True
)  # Create the directory if it doesn't exist

# Define the full path to save the reconstructed audio
reconstructed_audio_path = os.path.join(
    reconstructed_audio_dir, "reconstructed_audio.wav"
)

# Save the reconstructed waveform
torchaudio.save(
    reconstructed_audio_path,
    torch.FloatTensor(reconstructed_waveform).unsqueeze(0),
    sample_rate,
)

print(f"Reconstructed audio saved to {reconstructed_audio_path}")
