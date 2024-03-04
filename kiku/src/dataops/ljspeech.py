import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio


class LJSpeechDataset(Dataset):
    def __init__(self, directory, seq_len, transform=None):
        """
        Args:
            directory (string): Directory with all the WAV files.
            seq_len (int): Fixed length of the audio sequence.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.directory = directory
        self.seq_len = seq_len
        self.wav_files = [f for f in os.listdir(directory) if f.endswith(".wav")]
        self.transform = transform
        _, self.sample_rate = torchaudio.load(directory + "/" + self.wav_files[0])

    def __len__(self):
        return len(self.wav_files)

    def __getitem__(self, idx):
        wav_path = os.path.join(self.directory, self.wav_files[idx])
        waveform, sample_rate = torchaudio.load(wav_path)

        self.sample_rate = sample_rate

        # Truncate or Pad the waveform to have a fixed length of seq_len
        waveform = self._fix_length(waveform)

        if self.transform:
            waveform = self.transform(waveform)

        return waveform, sample_rate

    def _fix_length(self, waveform):
        if waveform.size(1) > self.seq_len:
            # Truncate the waveform
            waveform = waveform[:, : self.seq_len]
        elif waveform.size(1) < self.seq_len:
            # Pad the waveform
            pad_amount = self.seq_len - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount), "constant", 0)
        return waveform
