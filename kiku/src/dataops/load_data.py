import os

import numpy as np
import pandas as pd
import torch
from pydub import AudioSegment
from pytube import YouTube
from tqdm import tqdm

# TODO: ANYTHING OTHER THAN .PT -> TAKES UP SO MUCH SPACE!!!


def extract_youtube_info(file_path):
    df = pd.read_csv(file_path)
    youtube_info = df[["youtube_id", "start_time"]].values
    return youtube_info


def download_audio(youtube_id, save_path, start_time, duration=10):
    try:
        url = f"https://www.youtube.com/watch?v={youtube_id}"
        yt = YouTube(url)
        audio_stream = yt.streams.filter(only_audio=True).first()
        if not audio_stream:
            print(f"No audio stream found for {youtube_id}")
            return

        audio_file_path = audio_stream.download(
            output_path=save_path, filename=f"{youtube_id}_temp"
        )
        audio = AudioSegment.from_file(audio_file_path, format="mp4")

        audio = audio.set_channels(1)

        # Trim audio
        start_time_ms = start_time * 1000
        end_time_ms = (start_time + duration) * 1000
        audio = audio[start_time_ms:end_time_ms]

        # Save as mp3
        audio.export(f"{save_path}/{youtube_id}.mp3", format="mp3")

    except KeyError as e:
        print(f"KeyError: {e} for video id {youtube_id}.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if "audio_file_path" in locals():
            os.remove(audio_file_path)


def mp3_to_tensor(file_path):
    audio = AudioSegment.from_mp3(file_path)
    audio_array = np.array(audio.get_array_of_samples())
    audio_tensor = torch.tensor(audio_array, dtype=torch.float32)
    return audio_tensor


if __name__ == "__main__":
    print(os.getcwd())
    # Define paths
    csv_path = "./data/audiocaps/csv/train.csv"
    save_path = "./data/audiocaps/sound_tensors"
    os.makedirs(save_path, exist_ok=True)

    # Extract YouTube info from CSV
    youtube_info_list = extract_youtube_info(csv_path)

    # Download and preprocess audio
    for youtube_id, start_time in tqdm(youtube_info_list):
        download_audio(youtube_id, save_path, start_time)
