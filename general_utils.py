import os
import sys
import soundfile as sf
import torch
import pickle
import numpy as np
import datasets
from datasets import load_dataset
from huggingface_hub import login


def suppress_prints():
    sys.stdout = open(os.devnull, 'w')

"Enable prints after using the above"
def enable_prints():
    sys.stdout = sys.__stdout__

def play_audio(audio_data, sample_rate):
    # Use soundfile to play the audio
    sf.write('temp_audio.wav', audio_data, sample_rate)  # Save temporarily
    # Now play using an external player or load it back for processing
    print(f"Playing audio with sample rate {sample_rate}...")

def save_torch_model(model, path="DL_proj_latest_model.pth"):
    """Saves a PyTorch model."""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

"load the used dataset from ivrit.ai"

def load_ivrit_ai_dataset(token):
    login(token=token)
    ds = load_dataset("ivrit-ai/audio-vad",streaming=True,split="train")
    return ds

"create pkl files from a loaded ivrit.ai dataset"

def ivrit_ai_gen_pkl_files(ds,K_cnt=100):
    n1 = 0
    n2 = 1000
    for i in range(K_cnt):
        subset = []
        for batch_idx, batch in enumerate(ds):
            if n1 < batch_idx < n2:
                subset.append(batch)
            if batch_idx == n2:
                break
        filename = f'subset_list_{i:03d}.pkl'  # Create a filename with padded iteration index
        with open(filename, 'wb') as f:
            pickle.dump(subset, f)
            n1 = n1 + 1000
            n2 = n2 + 1000