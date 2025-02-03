import torch
import numpy as np
import torchaudio.transforms as T
import random
import torchaudio



""" 
pre-prosses for the auxliry data
take 1 sec (from start of activity) of sound
as the anchor , slightly augment to get positive take the next one for negetive

    Args:
        subset (list of dict): an unpack .pkl file from ivrit.ai dataset , a list of audio files
    Returns:
    a list of tuples:(anchor,positive,negetive) as mentiond above    
"""

def preprocess_dataset(subset):
    preprocessed = []
    for idx in range(len(subset) - 1):# tqdm(range(len(self.dataset) - 1), desc="Processing dataset"):
        # Anchor: Original audio sample
        audio_data_anchor = subset[idx]['audio']
        audio_attrb = subset[idx]['attrs']
        anchor_waveform = np.array(audio_data_anchor['array'])
        sr = audio_data_anchor['sampling_rate']

        start = int(np.floor(audio_attrb['end']))
        anchor_waveform = anchor_waveform[start:start + sr]
        # Positive: Augmented version of the anchor
        #print ("aug")
        positive_waveform =aug(anchor_waveform, samp_rate=sr)

        # Negative: Next audio sample
        audio_data_negative = subset[idx + 1]['audio']
        audio_nxtattrb = subset[idx + 1]['attrs']
        negative_waveform = np.array(audio_data_negative['array'])
        start = int(np.floor(audio_nxtattrb['end']))
        negative_waveform = negative_waveform[start:start + sr]

        # Convert to tensors
        anchor = torch.tensor(anchor_waveform, dtype=torch.float32)
        positive = torch.tensor(positive_waveform, dtype=torch.float32)
        negative = torch.tensor(negative_waveform, dtype=torch.float32)
        #print("mfcc")
        # Compute MFCC features
        anchor = get_mfcc(waveform=anchor, sample_rate=sr)
        positive = get_mfcc(waveform=positive, sample_rate=sr)
        negative = get_mfcc(waveform=negative, sample_rate=sr)

        preprocessed.append((anchor, positive, negative))
    return preprocessed


def get_mfcc(waveform, sample_rate, n_mfcc=10):
    """
    Compute MFCCs for a given waveform.

    Args:
        waveform (torch.Tensor): Input waveform.
        sample_rate (int): Sampling rate of the waveform.
        n_mfcc (int): Number of MFCCs to compute.

    Returns:
        torch.Tensor: Computed MFCCs with shape (1, n_mfcc, num_frames).
    """
    win_length_ms = 40
    hop_length_ms = 20

    # Convert window and hop lengths from milliseconds to samples
    win_length = int(sample_rate * (win_length_ms / 1000))
    hop_length = int(sample_rate * (hop_length_ms / 1000))

    # Calculate the number of frames
    num_frames = (len(waveform) - win_length) // hop_length + 1

    # Extract frames with 20ms stride
    frames = torch.stack(
        [waveform[i * hop_length: i * hop_length + win_length] for i in range(num_frames)],
        dim=0
    )

    # Apply Hamming window to each frame
    hamming_window = np.hamming(win_length)
    hamming_window = torch.from_numpy(hamming_window).float()
    frames = frames * hamming_window

    # Reshape waveform into a single tensor
    windowed_waveform = frames.reshape((1, num_frames * win_length)).float()

    # Compute MFCCs
    mfcc_transform = T.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={"n_fft": win_length, "hop_length": hop_length * 2}
    )
    TF_map = mfcc_transform(windowed_waveform)  # Shape: (1, n_mfcc, num_frames)
    return TF_map

"""
    Slightly augment a given waveform.

    Args:
        waveform (torch.Tensor): Input waveform.
        sample_rate (int): Sampling rate of the waveform.
    Returns:
        numpy wavwform : augmented waveform 
    """


def aug(waveform, samp_rate):
    # Apply volume change
    random.seed(42)
    torch.manual_seed(42)

    waveform = torch.from_numpy(waveform).float()
    volume_factor = random.uniform(0.8, 1.2)
    pitch_shift = random.randint(-8, 8)
    pre_emp_p = random.uniform(0.94, 0.99)

    vol_trans = torchaudio.transforms.Vol(volume_factor, gain_type="amplitude")
    pitch_trans = torchaudio.transforms.PitchShift(sample_rate=samp_rate, n_steps=pitch_shift,
                                                   bins_per_octave=64, n_fft=1024)
    pre_emp_trans = torchaudio.transforms.Preemphasis(pre_emp_p)

    waveform = vol_trans(waveform)
    waveform = pitch_trans(waveform)
    waveform = pre_emp_trans(waveform)

    return waveform.detach().numpy()