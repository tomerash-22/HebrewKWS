
import librosa
import  torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
import os
import random
import os
import random
import torchaudio
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split




class SpeechCommandsTripletDataset(Dataset):
    def __init__(self, data_dir, validation_split=0.2, sample_rate=16000, n_mfcc=10, transform=None):
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.transform = transform
        self.validation_split = validation_split

        # List all audio files in the directory and organize them by label
        self.files_by_label = self._get_files_by_label()
        # Flatten all the files to a list of (file_path, label)
        self.all_files = self._get_all_files()


    def _get_files_by_label(self):
        files_by_label = {}
        for label in os.listdir(self.data_dir):
            label_path = os.path.join(self.data_dir, label)
            if os.path.isdir(label_path):
                files_by_label[label] = [os.path.join(label_path, f) for f in os.listdir(label_path) if
                                         f.endswith('.wav')]
        return files_by_label

    def _get_all_files(self):
        all_files = []
        for label, files in self.files_by_label.items():
            for file in files:
                all_files.append((file, label))
        return all_files

    def _split_data(self):
        all_files = []
        for label, files in self.files_by_label.items():
            for file in files:
                all_files.append((file, label))

        random.shuffle(all_files)

        # Split into train and validation sets
        split_idx = int(len(all_files) * (1 - self.validation_split))
        train_files = all_files[:split_idx]
        val_files = all_files[split_idx:]

        return train_files, val_files

    def _get_mfcc(self, file_path,sample_rate=16000, n_mfcc=10):
        # Load the audio file
        waveform, sr = torchaudio.load(file_path, normalize=True)

        # Resample the audio if it's not at the desired sample rate
        if sr != sample_rate:
            waveform = T.Resample(sr, sample_rate)(waveform)
        if len(waveform[0]) < sample_rate : #less then 1s recording
            pad = self.sample_rate-len(waveform[0])
            waveform = torch.cat([waveform ,torch.zeros(1,pad)] , dim=1)
            #print("padded")
        elif len(waveform[0]) > sample_rate:
            # Truncate the waveform to the target length
            waveform = waveform[:, :sample_rate]
            #print("trun")
            # Compute frame sizes in samples
        win_length_ms = 40
        hop_length_ms = 20

        win_length = int(sample_rate * (win_length_ms / 1000))  # 40ms window
        hop_length = int(sample_rate * (hop_length_ms / 1000))  # 20ms stride
        # Manually extract frames with 20ms stride
        num_frames = (waveform.shape[1] - win_length) // hop_length + 1
        frames = torch.stack([waveform[:, i * hop_length: i * hop_length + win_length] for i in range(num_frames)],
                             dim=0)

        # Apply Hamming window to each frame
        hamming_window = np.hamming(win_length)
        frames = frames * hamming_window
        windowed_waveform = frames.reshape((1, num_frames * win_length)).float()
        # Compute the MFCCs
        mfcc_transform = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={"n_fft": win_length, "hop_length": hop_length * 2}
        )
        TF_map = mfcc_transform(windowed_waveform)  # 1,10,50 TF map
        #print(TF_map.shape)
        return TF_map

    def _create_triplet(self, anchor_file, anchor_label):
        # Positive: Same label
        positive_file = random.choice(self.files_by_label[anchor_label])

        # Negative: Random label that is not the same as anchor's label
        negative_label = random.choice([label for label in self.files_by_label.keys() if label != anchor_label])
        negative_file = random.choice(self.files_by_label[negative_label])

        # Load MFCCs for anchor, positive, and negative files
        anchor_mfcc = self._get_mfcc(anchor_file)
        positive_mfcc = self._get_mfcc(positive_file)
        negative_mfcc = self._get_mfcc(negative_file)

        return anchor_mfcc, positive_mfcc, negative_mfcc

    def __len__(self):
        return len(self.all_files)



    def __getitem__(self, idx):
        # Get a random anchor
        anchor_file, anchor_label = self.all_files[idx]
        # Create a triplet: anchor, positive, negative
        anchor_mfcc, positive_mfcc, negative_mfcc = self._create_triplet(anchor_file, anchor_label)

        if self.transform:
            anchor_mfcc = anchor_mfcc.permute(2, 1, 0)
            positive_mfcc = positive_mfcc.permute(2, 1, 0)
            negative_mfcc = negative_mfcc.permute(2, 1, 0)

        return anchor_mfcc, positive_mfcc, negative_mfcc

class Hebrew_test(Dataset):

    def __init__(self, data_dir, label, n_mfcc=10, transform=None):
        self.data_dir = data_dir

        self.n_mfcc = n_mfcc
        self.transform = transform
        self.label = label

        # List all audio files in the directory and organize them by label
        self.files_by_label = self._get_files_by_label()
        # Flatten all the files to a list of (file_path, label)
        self.all_files = self._get_all_files()
        self.label_files = self.files_by_label[label]

    def __len__(self):
        return len(self.label_files)  # Number of files for the given label

    def __getitem__(self, idx):
        anchor_file = self.label_files[idx]
        anchor_mfcc = self._get_mfcc_heb(anchor_file)
        return anchor_mfcc

    def _get_files_by_label(self):
        files_by_label = {}
        for label in os.listdir(self.data_dir):
            label_path = os.path.join(self.data_dir, label)
            if os.path.isdir(label_path):
                files_by_label[label] = [os.path.join(label_path, f) for f in os.listdir(label_path) if
                                         f.endswith('.wav')]
        return files_by_label

    def _get_all_files(self):
        all_files = []
        for label, files in self.files_by_label.items():
            for file in files:
                all_files.append((file, label))
        return all_files
    def _get_mfcc_heb(self, file_path, n_mfcc=10):
        # Load the audio file
        waveform, sr = torchaudio.load(file_path, normalize=True)
        try:
            len_d=len(waveform[1])
            if len_d>0:
                waveform=waveform[0]
        except Exception as e:
            print(e)

        if waveform.dim() == 2 and waveform.size(0) == 1:
            waveform= waveform.squeeze(0)
        # Resample the audio if it's not at the desired sample rate
        if len(waveform) < sr : #less then 1s recording
            pad = sr-len(waveform)
            waveform = torch.cat([waveform ,torch.zeros(pad)] , dim=0)
            #print("padded")
        elif len(waveform) > sr:
            # Truncate the waveform to the target length
            waveform = waveform[:sr]
            #print("trun")
            # Compute frame sizes in samples
        win_length_ms = 40
        hop_length_ms = 20

        win_length = int(sr * (win_length_ms / 1000))  # 40ms window
        hop_length = int(sr * (hop_length_ms / 1000))  # 20ms stride
        # Manually extract frames with 20ms stride
        num_frames = (len(waveform) - win_length) // hop_length + 1
        frames = torch.stack([waveform[i * hop_length: i * hop_length + win_length] for i in range(num_frames)],
                             dim=0)

        # Apply Hamming window to each frame
        hamming_window = np.hamming(win_length)
        hamming_window = torch.from_numpy(hamming_window).float()
        frames = frames * hamming_window
        windowed_waveform = frames.reshape((1, num_frames * win_length)).float()
        # Compute the MFCCs
        mfcc_transform = T.MFCC(
            sample_rate=sr,
            n_mfcc=n_mfcc,
            melkwargs={"n_fft": win_length, "hop_length": hop_length * 2}
        )
        TF_map = mfcc_transform(windowed_waveform)  # 1,10,50 TF map
        #print(TF_map.shape)
        return TF_map




# Function to create DataLoaders
def create_dataloaders(data_dir, validation_split=0.2, batch_size=32):
    dataset = SpeechCommandsTripletDataset(data_dir, validation_split=validation_split)

    # Split dataset into train and validation
    train_size = int(0.8 * len(dataset))  # 80% for training
    val_size = len(dataset) - train_size  # Remaining 20% for validation
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoader for training
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=0)

    # Create DataLoader for validation
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=0)

    return train_loader, val_loader


def create_test_dataloaders(data_dir, label):
    """
    Creates a test DataLoader for a given label, ensuring no overlap with training samples.

    Args:
        data_dir (str): Path to the dataset directory.
        label (str): Label for which to create the test DataLoader.
        train_samples (list): List of file paths used for training to exclude from the test set.

    Returns:
        DataLoader: DataLoader for testing.
    """
    test_dataset = Hebrew_test(data_dir, label)
    loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return loader

#
# val,train = create_dataloaders('SpeechCommands/speech_commands_v0.02',validation_split=0.2,batch_size=32)
#
# for anc,pos,neg in val:
#     break