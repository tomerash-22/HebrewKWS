
import soundfile as sf
import numpy as np
import torch
import random
from torch.utils.data import DataLoader,Dataset
from datasets import load_dataset
from huggingface_hub import login
#from datasets import Dataset
import  torchaudio
import torchaudio.transforms as T
import pickle
import os
from tqdm import tqdm
import warnings
import sys
import time
def enable_prints():
    sys.stdout = sys.__stdout__
class HebAudioDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Anchor: Original audio sample
        audio_data_anchor = self.dataset[idx]['audio']
        audio_attrb = self.dataset[idx]['attrs']
        anchor_waveform = np.array(audio_data_anchor['array'])  # Convert to numpy array
        sr = audio_data_anchor['sampling_rate']

        start = int(np.floor(audio_attrb['end']))
        anchor_waveform = anchor_waveform[start:start + sr]

        # Convert to tensors
        anchor = torch.tensor(anchor_waveform, dtype=torch.float32)
        anchor= self._get_mfcc(anchor, sample_rate=sr)

        return anchor
    def _get_mfcc(self, waveform, sample_rate, n_mfcc=10):
                # Load the audio file

                win_length_ms = 40
                hop_length_ms = 20

                win_length = int(sample_rate * (win_length_ms / 1000))  # 40ms window
                hop_length = int(sample_rate * (hop_length_ms / 1000))  # 20ms stride
                # Manually extract frames with 20ms stride
                num_frames = (len(waveform) - win_length) // hop_length + 1
                frames = torch.stack([waveform[ i * hop_length: i * hop_length + win_length] for i in range(num_frames)],
                                     dim=0)

                # Apply Hamming window to each frame
                hamming_window = np.hamming(win_length)
                hamming_window = torch.from_numpy(hamming_window).float()
                frames = frames * hamming_window
                windowed_waveform = frames.reshape((1, num_frames * win_length)).float()
                # Compute the MFCCs
                mfcc_transform = T.MFCC(
                    sample_rate=sample_rate,
                    n_mfcc=n_mfcc,
                    melkwargs={"n_fft": win_length, "hop_length": hop_length * 2}
                )
                TF_map = mfcc_transform(windowed_waveform)  # 1,10,50 TF map
                # print(TF_map.shape)
                return TF_map

class TripletAudioDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, save_path="preprocessed_dataset.pkl" ,force_reprocess=False):


            """
            Initialize the dataset. Preprocess the data if not already saved.
            :param dataset: The raw dataset to process.
            :param save_path1: Path to save or load the first part of the preprocessed data.
            :param save_path2: Path to save or load the second part of the preprocessed data.
            :param force_reprocess: Whether to force preprocessing even if saved files exist.
            """
            self.dataset = dataset
            self.save_path = save_path


            if not force_reprocess and os.path.exists(self.save_path):
                print(f"Loading preprocessed dataset from {self.save_path} ...")
                with open(self.save_path, "rb") as f1:
                    self.preprocessed_data = pickle.load(f1)
            else:
                print("Preprocessing dataset...")
                self.preprocessed_data=self._preprocess_dataset()
                print(f"Saving preprocessed dataset to {self.save_path}..")
                with open(self.save_path, "wb") as f1:
                    pickle.dump(self.preprocessed_data, f1)

    def __len__(self):
        return len(self.preprocessed_data)-1

    def __getitem__(self, idx):
        return self.preprocessed_data[idx]

    def _preprocess_dataset(self):
        preprocessed = []
        for idx in range(len(self.dataset) - 1):# tqdm(range(len(self.dataset) - 1), desc="Processing dataset"):
            # Anchor: Original audio sample
            audio_data_anchor = self.dataset[idx]['audio']
            audio_attrb = self.dataset[idx]['attrs']
            anchor_waveform = np.array(audio_data_anchor['array'])
            sr = audio_data_anchor['sampling_rate']

            start = int(np.floor(audio_attrb['end']))
            anchor_waveform = anchor_waveform[start:start + sr]
            # Positive: Augmented version of the anchor
            #print ("aug")
            positive_waveform = self._aug(anchor_waveform, samp_rate=sr)

            # Negative: Next audio sample
            audio_data_negative = self.dataset[idx + 1]['audio']
            audio_nxtattrb = self.dataset[idx + 1]['attrs']
            negative_waveform = np.array(audio_data_negative['array'])
            start = int(np.floor(audio_nxtattrb['end']))
            negative_waveform = negative_waveform[start:start + sr]

            # Convert to tensors
            anchor = torch.tensor(anchor_waveform, dtype=torch.float32)
            positive = torch.tensor(positive_waveform, dtype=torch.float32)
            negative = torch.tensor(negative_waveform, dtype=torch.float32)
            #print("mfcc")
            # Compute MFCC features
            anchor = self._get_mfcc(anchor, sample_rate=sr)
            positive = self._get_mfcc(positive, sample_rate=sr)
            negative = self._get_mfcc(negative, sample_rate=sr)

            preprocessed.append((anchor, positive, negative))
        return preprocessed


    def _get_mfcc(self, waveform, sample_rate, n_mfcc=10):
        # Load the audio file

        win_length_ms = 40
        hop_length_ms = 20

        win_length = int(sample_rate * (win_length_ms / 1000))  # 40ms window
        hop_length = int(sample_rate * (hop_length_ms / 1000))  # 20ms stride
        # Manually extract frames with 20ms stride
        num_frames = (len(waveform) - win_length) // hop_length + 1
        frames = torch.stack([waveform[ i * hop_length: i * hop_length + win_length] for i in range(num_frames)],
                             dim=0)

        # Apply Hamming window to each frame
        hamming_window = np.hamming(win_length)
        hamming_window = torch.from_numpy(hamming_window).float()
        frames = frames * hamming_window
        windowed_waveform = frames.reshape((1, num_frames * win_length)).float()
        # Compute the MFCCs
        mfcc_transform = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={"n_fft": win_length, "hop_length": hop_length * 2}
        )
        TF_map = mfcc_transform(windowed_waveform)  # 1,10,50 TF map
        # print(TF_map.shape)
        return TF_map

    def _aug(self, waveform, samp_rate):
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
        # # Apply time masking

        # waveform = torchaudio.transforms.TimeMasking(time_masking_param)(waveform)
        return waveform.detach().numpy()

class TripletPickleDataset(Dataset):

    def __init__(self, folder_path):
        """
        Initialize the dataset by loading all .pkl files from the specified folder.

        Args:
            folder_path (str): Path to the folder containing .pkl files.
        """
        self._data = []  # To store all triplets (anchor, positive, negative)
        self._load_pickle_files(folder_path)

    def _load_pickle_files(self, folder_path):
        """
        Load all .pkl files from the given folder and append their contents to self.data.
        """
        pickle_files = [f for f in os.listdir(folder_path) if f.endswith(".pkl")]
        for file_name in pickle_files:
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "rb") as file:
                triplet_list = pickle.load(file)  # Load the list of triplets
                with open(file_path, "rb") as file:
                    triplet_list = pickle.load(file)  # Load the list of triplets

                    # Filter the triplet list by excluding mismatched tensors
                    filtered_triplets = [
                        triplet for triplet in triplet_list
                        if all(tensor.shape == (1, 10, 50) for tensor in triplet)
                    ]

                    # Append the entire filtered list to self._data
                    self._data.extend(filtered_triplets)

    def __len__(self):
        """
        Return the total number of triplets in the dataset.
        """
        return len(self._data)

    def __getitem__(self, idx):

        return self._data[idx]
def get_mfcc( waveform, sample_rate, n_mfcc=10):
    # Load the audio file

    win_length_ms = 40
    hop_length_ms = 20

    win_length = int(sample_rate * (win_length_ms / 1000))  # 40ms window
    hop_length = int(sample_rate * (hop_length_ms / 1000))  # 20ms stride
    # Manually extract frames with 20ms stride
    num_frames = (len(waveform) - win_length) // hop_length + 1
    frames = torch.stack([waveform[ i * hop_length: i * hop_length + win_length] for i in range(num_frames)],
                         dim=0)

    # Apply Hamming window to each frame
    hamming_window = np.hamming(win_length)
    hamming_window = torch.from_numpy(hamming_window).float()
    frames = frames * hamming_window
    windowed_waveform = frames.reshape((1, num_frames * win_length)).float()
    # Compute the MFCCs
    mfcc_transform = T.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={"n_fft": win_length, "hop_length": hop_length * 2}
    )
    TF_map = mfcc_transform(windowed_waveform)  # 1,10,50 TF map
    # print(TF_map.shape)
    return TF_map


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
    # # Apply time masking

    # waveform = torchaudio.transforms.TimeMasking(time_masking_param)(waveform)
    return waveform.detach().numpy()


def play_audio(audio_data, sample_rate):
    # Use soundfile to play the audio
    sf.write('temp_audio.wav', audio_data, sample_rate)  # Save temporarily
    # Now play using an external player or load it back for processing
    print(f"Playing audio with sample rate {sample_rate}...")

def load_all_pickles_from_folder(folder_path):
    # Initialize an empty list to store all data
    combined_list = []

    # List all files in the directory
    for filename in os.listdir(folder_path):
        # Only consider .pkl files
        if filename.endswith('.pkl'):
            file_path = os.path.join(folder_path, filename)
            try:
                # Open and load each .pkl file
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                    # Assuming each .pkl contains a list, extend the combined list
                    if isinstance(data, list):
                        combined_list.extend(data)
                    else:
                        print(f"Warning: {filename} does not contain a list, skipping.")
            except Exception as e:
                print(f"Error loading {filename}: {e}")

    return combined_list

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


def process_and_dump_files(source_folder, dest_folder):
    """
    Iterate over .pkl files in the source folder, process each file,
    and dump the results into the destination folder.

    Args:
        source_folder (str): Path to the folder containing the source .pkl files.
        dest_folder (str): Path to the destination folder to save processed files.
        processing_function (callable): Function to process the loaded .pkl data.
    """
    # Ensure the destination folder exists
    os.makedirs(dest_folder, exist_ok=True)

    # Get a list of all .pkl files in the source folder
    pkl_files = [f for f in os.listdir(source_folder) if f.endswith('.pkl')]

    for i, file_name in enumerate(pkl_files, start=1):
        source_path = os.path.join(source_folder, file_name)
        dest_path = os.path.join(dest_folder, file_name)

        print(f"[{i}/{len(pkl_files)}] Processing file: {file_name}")

        try:
            # Load the .pkl file
            with open(source_path, 'rb') as f:
                data = pickle.load(f)

            # Process the data
            processed_data = preprocess_dataset(data)

            # Save the processed data to the destination folder
            with open(dest_path, 'wb') as f:
                pickle.dump(processed_data, f)

            print(f"Processed and saved: {file_name} -> {dest_path}")
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")

# Play one of the audio files
def create_triplet_aux_dataloader(folder_path,batch_size):
    subset=load_all_pickles_from_folder(folder_path)
    heb_data_set = TripletAudioDataset(subset,force_reprocess=True)
    loader = DataLoader(heb_data_set, batch_size=batch_size, shuffle=True,num_workers=0)
    return loader

def gen_triplet_aux_dataloader(folder_path,batch_size):
    heb_pkl_data_set = TripletPickleDataset(folder_path)
    loader = DataLoader(heb_pkl_data_set, batch_size=batch_size, shuffle=True,num_workers=0)
    return loader



def create_Hebrew_audio_dataset(subset):
    heb_data_set = HebAudioDataset(subset)
    loader = DataLoader(heb_data_set,batch_size=1,shuffle=False)
    return loader
# #
# enable_prints()
# source_fol = 'val_aux_pkl_files'
# dest_fol = 'val_aux_augmented'
#
# deb_load = gen_triplet_aux_dataloader(folder_path= 'val_aux_augmented' , batch_size=16 )
# for anc , pos,neg in deb_load:
#     break
#


