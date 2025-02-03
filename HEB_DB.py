

import numpy as np
import torch
import random
from torch.utils.data import DataLoader,Dataset
import  torchaudio
import pickle
import os
import sys
from pre_aug import get_mfcc , aug , preprocess_dataset
from general_utils import enable_prints , play_audio , suppress_prints
"""
       this file contains the nececery defs and classes to create an auxliry dataloader 
       using ivrit.ai database.         
"""
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
              return get_mfcc(waveform, sample_rate, n_mfcc)


"Genrate a dataset and loader from a folder full of .pkl files using this class"
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
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

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




""" 
Use this function to create a folder of triplets,
where the anchor is the original audio file
positive is an slightly augmented version of it
negetive is a diffrent audio file

Iterate over .pkl files in the source folder, process each file,
and dump the results into the destination folder.

    Args:
        source_folder (str): Path to the folder containing the source .pkl files.
        dest_folder (str): Path to the destination folder to save processed files.
"""
def process_and_dump_files(source_folder, dest_folder):

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



"""
    featch a "triplet" like dataset from existing folder created by process_and_dump_files

    Args:
        folder path : destination folder for prosses and dump , to featch from
        batch_size
    Returns:
        tourch dataloader
"""

def featch_triplet_aux_dataloader(folder_path,batch_size):
    heb_pkl_data_set = TripletPickleDataset(folder_path)
    loader = DataLoader(heb_pkl_data_set, batch_size=batch_size, shuffle=True, num_workers=0)
    return loader

def create_Hebrew_audio_dataset(subset):
    heb_data_set = HebAudioDataset(subset)
    loader = DataLoader(heb_data_set,batch_size=1,shuffle=False)
    return loader

 #
 # enable_prints()
# source_fol = 'val_aux_pkl_files'
# dest_fol = 'val_aux_augmented'
#  #
# deb_load = featch_triplet_aux_dataloader(folder_path= 'val_aux_augmented' , batch_size=16 )
# for anc , pos,neg in deb_load:
#   break



