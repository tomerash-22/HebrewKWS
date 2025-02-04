
import torch
from audio_models import DSCNN
from pre_prosses_dataloaders import create_test_dataloaders
from HEB_DB import create_Hebrew_audio_dataset
import pickle
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import random
from itertools import combinations, product
import matplotlib.pyplot as plt  # Importing for plotting histograms
import numpy as np
# Function to compute the embedding vector by averaging the outputs of the model
from testing_helper import  compute_embedding ,compute_distance \
    , predict_labels , calculate_metrics , gen_target_embedding,plot_TSNE,metrics_gamma1
import warnings
import sys
import os




filename = 'protonet_FAR_ACC_lists_fromaug0_01.pkl'
with open(filename, 'rb') as f:
    proto_lst=pickle.load(f)

ACC_means = []
ACC_std_devs = []
FAR_means = []
FAR_std_devs = []

labels = []

for dict_itam in proto_lst:
    for key,val in dict_itam.items():
       if key =='ACC':
           ACC_means.append(np.mean(val))
           ACC_std_devs.append(np.std(val))
       elif key=='FAR':
           FAR_means.append(np.mean(val))
           FAR_std_devs.append(np.std(val))



print("ACC AVG=" + str(np.mean(ACC_means)))
print("ACC AVG STD=" +str(np.mean(ACC_std_devs)))
print("FAR AVG" + str(np.mean(FAR_means)))
print("FAR AVG STD=" + str(np.mean(FAR_std_devs)))
