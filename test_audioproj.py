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
    , predict_labels , calculate_metrics , gen_target_embedding
import warnings
import sys
import os

# Suppress all prints globally
def suppress_prints():
    sys.stdout = open(os.devnull, 'w')

# Re-enable prints
def enable_prints():
    sys.stdout = sys.__stdout__



model = DSCNN()
# Path to the saved model
model_path = "DL_proj_latest_model.pth"
# Load the state dict from the saved model
model.load_state_dict(torch.load(model_path))
labels = ['לכם', 'שלום']
reps=100
far_list, acc_list = [], []
warnings.filterwarnings("ignore", category=UserWarning)  # Suppress user warnings
filename = 'subset_list_002.pkl'
with open(filename, 'rb') as f:
    subset_data = pickle.load(f)
gamma_values = np.linspace(0.1, 5, 50)
avg_FAR = []
avg_ACC = []

# Compute the 'unknown' embedding from the negative loader
neg_loader = create_Hebrew_audio_dataset(subset_data)
# unknown_embedding = compute_embedding(model, neg_loader)
test_embeddings = []
for idx, batch in enumerate(neg_loader):
    embedding = compute_embedding(model=model, data_loader=[batch])
    for emb in embedding:
        test_embeddings.append({'true_label': 'unknown', 'embedding': emb})
    if idx==20:
        break

for gamma in gamma_values:
    suppress_prints()
    cum_FAR,cum_ACC = 0,0
    for i in range (reps):
        label_embeddings={}
        random_indices_per_label , label_embeddings = gen_target_embedding\
            (labels,label_embeddings=label_embeddings,model=model)


          # Replace with your test embeddings
        for label in labels:
            pos_loader = create_test_dataloaders('hebrew_test', label)
            for idx, batch in enumerate(pos_loader):
                if idx not in random_indices_per_label[label]:
                    embedding = compute_embedding(model=model, data_loader=[batch])
                    # Store the true label and embedding in the dictionary
                    # Store each embedding with its true label
                    for emb in embedding:
                        test_embeddings.append({'true_label': label, 'embedding': emb})


        predictions = []
        true_labels=[]
        # Predict for each test embedding
        for test_embedding in test_embeddings:
            prediction = predict_labels(test_embedding['embedding'], label_embeddings, gamma)
            predictions.append(prediction)
            true_labels.append(test_embedding['true_label'])

        FAR, ACC = calculate_metrics(predictions, true_labels)
        cum_FAR = cum_FAR+FAR
        cum_ACC = cum_ACC+ACC
        # far_list.append(FAR)
        # acc_list.append(ACC)
    avg_FAR.append(cum_FAR / reps)
    avg_ACC.append(cum_ACC / reps)
    enable_prints()
    print(str(gamma))
# Print or return metrics
# Plot the results
plt.figure(figsize=(12, 6))

# Plot FAR
plt.subplot(1, 2, 1)
plt.plot(gamma_values, avg_FAR, label='FAR', color='red')
plt.title('False Acceptance Rate (FAR) vs Gamma')
plt.xlabel('Gamma')
plt.ylabel('FAR')
plt.grid(True)

# Plot ACC
plt.subplot(1, 2, 2)
plt.plot(gamma_values, avg_ACC, label='ACC', color='blue')
plt.title('Accuracy (ACC) vs Gamma')
plt.xlabel('Gamma')
plt.ylabel('Accuracy')
plt.grid(True)

plt.tight_layout()
plt.show()

# print(f"False Acceptance Rate (FAR): {cum_FAR/reps}")
# print(f"Accuracy (ACC): {cum_ACC/reps}")
# Function to compute distances between two embeddings (Cosine distance
# # Plot histograms for FAR and ACC results
# plt.figure(figsize=(12, 6))
#
# # Plot FAR histogram
# plt.subplot(1, 2, 1)
# plt.hist(far_list, bins=20, color='blue', alpha=0.7)
# plt.title('Histogram of FAR')
# plt.xlabel('False Acceptance Rate (FAR)')
# plt.ylabel('Frequency')
#
# # Plot ACC histogram
# plt.subplot(1, 2, 2)
# plt.hist(acc_list, bins=20, color='green', alpha=0.7)
# plt.title('Histogram of Accuracy (ACC)')
# plt.xlabel('Accuracy (ACC)')
# plt.ylabel('Frequency')
#
# # Show the histograms
# plt.tight_layout()
# plt.show()
