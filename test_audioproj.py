
import pickle
import matplotlib.pyplot as plt  # Importing for plotting histograms
import numpy as np
# Function to compute the embedding vector by averaging the outputs of the model
from testing_helper import  compute_embedding ,compute_distance \
    , predict_labels , calculate_metrics , gen_target_embedding,metrics_gamma1
import warnings
import sys
import os
from general_utils import  suppress_prints,enable_prints


# Full Model path list
# model_path_list=['DL_proj_AUX_reg_0.000.pth','DL_proj_AUX_reg_0.001.pth',
#                  "DL_proj_AUX_reg_0.010.pth",'DL_proj_AUX_reg_0.05.pth',
#                  'DL_proj_AUX_reg_0.10.pth','DL_proj_AUX_reg_0.50.pth','DL_proj_AUX_reg_1.00.pth'
#                  ,'DL_proj_AUX_reg_5.00.pth','DL_proj_AUX_reg_10.00.pth']

model_path_list=['DL_proj_AUX_reg_0.000.pth',
                 "DL_proj_AUX_reg_0.010.pth"]

labels = ['לכם','קדימה' ,'אחורה','שלום']
reps=1
far_list, acc_list = [], []

filename = 'subset_list_002.pkl'  # the .pkl file from ivrit.ai unused in training
with open(filename, 'rb') as f:
    subset_data = pickle.load(f)

# filename = 'protonet_FAR_ACC_lists_frombaseline.pkl'
# with open(filename, 'rb') as f:
#     FAR_ACC_doc= pickle.load(f)

# ROC cal
FAR_th_list = np.linspace(0 , 15, 150)
FAR_ACC_ROC_list =[]
for th in FAR_th_list:
    FAR_dict, ACC_dict = metrics_gamma1(labels=labels , subset_data=subset_data
    ,model_list=model_path_list,data_dir='hebrew_test' , num_runs=10,hebrew_flag=True,customize=True)
    FAR_ACC_ROC_list.append((FAR_dict,ACC_dict))
    print (str(th))

filename = 'classifers_ROC'
with open(filename, 'wb') as f:
    pickle.dump(FAR_ACC_ROC_list,f)


# cal mean and std from resukts

def cal_mean_std (dict):
    means = []
    std_devs = []
    labels = []

    for key, values in dict.items():
        means.append(np.mean(values))
        std_devs.append(np.std(values))
        labels.append(key)
    return means,std_devs,labels
