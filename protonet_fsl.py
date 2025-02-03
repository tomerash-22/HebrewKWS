
from audio_models import DSCNN
import pickle
import torch
import random
import numpy as np

from testing_helper import  compute_embedding ,compute_distance \
    , predict_labels , calculate_metrics , gen_target_embedding,test_FAR_ACC
import warnings
import sys
import os
from testing_helper import gen_label_load,gen_test_buffer,gen_unknowen_frompkl,achive_FAR_maxACC
from collections import defaultdict
import torch.optim as optim
from general_utils import  suppress_prints,enable_prints,save_torch_model

# generete the inital training data dict
def training_set (labels):
    random_indices_per_label = {}
    train_loader_dict = {}

    for lab in labels:
        random_indices_per_label,train_loader = gen_label_load(random_indices_per_label=random_indices_per_label,data_dir='hebrew_test',label=lab)
        train_loader_dict[lab] = train_loader
    return random_indices_per_label,train_loader_dict

# one training step like in the psuado-code
def training_step (random_indices_per_label,train_loader_dict,labels,model):

    Ck = defaultdict(list)
    Qk = defaultdict(list)

    V = random.sample(labels, k=3)

    for lab in V: #select class indices for episode
        Sk= random.choice(random_indices_per_label[lab])
        iter_loader = train_loader_dict[lab]

        # gen supprot and query dict
        for idx,batch in enumerate(iter_loader):
            if idx in random_indices_per_label[lab]:
                if idx == Sk:
                    Ck[lab] = model(batch)
                else:
                    Qk[lab].append(model(batch))
    loss=0
    for lab in V:
        # Compute distances and loss  # Stack queries
        for Q_emb in Qk[lab]:
            exp_sum = 0
            for lab_int in V:
                exp_sum = exp_sum + torch.exp(-1* torch.sum((Q_emb-Ck[lab_int])**2 ) )
            log_like = torch.log(exp_sum)
            int_loss = torch.sum((Q_emb-Ck[lab])**2) + log_like
            loss = loss+0.25*int_loss

    return loss

# training loop
def train_protonet(model,random_indices_per_label,train_loader_dict
                   ,epochs,save_path,episode,learning_rate=0.0005,
                    device="cuda" if torch.cuda.is_available() else "cpu"):
    model = model.to(device)
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=3, verbose=True)
    lr= learning_rate
    acc=0
    gamma_rtn=100
    for epoch in range(epochs):

        print(f"\nEpoch {epoch + 1}/{epochs}")
        model.train()
        epoch_loss = 0.0
        current_lr = optimizer.param_groups[0]['lr']
        if  current_lr != lr:
            enable_prints()
            lr=current_lr
            print("lr is " + str(lr))
            suppress_prints()

        for _ in range(episode):
            loss=training_step(random_indices_per_label=random_indices_per_label,train_loader_dict=train_loader_dict,
                               labels=labels,model=model)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        enable_prints()
        print("Epoch " + str(epoch) + "loss " +str(epoch_loss))
        suppress_prints()
        save_torch_model(model, save_path)
        # validation

        model.eval()  # Set to evaluation mode
        models.append(model)
        label_embeddings = {}
        gamma_vals = torch.linspace(0.1, 3, 100)
        _, tar_emb = gen_target_embedding(labels=labels, label_embeddings=label_embeddings,
                                          model=model, data_dir='hebrew_test',
                                          random_indices_per_label=random_indices_per_label)
        # unknowen_emb = gen_unknowen_frompkl(random_indices_per_label=random_indices_per_label,
        #                                     hebrew_flag=True, subset_data=subset_data, models=models)
        train_buffer = gen_test_buffer(labels=labels, data_dir='hebrew_test', models=models, subset_data=subset_data,
                                       random_indices_per_label=random_indices_per_label, hebrew_flag=True,
                                       avoid_idx=False)
        gamma, ACC_cur = achive_FAR_maxACC(train_buffer, tar_emb, gamma_vals)
        enable_prints()
        print("ACC achived" + str(ACC_cur))
        suppress_prints()
        if acc <= ACC_cur:
            if ACC_cur>acc:
                acc=ACC_cur
                save_torch_model(model, "DL_proto_best_ACC.pth")
                gamma_rtn=gamma

            elif gamma<gamma_rtn:
                save_torch_model(model, "DL_proto_best_ACC.pth")
                gamma_rtn = gamma

        else:
            scheduler.step(ACC_cur)
        models.pop()
    return gamma_rtn


#main

model = DSCNN()
# Path to the saved model
model_path = "DL_proj_latest_model.pth"
# Load the state dict from the saved model
model.load_state_dict(torch.load(model_path))
train_loader = defaultdict(list)
train_idx = defaultdict(list)
filename = 'subset_list_002.pkl'
with open(filename, 'rb') as f:
    subset_data = pickle.load(f)

labels = ['לכם','קדימה' , 'אחורה', 'שלום']
reps=100
far_list, acc_list = [], []
warnings.filterwarnings("ignore", category=UserWarning)  # Suppress user warnings
save_path = "DL_proto.pth"
learning_rate=0.0005
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
suppress_prints()
reps=10
model_list = [save_path]
models = []

test_reps=10
FAR_ACC_doc =[]
fails_cnt=0

#models loop (to check depency on the 5 samples selected

for it in range(reps):
    model = DSCNN()
    # Path to the saved model
    model_path = "DL_proj_AUX_reg_0.000.pth"
    # Load the state dict from the saved model
    model.load_state_dict(torch.load(model_path))

    random_indices_per_label,train_loader_dict = training_set(labels=labels)
    # Scheduler: Reduce LR when validation loss plateaus

    gamma = train_protonet(model=model , random_indices_per_label=random_indices_per_label,
                   train_loader_dict=train_loader_dict , epochs=20 ,episode=20,save_path=save_path)

    if gamma == 0:
        fails_cnt=fails_cnt+1
        continue
    model = DSCNN()
    model_path = "DL_proto_best_ACC.pth"
    model.load_state_dict(torch.load(model_path))
    models.append(model)
    FAR_cum=0
    ACC_cum =0
    FAR_lst=[]
    ACC_lst=[]
    label_embeddings = {}
    for i in range(test_reps):
        test_buf = gen_test_buffer(labels=labels,data_dir='hebrew_test',models=models,subset_data=subset_data,
                        random_indices_per_label=random_indices_per_label,hebrew_flag=True,avoid_idx=True)
        _, tar_emb = gen_target_embedding(labels=labels, label_embeddings=label_embeddings,
                                          model=model, data_dir='hebrew_test',
                                          random_indices_per_label=random_indices_per_label)
        FAR,ACC = test_FAR_ACC(test_buffer=test_buf ,gamma=gamma, label_emb=tar_emb)
        FAR_lst.append(FAR)
        ACC_lst.append(ACC)

    FAR_ACC_doc.append({'FAR':FAR_lst , 'ACC':ACC_lst})
    enable_prints()
    print ("FAR mean=" + str(np.mean(FAR_lst)) + "ACC mean = " +  str(np.mean(ACC_lst)) )
    print("iter="+str(it))
    suppress_prints()
    models.pop()
filename = 'protonet_FAR_ACC_lists_frombaseline.pkl'
with open(filename, 'wb') as f:
    pickle.dump(FAR_ACC_doc,f)