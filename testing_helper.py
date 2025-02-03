
from pre_prosses_dataloaders import create_test_dataloaders
from HEB_DB import create_Hebrew_audio_dataset
from general_utils import  suppress_prints ,enable_prints
import torch
import random
from collections import defaultdict
from audio_models import DSCNN
import numpy as np
import warnings


def compute_embedding(model, data_loader):
    """
    Compute embedding vector by averaging over `num_batches`.

    Args:
    - model: The model for generating embeddings.
    - data_loader: Data loader for fetching batches.
    - num_batches: Number of batches to average over for the embedding.
    - return_mean: Whether to return the mean embedding.

    Returns:
    - A tensor containing the mean embedding or a list of embeddings if return_mean is False.
    """
    model.eval()  # Set model to evaluation mode
    embeddings = []
    with torch.no_grad():  # Disable gradient computation for inference
        for idx, batch in enumerate(data_loader):
            output = model(batch)  # Pass batch through the model
            embeddings.append(output)
            return embeddings  # Return all embeddings without averaging


# Function to compute distances
def compute_distance(embedding1, embedding2):
    """
    Compute the L2 (Euclidean) distance between two embedding vectors.

    Args:
    - embedding1: First embedding tensor.
    - embedding2: Second embedding tensor.

    Returns:
    - L2 distance (Euclidean distance) between the two embeddings.
    """
    return torch.norm(embedding1 - embedding2, p=2)  # L2 distance



# Function to predict labels
def predict_labels(test_embedding, label_embeddings, gamma):
    """
    Predict the label for a given test embedding based on the distance to known label embeddings.
    If the minimum distance is greater than gamma, predict 'unknown'.

    Args:
    - test_embedding: The embedding to classify.
    - label_embeddings: Dictionary of known label embeddings (key: label, value: embedding).
    - gamma: Distance threshold for unknown prediction.

    Returns:
    - predicted_label: Predicted label or 'unknown'.
    """
    distances = {}

    # Calculate distances to each label embedding
    for label, embedding in label_embeddings.items():
        test_embedding = test_embedding.view(-1)
        distance = compute_distance(test_embedding, embedding)
        distances[label] = distance.item()

    # Find the label with the smallest distance
    predicted_label = min(distances, key=distances.get)
    min_distance = distances[predicted_label]

    # If the minimum distance is greater than gamma, predict 'unknown'
    if min_distance > gamma:
        return 'unknown'
    return predicted_label

# Function to calculate FAR and Accuracy
def calculate_metrics(predictions, true_labels):
    """
    Calculate FAR and Accuracy.

    Args:
    - predictions: List of predicted labels.
    - true_labels: List of true labels.

    Returns:
    - FAR: False Acceptance Rate.
    - Accuracy: Classification accuracy.
    """
    total_samples = len(true_labels)
    true_unknown = sum(1 for label in true_labels if label == 'unknown')
    false_accepts = 0

    for pred, true in zip(predictions, true_labels):
        if true == 'unknown' and pred != 'unknown':  # Misclassify "unknown" as known
            false_accepts += 1
        elif true != 'unknown' and ( pred != true and pred != 'unknown'):  # Misclassify one known label as another
            false_accepts += 1

    FAR = false_accepts / total_samples if total_samples > 0 else 0.0

    # Accuracy: Correct predictions / Total samples
    correct_predictions = sum(1 for pred, true in zip(predictions, true_labels) if pred == true)
    ACC = correct_predictions / total_samples if total_samples > 0 else 0.0

    return FAR, ACC


def gen_target_embedding (labels,label_embeddings,model,data_dir='hebrew_test',random_indices_per_label=None):
    """
        generate the target vector embedings

        Args:
        -labels: list of labels['str']
        -label_embedings empty dict
        -model: Feature extractor to use
        -data_dir: contains the recordings
        -random_indices_per_label:
        the idx of the indicies choosen for the target embeding, or none to randomly generate
        Returns:
            -label_embedings: dictonary of [key:'str' value:tensoe(1,64) ]
            -random_indices_per_label:
        """
    if random_indices_per_label == None:
        rand_gen = True
        random_indices_per_label = {}
    else:
        rand_gen = False
    for label in labels:
        pos_loader = create_test_dataloaders(data_dir=data_dir,label= label)
        # Get the number of batches in the loader
        if rand_gen:
            num_batches = len(pos_loader)
            # Select 5 random indices from the range of available batches
            random_indices = random.sample(range(num_batches), k=min(5, num_batches))  # Ensure it doesn't exceed num_batches
            random_indices_per_label[label] = random_indices  # Store the indices for the label
        else:
            random_indices = random_indices_per_label[label]
        embeddings = []
        try:
            for idx, batch in enumerate(pos_loader):
                if idx in random_indices:
                    embeddings = compute_embedding(model=model, data_loader=[batch])
        except IndexError:  # Catch IndexError
            continue

        embeddings = torch.cat(embeddings, dim=0)
        embeddings =torch.mean(embeddings, dim=0)
        label_embeddings[label] = embeddings
    return random_indices_per_label , label_embeddings

def gen_label_load(random_indices_per_label,data_dir , label,samps=5):
    """
            generate the indices to create target embeding
            and the relavant dataloader

            Args:
            -labels: list of labels['str']
            -samps: how meny samples to use
            -data_dir: contains the recordings
            -random_indices_per_label:
            the idx of the indicies choosen for the target embeding, or none to randomly generate
            Returns:
                -label_embedings: dictonary of [key:'str' value:tensoe(1,64) ]
                -random_indices_per_label:
    """

    pos_loader = create_test_dataloaders(data_dir=data_dir, label=label)
    # Get the number of batches in the loader
    num_batches = len(pos_loader)
    # Select 5 random indices from the range of available batches
    random_indices = random.sample(range(num_batches),
                                   k=min(samps, num_batches))
    random_indices_per_label[label] = random_indices
    return random_indices_per_label, pos_loader


def gen_unknowen_frompkl(random_indices_per_label,hebrew_flag,subset_data,models,avoid_idx):

    """
       test embedding list appended with the 50 unknowen samples

       Args:
        -hebrew_flag: True if the KWS is in hebrew , relavt for dataloader
        -subset_data: .pkl file containg the ivrit.ai inforamtion
        -models:List of DSCNN() models for multi-model analysis
        -avoid_idx: do you want to use given indicis or avoid using.

       Returns:
           test embedding:List({'true_label': 'unknown', 'embedding': emb, 'model': i}) ,
           with the unknowen embedings appended
    """

    if hebrew_flag:
        neg_loader = create_Hebrew_audio_dataset(subset_data)
        if avoid_idx:
            random_indices = random.sample(range(len(neg_loader) - 150), k=min(50, len(neg_loader) - 150))
        else:
            random_indices = random.sample(range(len(neg_loader) - 150, len(neg_loader)), k=min(50, 150))
    else:
        neg_loader = gen_label_load(random_indices_per_label=random_indices_per_label, \
                                    data_dir='english_unk', label='unknowen', samps=50)
        random_indices = random.sample(range(len(neg_loader[1])), k=min(50, len(neg_loader[1])))

        # Randomly select "unknown" embeddings
    if hebrew_flag:
        iter_loader = neg_loader
    else:
        iter_loader = neg_loader[1]
    test_embeddings = []
    for idx, batch in enumerate(iter_loader):
        if idx in random_indices:

            for i, model in enumerate(models):
                embedding = compute_embedding(model=model, data_loader=[batch])
                for emb in embedding:
                    test_embeddings.append({'true_label': 'unknown', 'embedding': emb, 'model': i})
    return test_embeddings

def gen_test_buffer(labels,data_dir,models,subset_data,random_indices_per_label,hebrew_flag=True,avoid_idx=True):
    """
          generate buffer for testing the model

          Args:
              -labels: list of 'str'
           -hebrew_flag: True if the KWS is in hebrew , relavt for dataloader
           -data_dir : dircotry where the recordings are
           -subset_data: .pkl file containg the ivrit.ai inforamtion
           -models:List of DSCNN() models for multi-model analysis
           random_indices_per_label- the labels that where used to create the target embeding
           -avoid_idx: do you want to use given indicis or avoid using.

          Returns:
              test embedding:List({'true_label': 'unknown', 'embedding': emb, 'model': i}) ,
              with the unknowen embedings appended
    """

# Step 2: Compute "unknown" embeddings from the negative loader (random sample)
    test_embeddings=gen_unknowen_frompkl(random_indices_per_label=random_indices_per_label ,hebrew_flag=hebrew_flag,
                                         subset_data=subset_data,models=models,avoid_idx=avoid_idx)
        # Step 3: Generate test embeddings for each label
    for label in labels:
        pos_loader = create_test_dataloaders(data_dir=data_dir, label=label)
        for idx, batch in enumerate(pos_loader):
            if avoid_idx:
                con = (idx not in random_indices_per_label[label])
            else:
                con = (idx in random_indices_per_label[label])

            if con:
                for i, model in enumerate(models):
                    embedding = compute_embedding(model=model, data_loader=[batch])
                    for emb in embedding:
                        test_embeddings.append({'true_label': label, 'embedding': emb, 'model': i})

    return test_embeddings

def metrics_gamma1(labels, subset_data, model_list,data_dir, num_runs=5,hebrew_flag=True,
                   gamma=1,customize=False,FAR_th=0.05):

    """
              implements the classifer for both custimzed gamma and const

              Args:
                -labels: list of 'str'
               -hebrew_flag: True if the KWS is in hebrew , relavt for dataloader
               -data_dir : dircotry where the recordings are
               -subset_data: .pkl file containg the ivrit.ai inforamtion
               -models_list :List of DSCNN() models for multi-model analysis
               -num_runs : each run the 5 samples of whom the target
               embeding is created is randomly sampled, as well as the 'unknowen buffer'
               customize: if True classifier "Zero shot optimize gamma" is apllied if not gamma=1
               ,FAR_th= for the "Zero shot optimize gamma" classifier
               random_indices_per_label- the labels that where used to create the target embeding

              Returns:
                  FAR,ACC dicts per model
        """

    avg_FAR_dict = defaultdict(list)
    avg_ACC_dict = defaultdict(list)
    models = []
    for mod in model_list:
        model = DSCNN()
        model.load_state_dict(torch.load(mod))
        model.eval()  # Set to evaluation mode
        models.append(model)
    FAR_dict = defaultdict(list)
    ACC_dict = defaultdict(list)

    for run in range(num_runs):
        enable_prints()
        print("run " + str(run) + "out of" + str(num_runs))
        suppress_prints()
        FAR_list, ACC_list = [], []
        random_indices_per_label = {}

        for label in labels:
            random_indices_per_label,pos_loader = gen_label_load(random_indices_per_label=random_indices_per_label,\
                                                                  data_dir=data_dir,label=label)
        # Step 1: Generate label embeddings for target labels

        label_embeddings_per_model = {}
        gamma_per_model ={}
        for i, mod in enumerate(model_list):
            model = DSCNN()
            model.load_state_dict(torch.load(mod))
            label_embeddings = {}
            _, label_embeddings = gen_target_embedding(
                labels, label_embeddings=label_embeddings, model=model,
                random_indices_per_label=random_indices_per_label,data_dir=data_dir
            )
            label_embeddings_per_model[i] = label_embeddings  # Store per model
            if customize:
                models_temp =[]
                models_temp.append(model)
                customize_buffer = gen_test_buffer(labels=labels , data_dir=data_dir ,models=models_temp,
                                                  subset_data=subset_data,random_indices_per_label=random_indices_per_label,
                                                  hebrew_flag=hebrew_flag,avoid_idx=False)
                gamma_per_model[i],_=achive_FAR_maxACC(train_buffer=customize_buffer,
                                                       tar_emb=label_embeddings_per_model[i],FARth=FAR_th)
                models_temp.pop()




        test_embeddings = gen_test_buffer(labels=labels,data_dir=data_dir,
           models=models,subset_data=subset_data,random_indices_per_label=random_indices_per_label,
                                          hebrew_flag=hebrew_flag,avoid_idx=True)

        for i in range(len(model_list)):  # Process separately for each model
            cum_FAR, cum_ACC = 0, 0
            predictions = []
            true_labels = []

            # Predict for each test embedding corresponding to this model
            for test_embedding in test_embeddings:
                if test_embedding['model'] == i:
                    if customize:
                       if gamma_per_model[i] != 0:
                            gamma=gamma_per_model[i]

                    prediction = predict_labels(test_embedding['embedding'], label_embeddings_per_model[i],
                                                gamma=gamma)
                    predictions.append(prediction)
                    true_labels.append(test_embedding['true_label'])

            # Calculate FAR and ACC for this model
            FAR, ACC = calculate_metrics(predictions, true_labels)

            FAR_dict[i].append(FAR)
            ACC_dict[i].append(ACC)

        print()
    return FAR_dict, ACC_dict

def test_FAR_ACC (test_buffer,gamma,label_emb):


    predictions=[]
    true_labels=[]
    for test_embedding in test_buffer:
        prediction = predict_labels(test_embedding['embedding'], label_emb,
                                    gamma=gamma)
        predictions.append(prediction)
        true_labels.append(test_embedding['true_label'])

        # Calculate FAR and ACC for this model
    FAR, ACC = calculate_metrics(predictions, true_labels)
    return FAR,ACC

def achive_FAR_maxACC (train_buffer,tar_emb,gamma_vals=np.linspace(0.5,3,200),FARth=0.05):
    gamma_rtn=0
    FAR_ACC_lst=[]
    for gamma in gamma_vals:
        predictions = []
        true_labels = []

        # Predict for each test embedding corresponding to this model
        for test_embedding in train_buffer:
            prediction = predict_labels(test_embedding['embedding'], tar_emb,gamma)
            predictions.append(prediction)
            true_labels.append(test_embedding['true_label'])

        # Calculate FAR and ACC for this model
        FAR, ACC = calculate_metrics(predictions, true_labels)
        if FAR <= FARth:
            FAR_ACC_lst.append((FAR,ACC,gamma))
    if FAR_ACC_lst:
        best_FAR, best_ACC,gamma_rtn = max(FAR_ACC_lst, key=lambda x: x[1])

    return gamma_rtn,best_ACC




