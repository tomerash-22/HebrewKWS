
from pre_prosses_dataloaders import create_test_dataloaders
import torch
import random
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

def gen_target_embedding (labels,label_embeddings,model):


    random_indices_per_label = {}
    for label in labels:
        pos_loader = create_test_dataloaders('hebrew_test', label)
        # Get the number of batches in the loader
        num_batches = len(pos_loader)
        # Select 5 random indices from the range of available batches
        random_indices = random.sample(range(num_batches), k=min(5, num_batches))  # Ensure it doesn't exceed num_batches
        random_indices_per_label[label] = random_indices  # Store the indices for the label
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

