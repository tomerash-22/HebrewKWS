import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from audio_models import DSCNN
from pre_prosses_dataloaders import create_dataloaders
from HEB_DB import create_triplet_aux_dataloader,gen_triplet_aux_dataloader
import pickle
import os
# Triplet Loss Function
def triplet_loss(anchor, positive, negative, margin=1.0):
    pos_dist = torch.sum((anchor - positive) ** 2, dim=-1)
    neg_dist = torch.sum((anchor - negative) ** 2, dim=-1)
    loss = torch.clamp(pos_dist - neg_dist + margin, min=0.0)
    return torch.mean(loss)

def save_torch_model(model, path="DL_proj_latest_model.pth"):
    """Saves a PyTorch model."""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def dynamic_aux_dataloader(folder_train, batch_size):
    """Helper function to dynamically create an auxiliary dataloader."""
    return create_triplet_aux_dataloader(folder_train, batch_size=batch_size)
def training_step (batch,model,optimizer,device,margin):
    anchor, positive, negative = [x.to(device) for x in batch]
    optimizer.zero_grad()
    embeddings_anc = model(anchor)
    embeddings_pos = model(positive)
    embeddings_neg = model(negative)
    loss = triplet_loss(embeddings_anc, embeddings_pos, embeddings_neg, margin=margin)
    return loss


def train_model(
        model, train_loader, val_loader, aux_val_loader, aux_train_loader, aux_reg,
        save_path, training_type, margin=1.0, epochs=5, learning_rate=0.001,
        device="cuda" if torch.cuda.is_available() else "cpu"
):
    # Move model to device
    model = model.to(device)
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    prev_lr = learning_rate

    # Training Loop
    for epoch in range(epochs):
        save_torch_model(model, save_path)
        print(f"\nEpoch {epoch + 1}/{epochs}")

        model.train()
        epoch_loss = 0.0

        if training_type != 'AUX':
            # Learning rate adjustment
            if epoch == 10:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = prev_lr / 2

            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != prev_lr:
                print(f"Learning Rate Updated: {new_lr:.6f}")
                prev_lr = new_lr

            # Main training loop
            train_batches = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}", unit="batch")
            for batch in train_batches:
                loss = training_step(batch=batch, model=model, optimizer=optimizer, device=device, margin=margin)
                epoch_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        else:
            # Auxiliary Training Loop
            train_batches = tqdm(zip(train_loader, aux_train_loader), desc=f"Training Epoch {epoch + 1}", unit="batch")
            for main_batch, aux_batch in train_batches:
                loss_main = training_step(batch=main_batch, model=model, optimizer=optimizer, device=device,
                                          margin=margin)
                loss_aux = training_step(batch=aux_batch, model=model, optimizer=optimizer, device=device,
                                         margin=margin)
                loss = loss_main + aux_reg * loss_aux
                epoch_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epoch_loss = epoch_loss / len(train_loader)
        print(f"Train Loss: {epoch_loss:.4f}")

        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            if training_type != 'AUX':
                val_batches = tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}", unit="batch")
                for batch in val_batches:
                    loss = training_step(batch=batch, model=model, optimizer=optimizer, device=device, margin=margin)
                    val_loss += loss.item()
            else:
                val_batches = tqdm(zip(val_loader, aux_val_loader), desc=f"Validation Epoch {epoch + 1}", unit="batch")
                for main_batch, aux_batch in val_batches:
                    loss_main = training_step(batch=main_batch, model=model, optimizer=optimizer, device=device,
                                              margin=margin)
                    loss_aux = training_step(batch=aux_batch, model=model, optimizer=optimizer, device=device,
                                             margin=margin)
                    loss = loss_main + aux_reg * loss_aux
                    val_loss += loss.item()

        val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {val_loss:.4f}")


# Data preparation
data_dir = 'SpeechCommands/speech_commands_v0.02'  # Update this with the path to your SpeechCommands dataset
train_loader, val_loader = create_dataloaders(data_dir, validation_split=0.2, batch_size=32)

folder_train = 'train_aux_augmented'
folder_val = 'val_aux_augmented'



val_aux = gen_triplet_aux_dataloader(folder_val,batch_size=16)
train_aux = gen_triplet_aux_dataloader(folder_train,batch_size=16)
# Build the model
model = DSCNN()

# Path to the saved model
model_path = "DL_proj_latest_model.pth"

# Load the state dict from the saved model
model.load_state_dict(torch.load(model_path))

# Train the model
train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    aux_val_loader=val_aux,
    aux_train_loader=train_aux,
    aux_reg = 0.01,
    margin=1.0,         # Margin for triplet loss
    epochs=5,          # Number of training epochs
    learning_rate=0.001, # Learning rate for the optimizer
    save_path="DL_proj_AUX_001.pth",
    training_type = 'AUX'
)
