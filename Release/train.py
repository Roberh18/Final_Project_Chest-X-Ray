import os
import logging
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
import time

from config import (
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    SCHEDULER_FACTOR,
    SCHEDULER_PATIENCE,
    EARLY_STOPPING_PATIENCE,
    LOG_PATH,
    MODEL_SAVE_PATH,
    DEVICE
)
from evaluation import evaluate_model
from models import build_model
from data import get_data_loaders, load_image_labels
import numpy as np

# Configure logging for the train module
os.makedirs(LOG_PATH, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_PATH, 'train.log'),
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None, device=DEVICE, num_epochs=NUM_EPOCHS):
    '''
    Trains the given model using the provided data loaders, criterion, and optimizer.
    Validates the model on the validation set after each epoch and saves the best model based on validation F1-score.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler.
        device (torch.device): Device to perform computations on.
        num_epochs (int): Number of epochs to train.

    Returns:
        tuple: Best validation F1-score, best epoch number, training metrics dictionary, validation metrics dictionary.
    '''
    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = 0.0
    best_epoch = 0

    # Early Stopping Variables
    epochs_no_improve = 0
    early_stop = False

    # Metrics Storage
    train_metrics = {
        'loss': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'jaccard_accuracy': []
    }
    val_metrics = {
        'loss': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'jaccard_accuracy': []
    }

    logging.info("Starting training process.")
    logging.info(f"Number of Epochs: {num_epochs}")
    logging.info(f"Learning Rate: {LEARNING_RATE}")
    logging.info(f"Weight Decay: {WEIGHT_DECAY}")
    if scheduler:
        logging.info(f"Scheduler: {scheduler}")

    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.time()  # Start time
        logging.info(f"\nEpoch {epoch}/{num_epochs}")
        logging.info("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = train_loader
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = val_loader

            running_loss = 0.0
            running_precision = 0.0
            running_recall = 0.0
            running_f1 = 0.0
            running_jaccard = 0.0
            total_batches = 0

            # Iterate over data.
            for inputs, labels in dataloader:
                if inputs is None or labels is None:
                    logging.warning("Encountered None in inputs or labels. Skipping this batch.")
                    continue

                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = torch.sigmoid(outputs) > 0.5

                    # Backward pass and optimization only in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item()
                total_batches += 1


            # After all batches in the phase
            epoch_loss, epoch_precision, epoch_recall, epoch_f1, epoch_jaccard = compute_epoch_metrics(model, dataloader, criterion, device)

            # Log metrics
            logging.info(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Precision: {epoch_precision:.4f} Recall: {epoch_recall:.4f} F1: {epoch_f1:.4f} Jaccard Accuracy: {epoch_jaccard:.4f}")

            # Store metrics
            if phase == 'train':
                train_metrics['loss'].append(epoch_loss)
                train_metrics['precision'].append(epoch_precision)
                train_metrics['recall'].append(epoch_recall)
                train_metrics['f1'].append(epoch_f1)
                train_metrics['jaccard_accuracy'].append(epoch_jaccard)
            else:
                val_metrics['loss'].append(epoch_loss)
                val_metrics['precision'].append(epoch_precision)
                val_metrics['recall'].append(epoch_recall)
                val_metrics['f1'].append(epoch_f1)
                val_metrics['jaccard_accuracy'].append(epoch_jaccard)

                # Deep copy the model if it has better F1
                if epoch_f1 > best_f1:
                    best_f1 = epoch_f1
                    best_epoch = epoch
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0

                    # Save the best model
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    model_filename = f"best_model_{timestamp}.pth"
                    model_path = os.path.join(MODEL_SAVE_PATH, model_filename)
                    torch.save(best_model_wts, model_path)
                    logging.info(f"Best model updated and saved to {model_path}.")
                else:
                    epochs_no_improve += 1
                    logging.info(f"No improvement in F1-score for {epochs_no_improve} consecutive epoch(s).")

                epoch_end_time = time.time()  # End time
                epoch_duration = epoch_end_time - epoch_start_time
                logging.info(f"Epoch {epoch} completed in {epoch_duration:.2f} seconds.\n")

                # Step the scheduler
                if scheduler:
                    scheduler.step(epoch_loss)

                # Early stopping
                if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                    logging.info(f"Early stopping triggered after {epochs_no_improve} epochs with no improvement.")
                    early_stop = True
                    break

        if early_stop:
            break

    logging.info(f"Training complete. Best Validation F1-score: {best_f1:.4f} at epoch {best_epoch}.")

    # Load best model weights
    model.load_state_dict(best_model_wts)

    return best_f1, best_epoch, train_metrics, val_metrics

def compute_epoch_metrics(model, dataloader, criterion, device=DEVICE):
    '''
    Computes loss and other metrics for an entire epoch.

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader for the dataset.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to perform computations on.

    Returns:
        tuple: Average loss, precision, recall, F1-score, Jaccard accuracy for the epoch.
    '''
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            if inputs is None or labels is None:
                logging.warning("Encountered None in inputs or labels. Skipping this batch.")
                continue

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            preds = torch.sigmoid(outputs) > 0.5
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = running_loss / len(dataloader)
    all_labels = np.vstack(all_labels)
    all_preds = np.vstack(all_preds)

    # Compute metrics using sklearn
    from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score

    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    jaccard = jaccard_score(all_labels, all_preds, average="samples")

    return avg_loss, precision, recall, f1, jaccard

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    '''
    Saves the model state to a file.

    Args:
        state (dict): State dictionary containing model state, optimizer state, etc.
        filename (str): Filename to save the checkpoint.
    '''
    torch.save(state, filename)
    logging.info(f"Checkpoint saved to {filename}.")

def load_checkpoint(model, optimizer, filename='checkpoint.pth.tar'):
    '''
    Loads the model state from a checkpoint file.

    Args:
        model (nn.Module): The model to load the state into.
        optimizer (torch.optim.Optimizer): The optimizer to load the state into.
        filename (str): Filename of the checkpoint.

    Returns:
        int: The epoch to resume training from.
        float: The best F1-score achieved so far.
    '''
    if os.path.isfile(filename):
        logging.info(f"Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename, map_location=DEVICE)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        best_f1 = checkpoint['best_f1']
        logging.info(f"Loaded checkpoint '{filename}' (epoch {epoch}) with best F1-score: {best_f1:.4f}")
        return epoch, best_f1
    else:
        logging.error(f"No checkpoint found at '{filename}'")
        return 0, 0.0

if __name__ == "__main__":
    # Example usage (This part can be removed or commented out in production)
    import numpy as np
    from data import get_data_loaders
    from models import build_model
    from config import CLASS_NAMES, MODEL_TYPE, config
    from evaluation import evaluate_model

    # Initialize DataLoaders
    train_loader, val_loader, test_loader = get_data_loaders(CLASS_NAMES, config)

    # Build Model
    model = build_model(MODEL_TYPE, class_weights=torch.tensor(config.CLASS_WEIGHTS).to(DEVICE))

    # Define Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(config.CLASS_WEIGHTS).to(DEVICE))
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Define Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        verbose=True
    )

    # Train the Model
    best_f1, best_epoch, train_metrics, val_metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
        num_epochs=NUM_EPOCHS
    )

    logging.info(f"Best Validation F1-score: {best_f1:.4f} at epoch {best_epoch}.")

    # Save Final Model
    final_model_path = os.path.join(MODEL_SAVE_PATH, f"final_model_{MODEL_TYPE}.pth")
    torch.save(model.state_dict(), final_model_path)
    logging.info(f"Final model saved to {final_model_path}.")
