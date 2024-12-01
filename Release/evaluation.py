import os
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    hamming_loss,
    jaccard_score,
    multilabel_confusion_matrix,
    classification_report
)
from matplotlib_venn import venn2
from config import (
    CLASS_NAMES,
    MODEL_TYPE,
    NUM_EPOCHS,
    IMAGE_SIZE,
    BATCH_SIZE,
    DROPOUT,
    LEARNING_RATE,
    VERSION,
    DEVICE,
    TEST_IMAGES_DIR,
    LOG_PATH,
    MODEL_SAVE_PATH
)

# Configure logging for the evaluation module
logging.basicConfig(filename=os.path.join(LOG_PATH, 'evaluation.log'),filemode='a', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def evaluate_model(model, data_loader, criterion, device=DEVICE, return_preds=False, return_outputs=False):
    '''
    Evaluates the model on a given dataset and computes various performance metrics.

    Args:
        model (torch.nn.Module): The trained PyTorch model to evaluate.
        data_loader (torch.utils.data.DataLoader): DataLoader for the dataset to evaluate.
        criterion (torch.nn.Module): Loss function.
        device (torch.device): Device to perform computations on.
        return_preds (bool): Whether to return the predictions.
        return_outputs (bool): Whether to return the raw outputs.

    Returns:
        tuple: Contains average loss, accuracy, overall accuracy, precision, recall, F1-score,
               hamming loss, specificity, and optionally predictions and labels.
    '''
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_outputs = [] if return_outputs else None
    all_specificity = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            if inputs is None or labels is None:
                logging.warning("Encountered None in inputs or labels. Skipping this batch.")
                continue

            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            preds = torch.sigmoid(outputs) > 0.5
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            if return_outputs:
                all_outputs.append(outputs.cpu().numpy())

    avg_loss = running_loss / len(data_loader)
    all_labels = np.vstack(all_labels)
    all_preds = np.vstack(all_preds)
    if return_outputs:
        all_outputs = np.vstack(all_outputs)

    # Compute metrics
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    hamming = hamming_loss(all_labels, all_preds)
    jaccard = jaccard_score(all_labels, all_preds, average="samples")

    # Compute confusion matrix-based metrics for specificity and overall accuracy
    cm = multilabel_confusion_matrix(all_labels, all_preds)
    tp = cm[:, 1, 1]
    tn = cm[:, 0, 0]
    fp = cm[:, 0, 1]
    fn = cm[:, 1, 0]

    specificity = tn / (tn + fp + 1e-8)  # Adding epsilon to avoid division by zero
    overall_accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    avg_specificity = np.nanmean(specificity)
    avg_overall_accuracy = np.nanmean(overall_accuracy)

    logging.info(f"Evaluation Results - Loss: {avg_loss:.4f}, Jaccard Accuracy: {jaccard:.4f}, "
                 f"Overall Accuracy: {avg_overall_accuracy:.4f}, Precision: {precision:.4f}, "
                 f"Recall: {recall:.4f}, F1-score: {f1:.4f}, Hamming Loss: {hamming:.4f}, "
                 f"Specificity: {avg_specificity:.4f}")

    if return_preds and return_outputs:
        return (avg_loss, jaccard, avg_overall_accuracy, precision, recall,
                f1, hamming, avg_specificity, all_preds, all_labels, all_outputs)
    elif return_preds:
        return (avg_loss, jaccard, avg_overall_accuracy, precision, recall,
                f1, hamming, avg_specificity, all_preds, all_labels)
    else:
        return (avg_loss, jaccard, avg_overall_accuracy, precision, recall,
                f1, hamming, avg_specificity)

def identify_misclassified_samples(test_labels, test_preds, image_names):
    '''
    Identifies misclassified samples by comparing true labels with predictions.

    Args:
        test_labels (ndarray): Ground truth binary labels.
        test_preds (ndarray): Predicted binary labels.
        image_names (list): List of image filenames corresponding to the labels.

    Returns:
        list: List of image filenames that were misclassified.
    '''
    misclassified = []
    for idx in range(len(test_labels)):
        if not np.array_equal(test_labels[idx], test_preds[idx]):
            misclassified.append(image_names[idx])
    logging.info(f"Number of misclassified samples: {len(misclassified)}")
    return misclassified

def generate_classification_report_custom(y_true, y_pred, class_names):
    '''
    Generates and saves a classification report based on true and predicted labels.

    Args:
        y_true (ndarray): Ground truth binary labels.
        y_pred (ndarray): Predicted binary labels.
        class_names (list): List of class names.
    '''
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    print("\nClassification Report:\n")
    print(report)

    # Save the report to a text file
    now = datetime.now()
    date = now.strftime("%d-%m-%y")
    time_str = now.strftime("%H-%M-%S")
    report_filename = f'classification_report_{MODEL_TYPE}_{date}_{time_str}.txt'
    with open(os.path.join(LOG_PATH, report_filename), 'w') as f:
        f.write(report)
    logging.info(f"Classification report saved to {report_filename}")

def plot_confusion_matrix_multi_label(y_true, y_pred, class_names):
    '''
    Plots a combined confusion matrix for multi-label classification.

    Args:
        y_true (ndarray): Ground truth binary labels.
        y_pred (ndarray): Predicted binary labels.
        class_names (list): List of class names.
    '''
    now = datetime.now()
    date = now.strftime("%d-%m-%y")
    time_str = now.strftime("%H-%M-%S")

    num_classes = len(class_names)
    combined_cm = np.zeros((num_classes, num_classes), dtype=int)

    for true_labels, pred_labels in zip(y_true, y_pred):
        true_indices = np.where(true_labels == 1)[0]
        pred_indices = np.where(pred_labels == 1)[0]
        for true_idx in true_indices:
            for pred_idx in pred_indices:
                combined_cm[true_idx, pred_idx] += 1

    plt.figure(figsize=(10, 8))
    sns.heatmap(combined_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Combined Confusion Matrix - Model: {MODEL_TYPE}')
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.tight_layout()

    filename = f'confusion_matrix_combined_{MODEL_TYPE}_{date}_{time_str}.png'
    plt.savefig(os.path.join(LOG_PATH, filename))
    plt.close()
    logging.info(f"Combined confusion matrix plot saved to {filename}")

def plot_classwise_confusion_matrix(cm, class_names):
    '''
    Plots confusion matrices for each class individually.

    Args:
        cm (ndarray): Multilabel confusion matrices for each class.
        class_names (list): List of class names.
    '''
    now = datetime.now()
    date = now.strftime("%d-%m-%y")
    time_str = now.strftime("%H-%M-%S")

    for i, (conf_matrix, class_name) in enumerate(zip(cm, class_names)):
        plt.figure(figsize=(4, 4))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f"Confusion Matrix for {class_name}")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()

        filename = f'confusion_matrix_{class_name}_{MODEL_TYPE}_{date}_{time_str}.png'
        plt.savefig(os.path.join(LOG_PATH, filename))
        plt.close()
        logging.info(f"Confusion matrix for {class_name} saved to {filename}")

def plot_venn_diagrams(y_true, y_pred, class_names):
    '''
    Plots Venn diagrams for each class showing the overlap between true and predicted labels.

    Args:
        y_true (ndarray): Ground truth binary labels.
        y_pred (ndarray): Predicted binary labels.
        class_names (list): List of class names.
    '''
    now = datetime.now()
    date = now.strftime("%d-%m-%y")
    time_str = now.strftime("%H-%M-%S")

    num_classes = len(class_names)
    cols = 4
    rows = (num_classes + cols - 1) // cols

    plt.figure(figsize=(20, 15))

    for class_idx in range(num_classes):
        true_set = set(np.where(y_true[:, class_idx] == 1)[0])
        pred_set = set(np.where(y_pred[:, class_idx] == 1)[0])

        plt.subplot(rows, cols, class_idx + 1)
        venn2([true_set, pred_set], set_labels=('True', 'Predicted'))
        plt.title(class_names[class_idx])

    plt.tight_layout()
    filename = f'venn_diagrams_{MODEL_TYPE}_{date}_{time_str}.png'
    plt.savefig(os.path.join(LOG_PATH, filename))
    plt.close()
    logging.info(f"Venn diagrams saved to {filename}")

def plot_metrics(train_metrics, val_metrics, test_metrics):
    '''
    Plots training and validation metrics over epochs.

    Args:
        train_metrics (dict): Dictionary containing training metrics.
        val_metrics (dict): Dictionary containing validation metrics.
        test_metrics (dict): Dictionary containing test metrics (optional).
    '''
    now = datetime.now()
    date = now.strftime("%d-%m-%y")
    time_str = now.strftime("%H-%M-%S")

    epochs_range = range(1, len(train_metrics['loss']) + 1)

    # Plot Loss
    plt.figure(figsize=(12, 6))
    plt.plot(epochs_range, train_metrics['loss'], label='Training Loss', color='blue', linestyle='-')
    plt.plot(epochs_range, val_metrics['loss'], label='Validation Loss', color='blue', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Loss over Epochs - Model: {MODEL_TYPE}')
    plt.legend()
    plt.tight_layout()
    filename = f'loss_{MODEL_TYPE}_{date}_{time_str}.png'
    plt.savefig(os.path.join(LOG_PATH, filename))
    plt.close()
    logging.info(f"Loss plot saved to {filename}")

    # Plot Precision, Recall, F1-Score
    plt.figure(figsize=(12, 6))
    plt.plot(epochs_range, train_metrics['precision'], label='Training Precision', color='green', linestyle='-')
    plt.plot(epochs_range, val_metrics['precision'], label='Validation Precision', color='green', linestyle='--')
    plt.plot(epochs_range, train_metrics['recall'], label='Training Recall', color='orange', linestyle='-')
    plt.plot(epochs_range, val_metrics['recall'], label='Validation Recall', color='orange', linestyle='--')
    plt.plot(epochs_range, train_metrics['f1'], label='Training F1-Score', color='red', linestyle='-')
    plt.plot(epochs_range, val_metrics['f1'], label='Validation F1-Score', color='red', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Scores')
    plt.title(f'Precision, Recall, F1-Score over Epochs - Model: {MODEL_TYPE}')
    plt.legend()
    plt.tight_layout()
    filename = f'precision_recall_f1_{MODEL_TYPE}_{date}_{time_str}.png'
    plt.savefig(os.path.join(LOG_PATH, filename))
    plt.close()
    logging.info(f"Precision, Recall, F1-Score plot saved to {filename}")

    # Plot Jaccard Accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(epochs_range, train_metrics['jaccard_accuracy'], label='Training Jaccard Accuracy', color='purple', linestyle='-')
    plt.plot(epochs_range, val_metrics['jaccard_accuracy'], label='Validation Jaccard Accuracy', color='purple', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Jaccard Accuracy')
    plt.title(f'Jaccard Accuracy over Epochs - Model: {MODEL_TYPE}')
    plt.legend()
    plt.tight_layout()
    filename = f'jaccard_accuracy_{MODEL_TYPE}_{date}_{time_str}.png'
    plt.savefig(os.path.join(LOG_PATH, filename))
    plt.close()
    logging.info(f"Jaccard Accuracy plot saved to {filename}")

def plot_combined_confusion_matrix(cm, class_names):
    '''
    Plots a combined confusion matrix by summing individual class confusion matrices.

    Args:
        cm (ndarray): Multilabel confusion matrices for each class.
        class_names (list): List of class names.
    '''
    now = datetime.now()
    date = now.strftime("%d-%m-%y")
    time_str = now.strftime("%H-%M-%S")

    combined_cm = np.sum(cm, axis=0)
    plt.figure(figsize=(6, 5))
    sns.heatmap(combined_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Combined Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    filename = f'Combined_ConfusionMatrix_{MODEL_TYPE}_{date}_{time_str}.png'
    plt.savefig(os.path.join(LOG_PATH, filename))
    plt.close()
    logging.info(f"Combined confusion matrix plot saved to {filename}")

def plot_confusion_matrix(cm, class_names):
    '''
    Plots confusion matrices for each class individually.

    Args:
        cm (ndarray): Multilabel confusion matrices for each class.
        class_names (list): List of class names.
    '''
    plot_classwise_confusion_matrix(cm, class_names)

def save_metrics_plots(train_metrics, val_metrics, best_val_f1, test_metrics):
    '''
    Saves all relevant metrics plots after training and testing phases.

    Args:
        train_metrics (dict): Dictionary containing training metrics.
        val_metrics (dict): Dictionary containing validation metrics.
        best_val_f1 (float): Best validation F1-score achieved during training.
        test_metrics (dict): Dictionary containing test metrics.
    '''
    # Plot Loss, Precision, Recall, F1-Score
    plot_metrics(train_metrics, val_metrics, test_metrics)

    # Plot Confusion Matrices
    plot_confusion_matrix_multi_label(test_metrics['labels'], test_metrics['preds'], CLASS_NAMES)
    cm = multilabel_confusion_matrix(test_metrics['labels'], test_metrics['preds'])
    plot_classwise_confusion_matrix(cm, CLASS_NAMES)

    # Plot Venn Diagrams
    plot_venn_diagrams(test_metrics['labels'], test_metrics['preds'], CLASS_NAMES)

    # Generate Classification Report
    generate_classification_report_custom(test_metrics['labels'], test_metrics['preds'], CLASS_NAMES)

def plot_classwise_confusion_matrix(cm, class_names):
    '''
    Plots confusion matrices for each class individually.

    Args:
        cm (ndarray): Multilabel confusion matrices for each class.
        class_names (list): List of class names.
    '''
    now = datetime.now()
    date = now.strftime("%d-%m-%y")
    time_str = now.strftime("%H-%M-%S")

    for i, (conf_matrix, class_name) in enumerate(zip(cm, class_names)):
        plt.figure(figsize=(4, 4))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f"Confusion Matrix for {class_name}")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()

        filename = f'confusion_matrix_{class_name}_{MODEL_TYPE}_{date}_{time_str}.png'
        plt.savefig(os.path.join(LOG_PATH, filename))
        plt.close()
        logging.info(f"Confusion matrix for {class_name} saved to {filename}")

def plot_venn_diagrams(y_true, y_pred, class_names):
    '''
    Plots Venn diagrams for each class showing the overlap between true and predicted labels.

    Args:
        y_true (ndarray): Ground truth binary labels.
        y_pred (ndarray): Predicted binary labels.
        class_names (list): List of class names.
    '''
    now = datetime.now()
    date = now.strftime("%d-%m-%y")
    time_str = now.strftime("%H-%M-%S")

    num_classes = len(class_names)
    cols = 4  # Number of columns in the grid
    rows = (num_classes + cols - 1) // cols  # Calculate rows needed for the grid

    plt.figure(figsize=(20, 15))

    for class_idx in range(num_classes):
        true_set = set(np.where(y_true[:, class_idx] == 1)[0])
        pred_set = set(np.where(y_pred[:, class_idx] == 1)[0])

        plt.subplot(rows, cols, class_idx + 1)
        venn2([true_set, pred_set], set_labels=('True', 'Predicted'))
        plt.title(class_names[class_idx])

    plt.tight_layout()
    filename = f'venn_diagrams_{MODEL_TYPE}_{date}_{time_str}.png'
    plt.savefig(os.path.join(LOG_PATH, filename))
    plt.close()
    logging.info(f"Venn diagrams saved to {filename}")

def generate_classification_report_custom(y_true, y_pred, class_names):
    '''
    Generates and saves a classification report based on true and predicted labels.

    Args:
        y_true (ndarray): Ground truth binary labels.
        y_pred (ndarray): Predicted binary labels.
        class_names (list): List of class names.
    '''
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    print("\nClassification Report:\n")
    print(report)

    # Save the report to a text file
    now = datetime.now()
    date = now.strftime("%d-%m-%y")
    time_str = now.strftime("%H-%M-%S")
    report_filename = f'classification_report_{MODEL_TYPE}_{date}_{time_str}.txt'
    with open(os.path.join(LOG_PATH, report_filename), 'w') as f:
        f.write(report)
    logging.info(f"Classification report saved to {report_filename}")

def plot_metrics(train_metrics, val_metrics, test_metrics):
    '''
    Plots training and validation metrics over epochs.

    Args:
        train_metrics (dict): Dictionary containing training metrics.
        val_metrics (dict): Dictionary containing validation metrics.
        test_metrics (dict): Dictionary containing test metrics (optional).
    '''
    now = datetime.now()
    date = now.strftime("%d-%m-%y")
    time_str = now.strftime("%H-%M-%S")

    epochs_range = range(1, len(train_metrics['loss']) + 1)

    # First Plot: Loss
    plt.figure(figsize=(12, 6))
    plt.plot(epochs_range, train_metrics['loss'], label='Training Loss', color='blue', linestyle='-')
    plt.plot(epochs_range, val_metrics['loss'], label='Validation Loss', color='blue', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Loss over Epochs - Model: {MODEL_TYPE}')
    plt.legend()
    plt.tight_layout()
    filename = f'loss_{MODEL_TYPE}_{date}_{time_str}.png'
    plt.savefig(os.path.join(LOG_PATH, filename))
    plt.close()
    logging.info(f"Loss plot saved to {filename}")

    # Second Plot: Precision, Recall, F1-Score
    plt.figure(figsize=(12, 6))
    plt.plot(epochs_range, train_metrics['precision'], label='Training Precision', color='green', linestyle='-')
    plt.plot(epochs_range, val_metrics['precision'], label='Validation Precision', color='green', linestyle='--')
    plt.plot(epochs_range, train_metrics['recall'], label='Training Recall', color='orange', linestyle='-')
    plt.plot(epochs_range, val_metrics['recall'], label='Validation Recall', color='orange', linestyle='--')
    plt.plot(epochs_range, train_metrics['f1'], label='Training F1-Score', color='red', linestyle='-')
    plt.plot(epochs_range, val_metrics['f1'], label='Validation F1-Score', color='red', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Scores')
    plt.title(f'Precision, Recall, F1-Score over Epochs - Model: {MODEL_TYPE}')
    plt.legend()
    plt.tight_layout()
    filename = f'precision_recall_f1_{MODEL_TYPE}_{date}_{time_str}.png'
    plt.savefig(os.path.join(LOG_PATH, filename))
    plt.close()
    logging.info(f"Precision, Recall, F1-Score plot saved to {filename}")

    # Third Plot: Jaccard Accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(epochs_range, train_metrics['jaccard_accuracy'], label='Training Jaccard Accuracy', color='purple', linestyle='-')
    plt.plot(epochs_range, val_metrics['jaccard_accuracy'], label='Validation Jaccard Accuracy', color='purple', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Jaccard Accuracy')
    plt.title(f'Jaccard Accuracy over Epochs - Model: {MODEL_TYPE}')
    plt.legend()
    plt.tight_layout()
    filename = f'jaccard_accuracy_{MODEL_TYPE}_{date}_{time_str}.png'
    plt.savefig(os.path.join(LOG_PATH, filename))
    plt.close()
    logging.info(f"Jaccard Accuracy plot saved to {filename}")

def plot_combined_confusion_matrix(cm, class_names):
    '''
    Plots a combined confusion matrix by summing individual class confusion matrices.

    Args:
        cm (ndarray): Multilabel confusion matrices for each class.
        class_names (list): List of class names.
    '''
    now = datetime.now()
    date = now.strftime("%d-%m-%y")
    time_str = now.strftime("%H-%M-%S")

    combined_cm = np.sum(cm, axis=0)
    plt.figure(figsize=(6, 5))
    sns.heatmap(combined_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Combined Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    filename = f'Combined_ConfusionMatrix_{MODEL_TYPE}_{date}_{time_str}.png'
    plt.savefig(os.path.join(LOG_PATH, filename))
    plt.close()
    logging.info(f"Combined confusion matrix plot saved to {filename}")

def save_metrics_plots(train_metrics, val_metrics, test_metrics):
    '''
    Saves all relevant metrics plots after training and testing phases.

    Args:
        train_metrics (dict): Dictionary containing training metrics.
        val_metrics (dict): Dictionary containing validation metrics.
        test_metrics (dict): Dictionary containing test metrics.
    '''
    # Plot Loss, Precision, Recall, F1-Score
    plot_metrics(train_metrics, val_metrics, test_metrics)

    # Plot Confusion Matrices
    plot_confusion_matrix_multi_label(test_metrics['labels'], test_metrics['preds'], CLASS_NAMES)
    cm = multilabel_confusion_matrix(test_metrics['labels'], test_metrics['preds'])
    plot_classwise_confusion_matrix(cm, CLASS_NAMES)

    # Plot Venn Diagrams
    plot_venn_diagrams(test_metrics['labels'], test_metrics['preds'], CLASS_NAMES)

    # Generate Classification Report
    generate_classification_report_custom(test_metrics['labels'], test_metrics['preds'], CLASS_NAMES)

def generate_classification_report_custom(y_true, y_pred, class_names):
    '''
    Generates and saves a classification report based on true and predicted labels.

    Args:
        y_true (ndarray): Ground truth binary labels.
        y_pred (ndarray): Predicted binary labels.
        class_names (list): List of class names.
    '''
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    print("\nClassification Report:\n")
    print(report)

    # Save the report to a text file
    now = datetime.now()
    date = now.strftime("%d-%m-%y")
    time_str = now.strftime("%H-%M-%S")
    report_filename = f'classification_report_{MODEL_TYPE}_{date}_{time_str}.txt'
    with open(os.path.join(LOG_PATH, report_filename), 'w') as f:
        f.write(report)
    logging.info(f"Classification report saved to {report_filename}")

def plot_combined_confusion_matrix(cm, class_names):
    '''
    Plots a combined confusion matrix by summing individual class confusion matrices.

    Args:
        cm (ndarray): Multilabel confusion matrices for each class.
        class_names (list): List of class names.
    '''
    now = datetime.now()
    date = now.strftime("%d-%m-%y")
    time_str = now.strftime("%H-%M-%S")

    combined_cm = np.sum(cm, axis=0)
    plt.figure(figsize=(6, 5))
    sns.heatmap(combined_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Combined Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    filename = f'Combined_ConfusionMatrix_{MODEL_TYPE}_{date}_{time_str}.png'
    plt.savefig(os.path.join(LOG_PATH, filename))
    plt.close()
    logging.info(f"Combined confusion matrix plot saved to {filename}")
