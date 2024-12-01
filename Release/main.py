import os
import logging
import torch
import config
from datetime import datetime

from config import (
    CLASS_NAMES,
    MODEL_TYPE,
    DEVICE,
    LOG_PATH,
    MODEL_SAVE_PATH,
    CLASS_WEIGHTS
)
from data import get_data_loaders
from models import build_model
from train import train_model
from evaluation import (
    evaluate_model,
    identify_misclassified_samples,
    save_metrics_plots,
    generate_classification_report_custom
)

def setup_logging():
    """
    Configures the logging settings for the main module.
    Logs are saved to 'main.log' within the LOG_PATH directory.
    """
    os.makedirs(LOG_PATH, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(LOG_PATH, 'main.log'),
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logging.getLogger().addHandler(logging.StreamHandler())  # Also log to console

def main():
    """
    Main function to orchestrate the training and evaluation.
    """
    # Setup logging
    setup_logging()
    logging.info("========== Starting Training ==========")
    
    # Check device
    device = DEVICE
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info("Using GPU for training.")
    else:
        device = torch.device('cpu')
        logging.info("GPU not available. Using CPU for training.")
    
    # Initialize DataLoaders
    logging.info("Initializing data loaders.")
    train_loader, val_loader, test_loader = get_data_loaders(CLASS_NAMES)
    logging.info("Data loaders initialized.")
    
    # Build Model
    logging.info(f"Building model: {MODEL_TYPE}")
    model = build_model(model_type=MODEL_TYPE, class_weights=torch.tensor(CLASS_WEIGHTS).to(device))
    logging.info(f"Model {MODEL_TYPE} built successfully.")
    
    # Define Loss Function
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(CLASS_WEIGHTS).to(device))
    logging.info("Loss function BCEWithLogitsLoss initialized.")
    
    # Define Optimizer
    # Assuming that the optimizer type is defined in config.py or hardcoded here
    # For this example, we'll use AdamW
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=1e-4,  # Example learning rate
                                  weight_decay=1e-5)  # Example weight decay
    logging.info("Optimizer AdamW initialized with learning rate 1e-4 and weight decay 1e-5.")
    
    # Define Learning Rate Scheduler (Optional)
    # Example using ReduceLROnPlateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    logging.info("Learning rate scheduler ReduceLROnPlateau initialized.")
    
    # Train the Model
    logging.info("Commencing training.")
    best_f1, best_epoch, train_metrics, val_metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=config.NUM_EPOCHS  # Example number of epochs
    )
    logging.info(f"Training completed. Best Validation F1-score: {best_f1:.4f} at epoch {best_epoch}.")
    
    # Save the Best Model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_model_filename = f"best_model_{MODEL_TYPE}_{timestamp}.pth"
    best_model_path = os.path.join(MODEL_SAVE_PATH, best_model_filename)
    torch.save(model.state_dict(), best_model_path)
    logging.info(f"Best model saved to {best_model_path}.")
    
    # Evaluate the Model on Test Set
    logging.info("Evaluating the model on the test set.")
    test_metrics = evaluate_model(model, test_loader, criterion, device, return_preds=True)
    
    # Unpack the tuple
    test_loss, test_jaccard, test_accuracy, test_precision, test_recall, test_f1, test_hamming_loss, test_specificity, test_preds, test_labels = test_metrics
    logging.info(f"Evaluation Results - Loss: {test_loss:.4f}, Jaccard Accuracy: {test_jaccard:.4f}, "
                 f"Overall Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, "
                 f"Recall: {test_recall:.4f}, F1-score: {test_f1:.4f}, "
                 f"Hamming Loss: {test_hamming_loss:.4f}, Specificity: {test_specificity:.4f}")
    logging.info("Model evaluation on test set completed.")
    
    # Identify Misclassified Samples
    logging.info("Identifying misclassified samples.")
    misclassified = identify_misclassified_samples(test_labels, test_preds, test_loader.dataset.image_names)
    logging.info(f"Number of misclassified samples: {len(misclassified)}")
    
    # Save Misclassified Samples to File
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    misclassified_path = os.path.join(LOG_PATH, f"misclassified_samples_{timestamp}.txt")
    with open(misclassified_path, 'w') as f:
        for filename in misclassified:
            f.write(f"{filename}\n")
    logging.info(f"Misclassified samples saved to {misclassified_path}.")
    
    logging.info("========== Training Completed Successfully ==========")

if __name__ == "__main__":
    main()
