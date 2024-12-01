import torch
import random
import numpy as np

VERSION = "2.0.1"             # Version of the model/configuration
NUM_CLASSES = 6               # Number of target classes for classification

TRAIN_IMAGES_DIR = "./train_images_filtered_80_10_10"  # Path to the training images directory
VAL_IMAGES_DIR = "./val_images_filtered_80_10_10"      # Path to the validation images directory
TEST_IMAGES_DIR = "./test_images_filtered_80_10_10"    # Path to the test images directory
DATA_CSV_PATH = "./Data_Entry_2017.csv"                # Path to the CSV file containing image labels
LOG_PATH = "./logs"                                    # Directory to store log files
MODEL_SAVE_PATH = "./models"                           # Directory to save trained models


BATCH_SIZE = 128                 # Batch size for training and evaluation
LEARNING_RATE = 0.001            # Initial learning rate for the optimizer
NUM_EPOCHS = 50                 # Number of training epochs
DROPOUT = 0.2                    # Dropout rate for regularization
IMAGE_SIZE = 224                 # Height and width to resize input images
FIRST_LAYER_SIZE = 3             # Number of input channels: 1 for grayscale, 3 for RGB
CALC_DATASET_MEAN_STD = False    # Whether to calculate dataset mean and std. Set to True if not precomputed
MODEL_TYPE = 'ResNet18_SEB_1'      # Type of model to use for classification
                                 # Options: 'DeeperCNN_8', 'DeeperCNN_8_SEB', 'ResNet50',
                                 # 'ResNet18', 'ResNet18_1', 'ResNet18_2',
                                 # 'ResNet18_SEB_1', 'ResNet18_SEB_2', 'DenseNet121'

OPTIMIZER_TYPE = 'AdamW'         # Optimizer to use for training
                                 # Options: 'Adam', 'AdamW', 'SGDm', 'RMSprop'

USE_SCHEDULER = True             # Whether to use a learning rate scheduler
SCHEDULER_FACTOR = 0.5           # Factor by which the learning rate will be reduced
SCHEDULER_PATIENCE = 2           # Number of epochs with no improvement after which LR will be reduced
 
FREEZE_LAYERS = False            # Whether to freeze layers in pre-trained models for fine-tuning
WEIGHT_DECAY = 5e-5              # Weight decay (L2 penalty) for the optimizer
EARLY_STOPPING_PATIENCE = 10     # Number of epochs with no improvement after which training will be stopped


LABEL_COUNTS = [10586, 10434, 5941, 5670, 3300, 2751]  # Number of samples per class
TOTAL_LABELS = sum(LABEL_COUNTS)                      # Total number of labels across all classes
CLASS_WEIGHTS = [TOTAL_LABELS / (len(LABEL_COUNTS) * count) for count in LABEL_COUNTS]  # Class weights for balancing

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Device to run the model on
print(f"Using device: {DEVICE}")

SEED = 42                    # Random seed for reproducibility
def set_seed(seed=SEED):
    """
    Sets the seed for generating random numbers to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensures deterministic behavior (may reduce performance)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

set_seed()


NUM_WORKERS = 4             # Number of subprocesses to use for data loading
PIN_MEMORY = True           # Whether to pin memory in DataLoader for faster data transfer to CUDA

CLASS_NAMES = ['No Finding', 'Infiltration', 'Effusion', 'Atelectasis', 'Nodule', 'Mass']  # List of class names


