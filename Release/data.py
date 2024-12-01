import os
import logging
from PIL import Image
import pandas as pd
import numpy as np
import torch
import config
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from config import (
    TRAIN_IMAGES_DIR,
    VAL_IMAGES_DIR,
    TEST_IMAGES_DIR,
    DATA_CSV_PATH,
    NUM_CLASSES,
    BATCH_SIZE,
    IMAGE_SIZE,
    CALC_DATASET_MEAN_STD,
    FIRST_LAYER_SIZE,
    DEVICE,
    LABEL_COUNTS,
    CLASS_WEIGHTS,
    NUM_WORKERS,
    PIN_MEMORY,
    CLASS_NAMES,
    MODEL_SAVE_PATH,
    LOG_PATH
)


os.makedirs(LOG_PATH, exist_ok=True)


# Ensure that logging is configured
logging.basicConfig(
    filename=os.path.join(LOG_PATH, 'data_loading.log'),
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def load_image_labels(csv_path, class_names):
    """
    Load image labels from a CSV file and create a mapping from image filenames to binary labels.

    Args:
        csv_path (str): Path to the CSV file containing image labels.
        class_names (list): List of class names to consider.

    Returns:
        dict: A dictionary mapping image filenames to their binary label lists.
    """
    try:
        data_df = pd.read_csv(csv_path)
        logging.info(f"CSV file loaded successfully from {csv_path}")
    except FileNotFoundError:
        logging.error(f"CSV file not found at path: {csv_path}")
        raise
    except pd.errors.EmptyDataError:
        logging.error(f"CSV file is empty at path: {csv_path}")
        raise
    except Exception as e:
        logging.error(f"Error reading CSV file at {csv_path}: {e}")
        raise

    image_labels = {}
    for _, row in data_df.iterrows():
        img_name = row["Image Index"]
        label_string = row["Finding Labels"]
        label_list = label_string.split("|")
        # Generate binary labels for the specified classes
        binary_labels = [1 if disease in label_list else 0 for disease in class_names]
        image_labels[img_name] = torch.tensor(binary_labels, dtype=torch.float32)

    logging.info(f"Image labels mapping created with {len(image_labels)} entries.")
    return image_labels

class ChestXrayDataset(Dataset):
    """
    Custom Dataset for loading Chest X-ray images and their corresponding labels.

    Args:
        root_dir (str): Directory with all the images.
        image_labels (dict): Dictionary mapping image filenames to their labels.
        transform (callable, optional): Optional transform to be applied on a sample.

    Attributes:
        root_dir (str): Directory with all the images.
        image_names (list): List of image filenames.
        transform (callable): Transformations to apply to the images.
        labels (list): List of label tensors corresponding to each image.
    """
    def __init__(self, root_dir, image_labels, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_names = [
            img for img in os.listdir(root_dir)
            if not img.startswith(".") and os.path.isfile(os.path.join(root_dir, img))
        ]
        self.labels = [image_labels.get(img, torch.zeros(NUM_CLASSES, dtype=torch.float32)) for img in self.image_names]
        logging.info(f"Dataset initialized with {len(self.image_names)} images from {root_dir}")

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.root_dir, img_name)

        try:
            image = Image.open(img_path).convert("RGB")  # Convert to RGB
        except (OSError, IOError) as e:
            logging.error(f"Error loading image {img_path}: {e}")
            return None, None

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def calculate_mean_std(dataset, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, device=DEVICE):
    """
    Calculate the mean and standard deviation of the dataset.

    Args:
        dataset (Dataset): PyTorch Dataset object.
        batch_size (int): Batch size for DataLoader.
        image_size (int): Size to which images are resized.
        device (torch.device): Device to perform computations on.

    Returns:
        tuple: Mean and standard deviation for each channel.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    sum_pixels = torch.zeros(3).to(device)
    sum_squared_pixels = torch.zeros(3).to(device)
    num_pixels = 0

    logging.info("Starting calculation of dataset mean and standard deviation.")
    for images, _ in loader:
        if images is None:
            continue
        images = images.to(device)
        sum_pixels += images.sum(dim=[0, 2, 3])
        sum_squared_pixels += (images ** 2).sum(dim=[0, 2, 3])
        num_pixels += images.numel() / images.size(1)  # Total pixels per channel

    mean = sum_pixels / num_pixels
    std = torch.sqrt((sum_squared_pixels / num_pixels) - (mean ** 2))
    mean = mean.cpu().numpy()
    std = std.cpu().numpy()

    logging.info(f"Calculated mean: {mean}")
    logging.info(f"Calculated std: {std}")

    return mean, std

def get_data_loaders(class_names):
    """
    Create DataLoaders for training, validation, and testing datasets.

    Args:
        class_names (list): List of class names.
        config (object): Configuration object containing all necessary parameters.

    Returns:
        tuple: DataLoaders for training, validation, and testing datasets.
    """
    # Load image labels
    image_labels = load_image_labels(config.DATA_CSV_PATH, class_names)

    # Define transforms
    if config.CALC_DATASET_MEAN_STD:
        # Temporary transform for calculating mean and std
        temp_transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
        ])
        temp_dataset = ChestXrayDataset(config.TRAIN_IMAGES_DIR, image_labels, transform=temp_transform)
        mean, std = calculate_mean_std(temp_dataset, batch_size=config.BATCH_SIZE, image_size=config.IMAGE_SIZE, device=config.DEVICE)
        logging.info(f"Computed dataset mean: {mean}, std: {std}")
    else:
        # Precomputed mean and std
        mean = [0.4964, 0.4964, 0.49643]
        std = [0.2473, 0.2473, 0.24730]
        logging.info(f"Using precomputed mean: {mean}")
        logging.info(f"Using precomputed std: {std}")

    # Define data transformations
    train_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.RandomResizedCrop(size=config.IMAGE_SIZE, scale=(0.9, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_test_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Create datasets
    train_dataset = ChestXrayDataset(config.TRAIN_IMAGES_DIR, image_labels, transform=train_transform)
    val_dataset = ChestXrayDataset(config.VAL_IMAGES_DIR, image_labels, transform=val_test_transform)
    test_dataset = ChestXrayDataset(config.TEST_IMAGES_DIR, image_labels, transform=val_test_transform)

    # WeightedRandomSampler for training to handle class imbalance
    label_counts = config.LABEL_COUNTS
    total_labels = config.TOTAL_LABELS
    class_weights = config.CLASS_WEIGHTS  # Already pre-calculated and found in config.py

    sample_weights = []
    for img_name in train_dataset.image_names:
        labels = image_labels.get(img_name, torch.zeros(NUM_CLASSES, dtype=torch.float32))
        weight = sum(class_weights[i] for i, label in enumerate(labels) if label == 1)
        sample_weights.append(weight)

    # Normalize sample weights
    mean_weight = np.mean(sample_weights)
    sample_weights = [weight / mean_weight for weight in sample_weights]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # Define collate function to filter out None samples
    def collate_fn(batch):
        batch = list(filter(lambda x: x[0] is not None and x[1] is not None, batch))
        if not batch:
            return None, None
        return torch.utils.data.dataloader.default_collate(batch)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, sampler=sampler, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, collate_fn=collate_fn)

    logging.info(f"Sample weights (first 10): {sample_weights[:10]}")
    logging.info(f"Total samples in training set: {len(sample_weights)}")

    return train_loader, val_loader, test_loader

