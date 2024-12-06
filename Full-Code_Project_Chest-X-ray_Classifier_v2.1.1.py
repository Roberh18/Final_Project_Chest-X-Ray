import os
import time
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    hamming_loss,
    jaccard_score,
    multilabel_confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import seaborn as sns



# ================================
# 1. Device Configuration
# ================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



# ============================================================
# 2.1 Hyperparameters: Constants and Configuration
# ============================================================

VERSION = "2.0.1"
NUM_CLASSES = 6
BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
DROPOUT = 0.2
IMAGE_SIZE = 224
FIRST_LAYER_SIZE = 3
CALC_DATASET_MEAN_STD = False

MODEL_TYPE = 'ResNet18_SEB'
OPTIMIZER_TYPE = 'AdamW'
USE_SCHEDULER = True
SCHEDULER_FACTOR = 0.3
SCHEDULER_PATIENCE = 2
FREEZE_LAYERS = False
WEIGHT_DECAY = 5e-5


# File paths
train_images_dir = "./train_images_filtered_80_10_10"
val_images_dir = "./val_images_filtered_80_10_10"
test_images_dir = "./test_images_filtered_80_10_10"
data_csv_dir = "./Data_Entry_2017.csv"


data_df = pd.read_csv(data_csv_dir)


# ==============================================================
# 3.1 Load Class Names and print details on dataset 
# ==============================================================


# Define class names 
class_names = ['No Finding', 'Infiltration', 'Effusion', 'Atelectasis', 'Nodule', 'Mass']

# Create a mapping from image index to labels
image_labels = {}
for idx, row in data_df.iterrows():
    img_name = row["Image Index"]
    label_string = row["Finding Labels"]
    label_list = label_string.split("|")
    # Generate binary labels for the 8 most represented classes
    binary_labels = [1 if disease in label_list else 0 for disease in class_names]
    image_labels[img_name] = binary_labels
    

# Check if folders exist and print the number of samples in each dataset directory
for path in [train_images_dir, val_images_dir, test_images_dir]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path not found: {path}")
    print(f"\nPath exists: {path} \n Samples: {len(os.listdir(path))}\n")





# =====================================================
# 3.2 Calculate Label Counts and Label Combinations per Dataset
# =====================================================


# For each dataset (train, val, test)
for dataset_name, images_dir in [('Training', train_images_dir), ('Validation', val_images_dir), ('Test', test_images_dir)]:
    image_names = [img for img in os.listdir(images_dir) if not img.startswith(".") and os.path.isfile(os.path.join(images_dir, img))]

    # Initialize counts
    label_counts = np.zeros(len(class_names), dtype=int)
    label_combo_counts = np.zeros(len(class_names), dtype=int)  # For counts of images with N labels (from 1 to 8)
    total_images = len(image_names)
    total_labels = 0

    # Process images
    for img_name in image_names:
        if img_name in image_labels:
            labels = image_labels[img_name]  # This is a list of 0s and 1s
            labels_np = np.array(labels)
            label_counts += labels_np.astype(int)
            num_labels = int(labels_np.sum())
            total_labels += num_labels
            if num_labels > 0 and num_labels <= len(class_names):
                label_combo_counts[num_labels-1] += 1  # Subtract 1 because index starts from 0
            else:
                print(f"Image {img_name} has invalid number of labels: {num_labels}")
        else:
            print(f"Image {img_name} not found in image_labels.")

    # Print label counts sorted from most represented label to least
    sorted_indices = np.argsort(-label_counts)  # Negative for descending order

    print(f"\n{'='*70}")
    print(f"Label Counts in {dataset_name} Dataset (Total Images: {total_images:,})")
    print(f"{'='*70}")
    print(f"{'Label':<20}{'Count':>15}{'Percentage':>15}")
    print(f"{'-'*70}")
    for idx in sorted_indices:
        label = class_names[idx]
        count = label_counts[idx]
        percentage = (count / total_labels) * 100 if total_labels > 0 else 0
        print(f"{label:<20}{count:>15,}{percentage:>14.2f}%")
    print(f"{'-'*70}")
    print(f"{'Total Labels:':<20}{total_labels:>15,}")
    
    # Print counts of images with N labels
    print(f"\nNumber of Images with N Labels in {dataset_name} Dataset:")
    print(f"{'-'*70}")
    print(f"{'Number of Labels':<20}{'Image Count':>15}{'Percentage':>15}")
    print(f"{'-'*70}")
    for num_labels, count in enumerate(label_combo_counts, start=1):
        percentage = (count / total_images) * 100 if total_images > 0 else 0
        print(f"{num_labels:<20}{count:>15,}{percentage:>14.2f}%")
    print(f"{'-'*70}")
    print(f"{'Total Images:':<20}{total_images:>15,}")

print(f"\nVersion: {VERSION}")
print(f"\nModel: {MODEL_TYPE}")
print(f"Optimizer: {OPTIMIZER_TYPE}\n")
print(f"\nBatch Size: {BATCH_SIZE}")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Number of Epochs: {NUM_EPOCHS}")
print(f"Dropout: {DROPOUT}")
print(f"Image Size: {IMAGE_SIZE}")
print(f"First Layer Size: {FIRST_LAYER_SIZE}")
print(f"Use Scheduler: {USE_SCHEDULER}")
print(f"Scheduler Factor: {SCHEDULER_FACTOR}")
print(f"Scheduler Patience: {SCHEDULER_PATIENCE}")
print(f"Freeze Layers: {FREEZE_LAYERS}")
print(f"Weight Decay: {WEIGHT_DECAY}\n")

print("GPU Details (nvidia-smi):")
os.system("nvidia-smi")
print(torch.cuda.memory_summary())


# ================================
# 4. Custom Classes & functions
# ================================

class ChestXrayDataset(Dataset):
    '''
    Description:
    - This class loads chest X-ray images from the specified root directory and stores their corresponding labels.
    
    Parameters:
    - root_dir: Path to the directory containing chest X-ray images.
    - image_labels (dict): Dictionary mapping image filenames to their labels.
    - transform (callable, optional): Optional transform to be applied on an image sample.

    '''
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_names = [
            img
            for img in os.listdir(root_dir)
            if not img.startswith(".") and os.path.isfile(os.path.join(root_dir, img))
        ]
        self.transform = transform

        # Precompute labels for all images
        self.labels = []
        for img_name in self.image_names:
            label = self._parse_labels(img_name)
            self.labels.append(label)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.root_dir, img_name)

        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            return None, None

        try:
            image = Image.open(img_path).convert("RGB")  # 'L' mode for grayscale
        except (OSError, IOError) as e:
            print(f"Error loading image {img_path}: {e}")
            return None, None

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
    
    def _parse_labels(self, img_name):
        if img_name in image_labels:
            binary_labels = image_labels[img_name]
            return torch.tensor(binary_labels, dtype=torch.float32)
        else:
            return torch.zeros(NUM_CLASSES, dtype=torch.float32)  


class SEBlock(nn.Module):
    '''Squeeze-and-Excitation Block'''
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_avg_pool(x).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y

def add_se_to_resnet(model, reduction=16):
    for name, module in model.named_children():
        if isinstance(module, nn.Sequential):  
            for block_name, block in module.named_children():
                if hasattr(block, 'conv2'):  
                    block.add_module("se", SEBlock(block.conv2.out_channels, reduction))
        elif isinstance(module, nn.Module):
            add_se_to_resnet(module, reduction)
    return model


# ================================
# Function to Calculate Mean and Std
# ================================

def calculate_mean_std(root_dir, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE):
    """
    Calculate the mean and standard deviation of the dataset.
    
    Parameters:
        root_dir (str): The root directory containing the dataset.
        image_size (int): The size to which images should be resized.
        batch_size (int): The batch size for loading the dataset.
    
    Returns:
        mean (torch.Tensor): The mean for each channel.
        std (torch.Tensor): The standard deviation for each channel.
    """
    calc_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    
    calc_dataset = ChestXrayDataset(root_dir=root_dir, transform=calc_transform)
    calc_loader = DataLoader(calc_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    sum_pixels = torch.zeros(3)
    sum_squared_pixels = torch.zeros(3)
    num_pixels = 0

    print("Calculating mean and std of dataset...")
    for images, _ in calc_loader:
        # Flatten images from (batch_size, channels, height, width) to (batch_size, channels, -1)
        batch_size, channels, height, width = images.size()
        num_pixels += batch_size * height * width
        
        # Sum pixel values and squared values
        sum_pixels += images.sum(dim=[0, 2, 3])  # Sum along batch, height, and width dimensions
        sum_squared_pixels += (images ** 2).sum(dim=[0, 2, 3])  # Sum of squares for variance calculation

    # Calculate mean and std for each channel
    mean = sum_pixels / num_pixels
    std = torch.sqrt((sum_squared_pixels / num_pixels) - (mean ** 2))

    # Print the calculated mean and std values
    print(f"Calculated mean: {mean}")
    print(f"Calculated std: {std}")

    return mean, std



# ================================
# 5. Data Transformations and DataLoaders
# ================================


if CALC_DATASET_MEAN_STD:
    mean, std = calculate_mean_std(train_images_dir, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)
else:
    # Precomputed mean and standard deviation for dataset
    mean = [0.4964, 0.4964, 0.49643]
    std = [0.2473, 0.2473, 0.24730]
    print(f"Pre-computed mean: {mean}")
    print(f"Pre-computed std: {std}")


# Define data transformations for training, validation, and testing
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),                       # Resize all images to a uniform size for consistency.
    transforms.RandomRotation(degrees=10),                             # Rotate images randomly
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),       # Small translations for variation
    transforms.RandomResizedCrop(size=IMAGE_SIZE, scale=(0.9, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),              # Slight brightness/contrast adjustment for realistic variation.
    transforms.ToTensor(),                                             # Convert PIL Image to a PyTorch tensor.
    transforms.Normalize(mean, std),                                   # Normalization helps stabilize gradients during training, leading to faster convergence and better performance.
])
val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])


# Create datasets
train_dataset = ChestXrayDataset(root_dir=train_images_dir, transform=train_transform)
val_dataset = ChestXrayDataset(root_dir=val_images_dir, transform=val_transform)
test_dataset = ChestXrayDataset(root_dir=test_images_dir, transform=val_transform)


def collate_fn(batch):
    # Filter out samples where image or label is None
    batch = list(filter(lambda x: x[0] is not None and x[1] is not None, batch))
    if not batch:
        # Return an empty batch if all samples are invalid
        return None, None
    return torch.utils.data.dataloader.default_collate(batch)




# ================================
# 6. Define Data Loader
# ================================

label_counts = [10586, 10434, 5941, 5670, 3300, 2751]  # Example counts for classes
total_labels = sum(label_counts)
class_weights = [total_labels / count for count in label_counts]
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)  # Move to GPU if needed

sample_weights = []
for img_name in os.listdir(train_images_dir):
    if img_name in image_labels:  # Check if the image has labels
        binary_labels = image_labels[img_name]  # Get the binary labels
        # Calculate the weight for this sample as the sum of class weights for active labels
        weight = sum(class_weights[i].item() for i, label in enumerate(binary_labels) if label == 1)
        sample_weights.append(weight)

# Normalize sample weights
mean_weight = np.mean(sample_weights)
sample_weights = [weight / mean_weight for weight in sample_weights]

# Create the WeightedRandomSampler using the calculated sample weights
sampler = WeightedRandomSampler(
    weights=sample_weights,  # Use the precomputed sample weights
    num_samples=len(sample_weights),  # Match the number of samples in the training set
    replacement=True  # Allow replacement to ensure balanced sampling
)

# Define DataLoaders for train, validation, and test sets
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)

# Debugging: Print statistics of sample weights to ensure they are calculated correctly
print(f"Sample weights (first 10): {sample_weights[:10]}")
print(f"Total samples in training set: {len(sample_weights)}")


# ================================
# 7.1 Custom Model Definitions
# ================================

class DeeperCNN_8(nn.Module):
    def __init__(self):
        super(DeeperCNN_8, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(DROPOUT)

        # Convolutional Block 1
        self.conv1_1 = nn.Conv2d(FIRST_LAYER_SIZE, 64, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Convolutional Block 2
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Convolutional Block 3
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Convolutional Block 4
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2, 2)

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Output Layer
        self.fc = nn.Linear(512, NUM_CLASSES)

    def forward(self, x):
        # Block 1
        residual = x
        x = self.relu(self.bn1_1(self.conv1_1(x)))
        x = self.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool1(x)
        # Residual Connection (Optional)
        # x += residual

        # Block 2
        x = self.relu(self.bn2_1(self.conv2_1(x)))
        x = self.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool2(x)

        # Block 3
        x = self.relu(self.bn3_1(self.conv3_1(x)))
        x = self.relu(self.bn3_2(self.conv3_2(x)))
        x = self.pool3(x)

        # Block 4
        x = self.relu(self.bn4_1(self.conv4_1(x)))
        x = self.relu(self.bn4_2(self.conv4_2(x)))
        x = self.pool4(x)

        # Global Average Pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)

        # Dropout
        x = self.dropout(x)

        # Output Layer
        x = self.fc(x)
        return x




class DeeperCNN_8_SEB(nn.Module):
    def __init__(self):
        '''Added Squeeze-and-Excitation (SE) block'''
        super(DeeperCNN_8_SEB, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(DROPOUT)

        # Convolutional Block 1
        self.conv1_1 = nn.Conv2d(FIRST_LAYER_SIZE, 64, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.se1 = SEBlock(64)  # Add SE block here
        self.pool1 = nn.MaxPool2d(2, 2)

        # Convolutional Block 2
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.se2 = SEBlock(128)  # Add SE block here
        self.pool2 = nn.MaxPool2d(2, 2)

        # Convolutional Block 3
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.se3 = SEBlock(256)  # Add SE block here
        self.pool3 = nn.MaxPool2d(2, 2)

        # Convolutional Block 4
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.se4 = SEBlock(512)  # Add SE block here
        self.pool4 = nn.MaxPool2d(2, 2)

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Output Layer
        self.fc = nn.Linear(512, NUM_CLASSES)

    def forward(self, x):
        # Block 1
        x = self.relu(self.bn1_1(self.conv1_1(x)))
        x = self.relu(self.bn1_2(self.conv1_2(x)))
        x = self.se1(x)  # Pass through SE block
        x = self.pool1(x)

        # Block 2
        x = self.relu(self.bn2_1(self.conv2_1(x)))
        x = self.relu(self.bn2_2(self.conv2_2(x)))
        x = self.se2(x)  # Pass through SE block
        x = self.pool2(x)

        # Block 3
        x = self.relu(self.bn3_1(self.conv3_1(x)))
        x = self.relu(self.bn3_2(self.conv3_2(x)))
        x = self.se3(x)  # Pass through SE block
        x = self.pool3(x)

        # Block 4
        x = self.relu(self.bn4_1(self.conv4_1(x)))
        x = self.relu(self.bn4_2(self.conv4_2(x)))
        x = self.se4(x)  # Pass through SE block
        x = self.pool4(x)

        # Global Average Pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)

        # Dropout
        x = self.dropout(x)

        # Output Layer
        x = self.fc(x)
        return x







# ===================================
# 7.1 Pre-trained Model Definitions
# ===================================




def get_model(model_type):
    if model_type == 'DenseNet121':
        model = models.densenet121(weights='IMAGENET1K_V2')
        num_features = model.classifier.in_features
        # Freeze early layers
        for param in model.features.parameters():
            param.requires_grad = False
        # Modify classifier
        model.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(512, NUM_CLASSES),
        )
    elif  model_type == 'ResNet50':
        model = models.resnet50(weights='IMAGENET1K_V2')
        num_features = model.fc.in_features
        # Freeze early layers based on FREEZE_LAYERS
        if FREEZE_LAYERS:
            for name, param in model.named_parameters():
                if 'layer4' not in name:
                    param.requires_grad = False
        # Modify classifier
        model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(512, NUM_CLASSES),
        )
    elif model_type == 'ResNet18':
        model = models.resnet18(weights='IMAGENET1K_V1')
        num_features = model.fc.in_features
        if FREEZE_LAYERS:
            # Freeze earlier layers for feature extraction
            for name, param in model.named_parameters():
                if 'layer4' not in name:
                    param.requires_grad = False
        # Modify classifier for multi-label classification
        model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(512, NUM_CLASSES),
        )
    elif model_type == 'ResNet18_1':
        model = models.resnet18(weights='IMAGENET1K_V1')
        num_features = model.fc.in_features
        if FREEZE_LAYERS:
            # Freeze earlier layers for feature extraction
            for name, param in model.named_parameters():
                if 'layer4' not in name:
                    param.requires_grad = False
        # Modify classifier for multi-label classification
        model.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(512, NUM_CLASSES),
        )
    elif model_type == 'ResNet18_2':
        model = models.resnet18(weights='IMAGENET1K_V1')
        num_features = model.fc.in_features
        if FREEZE_LAYERS:
            # Freeze earlier layers for feature extraction
            for name, param in model.named_parameters():
                if 'layer4' not in name:
                    param.requires_grad = False
        # Modify classifier for multi-label classification
        model.fc = nn.Sequential(
            nn.Linear(num_features, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(512, NUM_CLASSES),
        )
    elif model_type == 'ResNet18_SEB_1':
        model = models.resnet18(weights='IMAGENET1K_V1')
        model = add_se_to_resnet(model, reduction=16)  # Add SE blocks with a reduction ratio of 16
    
        num_features = model.fc.in_features
        if FREEZE_LAYERS:
            # Freeze earlier layers for feature extraction
            for name, param in model.named_parameters():
                if 'layer4' not in name:
                    param.requires_grad = False
    
        # Modify classifier for multi-label classification
        model.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(512, NUM_CLASSES),
    )
    elif model_type == 'ResNet18_SEB_2':
        model = models.resnet18(weights='IMAGENET1K_V1')
        model = add_se_to_resnet(model, reduction=8)  # Add SE blocks with a reduction ratio of 8
    
        num_features = model.fc.in_features
        if FREEZE_LAYERS:
             # Freeze earlier layers for feature extraction
            for name, param in model.named_parameters():
                if not any(layer in name for layer in ['layer4', 'layer3']):
                    param.requires_grad = False
                    
        # Modify classifier for multi-label classification
        model.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(DROPOUT),   
            nn.Linear(512, NUM_CLASSES),
    )
    else:
        raise ValueError(f"Invalid MODEL_TYPE: {model_type}")

    return model.to(device)



if MODEL_TYPE == 'DeeperCNN_6':
    model = DeeperCNN_6().to(device)
elif MODEL_TYPE == 'DeeperCNN_8':
    model = DeeperCNN_8().to(device)
elif MODEL_TYPE == 'DeeperCNN_8_SEB':
    model = DeeperCNN_8_SEB().to(device)
elif MODEL_TYPE == 'DeeperCNN_8_Res':
    model = DeeperCNN_8_Residual().to(device)
elif MODEL_TYPE in ['DenseNet121', 'ResNet50', 'ResNet18', 'ResNet18_1', 'ResNet18_2', 'ResNet18_SEB_1', 'ResNet18_SEB_2']:
    model = get_model(MODEL_TYPE)




# ==========================================
# 8. Loss Function, Optimizer, and Scheduler
# ==========================================

total_labels = sum(label_counts)

# Calculate weights
weights = [total_labels / (len(label_counts) * count) for count in label_counts]

# Assign weights in the same order as your dataset labels
class_weights = torch.tensor(weights, dtype=torch.float32).to(device)  # Move to GPU if using CUDA

# Define loss function with weights
criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)


if OPTIMIZER_TYPE == 'AdamW':
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
elif OPTIMIZER_TYPE == 'SGDm':
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)
elif OPTIMIZER_TYPE == "RMSprop":
    optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
elif OPTIMIZER_TYPE == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
else:
    raise ValueError(f"Invalid optimizer type: {OPTIMIZER_TYPE}")


scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE)


# ================================
# 9. Evaluate Model Function
# ================================



def evaluate_model(model, data_loader, criterion, return_preds=False, return_outputs=False):
    """
    Evaluates the model on a given dataset.
    """
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_outputs = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            if inputs is None or labels is None:
                continue

            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            if return_outputs:
                all_outputs.append(outputs.cpu().numpy())

            preds = torch.sigmoid(outputs) > 0.5
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = running_loss / len(data_loader)
    all_labels = np.vstack(all_labels)
    all_preds = np.vstack(all_preds)

    # Compute metrics
    precision = precision_score(
        all_labels, all_preds, average="macro", zero_division=0
    )
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    hamming = hamming_loss(all_labels, all_preds)
    accuracy = jaccard_score(all_labels, all_preds, average="samples")

    # Compute sensitivity, specificity, and accuracy based on confusion matrix
    cm = multilabel_confusion_matrix(all_labels, all_preds)
    tp = cm[:, 1, 1]
    tn = cm[:, 0, 0]
    fp = cm[:, 0, 1]
    fn = cm[:, 1, 0]

    specificity = tn / (tn + fp)
    overall_accuracy = (tp + tn) / (tp + tn + fp + fn)

    # Compute average across all classes
    avg_specificity = np.nanmean(specificity)
    avg_overall_accuracy = np.nanmean(overall_accuracy)


    if return_preds and return_outputs:
        return avg_loss, accuracy, avg_overall_accuracy, precision, recall, f1, hamming, avg_specificity, all_preds, all_labels, all_outputs
    elif return_preds:
        return avg_loss, accuracy, avg_overall_accuracy, precision, recall, f1, hamming, avg_specificity, all_preds, all_labels
    else:
        return avg_loss, accuracy, avg_overall_accuracy, precision, recall, f1, hamming, avg_specificity




def identify_misclassified_samples(test_labels, test_preds, test_image_names):
    misclassified_samples = []
    for i in range(len(test_labels)):
        if not np.array_equal(test_labels[i], test_preds[i]):
            misclassified_samples.append(test_image_names[i])
    return misclassified_samples





# ================================
# 10. Training and Validation Loop
# ================================

def train_model_with_scheduler(
    model, train_loader, val_loader, criterion, optimizer, scheduler=None, epochs=NUM_EPOCHS
):
    """
    Trains the model and evaluates it on the validation set after each epoch.
    """
    # Lists to store metrics
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    train_precisions, val_precisions = [], []
    train_recalls, val_recalls = [], []
    train_f1s, val_f1s = [], []
    train_hammings, val_hammings = [], []
    train_overall_accuracies, val_overall_accuracies = [], []
    train_specificities, val_specificities = [], []
    
    best_val_f1 = 0
    best_epoch = 0
            
    # Debugging: Check the distribution of sampled classes
    sampled_classes = []
    for img_name, weight in zip(os.listdir(train_images_dir), sample_weights):
        if img_name in image_labels:
            labels = image_labels[img_name]
            sampled_classes.extend([i for i, label in enumerate(labels) if label == 1])
    
    # Display the class distribution of the sampled dataset
    sampled_class_distribution = np.bincount(sampled_classes, minlength=NUM_CLASSES)
    print("\nSampled Class Distribution in Training:")
    for i, count in enumerate(sampled_class_distribution):
        print(f"{class_names[i]:<15}: {count} samples")


    for epoch in range(epochs):
        print(f"\nStarting Epoch {epoch + 1}/{epochs}")
        epoch_start_time = time.time()

        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        last_progress = -1

        for batch_idx, (inputs, labels) in enumerate(train_loader, 1):
            if inputs is None or labels is None:
                continue  # Skip this batch

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            preds = torch.sigmoid(outputs) > 0.5
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            # Debugging: Print the first batch labels and predictions for inspection, and then every 10th epoch
            if epoch % 10 == 0:
                if batch_idx == 1:
                    lbl = labels[:10]
                    prd = preds[:10].float()
                    print(f"\n[DEBUG] Labels (first batch): {lbl}")
                    print(f"[DEBUG] Predictions (first batch): {prd}")
                    print(f"[DEBUG] Number of positive labels in first batch: {labels.sum().item()}\n")

            # Progress bar
            progress = int(100 * batch_idx / len(train_loader))
            if progress % 10 == 0 and progress != last_progress:
                print(".", end="", flush=True)
                last_progress = progress
                

        # Compute training metrics
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        all_labels_np = np.vstack(all_labels)
        all_preds_np = np.vstack(all_preds)

        train_precision = precision_score(all_labels_np, all_preds_np, average="macro", zero_division=0)
        train_recall = recall_score(all_labels_np, all_preds_np, average="macro", zero_division=0)
        train_f1 = f1_score(all_labels_np, all_preds_np, average="macro", zero_division=0)
        train_hamming = hamming_loss(all_labels_np, all_preds_np)
        train_accuracy = jaccard_score(all_labels_np, all_preds_np, average="samples")

        # Compute confusion matrix-based metrics for specificity and overall accuracy
        cm = multilabel_confusion_matrix(all_labels_np, all_preds_np)
        tp = cm[:, 1, 1]
        tn = cm[:, 0, 0]
        fp = cm[:, 0, 1]
        fn = cm[:, 1, 0]

        train_specificity = np.divide(tn, tn + fp, out=np.zeros_like(tn, dtype=float), where=(tn + fp) != 0)
        train_overall_accuracy = np.divide(tp + tn, tp + tn + fp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + tn + fp + fn) != 0)

        avg_train_specificity = np.nanmean(train_specificity)
        avg_train_overall_accuracy = np.nanmean(train_overall_accuracy)

        # Store metrics for the training phase
        train_precisions.append(train_precision)
        train_recalls.append(train_recall)
        train_f1s.append(train_f1)
        train_hammings.append(train_hamming)
        train_accuracies.append(train_accuracy)
        train_specificities.append(avg_train_specificity)
        train_overall_accuracies.append(avg_train_overall_accuracy)

        # Validation metrics
        val_loss, val_accuracy, val_overall_accuracy, val_precision, val_recall, val_f1, val_hamming, val_specificity = evaluate_model(model, val_loader, criterion)
        val_losses.append(val_loss)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        val_f1s.append(val_f1)
        val_hammings.append(val_hamming)
        val_accuracies.append(val_accuracy)
        val_specificities.append(val_specificity)
        val_overall_accuracies.append(val_overall_accuracy)

        if scheduler:
            scheduler.step(val_loss)   # For type ReduceLROnPlateau

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            # Save the best model
            torch.save(model.state_dict(), f"best_model-{MODEL_TYPE}.pth")

        epoch_duration = time.time() - epoch_start_time
        print(f'\n\nEpoch {epoch + 1} Summary:')
        print(f'  Training Loss: {train_loss:.4f}, Accuracy (Jaccard): {train_accuracy:.4f}, Overall Accuracy: {avg_train_overall_accuracy:.4f}, Precision: {train_precision:.4f}, Recall (Sensitivity): {train_recall:.4f}, Specificity: {avg_train_specificity:.4f}, F1-score: {train_f1:.4f}')
        print(f'  Validation Loss: {val_loss:.4f}, Accuracy (Jaccard): {val_accuracy:.4f}, Overall Accuracy: {val_overall_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, Specificity: {val_specificity:.4f}, F1-score: {val_f1:.4f}')
        print(f'  Time taken: {epoch_duration:.2f} seconds')

    # =======================
    # Testing Phase
    # =======================

    # Load the best model
    model.load_state_dict(torch.load(f"best_model-{MODEL_TYPE}.pth", map_location=device)) # map_location=device parameter ensures compatibility across devices

    test_loss, test_accuracy, test_overall_accuracy, test_precision, test_recall, test_f1, test_hamming, test_specificity, test_preds, test_labels, test_outputs = evaluate_model(model, test_loader, criterion, return_preds=True, return_outputs=True)

    print(f'\nTest Loss: {test_loss:.4f}, Accuracy (Jaccard): {test_accuracy:.4f}, Overall Accuracy: {test_overall_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, Specificity: {test_specificity:.4f}, F1-score: {test_f1:.4f}')
    print(f'\nBest validation F1-score: {best_val_f1:.4f}, at epoch: {best_epoch}')

    # Classification Report
    class_report = classification_report(
        test_labels, test_preds, target_names=class_names, zero_division=0
    )
    print("\nClassification Report:\n")
    print(class_report)

    # =======================
    # Plotting Metrics
    # =======================

    # Plotting Metrics after training and testing phases
    plot_all_metrics(
        train_losses, val_losses,
        train_accuracies, val_accuracies,
        train_overall_accuracies, val_overall_accuracies,
        train_precisions, val_precisions,
        train_recalls, val_recalls,
        train_f1s, val_f1s,
        best_val_f1, test_accuracy,
        test_loss, test_precision,
        test_recall, test_f1
    )
    
    # Confusion Matrix
    cm = multilabel_confusion_matrix(test_labels, test_preds)
    plot_combined_confusion_matrix(cm, class_names)
    
    plot_venn_diagram(test_labels, test_preds, class_names=class_names)
    
    # Identify misclassified samples
    misclassified_samples = identify_misclassified_samples(test_labels, test_preds, test_dataset.image_names)
    print(f'Number of misclassified samples: {len(misclassified_samples)}')

    # Plot confusion matrix multi label
    plot_confusion_matrix_multi_label(test_labels, test_preds, class_names)

# ================================
# 11. Plotting Functions
# ================================

def plot_confusion_matrix_multi_label(y_true, y_pred, class_names):
    """
    Plots a combined confusion matrix for a multi-label classification problem.
    
    Args:
        y_true (ndarray): Ground truth binary labels.
        y_pred (ndarray): Predicted binary labels.
        class_names (list): List of class names.
    """
    # Get the current date and time
    now = datetime.now()
    
    # Extract date and time, and format them
    date = now.strftime("%d-%m-%y")  # Date in "dd-mm-yy" format
    time = now.strftime("%H:%M:%S")  # Time in "hh:mm:ss" format
    
    num_classes = len(class_names)
    
    # Initialize the combined confusion matrix with zeros
    combined_cm = np.zeros((num_classes, num_classes), dtype=int)
    
    # Iterate through each example
    for true_labels, pred_labels in zip(y_true, y_pred):
        # Find the indices of true labels and predicted labels
        true_indices = np.where(true_labels == 1)[0]
        pred_indices = np.where(pred_labels == 1)[0]
        
        # Update the combined confusion matrix accordingly
        for true_idx in true_indices:
            for pred_idx in pred_indices:
                combined_cm[true_idx, pred_idx] += 1

    # Plot heatmap for the combined confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(combined_cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Combined Confusion Matrix for Multi-Label Classification, Dataset-{train_images_dir}_Model-{MODEL_TYPE}')
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.tight_layout()
    
    # Generate dynamic filename and save the plot
    filename = f'confusion_MultiLabelMatrix_Epochs-{NUM_EPOCHS}_ImgSize-{IMAGE_SIZE}_BatchSize-{BATCH_SIZE}_Dropout-{DROPOUT}_LR-{LEARNING_RATE}_model-{MODEL_TYPE}_date-{date}_time-{time}.png'
    plt.savefig(filename)
    plt.close()

    
    

def plot_confusion_matrix(cm, class_names, best_val_f1, test_accuracy, test_loss, test_precision, test_recall, test_f1):
    """
    Plot the confusion matrix using seaborn heatmap.

    Args:
        cm (ndarray): Confusion matrix.
        class_names (list): List of class names.
    """
    # Get the current date and time
    now = datetime.now()
    
    # Extract date and time, and format them
    date = now.strftime("%d-%m-%y")  # Date in "dd-mm-yy" format
    time = now.strftime("%H:%M:%S")  # Time in "hh:mm:ss" format
    
    # For multi-label confusion matrix, cm is a list of matrices
    for i, (confusion_matrix, class_name) in enumerate(zip(cm, class_names)):
        plt.figure(figsize=(4, 4))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f"Confusion Matrix for {class_name}")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        # Generate dynamic filename
        filename = f'confusion_matrix_{class_name}_Epochs-{NUM_EPOCHS}_ImgSize-{IMAGE_SIZE}_BatchSize-{BATCH_SIZE}_Dropout-{DROPOUT}_LR-{LEARNING_RATE}_ValF1-{best_val_f1:.2f}_Model-{MODEL_TYPE}_TestAcc-{test_accuracy:.2f}_TestLoss-{test_loss:.4f}_Prec-{test_precision:.2f}_Recall-{test_recall:.2f}_F1-{test_f1:.2f}_date-{date}_time-{time}.png'
        plt.savefig(filename)
        plt.close()






def plot_combined_confusion_matrix(cm, class_names):
    """
    Plots a combined confusion matrix by summing individual class confusion matrices.

    Args:
        cm (ndarray): Multilabel confusion matrices for each class.
        class_names (list): List of class names.
    """
    # Get the current date and time
    now = datetime.now()
    
    # Extract date and time, and format them
    date = now.strftime("%d-%m-%y")  # Date in "dd-mm-yy" format
    time = now.strftime("%H:%M:%S")  # Time in "hh:mm:ss" format
    
    combined_cm = np.sum(cm, axis=0)
    plt.figure(figsize=(6, 5))
    sns.heatmap(combined_cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title("Combined Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(f'Combined_ConfusionMatrix_model-{MODEL_TYPE}_Version-{VERSION}_date-{date}_time-{time}.png')
    plt.close()


    
def plot_venn_diagram(true_labels, pred_labels, class_names):
    """
    Plots Venn diagrams for all classes on the same canvas.

    Args:
        true_labels (ndarray): Ground truth binary labels.
        pred_labels (ndarray): Predicted binary labels.
        class_names (list): List of class names.
    """
    # Get the current date and time for unique file naming
    now = datetime.now()
    date = now.strftime("%d-%m-%y")  # Date in "dd-mm-yy" format
    time = now.strftime("%H:%M:%S")  # Time in "hh:mm:ss" format

    num_classes = len(class_names)
    cols = 4  # Number of columns in the grid
    rows = (num_classes + cols - 1) // cols  # Calculate rows needed for the grid
    
    plt.figure(figsize=(20, 15))  # Adjust size as needed

    for class_idx in range(num_classes):
        true_set = set(np.where(true_labels[:, class_idx] == 1)[0])
        pred_set = set(np.where(pred_labels[:, class_idx] == 1)[0])

        plt.subplot(rows, cols, class_idx + 1)  # Create subplot
        venn2([true_set, pred_set], set_labels=('True', 'Predicted'))
        plt.title(class_names[class_idx])

    # Adjust layout and save the plot
    plt.tight_layout()
    filename = f'Venn_Diagrams_AllClasses_model-{MODEL_TYPE}_Version-{VERSION}_date-{date}_time-{time}.png'
    plt.savefig(filename)
    plt.close()



def plot_all_metrics(
    train_losses, val_losses,
    train_accuracies, val_accuracies,
    train_overall_accuracies, val_overall_accuracies,
    train_precisions, val_precisions,
    train_recalls, val_recalls,
    train_f1s, val_f1s,
    best_val_f1, test_accuracy,
    test_loss, test_precision,
    test_recall, test_f1
):
    """
    Plot training and validation metrics.

    Args:
        train_losses (list): List of training losses.
        val_losses (list): List of validation losses.
        train_accuracies (list): List of training accuracies.
        val_accuracies (list): List of validation accuracies.
        train_overall_accuracies (list): List of training overall accuracies.
        val_overall_accuracies (list): List of validation overall accuracies.
        train_precisions (list): List of training precisions.
        val_precisions (list): List of validation precisions.
        train_recalls (list): List of training recalls.
        val_recalls (list): List of validation recalls.
        train_f1s (list): List of training F1-scores.
        val_f1s (list): List of validation F1-scores.
        best_val_f1 (float): Best validation F1-score.
        test_accuracy (float): Test accuracy.
        test_loss (float): Test loss.
        test_precision (float): Test precision.
        test_recall (float): Test recall.
        test_f1 (float): Test F1-score.
    """
    # Get the current date and time
    now = datetime.now()
    
    # Extract date and time, and format them
    date = now.strftime("%d-%m-%y")  # Date in "dd-mm-yy" format
    time = now.strftime("%H:%M:%S")  # Time in "hh:mm:ss" format

    
    epochs_range = range(1, len(train_losses) + 1)
    
    # First Plot: Loss, Precision, Recall, F1-score
    fig, ax1 = plt.subplots(figsize=(12, 8))

    color = 'tab:blue'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs_range, train_losses, label='Training Loss', color=color, linestyle='-')
    ax1.plot(epochs_range, val_losses, label='Validation Loss', color=color, linestyle='--')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis

    color = 'tab:red'
    ax2.set_ylabel('Metrics', color=color)
    ax2.plot(epochs_range, train_precisions, label='Training Precision', color='green', linestyle='-')
    ax2.plot(epochs_range, val_precisions, label='Validation Precision', color='green', linestyle='--')
    ax2.plot(epochs_range, train_recalls, label='Training Recall', color='orange', linestyle='-')
    ax2.plot(epochs_range, val_recalls, label='Validation Recall', color='orange', linestyle='--')
    ax2.plot(epochs_range, train_f1s, label='Training F1-score', color=color, linestyle='-')
    ax2.plot(epochs_range, val_f1s, label='Validation F1-score', color=color, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)

    # Combine legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

    plt.title(f'Loss and Metrics over Epochs, Dataset-{train_images_dir}_Model-{MODEL_TYPE}')
    plt.tight_layout()
    # Generate dynamic filename
    filename = f'loss_and_metrics_Epochs-{NUM_EPOCHS}_ImgSize-{IMAGE_SIZE}_BatchSize-{BATCH_SIZE}_Dropout-{DROPOUT}_LR-{LEARNING_RATE}_ValF1-{best_val_f1:.2f}_Model-{MODEL_TYPE}_Version-{VERSION}_TestAcc-{test_accuracy:.3f}_TestLoss-{test_loss:.4f}_Prec-{test_precision:.2f}_Recall-{test_recall:.2f}_F1-{test_f1:.2f}_date-{date}_time-{time}.png'
    plt.savefig(filename)
    plt.close()

    # Second Plot: Jaccard Accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(epochs_range, train_accuracies, label='Training Jaccard Accuracy', color='purple', linestyle='-')
    plt.plot(epochs_range, val_accuracies, label='Validation Jaccard Accuracy', color='purple', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Jaccard Accuracy')
    plt.title(f'Training and Validation Jaccard Accuracy, Dataset-{train_images_dir}_Model-{MODEL_TYPE}')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    # Generate dynamic filename
    filename = f'jaccard_accuracy_Epochs-{NUM_EPOCHS}_ImgSize-{IMAGE_SIZE}_BatchSize-{BATCH_SIZE}_Dropout-{DROPOUT}_LR-{LEARNING_RATE}_ValF1-{best_val_f1:.2f}_Model-{MODEL_TYPE}_Version-{VERSION}_TestAcc-{test_accuracy:.3f}_TestLoss-{test_loss:.4f}_Prec-{test_precision:.2f}_Recall-{test_recall:.2f}_F1-{test_f1:.2f}_date-{date}_time-{time}.png'
    plt.savefig(filename)
    plt.close()

    # Third Plot: Overall Accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(epochs_range, train_overall_accuracies, label='Training Overall Accuracy', color='blue', linestyle='-')
    plt.plot(epochs_range, val_overall_accuracies, label='Validation Overall Accuracy', color='blue', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Overall Accuracy')
    plt.title(f'Training and Validation Overall Accuracy, Dataset-{train_images_dir}_Model-{MODEL_TYPE}')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    # Generate dynamic filename
    filename = f'overall_accuracy_Epochs-{NUM_EPOCHS}_ImgSize-{IMAGE_SIZE}_BatchSize-{BATCH_SIZE}_Dropout-{DROPOUT}_LR-{LEARNING_RATE}_ValF1-{best_val_f1:.2f}_Model-{MODEL_TYPE}_Version-{VERSION}_TestAcc-{test_accuracy:.3f}_TestLoss-{test_loss:.4f}_Prec-{test_precision:.2f}_Recall-{test_recall:.2f}_F1-{test_f1:.2f}_date-{date}_time-{time}.png'
    plt.savefig(filename)
    plt.close()



# ================================
# 12. Run the Training Process
# ================================

if __name__ == "__main__":
    print("\nStarting the training process...")
    train_model_with_scheduler(model, train_loader, val_loader, criterion, optimizer, scheduler)
