import torch
import torch.nn as nn
from torchvision import models
import logging
import os
from config import (
    NUM_CLASSES,
    DROPOUT,
    FREEZE_LAYERS,
    DEVICE,
    MODEL_TYPE,
    MODEL_SAVE_PATH,
    LOG_PATH,
    FIRST_LAYER_SIZE
)


# =====================================#
# Logging Configuration for models.py  #
# =====================================#

logging.basicConfig(filename=os.path.join(LOG_PATH, 'models.log'), filemode='a', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)



# =================================#
# 1. Squeeze-and-Excitation Block  #
# =================================#

class SEBlock(nn.Module):
    '''
    Squeeze-and-Excitation (SE) Block.
    Enhances the representational power of a network by enabling it to perform dynamic channel-wise feature recalibration.

    Args:
        in_channels (int): Number of input channels.
        reduction (int): Reduction ratio for the bottleneck in the SE block. Default is 16.
    '''
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Squeeze
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()  # Excitation
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_avg_pool(x).view(b, c)  # Squeeze
        y = self.fc(y).view(b, c, 1, 1)        # Excitation
        return x * y.expand_as(x)              # Scale

# ===============================#
# 2. Custom Model Definitions    #
# ===============================#

class DeeperCNN_6(nn.Module):
    '''
    A deeper CNN model with 6 layers tailored for multi-label classification.
    '''
    def __init__(self):
        super(DeeperCNN_6, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(DROPOUT)

        # Convolutional Block 1
        self.conv1 = nn.Conv2d(FIRST_LAYER_SIZE, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)  # Reduces spatial dimensions

        # Convolutional Block 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Convolutional Block 3
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Convolutional Block 4
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2, 2)

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Output Layer
        self.fc = nn.Linear(512, NUM_CLASSES)

    def forward(self, x):
        # Block 1
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        # Block 2
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        # Block 3
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        # Block 4
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)

        # Global Average Pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten

        # Dropout
        x = self.dropout(x)

        # Output Layer
        x = self.fc(x)
        return x

class DeeperCNN_8(nn.Module):
    '''
    A deeper CNN model with 8 layers tailored for multi-label classification.
    '''
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
        x = self.relu(self.bn1_1(self.conv1_1(x)))
        x = self.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool1(x)

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
        x = x.view(x.size(0), -1)  # Flatten

        # Dropout
        x = self.dropout(x)

        # Output Layer
        x = self.fc(x)
        return x

class DeeperCNN_8_SEB(nn.Module):
    '''
    A deeper CNN model with 8 layers and Squeeze-and-Excitation (SE) blocks.
    '''
    def __init__(self):
        super(DeeperCNN_8_SEB, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(DROPOUT)

        # Convolutional Block 1
        self.conv1_1 = nn.Conv2d(FIRST_LAYER_SIZE, 64, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.se1 = SEBlock(64)  # SE Block
        self.pool1 = nn.MaxPool2d(2, 2)

        # Convolutional Block 2
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.se2 = SEBlock(128)  # SE Block
        self.pool2 = nn.MaxPool2d(2, 2)

        # Convolutional Block 3
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.se3 = SEBlock(256)  # SE Block
        self.pool3 = nn.MaxPool2d(2, 2)

        # Convolutional Block 4
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.se4 = SEBlock(512)  # SE Block
        self.pool4 = nn.MaxPool2d(2, 2)

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Output Layer
        self.fc = nn.Linear(512, NUM_CLASSES)

    def forward(self, x):
        # Block 1
        x = self.relu(self.bn1_1(self.conv1_1(x)))
        x = self.relu(self.bn1_2(self.conv1_2(x)))
        x = self.se1(x)  # SE Block
        x = self.pool1(x)

        # Block 2
        x = self.relu(self.bn2_1(self.conv2_1(x)))
        x = self.relu(self.bn2_2(self.conv2_2(x)))
        x = self.se2(x)  # SE Block
        x = self.pool2(x)

        # Block 3
        x = self.relu(self.bn3_1(self.conv3_1(x)))
        x = self.relu(self.bn3_2(self.conv3_2(x)))
        x = self.se3(x)  # SE Block
        x = self.pool3(x)

        # Block 4
        x = self.relu(self.bn4_1(self.conv4_1(x)))
        x = self.relu(self.bn4_2(self.conv4_2(x)))
        x = self.se4(x)  # SE Block
        x = self.pool4(x)

        # Global Average Pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten

        # Dropout
        x = self.dropout(x)

        # Output Layer
        x = self.fc(x)
        return x

class DeeperCNN_8_Residual(nn.Module):
    '''
    A deeper CNN model with 8 layers and Residual connections.
    '''
    def __init__(self):
        super(DeeperCNN_8_Residual, self).__init__()
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
        x += residual  # Residual Connection (if dimensions match)
        x = self.pool1(x)

        # Block 2
        residual = x
        x = self.relu(self.bn2_1(self.conv2_1(x)))
        x = self.relu(self.bn2_2(self.conv2_2(x)))
        if residual.shape == x.shape:
            x += residual  # Residual Connection
        x = self.pool2(x)

        # Block 3
        residual = x
        x = self.relu(self.bn3_1(self.conv3_1(x)))
        x = self.relu(self.bn3_2(self.conv3_2(x)))
        if residual.shape == x.shape:
            x += residual  # Residual Connection
        x = self.pool3(x)

        # Block 4
        residual = x
        x = self.relu(self.bn4_1(self.conv4_1(x)))
        x = self.relu(self.bn4_2(self.conv4_2(x)))
        if residual.shape == x.shape:
            x += residual  # Residual Connection
        x = self.pool4(x)

        # Global Average Pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten

        # Dropout
        x = self.dropout(x)

        # Output Layer
        x = self.fc(x)
        return x

# =================================#
# 3. Pre-trained Model Definitions #
# =================================#

def add_se_to_resnet(model, reduction=16):
    '''
    Recursively adds SE blocks to all Bottleneck blocks in a ResNet model.

    Args:
        model (nn.Module): The ResNet model to modify.
        reduction (int): Reduction ratio for the SE blocks.

    Returns:
        nn.Module: Modified ResNet model with SE blocks.
    '''
    for name, module in model.named_children():
        if isinstance(module, models.resnet.Bottleneck):
            # Insert SE block after the second convolution
            se = SEBlock(module.conv2.out_channels, reduction)
            module.se = se
        else:
            add_se_to_resnet(module, reduction)
    return model

def get_pretrained_model(model_type):
    '''
    Retrieves and modifies a pre-trained model based on the specified type.

    Args:
        model_type (str): Type of the model to retrieve.

    Returns:
        nn.Module: Modified pre-trained model ready for training.
    '''
    if model_type == 'DenseNet121':
        model = models.densenet121(weights='IMAGENET1K_V1')
        num_features = model.classifier.in_features
        if FREEZE_LAYERS:
            for param in model.features.parameters():
                param.requires_grad = False
        # Modify classifier for multi-label classification
        model.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(512, NUM_CLASSES)
        )
        logging.info("Initialized DenseNet121 with modified classifier.")

    elif model_type == 'ResNet50':
        model = models.resnet50(weights='IMAGENET1K_V1')
        num_features = model.fc.in_features
        if FREEZE_LAYERS:
            for name, param in model.named_parameters():
                if 'layer4' not in name:
                    param.requires_grad = False
        # Modify fully connected layer for multi-label classification
        model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(512, NUM_CLASSES)
        )
        logging.info("Initialized ResNet50 with modified fully connected layers.")

    elif model_type == 'ResNet18':
        model = models.resnet18(weights='IMAGENET1K_V1')
        num_features = model.fc.in_features
        if FREEZE_LAYERS:
            for name, param in model.named_parameters():
                if 'layer4' not in name:
                    param.requires_grad = False
        # Modify fully connected layer for multi-label classification
        model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(512, NUM_CLASSES)
        )
        logging.info("Initialized ResNet18 with modified fully connected layers.")

    elif model_type == 'ResNet18_1':
        model = models.resnet18(weights='IMAGENET1K_V1')
        num_features = model.fc.in_features
        if FREEZE_LAYERS:
            for name, param in model.named_parameters():
                if 'layer4' not in name:
                    param.requires_grad = False
        # Modify classifier for multi-label classification with additional layers
        model.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(512, NUM_CLASSES)
        )
        logging.info("Initialized ResNet18_1 with extended fully connected layers.")

    elif model_type == 'ResNet18_2':
        model = models.resnet18(weights='IMAGENET1K_V1')
        num_features = model.fc.in_features
        if FREEZE_LAYERS:
            for name, param in model.named_parameters():
                if 'layer4' not in name:
                    param.requires_grad = False
        # Modify classifier for multi-label classification with more extended layers
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
            nn.Linear(512, NUM_CLASSES)
        )
        logging.info("Initialized ResNet18_2 with highly extended fully connected layers.")

    elif model_type == 'ResNet18_SEB_1':
        model = models.resnet18(weights='IMAGENET1K_V1')
        model = add_se_to_resnet(model, reduction=16)  # Add SE blocks with reduction ratio of 16

        num_features = model.fc.in_features
        if FREEZE_LAYERS:
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
            nn.Linear(512, NUM_CLASSES)
        )
        logging.info("Initialized ResNet18_SEB_1 with SE blocks and modified fully connected layers.")

    elif model_type == 'ResNet18_SEB_2':
        model = models.resnet18(weights='IMAGENET1K_V1')
        model = add_se_to_resnet(model, reduction=8)  # Add SE blocks with reduction ratio of 8

        num_features = model.fc.in_features
        if FREEZE_LAYERS:
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
            nn.Linear(512, NUM_CLASSES)
        )
        logging.info("Initialized ResNet18_SEB_2 with SE blocks and extended fully connected layers.")

    else:
        raise ValueError(f"Invalid MODEL_TYPE: {model_type}. Please choose a valid model type.")

    logging.info(f"Model {model_type} moved to device {DEVICE}.")
    return model.to(DEVICE)




def build_model(model_type, class_weights=None):
    '''
    Builds and returns the specified model based on the model type.

    Args:
        model_type (str): Type of the model to build.
        class_weights (torch.Tensor, optional): Tensor containing class weights for handling class imbalance.

    Returns:
        nn.Module: The constructed model.
    '''
    if model_type in ['DeeperCNN_6', 'DeeperCNN_8', 'DeeperCNN_8_SEB', 'DeeperCNN_8_Residual']:
        if model_type == 'DeeperCNN_6':
            model = DeeperCNN_6()
            logging.info("Initialized DeeperCNN_6 model.")
        elif model_type == 'DeeperCNN_8':
            model = DeeperCNN_8()
            logging.info("Initialized DeeperCNN_8 model.")
        elif model_type == 'DeeperCNN_8_SEB':
            model = DeeperCNN_8_SEB()
            logging.info("Initialized DeeperCNN_8_SEB model.")
        elif model_type == 'DeeperCNN_8_Residual':
            model = DeeperCNN_8_Residual()
            logging.info("Initialized DeeperCNN_8_Residual model.")
        else:
            raise ValueError(f"Unsupported custom model type: {model_type}")
        
        # Optionally freeze layers if required
        if FREEZE_LAYERS:
            for param in model.features.parameters():
                param.requires_grad = False
            logging.info("Frozen early layers of the custom model.")

        model = model.to(DEVICE)
        logging.info(f"Custom model {model_type} moved to device {DEVICE}.")

    elif model_type in ['DenseNet121', 'ResNet50', 'ResNet18', 'ResNet18_1', 'ResNet18_2', 'ResNet18_SEB_1', 'ResNet18_SEB_2']:
        model = get_pretrained_model(model_type)
    else:
        raise ValueError(f"MODEL_TYPE {model_type} is not supported. Please choose a valid model type.")

    return model



if __name__ == "__main__":
    # Example: Initialize a DeeperCNN_8_SEB model
    model = build_model('DeeperCNN_8_SEB')
    print(model)
