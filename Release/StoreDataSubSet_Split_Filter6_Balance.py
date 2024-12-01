import os
import shutil
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
import random

# ================================
# 1. Configuration
# ================================

# File paths
data_csv = './Data_Entry_2017.csv'
image_dir = '/home/roberh18/IKT450/Project__Chest_X _ay_CNN_model/ChestX-ray8_images/images'

# Reduction factor (No reduction of dataset here)
REDUCTION_FACTOR = 1  # Set this to the desired factor (e.g., 2 for half, 4 for a quarter)
NO_FINDING_REDUCTION_FACTOR = 4  # Reduction factor for "No Finding" images (e.g., keep 1/4 of them)

# Create directories for storing dataset splits (70/15/15)
output_dirs = {
    "train": "./train_images_filtered_80_10_10",
    "val": "./val_images_filtered_80_10_10",
    "test": "./test_images_filtered_80_10_10"
}

# Create output directories if they don't exist
for dir_name in output_dirs.values():
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

# ================================
# 2. Data Preparation
# ================================

# Load the metadata CSV
data_df = pd.read_csv(data_csv)

# Process 'Finding Labels' to get individual labels
data_df['Labels'] = data_df['Finding Labels'].str.split('|')

# Compute 'LabelCount' for each image
data_df['LabelCount'] = data_df['Labels'].apply(len)

# Flatten all labels into a single list
all_labels = data_df['Labels'].explode()

# Count the frequency of each label
label_counts = all_labels.value_counts()

# Get the 6 most represented labels
top_6_labels = label_counts.nlargest(6).index.tolist()
print("\nTop 6 labels:", top_6_labels)

# Function to check if all labels in the list are in top_6_labels
def all_labels_in_top(label_list):
    return all(label in top_6_labels for label in label_list)

# Filter the DataFrame to only include images where all labels are in the top 6 labels and have up to 3 labels
data_df['AllLabelsInTop'] = data_df['Labels'].apply(all_labels_in_top)
filtered_df = data_df[data_df['AllLabelsInTop'] & (data_df['LabelCount'] <= 3)]

# Calculate the counts of each label in the filtered dataset
label_counts_filtered = filtered_df['Labels'].explode().value_counts()
print("\nLabel counts in filtered dataset:")
print(label_counts_filtered)

# ================================
# 2.1 Reduce "No Finding" Images
# ================================

# Separate "No Finding" images from others
no_finding_df = filtered_df[filtered_df['Labels'].apply(lambda x: 'No Finding' in x)]
other_labels_df = filtered_df[~filtered_df['Labels'].apply(lambda x: 'No Finding' in x)]

# Randomly sample a fraction of "No Finding" images
reduced_no_finding_df = no_finding_df.sample(frac=1.0 / NO_FINDING_REDUCTION_FACTOR, random_state=42)

# Combine the reduced "No Finding" images with the other labels
balanced_filtered_df = pd.concat([reduced_no_finding_df, other_labels_df])

# Shuffle the dataset for good measure
balanced_filtered_df = balanced_filtered_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

# Recalculate label counts after balancing
balanced_label_counts = balanced_filtered_df['Labels'].explode().value_counts()
print("\nBalanced label counts in dataset:")
print(balanced_label_counts)

# ================================
# 2.2 Split Dataset into Train/Validation/Test
# ================================

# Split the balanced_filtered_df into train, val, and test sets

# First split into train and temp (70% train, 30% temp)
train_df, temp_df = train_test_split(
    balanced_filtered_df,
    test_size=0.2,
    random_state=42,
    stratify=balanced_filtered_df['LabelCount']
)

# Then split temp into val and test (each 15% of total data)
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42,
    stratify=temp_df['LabelCount']
)

# Optional: Print label counts and label combination counts in each split
def print_split_statistics(split_name, df):
    print(f"\n{split_name} Set Label Counts:")
    split_label_counts = df['Labels'].explode().value_counts()
    print(split_label_counts)
    
    label_count_dist = df['LabelCount'].value_counts().sort_index()
    print(f"\nNumber of images with N labels in {split_name} set:")
    print(label_count_dist)

print_split_statistics("Training", train_df)
print_split_statistics("Validation", val_df)
print_split_statistics("Test", test_df)

# ================================
# 3. Copy Images with Progress Indication
# ================================

def copy_images(data_frame, destination_folder):
    total_files = len(data_frame)
    print(f"\nCopying {total_files} images to {destination_folder}...")

    for idx, (_, row) in enumerate(data_frame.iterrows(), 1):
        img_name = row['Image Index']
        source_path = os.path.join(image_dir, img_name)
        destination_path = os.path.join(destination_folder, img_name)
        
        if os.path.exists(source_path):
            shutil.copy(source_path, destination_path)
        
        # Print progress every 1000 images
        if idx % 1000 == 0 or idx == total_files:
            print(f"Copied {idx}/{total_files} images")

# Copy images to respective directories
copy_images(train_df, output_dirs["train"])
copy_images(val_df, output_dirs["val"])
copy_images(test_df, output_dirs["test"])

# ================================
# 4. Print Summary
# ================================

train_samples = len(os.listdir(output_dirs["train"]))
val_samples = len(os.listdir(output_dirs["val"]))
test_samples = len(os.listdir(output_dirs["test"]))

print(f"\nDataset Summary After Copying Images:")
print(f"Train samples: {train_samples}")
print(f"Validation samples: {val_samples}")
print(f"Test samples: {test_samples}")

sys.exit("Stopping the program here.")




#OUTPUT:
'''
roberh18@jupyter-roberh18:~/IKT450/Project__Chest_X _ay_CNN_model$ python IKT450_Project_StoreDataSubSet_split_Filter6_UltraBalance_1.4.py

Top 6 labels: ['No Finding', 'Infiltration', 'Effusion', 'Atelectasis', 'Nodule', 'Mass']

Label counts in filtered dataset:
Labels
No Finding      60361
Infiltration    14885
Effusion         8502
Atelectasis      8113
Nodule           4743
Mass             3929
Name: count, dtype: int64

Balanced label counts in dataset:
Labels
No Finding      15090
Infiltration    14885
Effusion         8502
Atelectasis      8113
Nodule           4743
Mass             3929
Name: count, dtype: int64

Training Set Label Counts:
Labels
No Finding      12064
Infiltration    11929
Effusion         6796
Atelectasis      6513
Nodule           3780
Mass             3127
Name: count, dtype: int64

Number of images with N labels in Training set:
LabelCount
1    30120
2     5438
3     1071
Name: count, dtype: int64

Validation Set Label Counts:
Labels
No Finding      1512
Infiltration    1471
Effusion         861
Atelectasis      805
Nodule           468
Mass             410
Name: count, dtype: int64

Number of images with N labels in Validation set:
LabelCount
1    3765
2     680
3     134
Name: count, dtype: int64

Test Set Label Counts:
Labels
No Finding      1514
Infiltration    1485
Effusion         845
Atelectasis      795
Nodule           495
Mass             392
Name: count, dtype: int64

Number of images with N labels in Test set:
LabelCount
1    3766
2     679
3     134
Name: count, dtype: int64

Copying 36629 images to ./train_images_filtered_80_10_10...
Copied 1000/36629 images
Copied 2000/36629 images
Copied 3000/36629 images
Copied 4000/36629 images
Copied 5000/36629 images
Copied 6000/36629 images
Copied 7000/36629 images
Copied 8000/36629 images
Copied 9000/36629 images
Copied 10000/36629 images
Copied 11000/36629 images
Copied 12000/36629 images
Copied 13000/36629 images
Copied 14000/36629 images
Copied 15000/36629 images
Copied 16000/36629 images
Copied 17000/36629 images
Copied 18000/36629 images
Copied 19000/36629 images
Copied 20000/36629 images
Copied 21000/36629 images
Copied 22000/36629 images
Copied 23000/36629 images
Copied 24000/36629 images
Copied 25000/36629 images
Copied 26000/36629 images
Copied 27000/36629 images
Copied 28000/36629 images
Copied 29000/36629 images
Copied 30000/36629 images
Copied 31000/36629 images
Copied 32000/36629 images
Copied 33000/36629 images
Copied 34000/36629 images
Copied 35000/36629 images
Copied 36000/36629 images
Copied 36629/36629 images

Copying 4579 images to ./val_images_filtered_80_10_10...
Copied 1000/4579 images
Copied 2000/4579 images
Copied 3000/4579 images
Copied 4000/4579 images
Copied 4579/4579 images

Copying 4579 images to ./test_images_filtered_80_10_10...
Copied 1000/4579 images
Copied 2000/4579 images
Copied 3000/4579 images
Copied 4000/4579 images
Copied 4579/4579 images

Dataset Summary After Copying Images:
Train samples: 36629
Validation samples: 4579
Test samples: 4579
'''
