
1. Transfer all the files to Jupyterhub

2. Install necessary libraries / packages

3. Run every cell in "Download_dataset.ipynb"

4. Run the file "IKT450_Project_StoreDataSubSet_split_Filter6.py", image_dir filpath to the folder containing the dataset we downloaded should be in the working directory

5. You should now be able to run the classification model



Folder structure:
your_project/
│
├── config.py
├── data.py
├── models.py
├── train.py
├── evaluation.py
├── main.py
├── requirements.txt
├── models/                  # Directory to save trained models
├── logs/                    # Directory to save log files
├── ChestX-ray8_images/images
├── train_images_filtered_80_10_10/
├── val_images_filtered_80_10_10/
├── test_images_filtered_80_10_10/
├── Data_Entry_2017.csv
├── test_list.txt
├── train_val_list.txt
