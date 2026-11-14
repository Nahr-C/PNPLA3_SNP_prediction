import os
import cv2
import logging
import pandas as pd
import numpy as np
import SimpleITK as sitk

from PIL import Image
from torch.utils.data import Dataset

from sklearn.model_selection import StratifiedKFold, train_test_split

class VitDataset(Dataset):
    def __init__(self, img_id_list, img_folder, mask_folder, csv_file, snp='rs738409_G', transform=None):
        # Initialize the dataset with image ids, image and mask folder paths, CSV file, SNP type, and transformations
        self.img_id_list = img_id_list
        self.img_folder = img_folder
        self.mask_folder = mask_folder
        self.csv_data = pd.read_csv(csv_file)
        self.snp = snp
        self.transform = transform

        # Filter out samples with missing labels from the CSV file
        valid_ids = []
        for img_id in self.img_id_list:
            label = self.csv_data.loc[self.csv_data['ID'] == int(img_id), self.snp].values
            if len(label) > 0 and not pd.isna(label[0]):  # Keep only valid samples with non-missing labels
                valid_ids.append(img_id)
        self.img_id_list = valid_ids

    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.img_id_list)

    def __getitem__(self, idx):
        # Get the image ID at the specified index
        img_id = self.img_id_list[idx]

        # Construct the full image path
        img_path = os.path.join(self.img_folder, f'{img_id}_6.npy')

        image = np.load(img_path)
        image_array= image.astype(np.uint8)

        # # Read the image using OpenCV
        # image = cv2.imread(img_path)
        # # Add padding (12 pixels) to the top and bottom of the image to create a uniform size
        # top_padding = np.zeros((12, image.shape[1], image.shape[2]), dtype=np.uint8)
        # bottom_padding = np.zeros((12, image.shape[1], image.shape[2]), dtype=np.uint8)
        # image_array = np.vstack((top_padding, image, bottom_padding))

        # Construct the full path for the mask
        mask_path = os.path.join(self.mask_folder, f"{img_id}_mask.nii.gz")
        # Read the mask using SimpleITK
        mask = sitk.ReadImage(mask_path)
        mask_array = sitk.GetArrayFromImage(mask).astype(np.uint8)

        # Apply the mask to the image (mask out irrelevant parts)
        image = cv2.bitwise_and(image_array, image_array, mask=mask_array)

        # Find the bounding box of the liver region (non-zero pixels in the mask)
        coords = np.column_stack(np.where(mask_array > 0))
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        # Extract the region of interest (liver area)
        liver_image = image[y_min:y_max+1, x_min:x_max+1]

        # Make the bounding box square by adding padding if necessary
        height, width, _ = liver_image.shape
        if height > width:
            padding = (height - width) // 2
            liver_image = np.pad(liver_image, ((0, 0), (padding, padding), (0, 0)), mode='constant', constant_values=0)
        else:
            padding = (width - height) // 2
            liver_image = np.pad(liver_image, ((padding, padding), (0, 0), (0, 0)), mode='constant', constant_values=0)

        # Resize the image to 224x224 pixels using cubic interpolation
        liver_image = cv2.resize(liver_image, (224, 224), interpolation=cv2.INTER_CUBIC)

        # Retrieve the label from the CSV file (if SNP value is 2, map it to 1)
        label = self.csv_data.loc[self.csv_data['ID'] == int(img_id), self.snp].values[0]
        if label == 2:
            label = 1

        # Convert the numpy image array to a PIL image for transformations
        image = Image.fromarray(liver_image)
        if self.transform:
            image = self.transform(image)

        # Return the transformed image, its label, and the image ID
        return image, label, img_id


def build_cohort_CV(img_folder, csv_path, fold, snp='rs738409_G', type='Full'):
    """
    This function builds a cohort for cross-validation (CV) based on images and metadata from a CSV file.
    The cohort is filtered based on a specific SNP value and categorized by different criteria (e.g., Obese, Male, Age).
    The function also performs a stratified split to create training and testing datasets, and generates cross-validation folds.

    Args:
        img_folder (str): The path to the folder containing the image files.
        csv_path (str): The path to the CSV file containing metadata information.
        fold (int): The number of folds for cross-validation.
        snp (str, optional): The SNP column name in the CSV to filter the cohort by. Default is 'rs738409_G'.
        type (str, optional): The type of cohort (e.g., 'Obese', 'Male', 'Age_young'). Default is 'Full'.
    
    Returns:
        tuple: A tuple containing:
            - train_list (list): A list of image IDs for the training set.
            - test_list (list): A list of image IDs for the test set.
            - folds (list): A list of tuples representing the cross-validation folds (train indices, validation indices).
    """
    
    # Read the CSV file and filter rows where the SNP value is missing
    df = pd.read_csv(csv_path).dropna(subset=[snp])
    
    # Get the list of image filenames in the image folder
    img_list = [f for f in os.listdir(img_folder) if f.endswith('.npy')]
    
    logging.info('\n')
    logging.info(f'Build cohort: {type}')
    
    img_list_processed = []  # List to store processed image IDs
    labels = []  # List to store labels for the images
    
    num_absent = 0  # Counter for images not found in the CSV
    
    # Iterate through each image and process based on criteria
    for img_name in img_list:
        img_id = img_name.split('.')[0]
        img_id = img_id.split('_')[0]

        # Check if the image ID is in the CSV
        if int(img_id) not in df['ID'].values:
            num_absent += 1
            continue

        # Retrieve metadata for the image from the CSV
        snp_value = df.loc[df['ID'] == int(img_id), snp].values[0]   
        bmi_value = None
        sex_value = None
        age_value = None

        if 'BMI' in df.columns:
            bmi_value = df.loc[df['ID'] == int(img_id), 'BMI'].values[0]

        if 'SEX' in df.columns:
            sex_value = df.loc[df['ID'] == int(img_id), 'SEX'].values[0]

        if 'AGE' in df.columns:
            age_value = df.loc[df['ID'] == int(img_id), 'AGE'].values[0]

        # Filter images based on the given criteria (e.g., 'Obese', 'Male', 'Age_young', etc.)
        if snp_value in [0, 2] and img_id not in img_list_processed:
            if type == 'Obese' and bmi_value >= 30:
                img_list_processed.append(img_id)
                labels.append((snp_value // 2))  # 0 -> 0, 2 -> 1
            elif type == 'Notobese' and bmi_value < 30:
                img_list_processed.append(img_id)
                labels.append(snp_value // 2)  # 0 -> 0, 2 -> 1

            elif type == 'Male' and sex_value == 1:
                img_list_processed.append(img_id)
                labels.append(snp_value // 2)  # 0 -> 0, 2 -> 1
            elif type == 'Female' and sex_value == 0:
                img_list_processed.append(img_id)
                labels.append(snp_value // 2)  # 0 -> 0, 2 -> 1

            elif type == 'Age_young' and age_value <= 50:
                img_list_processed.append(img_id)
                labels.append(snp_value // 2)  # 0 -> 0, 2 -> 1
            elif type == 'Age_medium' and 51 <= age_value <= 60:
                img_list_processed.append(img_id)
                labels.append(snp_value // 2)  # 0 -> 0, 2 -> 
            elif type == 'Age_old' and age_value > 60:
                img_list_processed.append(img_id)
                labels.append(snp_value // 2)  # 0 -> 0, 2 -> 
                
            elif type == 'Full':
                img_list_processed.append(img_id)
                labels.append(snp_value // 2)  # 0 -> 0, 2 -> 1

    logging.info(f"Number of absent images in csv file: {num_absent/6}")

    if not img_list_processed:
        logging.info("No images processed, check your filters or data.")
        return [], [], []

    # Split the dataset into training and testing sets using stratified sampling
    train_list, test_list, train_labels, test_labels = train_test_split(
        img_list_processed, labels, test_size=0.15, stratify=labels, random_state=42)

    # Create the cross-validation folds using StratifiedKFold
    skf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=42)
    folds = [(train_idx, val_idx) for train_idx, val_idx in skf.split(train_list, train_labels)]
    
    return train_list, test_list, folds
