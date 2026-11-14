import os
import shutil
import random
import logging

from scipy.io import loadmat
from torch.utils.data import Dataset

from resnet_utils import read_subfolder, parse_augmentations, ensure_dir

# Custom dataset for training process
class MyResnet(Dataset):
    def __init__(self, data_dir, source_dir=None, mat_list_0=None, mat_list_1=None, transform=None):
        # Initialize the dataset for training
        self.data_dir = data_dir
        self.source_dir = source_dir
        self.mat_list_0 = mat_list_0
        self.mat_list_1 = mat_list_1
        self.transform = transform

        # Load all the data from the main directory
        self.data = []
        self._load_data_from_dir()

        # Check if we want to train the model with extra images with label 1
        if self.mat_list_1:
            self._load_additional_data(self.mat_list_1, label=1)
            logging.info(f'{len(self.mat_list_1)} additional artefact samples loaded.')

        # Check if we want to train the model with extra images with both labels
        if self.mat_list_0:
            self._load_additional_data(self.mat_list_0, label=0)
            logging.info(f'{len(self.mat_list_0)} additional correct samples loaded.\n')

    def _load_data_from_dir(self):
        # Helper method to load data from the main directory
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.mat'):
                mat_file = loadmat(os.path.join(self.data_dir, filename))
                water = mat_file['outputImages']['water'][0, 0]
                fat = mat_file['outputImages']['fat'][0, 0]
                epsilon = 1e-10  # To avoid division by zero
                ff = fat / (fat + water + epsilon)
                wf = 1 - ff  # Calculate water fraction
                self.data.append((ff, 0))  # Label 0 for fat fraction
                self.data.append((wf, 1))  # Label 1 for water fraction

    def _load_additional_data(self, mat_list, label):
        # Helper method to load additional data based on the provided mat list
        for mat_file_name in mat_list:
            mat_file_path = os.path.join(self.source_dir, f'{mat_file_name}.mat')
            mat_file = loadmat(mat_file_path)
            water = mat_file['outputImages']['water'][0, 0]
            fat = mat_file['outputImages']['fat'][0, 0]
            ff = fat / (water + fat + 0.0000000001)  # Calculate fat fraction
            self.data.append((ff, label))  # Append data with the specified label

    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieve the image and label at the specified index
        img, label = self.data[idx]
        if self.transform:
            img = self.transform(img)
            img = img.expand(3, -1, -1)  # Ensure 3 channels for the image
        return img, label
    
# Custom dataset for inference process
class MyResnetInference(Dataset):
    def __init__(self, data_dir, transform=None):
        # Initialize the dataset for inference
        self.data_dir = data_dir
        self.transform = transform
        self.file_list = os.listdir(data_dir)

    def _load_mat_data(self, file_path):
        # Helper method to load and process the .mat file data
        mat_data = loadmat(file_path)
        fat = mat_data['outputImages']['fat'][0, 0]
        water = mat_data['outputImages']['water'][0, 0]
        epsilon = 1e-10  # To avoid division by zero
        fat_fraction = fat / (fat + water + epsilon)
        return fat_fraction
    
    def __len__(self):
        # Return the total number of files in the dataset
        return len(self.file_list)

    def __getitem__(self, index):
        # Retrieve the processed fat fraction image at the specified index
        file_name = self.file_list[index]
        file_path = os.path.join(self.data_dir, file_name)

        # Load the .mat file data
        fat_fraction = self._load_mat_data(file_path)
        
        # Apply the transform if available
        if self.transform:
            fat_fraction_image = self.transform(fat_fraction)
            fat_fraction_image = fat_fraction_image.float()      
            fat_fraction_image = fat_fraction_image.expand(3, -1, -1)  # Ensure 3 channels for the image

        return file_name, fat_fraction_image
    
def check_id_existence(folder_path_1, folder_path_2):
    """
    Check if the IDs in folder_path_1 exist in folder_path_2.

    Args:
        folder_path_1 (str): Path to the first folder.
        folder_path_2 (str): Path to the second folder.

    Returns:
        list: List of IDs that exist in both folders.
    """
    ids_1 = set(os.path.splitext(f)[0] for f in os.listdir(folder_path_1) if f.endswith('.png'))
    ids_2 = set(os.path.splitext(f)[0] for f in os.listdir(folder_path_2) if f.endswith('.png'))

    # Find common IDs between the two folders
    common_ids = list(ids_1.intersection(ids_2))
    logging.info(f'Found {len(common_ids)} common IDs between the two folders.')

    return common_ids

def remove_common_ids(folder_path, common_ids):
    """
    Remove files from the folder whose filenames match the given IDs with a .png extension.

    Args:
        folder_path (str): Path to the folder.
        common_ids (list): List of IDs to remove.

    Returns:
        list: List of remaining .png file names after removal.
    """
    total_files = 0
    files_removed = 0
    remaining_files = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.png'):
            total_files += 1
            if os.path.splitext(file_name)[0] in common_ids:
                file_path = os.path.join(folder_path, file_name)
                os.remove(file_path)
                files_removed += 1
            else:
                remaining_files.append(file_name)

    files_left = total_files - files_removed
    logging.info(f'Removed {files_removed} common files compared to the last round.')
    logging.info(f'{files_left} new files left in the folder after removal.')

    return remaining_files

    
def copy_inferenced_artifact_samples(src_folder, dest_folder, common_ids):
    """
    Copy all .png files from the source folder to the destination folder.

    Args:
        src_folder (str): Path to the source folder containing .png files.
        dest_folder (str): Path to the destination folder.

    Returns:
        int: N
    """    
    png_files = remove_common_ids(src_folder, common_ids)

    # Copy each .png file to the destination folder
    for file_name in png_files:
        src_path = os.path.join(src_folder, file_name)
        dest_path = os.path.join(dest_folder, file_name)
        shutil.copy2(src_path, dest_path)

    return len(png_files)

def copy_inferenced_correct_samples(src_folder, dest_folder, num_samples, common_ids):
    """
    Collect image paths grouped by ID from all subfolders in the source folder,
    randomly sample a given number of IDs, and copy their images to the destination folder.

    Args:
        src_folder (str): Path to the main folder containing subfolders.
        dest_folder (str): Path to the destination folder.
        num_samples (int): Number of IDs to sample.

    Returns:
        int: Total number of images copied.
    """

    id_to_paths = {}

    # Traverse all subfolders and collect image paths grouped by ID
    for subfolder_name in os.listdir(src_folder):
        subfolder_path = os.path.join(src_folder, subfolder_name)

        # Remove common IDs from the subfolder
        remaining_files = remove_common_ids(subfolder_path, common_ids)

        for file_name in remaining_files:
            if file_name.endswith('.png'):
                # Extract ID (filename without extension)
                id_ = os.path.splitext(file_name)[0]

                full_path = os.path.join(subfolder_path, file_name)
                if id_ not in id_to_paths:
                    id_to_paths[id_] = []
                id_to_paths[id_].append(full_path)


    all_ids = list(id_to_paths.keys())

    # If available samples are fewer than required, take all IDs
    if len(all_ids) < num_samples:
        sampled_ids = all_ids
    else:
        sampled_ids = random.sample(all_ids, num_samples)

    copied_count = 0

    # Copy images associated with the sampled IDs to the destination folder
    for id_ in sampled_ids:
        for src_path in id_to_paths[id_]:
            dest_path = os.path.join(dest_folder, os.path.basename(src_path))
            shutil.copy2(src_path, dest_path)
            copied_count += 1

    return copied_count

def create_datasets(config, source_dir, mode):
    """
    Create train/validation datasets for the training loop.

    Args:
        config (dict): Configuration dictionary containing paths and augmentation settings.

    Returns:
        tuple: (train_dataset, val_dataset) where train_dataset is the training dataset instance and val_dataset is the validation dataset instance.
    """
    # Paths for train/validation/full data
    train_path = config['data']['train_path']
    val_path = config['data']['validation_path']

    # Paths for artefact/correct folders
    artefact_path = os.path.join(config['data']['output'], "artefact_samples")
    correct_path = os.path.join(config['data']['output'], "correct_samples")
    ensure_dir(artefact_path)
    ensure_dir(correct_path)

    # Paths for inferenced artefact/correct folders
    inferenced_artefact_path = os.path.join(config['data']['output'], "inference_artefact")
    inferenced_correct_path = os.path.join(config['data']['output'], "inference_correct")

    # Copy samples from inferenced folder
    if mode == "Auto":

        if os.path.exists(inferenced_artefact_path):
            # Check for common IDs between artefact folders and inferenced folders
            common_artifact_ids = check_id_existence(artefact_path, inferenced_artefact_path)

            # Copy new inferenced artefact samples from the inference folder to the artefact path
            logging.info(f'Checking Artefact samples...')
            num_artifact = copy_inferenced_artifact_samples(inferenced_artefact_path, 
                                                            artefact_path,
                                                            common_artifact_ids)
            logging.info(f'Copied {num_artifact} artefact samples from inference folder.\n\n')

        if os.path.exists(inferenced_correct_path):
            # Check for common IDs between correct folders and inferenced folders
            common_correct_ids = check_id_existence(correct_path, inferenced_correct_path)

            # Copy correct samples from the inference folder to the correct path    
            logging.info(f'Checking Correct samples...')
            num_correct = copy_inferenced_correct_samples(inferenced_correct_path, 
                                                          correct_path, 
                                                          num_artifact,
                                                          common_correct_ids)
            logging.info(f'Copied {num_correct} correct samples from inference folder.\n\n')

    elif mode == "Manual":
        logging.info("Manual mode selected. Skipping automatic sample copy.\n\n")
    
    # Read all samples from the artefact and correct paths
    artefact_list = read_subfolder(artefact_path)
    correct_list = read_subfolder(correct_path)

    # Split off 10% of each for the validation set
    num_artifact_val = int(0.1 * len(artefact_list))
    num_correct_val = int(0.1 * len(correct_list))
    
    # Select random samples for validation (10%)
    artefact_val = random.sample(artefact_list, num_artifact_val)
    correct_val = random.sample(correct_list, num_correct_val)
    
    # Remove validation samples from the training list
    artefact_train = [item for item in artefact_list if item not in artefact_val]
    correct_train = [item for item in correct_list if item not in correct_val]

    # Data augmentation 
    transform_train = parse_augmentations(config['training']['augmentation'])
    transform_val = parse_augmentations(config['evaluation']['augmentation'])

    # Create training dataset instance
    logging.info("For training dataset:")
    train_dataset = MyResnet(
        train_path, 
        source_dir=source_dir, 
        mat_list_0=correct_train, 
        mat_list_1=artefact_train, 
        transform=transform_train
    )
    
    # Create validation dataset instance
    logging.info("For validation dataset:")
    val_dataset = MyResnet(
        val_path, 
        source_dir=source_dir, 
        mat_list_0=correct_val, 
        mat_list_1=artefact_val, 
        transform=transform_val
    )

    logging.info("Datasets created successfully.")
    logging.info(f"Train Dataset: {len(train_dataset)} samples")
    logging.info(f"Validation Dataset: {len(val_dataset)} samples")

    return train_dataset, val_dataset

