import os
import logging
import numpy as np
import SimpleITK as sitk

from monai.data import Dataset
from unet_utils import parse_augmentations

class LiverDataset(Dataset):
    def __init__(self, image_folder, mask_folder, transforms):
        # Store full paths of all images and masks
        self.image_paths = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder)]
        self.mask_paths = [os.path.join(mask_folder, fname) for fname in os.listdir(mask_folder)]
        self.transforms = transforms  # Data augmentation or preprocessing transformations

    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.image_paths)

    def __getitem__(self, index):
        # Load the image using SimpleITK
        image = sitk.ReadImage(self.image_paths[index])
        image_array = sitk.GetArrayFromImage(image)
        
        # Extract the 3rd slice (index 2) from the image volume
        image_ff = image_array[2, :, :]
        
        # Load the corresponding mask
        mask_image = sitk.ReadImage(self.mask_paths[index])
        mask_array = sitk.GetArrayFromImage(mask_image)
        mask = mask_array[:, :].astype(np.int32)  # Ensure the mask is of integer type

        # Create a dictionary to hold both image and mask
        data_dict = {'image': image_ff, 'label': mask}

        # Apply any specified transforms
        if self.transforms:
            transformed = self.transforms(data_dict)

        # Return the transformed image and mask as a tuple
        return transformed['image'], transformed['label']

    
def create_datasets(config):
    """Create train/val dataset for training loop."""
    
    # Path to the base dataset directory specified in the configuration
    folder_path = config['data']['train_path']
    
    # Keys used for identifying data entries in transformations
    keys = ['image', 'label']

    # Construct paths to training and validation image and mask folders
    train_image_folder = os.path.join(folder_path, "train/image")
    train_mask_folder = os.path.join(folder_path, "train/mask")
    val_image_folder = os.path.join(folder_path, "val/image")
    val_mask_folder = os.path.join(folder_path, "val/mask")
    
    # Parse data augmentation configurations for training and validation
    transform_train = parse_augmentations(config['training']['augmentation'], keys)
    transform_val = parse_augmentations(config['evaluation']['augmentation'], keys)
    
    logging.info('Start to create dataset...')

    # Instantiate the training dataset
    train_dataset = LiverDataset(train_image_folder, 
                                 train_mask_folder, 
                                 transform_train)
    
    # Instantiate the validation dataset
    val_dataset = LiverDataset(val_image_folder, 
                                val_mask_folder, 
                                transform_val)

    logging.info("Datasets created successfully.")

    # Return both training and validation datasets
    return train_dataset, val_dataset
