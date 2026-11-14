import os
import yaml
import torch
import shutil
import logging
import numpy as np
import pandas as pd
import scipy.io as sio
import SimpleITK as sitk
import matplotlib.pyplot as plt

from torchvision.transforms import v2

def load_config(config_path):
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def ensure_dir(directory):
    """
    Ensure that a directory exists, creating it if necessary.

    Args:
        directory (str): Directory path to check/create.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def parse_augmentations(augmentations):
    """
    Parse the list of augmentation configurations and return a composed transformation.

    Args:
        augmentations (list): List of strings, each representing an augmentation function (e.g., "RandFlip", "ScaleIntensity").

    Returns:
        torchvision.transforms.Compose: Composed transformation of the specified augmentations.
    """
    aug_list = []  # List to hold the actual augmentation functions
    
    # Iterate through the list of augmentation function names (strings)
    for aug in augmentations:
        # Use eval to dynamically call the augmentation function based on its string name
        aug_list.append(eval(aug))
    
    # Return a Compose transformation that applies all the augmentations in the list sequentially
    return v2.Compose(aug_list)

def id2list(path, list):
    """
    Populate a list with filenames (without extensions) of PNG files from a given directory.

    Args:
        path (str): The directory path to search for files.
        list (list): The list to append filenames to.

    Returns:
        list: The updated list containing filenames without their extensions.
    """
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        
        # Check if it's a file and ends with '.png' (case insensitive)
        if os.path.isfile(file_path) and filename.lower().endswith(".png"):
            # Extract the filename without extension
            name_without_extension = os.path.splitext(filename)[0]
            list.append(name_without_extension)
    
    return list

def read_subfolder(folder_path):
    """
    Read filenames from a subfolder and return a list of unique IDs.

    Args:
        folder_path (str): The name of the subfolder to read files from.

    Returns:
        list: A list of unique IDs extracted from the filenames in the subfolder.
    """
    
    # Initialize an empty list to store IDs
    id_list = []
    
    # Populate the list with unique IDs by reading filenames from the subfolder
    id_list = list(set(id2list(folder_path, id_list)))
    
    return id_list

def save_train_metrics(file_path, train_loss_list, train_acc_list, val_loss_list, val_acc_list, lr_values, batch_size, weight_decay):
    """
    Save training metrics to a CSV file.

    Args:
        file_path (str): Path to save the CSV file containing training metrics.
        train_loss_list (list): List of training losses per epoch.
        train_acc_list (list): List of training accuracies per epoch.
        val_loss_list (list): List of validation losses per epoch.
        val_acc_list (list): List of validation accuracies per epoch.
        lr_values (list): List of learning rate values used during training.
        batch_size (int): Batch size used during training.
        weight_decay (float): Weight decay value used during training.

    Returns:
        None: Saves the data to a CSV file.
    """
    # Create a dictionary with all the training metrics
    data = {
        'Epoch': range(1, len(train_loss_list) + 1),  # Epoch numbers
        'Train Loss': train_loss_list,  # Training losses per epoch
        'Train Acc': train_acc_list,  # Training accuracies per epoch
        'Val Loss': val_loss_list,  # Validation losses per epoch
        'Val Acc': val_acc_list,  # Validation accuracies per epoch
        'lr_values': lr_values,  # Learning rate values used during training
        'batch_size': batch_size,  # Batch size used in training
        'weight_decay': weight_decay  # Weight decay used in training
    }

    # Convert the dictionary into a pandas DataFrame
    df = pd.DataFrame(data)

    # Save the DataFrame to the specified file path
    df.to_csv(file_path, index=False)

def plot_train_metrics(train_loss_list, train_acc_list, val_loss_list, val_acc_list, output_dir):
    """
    Plot training and validation metrics (loss and accuracy) and save the plot as a PNG file.

    Args:
        train_loss_list (list): List of training losses per epoch.
        train_acc_list (list): List of training accuracies per epoch.
        val_loss_list (list): List of validation losses per epoch.
        val_acc_list (list): List of validation accuracies per epoch.
        output_dir (str): Directory to save the plot.

    Returns:
        None: Saves the plot as a PNG image to the specified directory.
    """
    epochs = range(1, len(train_loss_list) + 1)  # Number of epochs

    # Create a figure with two subplots (for loss and accuracy)
    plt.figure(figsize=(12, 4))

    # Plot the training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss_list, label='Train Loss')  
    plt.plot(epochs, val_loss_list, label='Val Loss')  
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot the training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc_list, label='Train Acc')  
    plt.plot(epochs, val_acc_list, label='Val Acc')  
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Adjust layout to avoid overlapping
    plt.tight_layout()

    # Save the plot as a PNG image
    image_file_path = os.path.join(output_dir, "acc_loss_plot.png")
    plt.savefig(image_file_path)  
    plt.show()  

    logging.info(f"Plot saved in: {image_file_path}") 

def save_train_results(output_path, model, train_loss_list, train_acc_list, val_loss_list, val_acc_list, lr_values, batch_size, weight_decay):
    """
    Save training results including metrics, plots, and the trained model.

    Args:
        output_path (str): Directory to save the results.
        model (torch.nn.Module): Trained model to be saved.
        train_loss_list (list): List of training losses for each epoch.
        train_acc_list (list): List of training accuracies for each epoch.
        val_loss_list (list): List of validation losses for each epoch.
        val_acc_list (list): List of validation accuracies for each epoch.
        lr_values (list): List of learning rate values used during training.
        batch_size (int): Batch size used during training.
        weight_decay (float): Weight decay used during training.

    Returns:
        None: Saves the model, metrics, and plots in the specified directory.
    """
    ensure_dir(output_path)  # Ensure the output directory exists

    # Save training metrics to a CSV file
    metrics_file_path = os.path.join(output_path, 'metrics.csv')
    save_train_metrics(metrics_file_path, train_loss_list, train_acc_list, val_loss_list, val_acc_list, lr_values, batch_size, weight_decay)
    
    # Save the trained model's state_dict (weights)
    model_path = os.path.join(output_path, 'resnet_model.pth')
    torch.save(model.state_dict(), model_path)
    logging.info(f"Model saved in {model_path}")  

    # Plot the training metrics (loss and accuracy) and save the plot as a PNG image
    plot_train_metrics(train_loss_list, train_acc_list, val_loss_list, val_acc_list, output_path)
    logging.info("Training metrics plot saved.")


def save_inference_results(file_names, probabilities, predicted_labels, output_path):
    """
    Save the inference results (file names, probabilities, and predicted labels) to a CSV file.

    Args:
        file_names (list): List of file names (e.g., NIfTI or image file names).
        probabilities (list): List of probabilities for label 0 (the probability values).
        predicted_labels (list): List of predicted labels (the classification results).
        output_path (str): Directory where the CSV results file will be saved.
    
    Returns:
        pandas.DataFrame: DataFrame containing the inference results (file names, probabilities, predicted labels).
    """
    # Prepare the data dictionary for the DataFrame
    results = {'File Name': file_names, 'label_0_prob': probabilities, 'label': predicted_labels}
    
    # Convert the dictionary into a pandas DataFrame
    df = pd.DataFrame(results)
    
    # Define the path for the results CSV file
    results_file_path = os.path.join(output_path, 'inference_results.csv')
    
    # Save the DataFrame as a CSV file
    df.to_csv(results_file_path, index=False)

    # Log the information about where the results have been saved
    logging.info(f"Inference results saved to {results_file_path}.")
    
    # Return the DataFrame containing the results
    return df


def plot_data(df, mat_dir, path):
    """
    Plot and save images from the .mat files based on the DataFrame.

    Args:
        df (pandas.DataFrame): DataFrame containing the file names and probabilities.
        mat_dir (str): Directory containing the .mat files.
        path (str): Directory to save the plotted images.
    
    Description:
        This function iterates through the DataFrame `df`, loads .mat files containing medical image data, 
        processes the data, and saves the visualizations (water, fat, ff, wf, r2star) as PNG images in the specified `path`.
    """
    for index, row in df.iterrows():
        # Get the file name and create the full path to the .mat file
        file_name = row['File Name']  
        file_path = os.path.join(mat_dir, file_name) 

        # Load data from the .mat file
        data = sio.loadmat(file_path)
        water = data['outputImages']['water'][0, 0]  # Extract the 'water' image
        fat = data['outputImages']['fat'][0, 0]  # Extract the 'fat' image
        ff = fat / (water + fat + 0.00000001)  # Calculate fat fraction (ff) with small epsilon for stability
        wf = 1 - ff  # Calculate water fraction (wf)
        r2star = data['outputImages']['r2star'][0, 0]  # Extract the 'r2star' image
        
        # Get the probability for label_0 from the DataFrame
        label_0_prob = row['label_0_prob']  

        # Plot and save the images as PNG
        plt.figure(figsize=(20, 4))  # Create a figure with 5 subplots

        # Plot each image
        plt.subplot(1, 5, 1)
        plt.imshow(water, cmap='gray', clim=[0, 100])
        plt.title('water')

        plt.subplot(1, 5, 2)
        plt.imshow(fat, cmap='gray', clim=[0, 100])
        plt.title('fat')

        plt.subplot(1, 5, 3)
        plt.imshow(ff, cmap='jet', clim=[0, 1])  
        plt.title('ff')

        plt.subplot(1, 5, 4)
        plt.imshow(wf, cmap='jet', clim=[0, 1])  
        plt.title('wf')

        plt.subplot(1, 5, 5)
        plt.imshow(r2star, cmap='gray', clim=[0, 100])  
        plt.title('r2star')
        
        # Add a title with the label_0_prob value
        plt.suptitle(f'label_0_prob: {label_0_prob:.2f}', fontsize=14, y=0.95)

        plt.tight_layout()  

        # Save the figure as a PNG file in the output folder
        output_filename = f"{file_name.replace('.mat', '')}.png"  
        output_imagepath = os.path.join(path, output_filename)
        plt.savefig(output_imagepath)  

        plt.close()  

        logging.info(f"Image saved at {output_imagepath}")  


def save_artefact_images(df, mat_dir, output_path):
    """
    Plot and save artefact images for label 1 samples.

    Args:
        df (pandas.DataFrame): DataFrame containing the inference results.
        mat_dir (str): Directory containing the .mat files.
        output_path (str): Directory to save the artefact images.
    """
    # Filter rows where the label is 1 (artefact images)
    rows_label_1 = df[df['label'] == 1]

    # Set output folder for saving the artefact images
    output_folder = os.path.join(output_path, "inference_artefact")
    ground_truth_folder = os.path.join(output_path, "artefact_samples")

    # Delete the existing folder if it exists
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    
    # Ensure the output folder exists
    ensure_dir(output_folder)

    # Plot and save the data for the filtered rows
    plot_data(rows_label_1, mat_dir, output_folder)

    # remove ground truth images
    remove_artefact_nifti(ground_truth_folder, output_folder)

    logging.info('All artefact images have been saved successfully.')


def data_bins(lower_threshold, upper_threshold, df, output_path):
    """
    Filter the DataFrame based on probability thresholds and create directories for saving images.

    Args:
        lower_threshold (float): Lower threshold for label 0 probability.
        upper_threshold (float): Upper threshold for label 0 probability.
        df (pandas.DataFrame): DataFrame containing the inference results.
        output_path (str): Directory to save the images.

    Returns:
        tuple: (df_bin, bin_path) where df_bin is the filtered DataFrame and bin_path is the directory path for saving images.
    """
    # Filter the DataFrame based on the provided probability thresholds
    df_bin = df[(df['label_0_prob'] >= lower_threshold) & (df['label_0_prob'] < upper_threshold)]
    
    logging.info(f"Interval [{lower_threshold}, {upper_threshold}) has {df_bin.shape[0]} samples.")

    # Set the output folder for saving the images
    output_folder = os.path.join(output_path, "inference_correct")
    
    # Create a specific folder for the current threshold range (bin)
    bin_path = os.path.join(output_folder, f'bin{lower_threshold}_{upper_threshold}')

    # Delete the bin folder if it exists
    if os.path.exists(bin_path):
        shutil.rmtree(bin_path)
    
    # Ensure the output directories exist
    ensure_dir(output_folder)
    ensure_dir(bin_path)

    return df_bin, bin_path


def data_randomdraw(df):
    """
    Randomly draw 100 samples from the DataFrame if it contains more than 100 samples.

    Args:
        df (pandas.DataFrame): DataFrame to draw samples from.

    Returns:
        pandas.DataFrame: DataFrame containing the drawn samples.
    """
    # Check if the DataFrame has more than 100 samples
    if df.shape[0] > 50:
        # Randomly sample 100 samples from the DataFrame
        df_sample = df.sample(n=50)
    else:
        # If there are fewer than or equal to 100 samples, use the whole DataFrame
        df_sample = df
        
    return df_sample


def plot_image(df, lower_threshold, upper_threshold, mat_dir, output_path):
    """
    Plot and save images for samples within specified probability thresholds.

    Args:
        df (pandas.DataFrame): DataFrame containing the inference results.
        lower_threshold (float): Lower threshold for label 0 probability.
        upper_threshold (float): Upper threshold for label 0 probability.
        mat_dir (str): Directory containing the .mat files.
        output_path (str): Directory to save the images.
    """
    # Filter the DataFrame based on the provided probability thresholds and create directories for saving images
    df_bin, bin_path = data_bins(lower_threshold, upper_threshold, df, output_path)
    
    # Randomly sample data from the filtered DataFrame
    df_sample = data_randomdraw(df_bin)
    
    # Plot and save the images for the sampled data
    plot_data(df_sample, mat_dir, bin_path)


def save_correct_images(df, mat_dir, output_path):
    """
    Plot and save correct images for different probability threshold intervals.

    Args:
        df (pandas.DataFrame): DataFrame containing the inference results.
        mat_dir (str): Directory containing the .mat files.
        output_path (str): Directory to save the correct images.
    """
    # Plot and save images for samples within the interval [0.5, 0.6]
    plot_image(df, 0.5, 0.6, mat_dir, output_path)
    
    # Plot and save images for samples within the interval [0.6, 0.7]
    plot_image(df, 0.6, 0.7, mat_dir, output_path)
    
    # Plot and save images for samples within the interval [0.7, 0.8]
    plot_image(df, 0.7, 0.8, mat_dir, output_path)
    
    # Plot and save images for samples within the interval [0.8, 0.9]
    plot_image(df, 0.8, 0.9, mat_dir, output_path)
    
    # Plot and save images for samples within the interval [0.9, 0.95]
    plot_image(df, 0.9, 0.95, mat_dir, output_path)
    
    # Plot and save images for samples within the interval [0.95, 1.0]
    plot_image(df, 0.95, 1.0, mat_dir, output_path)


def convert_mat_to_nifti(input_folder, output_folder):
    """
    Convert MAT files to NIfTI format.

    Args:
        input_folder (str): Path to the input folder containing MAT files.
        output_folder (str): Path to the output folder to save NIfTI files.
    """
    # List all MAT files in the input folder
    mat_files = [f for f in os.listdir(input_folder) if f.endswith('.mat')]

    count = 0
    # Iterate over each MAT file
    for mat_file in mat_files:
        # Construct the full path to the MAT file
        mat_path = os.path.join(input_folder, mat_file)
        
        # Load data from the MAT file
        mat_data = sio.loadmat(mat_path)
        images_struct = mat_data['outputImages']

        # Extract individual images from the structure
        water_img = np.array(images_struct['water'][0, 0])
        fat_img = np.array(images_struct['fat'][0, 0])
        ff_img = fat_img / (water_img + fat_img + 0.000000001)  # Calculate fat fraction image
        r2star_img = np.array(images_struct['r2star'][0, 0])

        # For ITK-SNAP visualization (optional)
        # ff_img = ff_img * 1000
        # ff_img = ff_img.astype(np.uint16)

        # Stack the four images into a 3D array
        stacked_images = np.stack([water_img, fat_img, ff_img, r2star_img], axis=0)
        
        # Define the output NIfTI file path, keeping the same file name
        output_file = os.path.splitext(mat_file)[0] + '.nii.gz'
        output_path = os.path.join(output_folder, output_file)
        
        # Define the origin and spacing, not important for our 2D single slice images
        origin_3d = (-216.81655311584, -227.68611907959, 0)
        spacing_3d = (2.5, 2.5, 1)
        
        # Create a SimpleITK image object
        sitk_img = sitk.GetImageFromArray(stacked_images)
        
        # Set image properties
        sitk_img.SetOrigin(origin_3d)
        sitk_img.SetSpacing(spacing_3d)
        
        # Save the image as NIfTI format
        sitk.WriteImage(sitk_img, output_path)

        logging.info(f"{count + 1}/{len(mat_files)}: Converted {mat_file} to {output_file}")
        count += 1

    logging.info("Conversion completed.")


def remove_artefact_nifti(png_folder, nii_folder):
    """
    Remove corresponding .nii.gz files based on the presence of .png files.

    Args:
        png_folder (str): Path to the folder containing artifact PNG images.
        nii_folder (str): Path to the folder containing full_set .nii.gz files, remove images from this folder.
    """
    # Get all file names in the PNG folder
    png_files = os.listdir(png_folder)
    nii_files = os.listdir(nii_folder)

    # Iterate over the files in the PNG folder
    for png_file in png_files:
        if png_file.endswith('.png'):
            # Extract the numeric part from the file name
            file_number = os.path.splitext(png_file)[0]
            
            # Construct the corresponding .nii.gz file name
            nii_file_to_check = file_number + '.nii.gz'
            
            # Check if the corresponding file exists in the NII folder
            if nii_file_to_check in nii_files:
                nii_file_path = os.path.join(nii_folder, nii_file_to_check)
                
                # Delete the .nii.gz file
                os.remove(nii_file_path)

            if png_file in nii_files:
                nii_file_path = os.path.join(nii_folder, png_file)

                # Delete the .nii.gz file
                os.remove(nii_file_path)
                
            logging.info(f"Deleted {nii_file_path}")

    logging.info("Deletion operation completed")

