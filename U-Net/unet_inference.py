import os
import SimpleITK as sitk
import time
import torch
import random
import logging
import shutil

from pathlib import Path
from monai.transforms import Compose, EnsureChannelFirst, ScaleIntensity, ToTensor
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from unet_utils import *

def load_model(model_path, device):
    """
    Load a pre-trained model and move it to the specified device.

    Args:
        model_path (str): Path to the trained model file.
        device (str): The device on which the model should run ('cuda' or 'cpu').

    Returns:
        model (torch.nn.Module): The loaded U-Net model, set to evaluation mode.
    """
    # Initialize the U-Net model architecture
    model = UNet(spatial_dims=2, in_channels=1, out_channels=2, 
                 channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), 
                 num_res_units=2, norm=Norm.BATCH).to(device)
    
    # Load the trained weights into the model
    model.load_state_dict(torch.load(model_path))
    
    # Set the model to evaluation mode (turn off dropout, batchnorm, etc.)
    model.eval()
    
    return model


def inference(model, nifti_files, output_folder, device, infer_transforms):
    """
    Perform inference using the given model on a list of NIfTI files and save the segmentation results.

    Args:
        model (torch.nn.Module): The pre-trained U-Net model used for inference.
        nifti_files (list of str): List of input NIfTI file paths for inference.
        output_folder (str): Path to the folder where segmentation results will be saved.
        device (str): Device for inference ('cuda' or 'cpu').
        infer_transforms (callable): Data preprocessing transformations to apply to the input data.

    Returns:
        None
    """
    # Ensure the output directory exists
    ensure_dir(output_folder)
    
    # Initialize counter for processed files
    number = 1
    
    # Iterate through all the NIfTI files
    for nifti_file_path in nifti_files:
        # Read the NIfTI image using SimpleITK
        image = sitk.ReadImage(nifti_file_path)
        origin_3d = image.GetOrigin()  # Get original image origin (3D)
        spacing_3d = image.GetSpacing()  # Get spacing between pixels (3D)

        # Convert the image to a numpy array and extract the middle slice (2D)
        image_array = sitk.GetArrayFromImage(image)
        image_ff = image_array[2, :, :]  # Choose the 2D slice at index 2
        
        # Apply the preprocessing transformations
        input_data = infer_transforms(image_ff)
        
        # Move the input data to the specified device and add batch dimension
        input_data = input_data.to(device).unsqueeze(0)
        
        # Perform inference with no gradient computation
        with torch.no_grad():
            output = model(input_data)
        
        # Convert the model output to numpy array (segmentation mask)
        output = torch.argmax(output, dim=1)
        output_array = output.cpu().numpy()[0, ...]  # Extract the predicted mask

        # Create a SimpleITK image from the output array
        output_image = sitk.GetImageFromArray(output_array)
        output_image.SetOrigin(origin_3d)  # Set original image origin
        output_image.SetSpacing(spacing_3d)  # Set original spacing
        
        # Build the output file path
        output_file_name = os.path.basename(nifti_file_path).replace('.nii.gz', '_mask.nii.gz')
        output_file_path = os.path.join(output_folder, output_file_name)
        
        # Save the output image to the specified path
        print(f'{number} files are processed successfully!')
        sitk.WriteImage(output_image, output_file_path)

        # Increment the processed file counter
        number += 1

    # Final message when all files are processed
    print("Inference completed!")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,  # Set the logging level to INFO
        format='%(message)s'  # Define the logging format
    )

    # Load the configuration file for the experiment
    config = load_config("unet_config.yaml")

    # Get configuration parameters for input and output paths, sample number, and DICOM folder
    repo_path = Path(__file__).resolve().parent.parent
    input_folder = os.path.join(repo_path, "dataset", "fullcohort_2_nifti_cleaned")
    output_folder = config['data']['output']
    sample_number = config['evaluation']['sample_number']
    dicom_folder = config['data']['dicom_path']

    # Set up output folders for masks and plots
    output_mask_folder = os.path.join(output_folder, "inference")
    output_plot_folder = os.path.join(output_folder, "inference_plot")

    # Delete the existing folder if it exists
    delete_dir(output_mask_folder)
    delete_dir(output_plot_folder)
    
    # Ensure the output directories exist
    ensure_dir(output_mask_folder)
    ensure_dir(output_plot_folder)

    # Set the path for the trained model
    model_path = os.path.join(output_folder, 'unet_model.pth')

    # Set up the device for inference (GPU or CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load the pre-trained model
    model = load_model(model_path, device)
    
    # Define the transformations for data preprocessing during inference
    infer_transforms = Compose([
        EnsureChannelFirst(channel_dim='no_channel'),
        ScaleIntensity(),
        ToTensor()
    ])

    # Get all NIfTI file paths from the input folder
    all_nifti_files = [os.path.join(input_folder, fname) for fname in os.listdir(input_folder) if fname.endswith('.nii.gz')]

    # Check if full cohort or a random sample is needed for inference
    if sample_number == "full_cohort":
        logging.info("Using full cohort for inference.")
        time.sleep(3)  # Simulate delay for user awareness
        nifti_files = all_nifti_files  # Use all NIfTI files for inference

        # Perform inference on the full cohort of files
        inference(model, nifti_files, output_mask_folder, device, infer_transforms)

        # Copy inferenced mask to the dataset folder
        shutil.copytree(output_mask_folder, os.path.join(repo_path, "dataset", "fullcohort_2_mask"), dirs_exist_ok=True)

    elif isinstance(sample_number, int):
        logging.info(f"Randomly sampling {sample_number} files for inference.")
        time.sleep(3)  # Simulate delay for user awareness
        nifti_files = random.sample(all_nifti_files, min(sample_number, len(all_nifti_files)))  # Randomly select files

        # Perform inference on the randomly selected sample
        inference(model, nifti_files, output_mask_folder, device, infer_transforms)

        # Plot the randomly selected samples for evaluation purposes
        logging.info(f"Plotting {sample_number} files.")
        time.sleep(3)  # Simulate delay for user awareness
        plot_masked_img(input_folder, dicom_folder, output_mask_folder, output_plot_folder)

    else:
        # Raise an error if the sample_number is not valid
        raise ValueError("sample_number must be 'full_cohort' or an integer.")


