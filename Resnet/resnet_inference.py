import os
import logging
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

from pathlib import Path
from torch.utils.data import DataLoader
from resnet_utils import *
from resnet_datasets import MyResnetInference


def load_model(model_path, num_classes=2):
    """
    Load the pre-trained ResNet model.

    Args:
        model_path (str): Path to the saved model checkpoint.
        num_classes (int): Number of output classes for the model.

    Returns:
        tuple: (model, device) where model is the loaded ResNet model and device is the device (CPU or GPU) used.
    """
    logging.info("Loading pre-trained ResNet model...")
    
    model = models.resnet50()
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes) 

    # Use GPU if possible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Load the saved model checkpoint
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)

    # Set the model to evaluation mode
    model.eval()

    logging.info("Model loaded and set to evaluation mode.")
    return model, device


def predict(model, dataloader, device):
    """
    Perform inference and return filenames, probabilities, and predictions.

    Args:
        model (torch.nn.Module): The trained model for inference.
        dataloader (torch.utils.data.DataLoader): DataLoader for the inference dataset.
        device (torch.device): Device to perform inference on (CPU or GPU).

    Returns:
        tuple: (file_names_list, probs, predictions) where file_names_list is a list of filenames,
               probs is a numpy array of probabilities, and predictions is a list of predicted labels.
    """
    # Initialize lists to store file names, probabilities, and predictions
    file_names_list = []
    probs_list = []
    predictions = []

    # Perform inference with no gradient calculation (to save memory and computations)
    with torch.no_grad():
        # Iterate through batches of images in the dataloader
        for file_names, images in dataloader:
            # Move images to the specified device (CPU or GPU)
            images = images.to(device)
            
            # Get model outputs (raw logits)
            outputs = model(images)
            
            # Apply softmax to outputs to get probabilities for each class
            probabilities = torch.softmax(outputs, dim=1)
            
            # Store the probabilities for class 0 (assuming binary classification)
            probs_list.append(probabilities[:, 0].cpu().numpy())
            
            # Get the predicted class labels (max probability)
            predicted_labels = torch.argmax(outputs, dim=1).cpu().numpy()
            
            # Append the predictions and filenames to the lists
            predictions.extend(predicted_labels)
            file_names_list.extend(file_names)
            
            # Log the prediction for the current batch
            # logging.info(f'{file_names} patient is predicted as label {predicted_labels} with probability {probabilities}')

    # Concatenate the probability arrays into a single numpy array
    probs = np.concatenate(probs_list)

    return file_names_list, probs, predictions




if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,  # Set the logging level to INFO
        format='%(message)s'  # Define the logging format
    )
    
    # Load configuration
    config = load_config("resnet_config.yaml")
    transform_inference = parse_augmentations(config['evaluation']['augmentation'])

    # Define paths
    output_path = config['data']['output']
    ensure_dir(output_path)
    model_path = os.path.join(output_path, 'resnet_model.pth')
    
    # Get the source directory for the dataset
    repo_path = Path(__file__).resolve().parent.parent
    mat_dir = os.path.join(repo_path, "dataset", "fullcohort_1_wfs")

    # Load model and set device
    model, device = load_model(model_path)

    # Create dataset and dataloader
    num_workers = int(os.cpu_count() / 2)
    dataset = MyResnetInference(mat_dir, transform=transform_inference)
    logging.info(f'{num_workers * 2} available CPU cores detected. Half ({num_workers}) used for inference.')

    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=num_workers)

    # Perform inference
    file_names, probabilities, predicted_labels = predict(model, dataloader, device)

    # Save results to CSV
    df = save_inference_results(file_names, probabilities, predicted_labels, output_path)

    # Plot and save artefact images
    save_artefact_images(df, mat_dir, output_path)

    # Plot and save correct images with different threshold intervals
    save_correct_images(df, mat_dir, output_path)
