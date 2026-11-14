import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, confusion_matrix

from vit_dataset import VitDataset, build_cohort_CV
from vit_utils import *
    

def train(model, train_loader, criterion, optimizer, scheduler, device):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The neural network model to be trained.
        train_loader (DataLoader): DataLoader object for the training data.
        criterion (loss function): The loss function used to compute the error.
        optimizer (Optimizer): Optimizer used to update model parameters.
        scheduler (scheduler): Learning rate scheduler to adjust the learning rate.
        device (torch.device): The device (CPU or GPU) where the model and data are located.

    Returns:
        avg_loss (float): Average loss over the training data.
        train_auroc (float): Area Under the Receiver Operating Characteristic Curve (AUROC) score.
    """
    
    # Set the model to training mode
    model.train()
    
    total_loss = 0  # Accumulate the total loss
    y_true = []  # List to store true labels
    y_scores = []  # List to store predicted scores for AUROC calculation

    # Iterate through the training data in batches
    for images, labels, _ in train_loader:
        images = images.to(device)  # Move images to the specified device (CPU/GPU)
        labels = labels.long().to(device)  # Move labels to the specified device

        # Zero the gradients from the previous step
        optimizer.zero_grad()
        
        # Forward pass: get model outputs
        outputs = model(images)

        # Check if any NaN values are present in the output logits
        if np.isnan(outputs.logits.cpu().detach().numpy()).any():
            logging.info("NaN detected in output logits")
            continue  # Skip this batch or choose a different handling strategy

        # Compute the loss
        loss = criterion(outputs.logits.float(), labels)

        # Check if the computed loss contains NaN values
        if np.isnan(loss.item()):
            logging.info("NaN detected in loss")
            continue  # Skip this batch or choose a different handling strategy

        # Backward pass: compute gradients
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        # Update the model's parameters based on the computed gradients
        optimizer.step()

        # Accumulate the total loss
        total_loss += loss.item()
        # Append the true labels to the y_true list
        y_true.extend(labels.cpu().detach().numpy())

        # Apply softmax to logits to get probabilities
        probs = torch.softmax(outputs.logits, dim=1)
        probs_numpy = probs[:, 1].cpu().detach().numpy()  # Get probabilities for the positive class

        # Append predicted scores (probabilities) to the y_scores list
        y_scores.extend(probs_numpy)

    # Compute the average loss for the epoch
    avg_loss = total_loss / len(train_loader)

    # Compute the AUROC score for the training data
    train_auroc = roc_auc_score(y_true, y_scores) if y_scores else float('nan')

    # Step the scheduler to adjust the learning rate
    scheduler.step()

    return avg_loss, train_auroc


def validate(model, val_loader, criterion, device):
    """
    Validates the model on the validation dataset.

    Args:
        model (nn.Module): The trained neural network model.
        val_loader (DataLoader): DataLoader object for the validation data.
        criterion (loss function): The loss function used to compute the error.
        device (torch.device): The device (CPU or GPU) where the model and data are located.

    Returns:
        avg_loss (float): Average loss over the validation data.
        val_auroc (float): Area Under the Receiver Operating Characteristic Curve (AUROC) score.
    """
    
    # Set the model to evaluation mode (disables dropout, batch norm, etc.)
    model.eval()
    
    total_loss = 0  # Accumulate the total loss
    y_true = []  # List to store true labels
    y_scores = []  # List to store predicted scores for AUROC calculation

    # Disable gradient computation for validation (saves memory and computation)
    with torch.no_grad():
        # Iterate through the validation data in batches
        for images, labels, _ in val_loader:
            images = images.to(device)  # Move images to the specified device (CPU/GPU)
            labels = labels.long().to(device)  # Move labels to the specified device

            # Forward pass: get model outputs (logits)
            outputs = model(images, output_attentions=True)

            # Compute the loss
            loss = criterion(outputs.logits.float(), labels)
            total_loss += loss.item()

            # Append the true labels to the y_true list
            y_true.extend(labels.cpu().detach().numpy())

            # Apply softmax to logits to get probabilities
            probs = torch.softmax(outputs.logits, dim=1)
            # Append predicted scores (probabilities) for the positive class to the y_scores list
            y_scores.extend(probs[:, 1].cpu().detach().numpy())

    # Compute the average loss for the epoch
    avg_loss = total_loss / len(val_loader)

    # Compute the AUROC score for the validation data
    val_auroc = roc_auc_score(y_true, y_scores)

    return avg_loss, val_auroc


def get_predictions(model, dataloader, device):
    """
    Generates predictions from the model for the given data.

    Args:
        model (nn.Module): The trained neural network model.
        dataloader (DataLoader): DataLoader object for the dataset (could be validation or test set).
        device (torch.device): The device (CPU or GPU) where the model and data are located.

    Returns:
        all_preds (np.ndarray): Array of predicted labels.
        all_labels (np.ndarray): Array of true labels.
    """
    
    # Set the model to evaluation mode (disables dropout, batch norm, etc.)
    model.eval()
    
    all_preds = []  # List to store predicted labels
    all_labels = []  # List to store true labels

    # Disable gradient computation for prediction (saves memory and computation)
    with torch.no_grad():
        # Iterate through the data in batches
        for data in dataloader:
            inputs, labels, _ = data
            inputs, labels = inputs.to(device), labels.to(device)  # Move inputs and labels to device

            # Forward pass: get model outputs (logits)
            outputs = model(inputs)
            
            # Extract the logits from the model outputs
            logits = outputs.logits
            
            # Get the predicted labels by taking the class with the highest probability
            _, preds = torch.max(logits, 1)
            
            # Append the predictions and true labels to the respective lists
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert the lists of predictions and labels to numpy arrays
    return np.array(all_preds), np.array(all_labels)

def initialize_optimizer(model, train_dataset, config, device):
    """
    Initializes the optimizer, learning rate scheduler, and loss function for training.

    Args:
        model (nn.Module): The neural network model to be optimized.
        train_dataset (Dataset): The training dataset used to compute class weights.
        config (dict): Configuration dictionary containing optimizer settings, such as learning rate and weight decay.
        device (torch.device): The device (CPU or GPU) where the model and data are located.

    Returns:
        optimizer (Optimizer): The initialized optimizer (AdamW).
        scheduler (lr_scheduler): The initialized learning rate scheduler (CosineAnnealingLR).
        criterion (Loss function): The initialized loss function (CrossEntropyLoss) with class weights.
    """
    
    # Initialize the AdamW optimizer with model parameters, learning rate, and weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['optimizer']['learning_rate'],
        weight_decay=config['training']['optimizer']['weight_decay']
    )
    
    # Initialize the learning rate scheduler using Cosine Annealing
    scheduler = CosineAnnealingLR(optimizer, T_max=config['training']['epochs'], eta_min=0)
    
    # Compute the class weights based on the training dataset class distribution
    train_count_0, train_count_1 = count_labels(train_dataset)
    
    # Calculate the class weights (inverse frequency method)
    weight_class0 = train_count_1 / (train_count_0 + train_count_1)
    weight_class1 = train_count_0 / (train_count_0 + train_count_1)

    # Create a tensor containing the class weights, and move it to the appropriate device
    weights = torch.tensor([weight_class0, weight_class1], dtype=torch.float).to(device)
    
    # Initialize the loss function (CrossEntropyLoss) with the computed class weights
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    return optimizer, scheduler, criterion



def train_loop(model, train_loader, val_loader, device, criterion, optimizer, scheduler, num_epochs):
    """
    Runs the training loop for multiple epochs, including training and validation phases.

    Args:
        model (nn.Module): The neural network model to be trained.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        device (torch.device): The device (CPU or GPU) where the model and data are located.
        criterion (Loss function): The loss function used to compute the error.
        optimizer (Optimizer): The optimizer used to update model parameters.
        scheduler (lr_scheduler): The learning rate scheduler to adjust the learning rate.
        num_epochs (int): The number of epochs to train the model.

    Returns:
        best_model_state (dict): The state dictionary of the best model (based on validation AUROC).
        best_val_auroc (float): The best validation AUROC score achieved during training.
    """
    
    best_val_auroc = 0.0  # Initialize the best validation AUROC to track improvements
    best_model_state = None  # Variable to store the best model's state

    # Iterate over the specified number of epochs
    for epoch in range(num_epochs):
        # Training phase: train the model and compute the training loss and AUROC
        train_loss, train_auroc = train(model, train_loader, criterion, optimizer, scheduler, device)
        
        # Validation phase: validate the model and compute the validation loss and AUROC
        val_loss, val_auroc = validate(model, val_loader, criterion, device)
        
        # Get the current learning rate from the optimizer
        current_learning_rate = optimizer.param_groups[0]['lr']
        
        # Print the training and validation results for the current epoch
        logging.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Train AUROC: {train_auroc}, Val Loss: {val_loss}, Val AUROC: {val_auroc}")
        logging.info(f"Learning rate: {current_learning_rate}")
        
        # If the validation AUROC has improved, update the best model state
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_model_state = model.state_dict()

    return best_model_state, best_val_auroc


def analyse_model(model, dataloader, device, fold, output_path, type=None):
    """
    Analyzes the model's performance by calculating various metrics (e.g., sensitivity, specificity, AUROC)
    and saving confusion matrices for a given fold.

    Args:
        model (nn.Module): The trained model to be analyzed.
        dataloader (DataLoader): DataLoader for the dataset (e.g., training, validation, or test set).
        device (torch.device): The device (CPU or GPU) where the model and data are located.
        fold (int): The current fold in cross-validation (1-indexed).
        output_path (str): Path where the output confusion matrix image will be saved.
        type (str, optional): The type of dataset being analyzed (e.g., "Train", "Val", "Test").

    Returns:
        sensitivity (float): The sensitivity (true positive rate) of the model for the dataset.
        specificity (float): The specificity (true negative rate) of the model for the dataset.
        auc (float): The Area Under the ROC Curve (AUC) for the model on the dataset.
    """
    
    # Get the predictions and true labels from the model using the dataloader
    predictions, labels = get_predictions(model, dataloader, device)
    
    # Compute the confusion matrix based on predictions and true labels
    cm = confusion_matrix(labels, predictions)
    
    # Define the path to save the confusion matrix image
    cm_path = os.path.join(output_path, f'{type}_fold{fold+1}.png')
    
    # Plot and save the confusion matrix
    plot_confusion_matrix(cm, f"{type} Confusion Matrix for Fold {fold+1}", cm_path)
    
    # Calculate sensitivity (True Positive Rate) and specificity (True Negative Rate)
    sensitivity, specificity = calculate_sensitivity_specificity(labels, predictions)
    
    # Print the calculated sensitivity and specificity for the current fold and dataset type
    logging.info(f"{type}-Set for Fold {fold+1}: Sensitivity: {sensitivity:.3f}; Specificity: {specificity:.3f}")
    
    # Compute the ROC curve data and AUC for the model on the dataset
    fpr, tpr, auc, _ = get_roc_curve_data(model, dataloader, device)

    return sensitivity, specificity, auc


def cross_validation_loop(folds, img_list_processed, data_dir, mask_dir, label_csv, augmentations_train, augmentations_val, config, device, output_path):
    """
    Runs cross-validation for the model, training and evaluating on different folds of the dataset.

    Args:
        folds (list of tuples): List of train-validation splits for cross-validation. Each tuple contains train indices and validation indices.
        img_list_processed (list): List of processed image file paths for the dataset.
        data_dir (str): Directory where the image data is stored.
        mask_dir (str): Directory where the mask data is stored.
        label_csv (str): Path to the CSV file containing the labels for the dataset.
        augmentations_train (callable): Augmentation function applied to the training dataset.
        augmentations_val (callable): Augmentation function applied to the validation dataset.
        config (dict): Configuration dictionary containing training parameters like batch size, number of epochs, etc.
        device (torch.device): The device (CPU or GPU) where the model and data are located.
        output_path (str): Directory where the results and models will be saved.

    Returns:
        overall_dict (dict): Dictionary containing the performance metrics (sensitivity, specificity, AUROC) for both training and validation sets across all folds.
    """
    
    # Dictionary to store the results of sensitivity, specificity, and AUC for each fold
    overall_dict = {'train_sensitivity': [],
                    'train_specificity': [],
                    'train_auc': [],
                    'val_sensitivity': [],
                    'val_specificity': [],
                    'val_auc': []
    }

    # Iterate through each fold in the cross-validation splits
    for fold, (train_idx, val_idx) in enumerate(folds):
        logging.info(f"Training on fold {fold+1}...")

        # Prepare the datasets for the current fold
        train_list = [img_list_processed[i] for i in train_idx]
        val_list = [img_list_processed[i] for i in val_idx]

        # Create dataset objects for training and validation
        train_dataset = VitDataset(train_list, data_dir, mask_dir, label_csv, transform=augmentations_train)
        val_dataset = VitDataset(val_list, data_dir, mask_dir, label_csv, transform=augmentations_val)
        
        # Create DataLoader objects for the training and validation datasets
        train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['evaluation']['batch_size'])

        # Print the class distribution in the training and validation sets
        train_count_0, train_count_1 = count_labels(train_dataset)
        val_count_0, val_count_1 = count_labels(val_dataset)
        logging.info(f"[Fold{fold+1}]Train Dataset: Label 0: {train_count_0}, Label 1: {train_count_1}")
        logging.info(f"[Fold{fold+1}]Validation Dataset: Label 0: {val_count_0}, Label 1: {val_count_1}")
        
        # Initialize the model using the configuration and device
        model = initialize_model(config, device)
        
        # Initialize optimizer, learning rate scheduler, and loss function (criterion)
        optimizer, scheduler, criterion = initialize_optimizer(model, train_dataset, config, device)
        
        # Train and validate the model for the specified number of epochs
        num_epochs = config['training']['epochs']
        best_model_state, best_val_auroc = train_loop(model, train_loader, val_loader, device, criterion, optimizer, scheduler, num_epochs)
        
        # Save the best model for the current fold
        ensure_dir(output_path)
        torch.save(best_model_state, f'{output_path}/best_model_fold{fold+1}.pth')
        
        # Load the best model state for analysis
        model.load_state_dict(best_model_state)
        
        # Analyze the model performance on both the train and validation sets
        train_sensitivity, train_specificity, train_auc = analyse_model(model, train_loader, device, fold, output_path, type="train")
        val_sensitivity, val_specificity, val_auc = analyse_model(model, val_loader, device, fold, output_path, type="val")

        # Store the metrics for the current fold in the overall dictionary
        overall_dict['train_sensitivity'].append(train_sensitivity)
        overall_dict['train_specificity'].append(train_specificity)
        overall_dict['train_auc'].append(train_auc)
        overall_dict['val_sensitivity'].append(val_sensitivity)
        overall_dict['val_specificity'].append(val_specificity)
        overall_dict['val_auc'].append(val_auc)
        
        # Print the best validation AUROC for the current fold
        logging.info(f"Fold {fold+1}, Best Val AUROC: {best_val_auroc}")

    # Return the overall dictionary containing the metrics for all folds
    return overall_dict


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,  # set level to INFO
        format='%(message)s' # output format
    )

    # Load the configuration file
    config = load_config("vit_config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Parse augmentations
    augmentations_train = parse_augmentations(config['training']['augumentation'])
    augmentations_val = parse_augmentations(config['evaluation']['augumentation'])

    # Get the source directory for the dataset
    repo_path = Path(__file__).resolve().parent.parent
    
    # Prepare data and paths
    data_dir = config['data']['data_dir']
    output_path = config['data']['output']

    mask_dir = os.path.join(repo_path, "dataset", "fullcohort_2_mask")
    label_csv = os.path.join(repo_path, "dataset", "pdff_values_sigmaclip.csv")
    
    # Build train/val/test list
    img_list_processed, test_list, folds = build_cohort_CV(data_dir, label_csv, fold=config['training']['cv_fold_num'],
                                                           snp=config['data']['snp'], type=config['data']['cohort'])

    # Cross-validation loop
    overall_dict = cross_validation_loop(folds, img_list_processed, data_dir, mask_dir, label_csv, augmentations_train, augmentations_val, config, device, output_path)
    

    ###### test #####
    model = initialize_model(config, device)
    test_pd = pd.DataFrame(test_list, columns=['data'])
    test_pd_path = os.path.join(config['data']['output'], "test_list.csv")
    test_pd.to_csv(test_pd_path, index=False)

    # Prepare test dataset and loader
    test_dataset = VitDataset(test_list, data_dir, mask_dir, label_csv, transform=augmentations_val)
    test_loader = DataLoader(test_dataset, batch_size=config['evaluation']['batch_size'])

    test_count_0, test_count_1 = count_labels(test_dataset)
    logging.info(f"Test Dataset: Label 0: {test_count_0}, Label 1: {test_count_1}\n")

    # Get ensemble predictions and test labels
    ensemble_predictions, test_labels = get_ensemble_predictions(model, test_loader, device, config, output_path)

    # Compute AUC for test set
    test_auc = roc_auc_score(test_labels, ensemble_predictions)
    logging.info(f"Ensemble Test AUC: {test_auc:.4f}")

    # Select best threshold and calculate Sensitivity/Specificity
    best_threshold, best_sensitivity, best_specificity = evaluate_predictions(ensemble_predictions, 
                                                                              test_labels,
                                                                              threshold=config['model']['classification_threshold'])

    # Convert probabilities to binary labels using best threshold
    binary_predictions = (ensemble_predictions >= best_threshold).astype(int)

    # Save attention maps based on predictions
    attention_dir = os.path.join(output_path, 'attention_map')
    ensure_dir(attention_dir)
    save_attention_maps_for_class(model, test_loader, best_threshold, device, output_path, config)

    # Check for common samples and average attention maps
    final_attentionmap_path = f"{attention_dir}/final_attention_maps"
    ensure_dir(final_attentionmap_path)
    find_common_samples_and_average(attention_dir, final_attentionmap_path, config)

    # Calculate AUC bootstrap confidence intervals
    confidence_lower, confidence_upper = bootstrap_auc(ensemble_predictions, test_labels)
    logging.info(f"Test AUC: {test_auc:.3f} | 95% CI: [{confidence_lower:.3f}, {confidence_upper:.3f}]")

    # Plot ROC curve and save it
    plot_save_curves(ensemble_predictions, test_labels, output_path, curve_type="auc_roc", cohort=config['data']['cohort'])

    # Plot PR curve and save it
    plot_save_curves(ensemble_predictions, test_labels, output_path, curve_type="auc_pr", cohort=config['data']['cohort'])

    # Plot confustion matrix and save it
    test_cm = confusion_matrix(test_labels, binary_predictions)
    test_cm_path = os.path.join(output_path, 'test_CM.png')
    plot_confusion_matrix(test_cm, "Test Confusion Matrix", test_cm_path)

    logging.info(f"Final AUC: {test_auc:.3f}, Sensitivity: {best_sensitivity:.3f}, Specificity: {best_specificity:.3f}")