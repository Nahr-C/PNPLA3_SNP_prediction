import os
import cv2
import yaml
import torch
import logging
import matplotlib.pyplot as plt
import numpy as np

from transformers import ViTConfig, ViTForImageClassification
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve


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
    Parse and apply augmentation functions to create a composed augmentation pipeline.

    Args:
        augmentations (list): A list of strings, where each string represents an augmentation function.
                              These strings are evaluated and converted into actual augmentation functions.

    Returns:
        v2.Compose: A composed augmentation pipeline created by applying the specified augmentations.
    """
    
    # Initialize an empty list to store augmentation functions
    aug_list = []
    
    # Loop through each augmentation specification
    for aug in augmentations:
        # Use eval to convert the string into an actual function call
        aug_list.append(eval(aug))
    
    # Return a composed augmentation pipeline
    return v2.Compose(aug_list)

def count_labels(dataset):
    """
    Count the occurrences of each label (0 and 1) in a dataset.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to be processed, which is expected to return data in the format
                                             (image, label, image_id).

    Returns:
        tuple: A tuple containing two integers:
            - count_0 (int): The number of samples with label 0.
            - count_1 (int): The number of samples with label 1.
    """
    
    # Initialize counters for label 0 and label 1
    count_0 = 0
    count_1 = 0
    
    # Create a DataLoader to efficiently iterate through the dataset
    # A batch size of 256 and shuffle set to False (no need to shuffle for label counting)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    # Iterate over the dataset in batches
    for _, labels, _ in loader:
        # For each batch, count the occurrences of each label (0 or 1)
        count_0 += torch.sum(labels == 0).item()
        count_1 += torch.sum(labels == 1).item()
    
    # Return the counts of labels 0 and 1
    return count_0, count_1

def initialize_model(config, device):
    """
    Initializes the Vision Transformer (ViT) model for image classification based on the provided configuration.

    Args:
        config (dict): Configuration dictionary containing model parameters such as image size, 
                       number of channels, number of labels, and attention heads.
        device (torch.device): The device (CPU or GPU) where the model will be loaded.

    Returns:
        model (ViTForImageClassification): The initialized Vision Transformer model.
    """
    
    # Set up the configuration for the ViT model
    model_config = ViTConfig(
        image_size=config['model']['image_size'],
        num_channels=config['model']['num_channels'],
        num_labels=config['model']['num_labels'],
        num_attention_heads=config['model']['num_attention_heads'],
    )
    
    # Load the pre-trained Vision Transformer model for image classification
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224', 
        num_labels=2,  
        ignore_mismatched_sizes=True  
    )
    
    # Initialize the classifier layer weights and biases
    model.classifier.weight.data.normal_(mean=0.0, std=model_config.initializer_range)  
    model.classifier.bias.data.zero_()  
    
    # Move the model to the specified device (CPU or GPU)
    model = model.to(device)
    
    return model

def load_model_for_inference(model, model_path, device):
    """
    Loads a pre-trained model for inference, moves it to the specified device (CPU or GPU),
    and sets it to evaluation mode.

    Args:
        model (torch.nn.Module): The model architecture to be loaded.
        model_path (str): Path to the pre-trained model file (.pth).
        device (torch.device): The device (CPU or GPU) on which to load the model.

    Returns:
        model (torch.nn.Module): The loaded model moved to the specified device and set to evaluation mode.
    """
    
    # Load the model's state_dict from the specified file path
    model.load_state_dict(torch.load(model_path))
    
    # Move the model to the specified device (CPU or GPU)
    model.to(device)
    
    # Set the model to evaluation mode (important for inference, disables dropout, batch norm updates, etc.)
    model.eval()
    
    return model

def get_roc_curve_data(model, data_loader, device):
    """
    Generate the ROC curve data for the model on the given dataset.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        data_loader (torch.utils.data.DataLoader): DataLoader containing the dataset for evaluation.
        device (torch.device): The device to perform evaluation on (CPU or GPU).

    Returns:
        tuple: (fpr, tpr, roc_auc, all_attentions)
            - fpr (numpy.ndarray): False Positive Rate at different thresholds.
            - tpr (numpy.ndarray): True Positive Rate at different thresholds.
            - roc_auc (float): Area Under the ROC Curve.
            - all_attentions (list): (Empty list in this case, may be used for attention outputs).
    """
    
    # Set the model to evaluation mode
    model.eval()
    
    # Initialize lists to store true labels, predictions, and attention weights
    all_labels = []
    all_preds = []
    all_attentions = []
    
    # Iterate over the data in the dataloader
    with torch.no_grad():
        for images, labels, _ in data_loader:
            images = images.to(device)
            labels = labels.long().to(device)
            
            # Get the model's predictions and attention outputs
            outputs = model(images, output_attentions=True)
            
            # Apply softmax to get probabilities for class 1
            preds = outputs.logits.softmax(dim=1)[:, 1].cpu().numpy()
            
            # Collect the true labels and predictions
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)
    
    # Calculate the False Positive Rate, True Positive Rate, and thresholds for the ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    
    # Compute the Area Under the Curve (AUC)
    roc_auc = auc(fpr, tpr)
    
    return fpr, tpr, roc_auc, all_attentions

def get_ensemble_predictions(model, test_loader, device, config, output_path):
    """
    This function generates ensemble predictions for a given model by loading the model weights
    from different folds (cross-validation) and aggregating the predictions.
    
    Args:
        model (torch.nn.Module): The model architecture.
        test_loader (DataLoader): The DataLoader object that provides batches of test data.
        device (torch.device): The device (CPU or GPU) to run the model on.
        config (dict): Configuration dictionary that contains the training setup, including the number of folds.
        output_path (str): The directory where the trained model weights are saved.
    
    Returns:
        tuple: A tuple containing:
            - ensemble_predictions (numpy.ndarray): The average prediction across all folds.
            - test_labels (list): The true labels for the test set.
    """

    # Initialize lists to store predictions and labels
    test_predictions = []  
    test_labels = []       

    # Loop through each fold in the cross-validation process
    for fold_index in range(config['training']['cv_fold_num']):
        
        # Define the model path based on the fold index
        model_path = f"{output_path}/best_model_fold{fold_index+1}.pth"
        
        # Load the trained model for the current fold
        model = load_model_for_inference(model, model_path, device)

        fold_predictions = []  
        fold_ids = []          

        # Set the model to evaluation mode and perform inference
        with torch.no_grad():
            # Loop over the test data
            for images, labels, img_ids in test_loader:
                # Move the data to the appropriate device
                images = images.to(device)
                labels = labels.to(device)
                
                # Get the model's outputs (logits) for the current batch
                outputs = model(images)
                
                # Apply softmax to get probabilities for the positive class (class 1)
                probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
                
                # Append predictions and image IDs to the fold's lists
                fold_predictions.extend(probs)
                fold_ids.extend(img_ids)
                
                # Store the true labels only once (in the first fold)
                if fold_index == 0:  
                    test_labels.extend(labels.cpu().numpy())
        
        # Append the current fold's predictions to the list of all fold predictions
        test_predictions.append(fold_predictions)

    # Compute the average predictions across all folds (ensemble)
    ensemble_predictions = np.mean(test_predictions, axis=0)
    
    return ensemble_predictions, test_labels

def evaluate_predictions(ensemble_predictions, test_labels, threshold="Youden_index"):
    """
    This function evaluates the performance of the ensemble model's predictions by calculating
    sensitivity and specificity at different thresholds, and prints the results.
    
    Args:
        ensemble_predictions (numpy.ndarray): The predicted probabilities for the positive class.
        test_labels (numpy.ndarray): The true labels of the test data.
        threshold (str or float, optional): The threshold for classification. If "Youden_index", the optimal threshold 
                                            based on Youden's index will be used. Otherwise, a specific threshold 
                                            value can be provided.
    
    Returns:
        tuple: A tuple containing:
            - best_threshold (float): The threshold chosen for classification.
            - best_sensitivity (float): The sensitivity at the chosen threshold.
            - best_specificity (float): The specificity at the chosen threshold.
    """
    
    # Compute the false positive rate (fpr), true positive rate (tpr), and corresponding thresholds
    fpr, tpr, thresholds = roc_curve(test_labels, ensemble_predictions)
    
    # Compute specificity as 1 - false positive rate
    specificity = 1 - fpr
    
    if threshold == "Youden_index":
        # If the threshold method is Youden's index, find the best threshold
        # Youden's index = Sensitivity + Specificity - 1
        sensitivity_specificity_sum = tpr + specificity - 1
        best_threshold_index = np.argmax(sensitivity_specificity_sum)
        best_threshold = thresholds[best_threshold_index]

        # Get sensitivity and specificity at the best threshold
        best_sensitivity = tpr[best_threshold_index]
        best_specificity = specificity[best_threshold_index]

        # Print the results for Youden's index threshold
        logging.info(f"Best Sensitivity + Specificity: {sensitivity_specificity_sum[best_threshold_index]:.4f}, Best threshold: {best_threshold}")
        logging.info(f"Sensitivity at best threshold: {best_sensitivity:.4f}")
        logging.info(f"Specificity at best threshold: {best_specificity:.4f}")
        
    else:
        # If a specific threshold is provided, use it directly
        best_threshold = threshold

        # Find the closest threshold to the provided threshold and compute sensitivity and specificity
        best_sensitivity = tpr[np.argmin(np.abs(thresholds - best_threshold))]
        best_specificity = specificity[np.argmin(np.abs(thresholds - best_threshold))]
        
        # Print the results for the chosen threshold
        logging.info(f"Threshold for binary classification: {best_threshold}")
        logging.info(f"Sensitivity at best threshold: {best_sensitivity:.4f}")
        logging.info(f"Specificity at best threshold: {best_specificity:.4f}")
    
    return best_threshold, best_sensitivity, best_specificity


def calculate_sensitivity_specificity(y_true, y_pred):
    """
    Calculate sensitivity and specificity from the true labels and predicted labels.

    Args:
        y_true (array-like): True labels of the dataset.
        y_pred (array-like): Predicted labels from the model.

    Returns:
        tuple: (sensitivity, specificity)
            - sensitivity (float): The true positive rate (recall), calculated as TP / (TP + FN).
            - specificity (float): The true negative rate, calculated as TN / (TN + FP).
    """
    
    # Calculate the confusion matrix from the true labels and predictions
    cm = confusion_matrix(y_true, y_pred)
    
    # Extract true negatives (tn), false positives (fp), false negatives (fn), and true positives (tp)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate sensitivity (true positive rate)
    sensitivity = tp / (tp + fn)
    
    # Calculate specificity (true negative rate)
    specificity = tn / (tn + fp)
    
    return sensitivity, specificity

def save_attention_maps_for_class(model, test_loader, best_threshold, device, output_path, config):
    """
    This function processes the test data through the model, computes attention maps for each sample, 
    and saves them based on the class of the prediction (True Positive, True Negative, False Positive, False Negative).
    
    Args:
        model (torch.nn.Module): The model architecture.
        test_loader (DataLoader): The DataLoader object that provides batches of test data.
        best_threshold (float): The threshold used to classify predictions as positive or negative.
        device (torch.device): The device (CPU or GPU) to run the model on.
        output_path (str): The directory where the model and attention maps are saved.
        config (dict): Configuration dictionary that contains the training setup, including the number of folds.
    """
    
    # Loop through each fold in the cross-validation process
    for fold_index in range(config['training']['cv_fold_num']):
        
        # Define the model path based on the fold index
        model_path = f"{output_path}/best_model_fold{fold_index+1}.pth"
        
        # Load the trained model for the current fold
        model = load_model_for_inference(model, model_path, device)
        
        # Define the directory path to save attention maps for the current fold
        attention_path = f"{output_path}/attention_map/fold{fold_index+1}"

        # Set the model to evaluation mode and perform inference
        with torch.no_grad():
            # Loop over the test data
            for images, labels, img_ids in test_loader:
                # Move the data to the appropriate device
                images = images.to(device)
                labels = labels.to(device)
                
                # Get the model's outputs (logits) and attention maps
                outputs = model(images, output_attentions=True)
                attentions = outputs.attentions[-1]  # Take the last attention layer
                probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()  # Get probabilities for the positive class
                binary_preds = (probs >= best_threshold).astype(int)  # Convert to binary predictions based on the threshold

                # Loop through each prediction in the batch
                for i in range(len(binary_preds)):
                    # Determine the class label (TP, TN, FP, FN) based on the prediction and true label
                    if binary_preds[i] == 1 and labels[i] == 1:
                        class_label = "TP"  # True Positive
                    elif binary_preds[i] == 0 and labels[i] == 0:
                        class_label = "TN"  # True Negative
                    elif binary_preds[i] == 1 and labels[i] == 0:
                        class_label = "FP"  # False Positive
                    elif binary_preds[i] == 0 and labels[i] == 1:
                        class_label = "FN"  # False Negative

                    # Save the attention maps along with the corresponding class label and image ID
                    save_attention_maps(attentions[i:i+1], images[i:i+1], attention_path, img_ids[i], class_label, device)


def find_common_samples_and_average(base_dir, final_output_dir, config):
    """
    This function identifies common samples across different folds for each category (TP, TN, FP, FN),
    averages the corresponding images, and saves the averaged images to a specified output directory.

    Args:
        base_dir (str): The directory containing the images for each fold, organized by category.
        final_output_dir (str): The directory where the averaged images will be saved.
        config (dict): Configuration dictionary containing the number of cross-validation folds.
    """
    
    # Define categories for which we need to process the images
    categories = ["TP", "TN", "FP", "FN"]

    # Loop through each category (True Positive, True Negative, False Positive, False Negative)
    for category in categories:
        sample_sets = []  
        
        # Loop through each fold and collect the set of sample filenames for the current category
        for fold in range(config['training']['cv_fold_num']):
            fold_dir = os.path.join(base_dir, f"fold{fold+1}", category)
            samples = set(os.listdir(fold_dir))  
            sample_sets.append(samples)
        
        # Find the common samples across all folds for the current category
        common_samples = set.intersection(*sample_sets)
        
        # Create the output directory for the current category if it doesn't exist
        os.makedirs(os.path.join(final_output_dir, category), exist_ok=True)
        
        # Loop through each common sample and compute the average image
        for sample in common_samples:
            images = []  
            
            # Loop through each fold and read the image for the current sample
            for fold in range(config['training']['cv_fold_num']):
                img_path = os.path.join(base_dir, f"fold{fold+1}", category, sample)
                img = cv2.imread(img_path)  
                images.append(img)
            
            # Compute the average of the images across all folds and convert to uint8
            average_image = np.mean(images, axis=0).astype(np.uint8)
            
            # Save the averaged image to the output directory
            cv2.imwrite(os.path.join(final_output_dir, category, sample), average_image)


def bootstrap_auc(ensemble_predictions, test_labels, n_bootstraps=1000):
    """
    This function calculates the confidence intervals for the AUC (Area Under the Curve) score 
    using bootstrapping. It generates multiple bootstrapped samples from the data, computes the 
    AUC score for each sample, and returns the 95% confidence interval for the AUC.

    Args:
        ensemble_predictions (numpy.ndarray): The predicted probabilities for the positive class.
        test_labels (numpy.ndarray): The true labels of the test data.
        n_bootstraps (int, optional): The number of bootstrap samples to generate. Default is 1000.
    
    Returns:
        tuple: A tuple containing:
            - confidence_lower (float): The lower bound of the 95% confidence interval for the AUC.
            - confidence_upper (float): The upper bound of the 95% confidence interval for the AUC.
    """

    # List to store the AUC scores for each bootstrapped sample
    bootstrapped_aucs = []

    # Perform bootstrapping n_bootstraps times
    for _ in range(n_bootstraps):
        # Generate a random sample of indices with replacement
        indices = np.random.randint(0, len(ensemble_predictions), len(ensemble_predictions))
        
        # Skip if the bootstrapped sample does not have both positive and negative labels
        if len(np.unique(np.array(test_labels)[indices])) < 2:
            continue

        # Compute the AUC score for the bootstrapped sample
        score = roc_auc_score(np.array(test_labels)[indices], np.array(ensemble_predictions)[indices])
        
        # Append the computed AUC score to the list
        bootstrapped_aucs.append(score)

    # Sort the AUC scores to calculate the confidence interval
    sorted_scores = np.array(bootstrapped_aucs)
    sorted_scores.sort()

    # Calculate the 2.5th and 97.5th percentiles for the 95% confidence interval
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]

    return confidence_lower, confidence_upper


def plot_confusion_matrix(cm, title, save_path):
    """
    Plot and save the confusion matrix as an image.

    Args:
        cm (array-like): The confusion matrix to be plotted.
        title (str): The title of the confusion matrix plot.
        save_path (str): The file path where the confusion matrix image will be saved.
    """
    
    # Create a ConfusionMatrixDisplay object using the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    
    # Plot the confusion matrix with a blue color map
    disp.plot(cmap=plt.cm.Blues)
    
    # Set the title of the plot
    plt.title(title)

    # Save the plot as an image at the specified path
    plt.savefig(save_path)

    # Clear the current plot to free up memory and avoid overlap with future plots
    plt.clf()
    plt.close()

def print_cross_validation_metrics(overall_dict):
    """
    Print the mean metrics for training and validation sets after cross-validation.

    Args:
        overall_dict (dict): A dictionary containing metrics for training and validation sets. 
            The dictionary should have the following keys:
            - 'train_auc': A list of AUC scores for the training set.
            - 'val_auc': A list of AUC scores for the validation set.
            - 'train_sensitivity': A list of sensitivity values for the training set.
            - 'train_specificity': A list of specificity values for the training set.
            - 'val_sensitivity': A list of sensitivity values for the validation set.
            - 'val_specificity': A list of specificity values for the validation set.

    Returns:
        None: The function prints the mean AUC, sensitivity, and specificity for both the training and validation sets.
    """
    # Calculate the mean AUC for training and validation sets
    mean_train_auc = np.mean(overall_dict['train_auc'])
    mean_val_auc = np.mean(overall_dict['val_auc'])

    # Calculate the mean sensitivity for training and validation sets
    mean_train_sensitivity = np.mean(overall_dict['train_sensitivity'])
    mean_train_specificity = np.mean(overall_dict['train_specificity'])

    # Calculate the mean specificity for training and validation sets
    mean_val_sensitivity = np.mean(overall_dict['val_sensitivity'])
    mean_val_specificity = np.mean(overall_dict['val_specificity'])

    logging.info("\nCross-validation complete. Model training complete. Model saved.")
    logging.info(f"Train: AUC: {mean_train_auc:.3f} | Sensitivity: {mean_train_sensitivity:.3f} | specificity: {mean_train_specificity:.3f}")
    logging.info(f"Valid: AUC: {mean_val_auc:.3f} | Sensitivity: {mean_val_sensitivity:.3f} | specificity: {mean_val_specificity:.3f}")

def plot_and_save_confusion_matrix(test_labels, binary_predictions, output_path):
    """
    Generate and save the confusion matrix for the test set.

    Args:
        test_labels (list or np.ndarray): True labels for the test set.
        binary_predictions (list or np.ndarray): Binary predictions made by the model.
        output_path (str): The directory where the confusion matrix image will be saved.

    Returns:
        None: The function generates and saves the confusion matrix plot as a PNG image.
    """
    # Generate the confusion matrix from the test labels and binary predictions
    test_cm = confusion_matrix(test_labels, binary_predictions)

    # Define the path where the confusion matrix image will be saved
    test_cm_path = os.path.join(output_path, 'test_CM.png')

    # Plot the confusion matrix and save it as an image
    plot_confusion_matrix(test_cm, "Test Confusion Matrix", test_cm_path)

def calculate_metrics(test_labels, ensemble_predictions):
    """
    Calculate the optimal threshold, sensitivity, specificity, and Youden index based on the ROC curve.

    Args:
        test_labels (list or np.ndarray): True labels for the test set.
        ensemble_predictions (list or np.ndarray): Model's probability predictions for the positive class.

    Returns:
        tuple: 
            - best_threshold (float): The optimal threshold for classification.
            - best_sensitivity (float): Sensitivity (True Positive Rate) at the optimal threshold.
            - best_specificity (float): Specificity (True Negative Rate) at the optimal threshold.
            - best_youden_index (float): The sum of sensitivity and specificity at the optimal threshold.
    """
    # Compute the false positive rate, true positive rate, and thresholds
    fpr, tpr, thresholds = roc_curve(test_labels, ensemble_predictions)

    # Calculate specificity as 1 - false positive rate
    specificity = 1 - fpr

    # Calculate the sum of sensitivity and specificity (Youden index, helps to find the optimal threshold)
    sensitivity_specificity_sum = tpr + specificity - 1

    # Identify the index of the best threshold, where sensitivity + specificity is maximized
    best_threshold_index = np.argmax(sensitivity_specificity_sum)

    # Extract the best threshold, sensitivity, and specificity based on the optimal threshold index
    best_threshold = thresholds[best_threshold_index]
    best_sensitivity = tpr[best_threshold_index]
    best_specificity = specificity[best_threshold_index]
    
    # Return the best threshold, sensitivity, specificity, and the sum of sensitivity and specificity at that threshold
    return best_threshold, best_sensitivity, best_specificity, sensitivity_specificity_sum[best_threshold_index]


# def compute_attention_mask(attentions, device, head_fusion="mean", discard_ratio=0.9):
#     """
#     Compute an attention mask from the attention weights of a model.

#     This function aggregates the attention weights across different attention heads and layers 
#     and generates a mask based on the top-k attention values, with the option to discard a 
#     portion of them. The mask is normalized and reshaped into a square format.

#     Args:
#         attentions (list of torch.Tensor): List of attention tensors, each with shape 
#                                            [batch_size, num_heads, seq_len, seq_len].
#         device (torch.device): Device to perform the operations (CPU or GPU).
#         head_fusion (str, optional): The method for aggregating attention heads, can be "mean", 
#                                       "max", or "min". Default is "mean".
#         discard_ratio (float, optional): Ratio of top-k attention values to discard. Default is 0.9.

#     Returns:
#         np.ndarray: A 2D numpy array representing the attention mask, normalized to [0,1] and 
#                     reshaped into a square form.
#     """
#     # Initialize the result as the identity matrix
#     result = torch.eye(attentions[0].size(-1)).to(device)
    
#     with torch.no_grad():
#         for attention in attentions:
#             # Aggregate attention heads based on the specified fusion method
#             if head_fusion == "mean":
#                 attention_heads_fused = attention.mean(dim=0)
#             elif head_fusion == "max":
#                 attention_heads_fused = attention.max(dim=0)[0]
#             elif head_fusion == "min":
#                 attention_heads_fused = attention.min(dim=0)[0]
#             else:
#                 raise ValueError("Attention head fusion type not supported")
            
#             attention_heads_fused = attention_heads_fused.to(device)

#             # Flatten the attention matrix and discard the top-k values based on the discard ratio
#             flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
#             _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)
#             indices = indices[indices != 0]
#             flat[0, indices] = 0

#             # Apply the identity matrix and normalize
#             I = torch.eye(attention_heads_fused.size(-1)).to(device)
#             a = (attention_heads_fused + 1.0 * I) / 2
#             a = a / a.sum(dim=-1)

#             # Update the result by multiplying with the aggregated attention matrix
#             result = torch.matmul(a, result)

#     # Reshape the result into a 2D mask and normalize to [0, 1]
#     mask = result[0, 1:]
#     width = int(mask.size(-1) ** 0.5)
#     mask = mask.reshape(width, width).cpu().numpy()
#     mask = mask / np.max(mask)

#     return mask


def compute_attention_mask(attentions, device, head_fusion="mean", discard_ratio=0.9):
    """
    Computes the attention mask by processing multiple attention layers and applying head fusion, residual connections, and normalization.

    Args:
        attentions (list): A list of attention maps, where each element is a tensor of shape [num_heads, seq_len, seq_len].
        device (str): The device on which computations are performed (e.g., 'cpu' or 'cuda').
        head_fusion (str, optional): The method to combine attention heads ('mean', 'max', or 'min'). Defaults to 'mean'.
        discard_ratio (float, optional): The ratio of the attention values to discard when computing the mask. Defaults to 0.9.

    Returns:
        numpy.ndarray: The computed attention mask, normalized to the range [0, 1] and reshaped.
    """

    with torch.no_grad():
        num_layers = len(attentions)  # Get the number of attention layers
        seq_len = attentions[0].size(-1)  # Get the sequence length (assumed to be the same across layers)

        # Step 1: Head Fusion (Mean, Max, Min)
        if head_fusion == "mean":
            # Compute the mean across the attention heads for each layer
            fused_attentions = torch.stack([att.mean(dim=0) for att in attentions])  # [num_layers, tokens, tokens]
        elif head_fusion == "max":
            # Compute the max across the attention heads for each layer
            fused_attentions = torch.stack([att.max(dim=0)[0] for att in attentions])
        elif head_fusion == "min":
            # Compute the min across the attention heads for each layer
            fused_attentions = torch.stack([att.min(dim=0)[0] for att in attentions])
        else:
            raise ValueError("Attention head fusion type not supported")
        
        # Step 2: Add Residual Connection & Normalize
        # Create an identity matrix of size seq_len
        identity_matrix = torch.eye(seq_len).to(device)
        # Add the identity matrix to each attention map (residual connection)
        aug_att_mat = fused_attentions + identity_matrix.unsqueeze(0)  # Broadcast identity matrix across layers
        # Normalize the attention matrix by its row sums
        aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1, keepdim=True)  # Normalize
        
        # Step 3: Compute Joint Attention (Cumulative Attention)
        # Initialize joint_attentions tensor to accumulate the attention values
        joint_attentions = torch.zeros_like(aug_att_mat).to(device)
        joint_attentions[0] = aug_att_mat[0]  # First layer's attention
        # Compute cumulative attention across layers
        for n in range(1, num_layers):
            joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])
        
        # Step 4: Extract Final Attention Map (Last Layer)
        v = joint_attentions[-1]  # Use final cumulative attention
        
        # Step 5: Remove CLS Token & Reshape
        # Extract the attention map without the CLS token (first token)
        mask = v[0, 1:].reshape(int((seq_len - 1) ** 0.5), -1).detach().cpu().numpy()
        # Normalize the mask to the range [0, 1]
        mask = mask / mask.max()  # Normalize to [0,1]
        
        return mask


def save_attention_maps(attentions, images, output_dir, sample_name, class_label, device, head_fusion="min", discard_ratio=0.2):
    """
    Saves attention maps overlaid on images for visualization.

    Args:
        attentions (list): A list of attention maps obtained from the model.
        images (list): A list of images corresponding to the attention maps.
        output_dir (str): The directory where the output images will be saved.
        sample_name (str): The name of the sample to be used in the saved file name.
        class_label (str): The class label of the image (used for organizing the output).
        device (str): The device on which the attention map computation is performed (e.g., 'cpu' or 'cuda').
        head_fusion (str, optional): Method used to combine attention heads ('min' or 'max'). Defaults to 'min'.
        discard_ratio (float, optional): The ratio of the attention values to discard when computing the mask. Defaults to 0.2.

    Returns:
        None
    """

    for i, attention in enumerate(attentions):
        # Step 1: Compute Attention Mask using the provided head fusion method
        mask = compute_attention_mask([attention], device, head_fusion, discard_ratio)
        # Resize the mask to the same size as the input image (224x224)
        mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_LINEAR)

        # Step 2: Preprocess image for visualization
        img = images[i].cpu().numpy().transpose(1, 2, 0)  # Convert image to numpy format (H, W, C)

        # Normalize image to the range [0, 1] for better visualization
        img = (img - img.min()) / (img.max() - img.min())

        # Step 3: Plot and save the images with attention map
        plt.figure(figsize=(10, 5))  # Set the figure size for the plot

        # Plot the original image in the first subplot
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title("Original Image")  # Title for the original image
        plt.axis("off")  # Turn off axis labels

        # Plot the image with the overlayed attention map in the second subplot
        plt.subplot(1, 2, 2)
        plt.imshow(img)
        plt.imshow(mask, cmap='jet', alpha=0.5)  # Overlay the attention map with transparency
        plt.title("Attention Map")  # Title for the attention map
        plt.axis("off")  # Turn off axis labels

        # Step 4: Save the output image
        os.makedirs(os.path.join(output_dir, class_label), exist_ok=True)  # Create class-specific output directory
        # Save the figure with the attention map
        plt.savefig(os.path.join(output_dir, class_label, f"{sample_name}.png"))
        plt.close()  # Close the plot to release memory


def plot_save_curves(predictions, labels, output_path, curve_type, cohort):
    """
    This function plots and saves either an ROC curve or a PR curve based on the specified curve type.
    The function calculates the AUC score and includes it in the plot title and legend.

    Args:
        predictions (numpy.ndarray): The predicted probabilities for the positive class.
        labels (numpy.ndarray): The true labels of the test data.
        output_path (str): The directory where the plot will be saved.
        curve_type (str): The type of curve to plot. Either 'auc_roc' for ROC curve or 'auc_pr' for PR curve.
        cohort (str): A string identifying the cohort or dataset for naming the output file.
    
    Raises:
        ValueError: If an invalid curve_type is provided (not 'auc_roc' or 'auc_pr').
    """
    
    # Check the curve type and plot the corresponding curve
    if curve_type.lower() == "auc_roc":
        # Compute the ROC curve and AUC score
        x, y, _ = roc_curve(labels, predictions)
        auc_score = roc_auc_score(labels, predictions)
        xlabel, ylabel = "False Positive Rate", "True Positive Rate"
        title = "Test ROC Curve"
        filename = f"Test_AUC_ROC_{cohort}.png"
    
    elif curve_type.lower() == "auc_pr":
        # Compute the Precision-Recall curve and AUC score
        x, y, _ = precision_recall_curve(labels, predictions)
        auc_score = auc(y, x)  # PR curve AUC is computed with recall on x-axis
        xlabel, ylabel = "Recall", "Precision"
        title = "Test PR Curve"
        filename = f"Test_PR_{cohort}.png"
    
    else:
        # Raise an error if the curve_type is not valid
        raise ValueError("Invalid curve_type. Choose 'auc_roc' or 'auc_pr'.")
    
    # Plot the curve
    plt.figure()
    plt.plot(x, y, color='r', lw=2, label=f'{title} (AUC = {auc_score:.2f})')
    
    # Add the diagonal line for the ROC curve (representing random performance)
    if curve_type.lower() == "auc_roc":
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', alpha=.8)
    
    # Set the limits and labels for the plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="lower right")
    
    # Save the plot to the specified output path
    save_path = os.path.join(output_path, filename)
    plt.savefig(save_path)
    plt.close()  # Close the plot to free memory
    logging.info(f"Saved {title} at {save_path}")


