import logging
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim

from pathlib import Path
from torch.utils.data import DataLoader
from resnet_utils import *
from resnet_datasets import create_datasets


def create_model(num_classes=2):
    """
    Load and initialize the ResNet model.

    Args:
        num_classes (int): Number of output classes for the model.

    Returns:
        tuple: (model, device) where model is the initialized ResNet model and device is the device (CPU or GPU) used.
    """
    logging.info('Loading ResNet model...')
    
    # Load the pre-trained ResNet model, change the output layer to match the number of classes
    model = torchvision.models.resnet50(weights="IMAGENET1K_V2")
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Handle multiple GPUs
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    logging.info(f'Model created on {device}')
    return model, device

def train(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch.

    Args:
        model (torch.nn.Module): The model to train.
        dataloader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (torch.device): Device to perform training on (CPU or GPU).

    Returns:
        tuple: (average_loss, accuracy) where average_loss is the average training loss and accuracy is the training accuracy.
    """
    # Set the model to training mode
    model.train()  
    
    # Initialize variables to track total loss, correct predictions, and total samples
    total_loss = 0.0
    correct = 0
    total = 0

    # Iterate through batches of images and labels in the dataloader
    for images, labels in dataloader:
        # Move images and labels to the specified device (CPU or GPU)
        images = images.to(device, dtype=torch.float)
        labels = labels.to(device)

        # Zero the gradients before the backward pass
        optimizer.zero_grad()

        # Perform a forward pass through the model
        outputs = model(images)

        # Calculate the loss
        loss = criterion(outputs, labels)

        # Backpropagate the gradients
        loss.backward()

        # Update the model parameters based on the gradients
        optimizer.step()

        # Accumulate the total loss for this batch
        total_loss += loss.item()

        # Get the predicted class labels
        _, predicted = outputs.max(1)

        # Update the total number of samples and correct predictions
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    # Return the average loss and accuracy for the epoch
    return total_loss / len(dataloader), 100.0 * correct / total


def validate(model, dataloader, criterion, device):
    """
    Validate the model on the validation dataset.

    Args:
        model (torch.nn.Module): The model to validate.
        dataloader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        criterion (torch.nn.Module): Loss function.
        device (torch.device): Device to perform validation on (CPU or GPU).

    Returns:
        tuple: (average_loss, accuracy) where average_loss is the average validation loss and accuracy is the validation accuracy.
    """
    # Set the model to evaluation mode (disables dropout, batch norm, etc.)
    model.eval()  
    
    # Initialize variables to track total loss, correct predictions, and total samples
    total_loss = 0.0
    correct = 0
    total = 0

    # Disable gradient calculations since we're only performing inference
    with torch.no_grad():
        # Iterate through batches of images and labels in the dataloader
        for images, labels in dataloader:
            # Move images and labels to the specified device (CPU or GPU)
            images = images.to(device, dtype=torch.float)
            labels = labels.to(device)

            # Perform a forward pass through the model
            outputs = model(images)

            # Calculate the loss
            loss = criterion(outputs, labels)

            # Accumulate the total loss for this batch
            total_loss += loss.item()

            # Get the predicted class labels
            _, predicted = outputs.max(1)

            # Update the total number of samples and correct predictions
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    # Return the average loss and accuracy for the validation dataset
    return total_loss / len(dataloader), 100.0 * correct / total


def train_loop(model, train_loader, val_loader, config, device):
    """
    Run the training loop for the specified number of epochs.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        config (dict): Configuration dictionary containing training parameters.
        device (torch.device): Device to perform training on (CPU or GPU).

    Returns:
        tuple: (train_loss_list, train_acc_list, val_loss_list, val_acc_list, lr_values) where each list contains the respective metrics for each epoch.
    """
    num_epochs = config['training']['epochs']
    learning_rate = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    
    # Define loss function, optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Lists to store metrics
    train_loss_list, train_acc_list = [], []
    val_loss_list, val_acc_list = [], []
    lr_values = []

    # Training loop
    logging.info('Training started...')
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        current_lr = scheduler.get_last_lr()[0]
        
        # Store metrics
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        lr_values.append(current_lr)
        
        scheduler.step()

        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, LR: {current_lr}")

        # Save the best model so far based on validation loss in case of early stopping
        if val_loss == min(val_loss_list):
            model_path = os.path.join(config['data']['output'], 'resnet_model.pth')
            torch.save(model.state_dict(), model_path)
    
    return train_loss_list, train_acc_list, val_loss_list, val_acc_list, lr_values

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,  # Set the logging level to INFO
        format='%(message)s'  # Define the logging format
    )
    
    # Load configuration
    config = load_config("resnet_config.yaml")

    batch_size = config['training']['batch_size']
    mode = config['training']['mode']
    output_path = config['data']['output']
    ensure_dir(output_path)

    # Get the source directory for the dataset
    repo_path = Path(__file__).resolve().parent.parent
    mat_dir = os.path.join(repo_path, "dataset", "fullcohort_1_wfs")

    # Create datasets and dataloaders
    logging.info('Creating datasets and dataloaders...\n')

    train_dataset, val_dataset = create_datasets(config, mat_dir, mode)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    logging.info('Dataloader created')

    # Create and train model
    model, device = create_model()
    train_loss_list, train_acc_list, val_loss_list, val_acc_list, lr_values = train_loop(model, train_loader, val_loader, config, device)

    # Save results and model
    save_train_results(output_path, model, train_loss_list, train_acc_list, val_loss_list, val_acc_list, lr_values, batch_size, config['training']['weight_decay'])