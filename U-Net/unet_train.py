import logging
import torch.optim as optim
import torch

from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.data import DataLoader

from unet_utils import *
from unet_datasets import create_datasets


def create_model():
    """
    Create and initialize a 2D U-Net model for segmentation tasks.

    Returns:
        model (torch.nn.Module): The initialized U-Net model moved to the appropriate device.
        device (torch.device): The device (GPU or CPU) where the model is loaded.
    """
    logging.info('Loading U-Net model...')
    
    # Select GPU if available, otherwise use CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize a 2D U-Net model with specified architecture parameters
    model = UNet(
        spatial_dims=2,
        in_channels=1,       # Input channel size (e.g., grayscale image)
        out_channels=2,      # Number of output classes
        channels=(16, 32, 64, 128, 256),  # Number of feature maps at each level
        strides=(2, 2, 2, 2),             # Stride values for downsampling
        num_res_units=2,     # Number of residual units in each block
        norm=Norm.BATCH      # Batch normalization
    ).to(device)  # Move model to selected device
    
    logging.info(f'Model created on {device}')
    return model, device


def train(model, train_loader, optimizer, loss_function, dice_metric, device):
    """
    Train the model for one epoch using the given training data.

    Args:
        model (torch.nn.Module): The neural network model to be trained.
        train_loader (DataLoader): DataLoader providing training batches.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        loss_function (callable): Loss function used to compute the training loss.
        dice_metric (Metric): Dice score metric used for evaluating segmentation accuracy.
        device (torch.device): The device (CPU or GPU) to perform computation on.

    Returns:
        epoch_loss (float): Average loss over the entire epoch.
        train_dice_score (float): Average Dice score over the training data.
    """
    model.train()  # Set the model to training mode
    epoch_loss = 0  # Cumulative loss for the epoch
    step = 0        # Step counter for averaging

    for batch_data in train_loader:
        step += 1

        # Move input and label tensors to the target device
        inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
        inputs = inputs.float()
        labels_original = labels.clone()  # Keep original labels for metric calculation

        optimizer.zero_grad()  # Clear the previous gradients

        # Forward pass
        outputs = model(inputs)

        # Prepare labels and compute loss
        labels = labels.squeeze(1).long()  # Remove channel dimension and convert to long type
        loss = loss_function(outputs, labels)

        loss.backward()     # Backward pass to compute gradients
        optimizer.step()    # Update model parameters

        # Calculate Dice score for current batch
        outputs_argmax = torch.argmax(outputs, dim=1, keepdim=True)  # Get predicted class labels
        dice_metric(y_pred=outputs_argmax, y=labels_original)

        epoch_loss += loss.item()  # Accumulate loss

    # Compute average Dice score and reset the metric
    train_dice_score = dice_metric.aggregate().item()
    dice_metric.reset()

    # Compute average loss for the epoch
    epoch_loss /= step

    return epoch_loss, train_dice_score


def validate(model, val_loader, loss_function, dice_metric, device):
    """
    Evaluate the model on the validation dataset.

    Args:
        model (torch.nn.Module): The trained model to be evaluated.
        val_loader (DataLoader): DataLoader providing validation batches.
        loss_function (callable): Loss function used to compute validation loss.
        dice_metric (Metric): Dice score metric used to evaluate segmentation quality.
        device (torch.device): The device (CPU or GPU) for computation.

    Returns:
        epoch_loss_val (float): Average loss over the validation dataset.
        val_dice_score (float): Average Dice score on the validation data.
    """
    model.eval()  # Set the model to evaluation mode
    epoch_loss_val = 0  # Cumulative validation loss
    val_steps = 0       # Number of validation steps

    with torch.no_grad():  # Disable gradient computation for evaluation
        for val_data in val_loader:
            val_steps += 1

            # Move validation data to the selected device
            val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
            val_images = val_images.float()

            # Forward pass
            val_outputs = model(val_images)

            # Calculate Dice score
            val_outputs_argmax = torch.argmax(val_outputs, dim=1, keepdim=True)
            dice_metric(y_pred=val_outputs_argmax, y=val_labels)

            # Prepare labels and compute loss
            val_labels = val_labels.squeeze(1).long()
            val_loss = loss_function(val_outputs, val_labels)

            # Accumulate validation loss
            epoch_loss_val += val_loss.item()

    # Compute average Dice score and reset metric
    val_dice_score = dice_metric.aggregate().item()
    dice_metric.reset()

    # Compute average validation loss
    epoch_loss_val /= val_steps

    return epoch_loss_val, val_dice_score


def train_loop(model, train_loader, val_loader, device, config, output_folder):
    """
    Perform the full training loop for the model, including validation and saving the best model.

    Args:
        model (torch.nn.Module): The U-Net model to be trained.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        device (torch.device): The device (CPU or GPU) used for computation.
        config (dict): Configuration dictionary containing training parameters.
        output_folder (str): Path to the folder where the best model will be saved.

    Returns:
        None
    """
    # Retrieve training hyperparameters from config
    num_epochs = config['training']['epochs']
    learning_rate = config['training']['learning_rate']

    # Define loss function, optimizer, and metric
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    dice_metric = DiceMetric(include_background=False, reduction='mean')

    best_val_dice = 0.0       # Best validation Dice score seen so far
    best_model_path = None    # Path to the best model

    for epoch in range(num_epochs):
        # Train the model for one epoch
        train_loss, train_dice = train(model, train_loader, optimizer, loss_function, dice_metric, device)
        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")

        # Validate the model on the validation set
        val_loss, val_dice = validate(model, val_loader, loss_function, dice_metric, device)
        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Validation Dice: {val_dice:.4f}")

        # Save model if it has the best validation Dice score so far
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_model_path = f"{output_folder}/unet_model.pth"
            torch.save(model.state_dict(), best_model_path)

        # (Optional) Save model at each epoch
        # torch.save(model.state_dict(), f"{output_folder}/model_{epoch+1}.pth")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,  # Set the logging level to INFO
        format='%(message)s'  # Define the logging format
    )

    # Load configuration from YAML file
    config = load_config("unet_config.yaml")

    # Extract training and evaluation settings from config
    batch_size_train = config['training']['batch_size']
    batch_size_val = config['evaluation']['batch_size']
    output_path = config['data']['output']

    # Ensure the output directory exists
    ensure_dir(output_path)

    # Create datasets and corresponding dataloaders
    train_dataset, val_dataset = create_datasets(config)
    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_val)

    # Log dataset statistics
    logging.info(f'Detect {len(train_loader.dataset)} images in train dataloader! '
                 f'{len(val_loader.dataset)} images in val dataloader!')
    logging.info('Dataloader created')

    # Plot sample images from the training dataloader for visualization/debugging
    plot_images_from_dataloader(train_loader, num_images=5)

    # Create U-Net model and determine the computation device
    model, device = create_model()

    # Run the full training loop
    train_loop(model, train_loader, val_loader, device, config, output_path)



