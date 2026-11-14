import os
import pandas as pd

from sklearn.metrics import roc_auc_score, confusion_matrix
from torch.utils.data import DataLoader
from vit_dataset import VitDataset
from pathlib import Path
from vit_utils import *

if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,  # set level to INFO
        format='%(message)s' # output format
    )

    # Load configuration
    config = load_config("vit_config.yaml")
    output_path = config['data']['output']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = initialize_model(config, device)
    test_path = os.path.join(output_path, "test_list.csv")
    test_pd = pd.read_csv(test_path)
    test_list = test_pd['data'].tolist()

    # Get the source directory for the dataset
    repo_path = Path(__file__).resolve().parent.parent

    # Prepare data and paths
    data_dir = config['data']['data_dir']
    augmentations_val = parse_augmentations(config['evaluation']['augumentation'])
    snp = config['data']['snp']

    mask_dir = os.path.join(repo_path, "dataset", "fullcohort_2_mask")
    label_csv = os.path.join(repo_path, "dataset", "pdff_values_sigmaclip.csv")

    # Prepare test dataset and loader
    test_dataset = VitDataset(test_list, data_dir, mask_dir, label_csv, snp=snp, transform=augmentations_val)
    test_loader = DataLoader(test_dataset, batch_size=config['evaluation']['batch_size'], shuffle=False)

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

    # 查找每折中的共有样本并进行平均
    final_attentionmap_path = f"{attention_dir}/final_attention_maps"
    ensure_dir(final_attentionmap_path)
    find_common_samples_and_average(attention_dir, final_attentionmap_path, config)

    # Calculate AUC bootstrap confidence intervals
    confidence_lower, confidence_upper = bootstrap_auc(ensemble_predictions, test_labels)
    logging.info(f"Test AUC: {test_auc:.3f} | 95% CI: [{confidence_lower:.3f}, {confidence_upper:.3f}]")

    # Plot ROC curve 
    plot_save_curves(ensemble_predictions, test_labels, output_path, curve_type="auc_roc", cohort=config['data']['cohort'])

    # Plot PR curve and save it
    plot_save_curves(ensemble_predictions, test_labels, output_path, curve_type="auc_pr", cohort=config['data']['cohort'])

    # Plot confusion matrix for test set
    test_cm = confusion_matrix(test_labels, binary_predictions)
    test_cm_path = os.path.join(output_path, 'test_CM.png')
    plot_confusion_matrix(test_cm, "Test Confusion Matrix", test_cm_path)

    logging.info(f"Final AUC: {test_auc:.3f}, Sensitivity: {best_sensitivity:.3f}, Specificity: {best_specificity:.3f}")