import os
import logging
from pathlib import Path
from resnet_utils import *

if __name__ == "__main__":
    # load the configuration file
    config = load_config("resnet_config.yaml")

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,  # Set the logging level to INFO
        format='%(message)s'  # Define the logging format
    )

    # Define the input and output folders for converting mat files to nifti
    repo_path = Path(__file__).resolve().parent.parent
    mat_dir = os.path.join(repo_path, "dataset", "fullcohort_1_wfs")

    output_folder = os.path.join(repo_path, "dataset", "fullcohort_2_nifti_cleaned")
    ensure_dir(output_folder)

    # Convert mat files to nifti format
    logging.info("Converting files for U-Net model...")
    convert_mat_to_nifti(mat_dir, output_folder)
    logging.info("Convert completed!")

    # Define the input and output folders  for removing the artifact/unqualified images from the full_set
    artifect_folder = os.path.join(repo_path, "ResNet", "output", "artefact_samples")
    remove_artefact_nifti(artifect_folder, output_folder)
    logging.info("Unqualified images are removed from dataset!")