import os
from pathlib import Path
from unet_utils import *

if __name__ == "__main__":
    # load the configuration file
    config = load_config("unet_config.yaml")

    # Define the repo
    repo_path = Path(__file__).resolve().parents[1]

    # Calculate the PDFF values
    mri_folder = os.path.join(repo_path, "dataset", "fullcohort_2_nifti_cleaned")
    mask_folder = os.path.join(repo_path, "dataset", "fullcohort_2_mask")
    output_folder = os.path.join(repo_path, "dataset")

    pdff_df = pdff_calculation(mri_folder, mask_folder, output_folder, config)

    # Import csv file here also works
    # import pandas as pd
    # pdff_df = pd.read_csv('PATH')

    # Extract MRI images for SLD cohort
    dicom_folder = config['data']['dicom_path']
    sld_folder = os.path.join(repo_path, "Vision-Transformer", "training_dataset", "SLD_cohort")
    ensure_dir(sld_folder)

    sld_df = pdff_df[pdff_df['PDFF_median'] >= 0.05]
    process_set("SLD", sld_df, sld_folder, dicom_folder)

    # EXTRACT MRI images for nonSLD cohort
    nonSLD_folder = os.path.join(repo_path, "Vision-Transformer", "training_dataset", "nonSLD_cohort")
    ensure_dir(nonSLD_folder)

    nonsld_df = pdff_df[pdff_df['PDFF_median'] < 0.05]
    process_set("nonSLD", nonsld_df, nonSLD_folder, dicom_folder)



    
