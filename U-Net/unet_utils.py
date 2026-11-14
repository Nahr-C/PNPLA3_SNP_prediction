import os
import yaml
import shutil
import logging
import pydicom
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt

from monai.transforms import Compose, EnsureChannelFirstd, ScaleIntensityd, RandFlipd, RandAffined, RandGaussianNoised, ToTensord
from matplotlib.colors import ListedColormap


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

def delete_dir(directory):
    """
    Delete a directory and all its contents.

    Args:
        directory (str): Directory path to delete.
    """
    if os.path.exists(directory):
        shutil.rmtree(directory)    

def parse_augmentations(augmentations, keys):
    """
    Parse the list of augmentation operations and return a composed transformation.

    This function takes a list of augmentation operations as strings, maps them to 
    the corresponding transformation objects, and returns a composed transformation 
    that applies all these augmentations sequentially.

    Parameters:
    - augmentations: A list of augmentation operation names, such as 'RandFlipd_axis0', 'ScaleIntensityd', etc.
    - keys: A list of keys used for augmentation, typically including 'image' and 'label'.

    Returns:
    - Compose: The composed transformation object that applies all augmentations in sequence.
    """
    
    aug_list = []  # Initialize an empty list to hold augmentation operations
    
    # Create a mapping of augmentation strings to the actual transformation objects
    transform_mapping = {
        "EnsureChannelFirstd": EnsureChannelFirstd(keys=keys, channel_dim='no_channel'),
        "ScaleIntensityd": ScaleIntensityd(keys=keys[0]),
        "RandFlipd_axis0": RandFlipd(keys, prob=0.5, spatial_axis=0),
        "RandFlipd_axis1": RandFlipd(keys, prob=0.5, spatial_axis=1),
        "RandAffined": RandAffined(keys=keys, prob=0.5, rotate_range=(np.pi / 10)),
        "RandGaussianNoised": RandGaussianNoised(keys=keys[0], prob=0.2, mean=0.0, std=0.1),
        "ToTensord": ToTensord(keys=keys)
    }
    
    for aug in augmentations:
        # Use the mapping to find the corresponding augmentation operation (safer than eval)
        if aug in transform_mapping:
            aug_list.append(transform_mapping[aug])  # Add the transformation to the list
        else:
            # Raise an error if an unknown augmentation is provided
            raise ValueError(f"Unknown augmentation: {aug}")
    
    # Return the composed transformation that applies all the augmentations sequentially
    return Compose(aug_list)


def plot_images_from_dataloader(dataloader, num_images=5):
    """
    Plot a specified number of images and their corresponding masks from a dataloader.

    This function iterates through the dataloader, retrieves image and mask pairs, and 
    displays them using matplotlib. It visualizes the first channel of both the image 
    and the mask for each sample.

    Parameters:
    - dataloader: PyTorch DataLoader that provides batches of images and masks.
    - num_images: The number of image-mask pairs to plot (default is 5).
    """
    for i, batch in enumerate(dataloader):
        if i >= num_images:  # Stop after plotting the specified number of images
            break

        image, mask = batch  # Retrieve the image and mask from the batch

        # Convert the image and mask to numpy arrays for plotting
        image = image.detach().cpu().numpy()
        mask = mask.detach().cpu().numpy()

        # Rearrange the image dimensions for correct plotting (H, W, C, N)
        image = np.transpose(image, (1, 2, 3, 0))

        # Prepare the mask for visualization by converting it to float and handling background
        mask_array = mask.astype(float)
        mask_array[mask_array == 0.0] = np.nan  # Set background to NaN for transparency
        mask = np.transpose(mask_array, (1, 2, 3, 0))

        # Define a colormap for the mask (dark red with full opacity)
        colors = [(1, 0, 0, 1)]  # Red color for the mask
        cmap_red = ListedColormap(colors)

        # Plot the image and the mask side by side
        plt.subplot(1, 2, 1)
        plt.imshow(image[0, ..., 0], cmap='gray')  # Display the first channel of the image
        plt.title("Image")

        plt.subplot(1, 2, 2)
        plt.imshow(mask[0, ..., 0], cmap=cmap_red)  # Display the first channel of the mask
        plt.title("Mask")

        plt.show()  # Display the plot


def display_image_withseg(filename, mask_array, ax, num, cmap='gray', clim=[0, 100]):
    """
    Display an image slice with a superimposed mask on the given axis.

    This function shows a specific slice of an image and overlays a mask on top of it. 
    The mask is displayed in dark red with partial transparency.

    Parameters:
    - filename: The image data (assumed to be a 3D array) from which a specific slice is displayed.
    - mask_array: The mask data to be overlaid on the image. It should match the size of the image slice.
    - ax: The matplotlib axis object on which the image and mask will be displayed.
    - num: The slice index to be displayed from the image.
    - cmap: The colormap to use for displaying the image (default is 'gray').
    - clim: The color limits for the image colormap (default is [0, 100]).
    """
    img_array = filename

    # Create a colormap with NaN mapped to transparent and 1 mapped to dark red
    colors = [(1, 0, 0, 1)]  # Red color with full opacity for the mask
    cmap_red = ListedColormap(colors)

    # Display the image slice with the specified colormap and color limits
    ax.imshow(img_array[num, :, :], cmap=cmap, clim=clim)

    # Overlay the mask with partial transparency
    ax.imshow(mask_array[:, :], cmap=cmap_red, alpha=0.5)

    # Hide the axis
    ax.axis('off')


def display_image(filename, ax, title, num, cmap='gray', clim=[0, 100]):
    """
    Display an image slice on the given axis.

    This function displays a specific slice of an image on the provided axis. 
    The slice is visualized using a colormap and can have customizable color limits.

    Parameters:
    - filename: The image data (assumed to be a 3D array) from which a specific slice is displayed.
    - ax: The matplotlib axis object on which the image will be displayed.
    - title: The title to be displayed on the image.
    - num: The slice index to be displayed from the image.
    - cmap: The colormap to use for displaying the image (default is 'gray').
    - clim: The color limits for the image colormap (default is [0, 100]).
    """
    img_array = filename

    # Display the image slice with the specified colormap and color limits
    ax.imshow(img_array[num, :, :], cmap=cmap, clim=clim)

    # Set the title of the subplot
    ax.set_title(title, fontsize=15)

    # Hide the axis for better visualization
    ax.axis('off')


def get_dicom_file(folder_path):
    """
    Retrieve the file path of the DICOM file with the smallest SeriesNumber from a given folder.

    This function reads the DICOM series in a specified folder and selects the DICOM file 
    associated with the sequence that has the smallest SeriesNumber.

    Parameters:
    - folder_path: The path to the folder containing the DICOM files.

    Returns:
    - The file path of the DICOM file with the smallest SeriesNumber (sorted by filename).
    """
    reader = sitk.ImageSeriesReader()

    # Get the IDs of all DICOM series in the folder
    series_ids = reader.GetGDCMSeriesIDs(folder_path)

    # List to store the SeriesNumber of each DICOM series
    series_numbers = []

    # Loop through each series ID and extract its SeriesNumber
    for series_id in series_ids:
        dicom_names = reader.GetGDCMSeriesFileNames(folder_path, series_id)
        ds = sitk.ReadImage(dicom_names[0])  # Read the first file of the series to get metadata
        series_number = int(ds.GetMetaData('0020|0011'))  # Extract SeriesNumber from metadata
        
        series_numbers.append(series_number)

    # Find the series with the minimum SeriesNumber
    first_series_id = series_ids[np.argmin(series_numbers)]

    # Get the filenames of the selected series and sort them
    first_series_filenames = list(reader.GetGDCMSeriesFileNames(folder_path, first_series_id))
    first_series_filenames.sort()

    # Return the fourth file (index 3) in the sorted list of filenames (most contrast)
    return first_series_filenames[3]



def display_original(filename, ax, title, cmap='gray'):
    """
    Display the first slice of the image array with a specified colormap.

    Parameters:
    - filename: The image array to be displayed.
    - ax: The axes object where the image will be plotted.
    - title: The title to be displayed on top of the image.
    - cmap: The colormap to use for displaying the image (default is 'gray').

    This function displays the first slice of the image array along the z-axis.
    """
    img_array = filename
    
    # Display the first slice of the image with the given colormap
    ax.imshow(img_array[0, :, :], cmap=cmap)
    
    # Set the title for the plot
    ax.set_title(title, fontsize=20)
    
    # Remove axis for a cleaner display
    ax.axis('off')


def plot_masked_img(img_nifti_folder, img_dicom_folder, mask_folder, output_folder):
    """
    Plot original DICOM images and their corresponding NIfTI images and masks side by side, and save the results.

    Parameters:
    - img_nifti_folder: Folder path containing NIfTI image files.
    - img_dicom_folder: Folder path containing DICOM image files.
    - mask_folder: Folder path containing the mask files corresponding to NIfTI images.
    - output_folder: Folder path to save the output plotted images.

    This function processes NIfTI images and their corresponding masks, along with DICOM images, and
    displays them in a side-by-side layout. It saves the result as PNG images in the specified output folder.
    """
    
    # Get the list of sample files (only those with '_mask.nii.gz' suffix)
    sample_files = [file for file in os.listdir(mask_folder) if file.endswith('_mask.nii.gz')]
    
    # Extract file numbers from the mask filenames (assumes the number is at the start of the file)
    file_numbers = [int(file.split('_')[0]) for file in sample_files]

    count = 0
    for number in file_numbers:
        # Get paths for NIfTI images, masks, and the corresponding DICOM file
        img_nifti_path = os.path.join(img_nifti_folder, f'{number}.nii.gz')
        mask_path = os.path.join(mask_folder, f'{number}_mask.nii.gz')
        img_nifti = sitk.ReadImage(img_nifti_path)
        mask = sitk.ReadImage(mask_path)
        
        dicom_patient_folder = os.path.join(img_dicom_folder, str(number))
        img_dicom_path = get_dicom_file(dicom_patient_folder)
        img_dicom = sitk.ReadImage(img_dicom_path)

        # Convert images to numpy arrays for processing
        img_nifti_array = sitk.GetArrayFromImage(img_nifti)
        mask_array = sitk.GetArrayFromImage(mask)
        img_dicom_array = sitk.GetArrayFromImage(img_dicom)
        
        # Convert mask to float and set zeros to NaN for better visualization
        mask_array = mask_array.astype(float)
        mask_array[mask_array == 0.0] = np.nan

        # Create a figure with subplots for displaying the images
        fig, axs = plt.subplots(2, 5, figsize=(25, 10))

        axs[1, 0].axis('off')
        display_image_withseg(img_nifti_array, mask_array, axs[1, 1],  0)
        display_image_withseg(img_nifti_array, mask_array, axs[1, 2], 1)
        display_image_withseg(img_nifti_array, mask_array, axs[1, 3], 2, clim=[0,1], cmap='jet')
        display_image_withseg(img_nifti_array, mask_array, axs[1, 4], 3)

        display_original(img_dicom_array, axs[0, 0], 'Original')
        display_image(img_nifti_array, axs[0, 1], 'water', 0)
        display_image(img_nifti_array, axs[0, 2], 'fat', 1)
        display_image(img_nifti_array, axs[0, 3], 'ff', 2, clim=[0,1], cmap='jet')
        display_image(img_nifti_array, axs[0, 4], 'r2star', 3)

        # Adjust layout to make space for the title
        plt.tight_layout()
        pos1 = axs[0, 0].get_position()
        axs[0, 0].set_position([pos1.x0, pos1.y0 - 0.25,  pos1.width, pos1.height])

        # Save the plotted figure as a PNG file
        plot_name = f"{number}.png"
        plot_path = os.path.join(output_folder, plot_name)
        plt.savefig(plot_path)
        plt.close()

        count += 1
        logging.info(f'{count} files are plotted successfully!')

def sigma_clipping(img_array, mask_array, sigma, max_iters, type):
    """
    Iterative lower sigma clipping threshold calculation.
    """
    current_mask = mask_array > 0

    for i in range(max_iters):
        masked_data = img_array[current_mask]
        if masked_data.size == 0:
            raise ValueError("No valid data in masked area during iteration.")

        median_val = np.median(masked_data)
        std_val = np.std(masked_data)

        if type == 'upper':
            threshold = median_val + sigma * std_val
            new_mask = (mask_array > 0) & (img_array <= threshold)

        elif type == 'lower':
            threshold = median_val - sigma * std_val
            new_mask = (mask_array > 0) & (img_array >= threshold)

        else:
            raise ValueError("Invalid type. Use 'upper' or 'lower'.")

        if np.array_equal(new_mask, current_mask):
            break

        current_mask = new_mask

    return threshold

def normalise_seg(img_array, mask_array):

    # find max and min value in masked area
    min_val = img_array[mask_array > 0].min()
    max_val = img_array[mask_array > 0].max()

    # create a copy to store normalized image
    normalised_img = np.copy(img_array)

    # normalize masked area
    normalised_img[mask_array > 0] = (normalised_img[mask_array > 0] - min_val) / (max_val - min_val)

    return normalised_img

def pdff_calculation(mri_folder, mask_folder, output_folder, config):
    # 初始化DataFrame
    df = pd.DataFrame(columns=['ID', 'eid', 'PDFF_mean', 'PDFF_median'])

    # 定义一个空的列表，用于存放即将创建的临时DataFrames
    df_list = []

    r2star_upper_sigma = config['PDFF_calculation']['r2star_upper_sigma']
    r2star_lower_sigma = config['PDFF_calculation']['r2star_lower_sigma']
    ff_upper_sigma = config['PDFF_calculation']['ff_upper_sigma']
    max_iters = config['PDFF_calculation']['max_iters']

    # 遍历mri文件夹
    number = 1
    for mask_file in os.listdir(mask_folder):
        if mask_file.endswith(".nii.gz"):

            base_name = mask_file.split('_')[0]
            mri_file = base_name + '.nii.gz'
            mri_path = os.path.join(mri_folder, mri_file)

            mask_path = os.path.join(mask_folder, mask_file)

            # 读取MRI图像
            mri_image = sitk.ReadImage(mri_path)
            mri_array = sitk.GetArrayFromImage(mri_image)
            mri_ff = mri_array[2, :, :]
            mri_r2star = mri_array[3, :, :]

            # 读取mask图像
            mask_image = sitk.ReadImage(mask_path)
            mask_array = sitk.GetArrayFromImage(mask_image)

            # normalise masked area in r2star image
            normalised_r2star = normalise_seg(mri_r2star, mask_array)
            normalise_ff = normalise_seg(mri_ff, mask_array)

            # sigma clipping for lower threshold, and determined value for upper threshold
            r2star_lower_threshold = sigma_clipping(normalised_r2star, mask_array, r2star_lower_sigma, max_iters, type='lower')
            r2star_upper_threshold = sigma_clipping(normalised_r2star, mask_array, r2star_upper_sigma, max_iters, type='upper')
            ff_upper_threshold = sigma_clipping(normalise_ff, mask_array, ff_upper_sigma, max_iters, type='upper')

            original_mask_array = np.copy(mask_array)
            mask_array[(mask_array > 0) & ((normalised_r2star > r2star_upper_threshold) | (normalised_r2star < r2star_lower_threshold))] = 0
            mask_array[(mask_array > 0) & (normalise_ff > ff_upper_threshold)] = 0

            mask_ff = mask_array[:, :]

            # 应用mask并计算PDFF值
            masked_ff = mri_ff[mask_ff == 1]
            pdff_mean = np.mean(masked_ff)
            pdff_median = np.median(masked_ff)

            # 创建一个包含当前结果的临时DataFrame并添加到列表中
            temp_df = pd.DataFrame({'ID': [base_name], 'PDFF_mean': [pdff_mean], 'PDFF_median': [pdff_median]})
            df_list.append(temp_df)

            number += 1
            print(f'{number} files have been calculated!')

    # 使用concat连接所有的临时DataFrames
    df = pd.concat(df_list, ignore_index=True)
    df['ID'] = pd.to_numeric(df['ID'], errors='coerce')
    df = df.sort_values(by='ID')

    # 保存为CSV文件
    csv_path = os.path.join(output_folder, "pdff_values_sigmaclip.csv")
    df.to_csv(csv_path, index=False)

    return df

def extract_image(file_name, file_path, first_series_number, output_folder):
    """
    Extract pixel array from a DICOM file, pad it vertically, and save as .npy for training.

    Args:
        file_name (str): Base name to use for output file.
        file_path (str): Path to the DICOM file.
        first_series_number (int): SeriesNumber to match.
        output_folder (str): Folder where .npy files will be saved.
    """
    echo_time_dict = {
        "1.2": "1",
        "3.2": "2",
        "5.2": "3",
        "7.2": "4",
        "9.2": "5",
        "11.2": "6"
    }

    if file_path.endswith('.dcm'):
        try:
            info = pydicom.dcmread(file_path)

            if info.SeriesNumber == first_series_number:
                pixel_array = info.pixel_array

                # 如果是灰度图，转换为 3 通道
                if pixel_array.ndim == 2:
                    pixel_array = np.stack([pixel_array]*3, axis=-1)

                # 添加 top & bottom padding（每边 12 像素）
                top_padding = np.zeros((12, pixel_array.shape[1], pixel_array.shape[2]), dtype=np.uint8)
                bottom_padding = np.zeros((12, pixel_array.shape[1], pixel_array.shape[2]), dtype=np.uint8)
                padded_array = np.vstack((top_padding, pixel_array, bottom_padding))

                echo_time = str(info.EchoTime)
                echo_idx = echo_time_dict[echo_time]  # assume always valid

                output_path = os.path.join(output_folder, f"{file_name}_{echo_idx}.npy")
                np.save(output_path, padded_array)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")


def check_Seriesnumber(folder_path):   
    # empty list to store all series numbers
    series_numbers = []    
    
    # traverse patient folder and get series number of each DICOM file
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if file_path.endswith('.dcm'):
            file_path = os.path.normpath(file_path)
            ds = pydicom.dcmread(file_path)
            series_number = ds.SeriesNumber
            series_numbers.append(series_number)
    # remove same value is the list and choose the smallest sequence
    unique_series_numbers = np.unique(series_numbers)
    first_series_numbers = np.sort(unique_series_numbers)[0]
    
    return first_series_numbers

def process_set(name_dataset, pdff_df, output_folder, dicom_folder):

    count = 1

    # Read patient IDs from DataFrame (assumes column name is 'ID')
    patient_list = pdff_df['ID'].astype(str).tolist()

    # traverse patient folder in target folder
    for folder_id in patient_list:
        patient_folder_path = os.path.join(dicom_folder, folder_id)
        # find first(smallest) series number
        first_series_number = check_Seriesnumber(patient_folder_path)

        # traverse all DICOM files in a single patient folder
        for file_name in os.listdir(patient_folder_path):
            file_path = os.path.join(patient_folder_path, file_name)

            # extract images from DICOM files    
            extract_image(folder_id, file_path, first_series_number, output_folder)
    
        print(f'Processing {name_dataset} folder, {count} patients have been processed!')
        count += 1