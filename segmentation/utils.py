import os
import numpy as np
import cv2
import albumentations as albu
from sklearn.model_selection import train_test_split 
from albumentations.pytorch import ToTensorV2
import torch
import csv


# Custom imports
from .constants import VisualisationConstants


class preprocessing():
    @ staticmethod
    def train_val_split(x_trainVal_dir, y_trainVal_dir, val_size = 0.2, random_state = 42):
        # Getting the names of the images
        images_idx = sorted(os.listdir(x_trainVal_dir)) # Sorting to match with labels
        labels_idx = sorted(os.listdir(y_trainVal_dir)) # Sorting to match with images


        # Splitting into training and validation
        X_train, X_val, y_train, y_val = train_test_split(images_idx,
                                                        labels_idx,
                                                        test_size=val_size,
                                                        random_state=random_state
                                                        )
        # Concatenating the data directory onto each of the names
        X_train_fps = [os.path.join(x_trainVal_dir, image_idx) for image_idx in X_train]
        y_train_fps = [os.path.join(y_trainVal_dir, label_idx) for label_idx in y_train]

        X_val_fps = [os.path.join(x_trainVal_dir, image_idx) for image_idx in X_val]
        y_val_fps = [os.path.join(y_trainVal_dir, label_idx) for label_idx in y_val]
        
        return X_train_fps, X_val_fps, y_train_fps, y_val_fps

    @ staticmethod
    def get_testing_paths(x_test_dir, y_test_dir):
        # Getting the names of the images
        images_idx = sorted(os.listdir(x_test_dir)) # Sorting to match with labels
        labels_idx = sorted(os.listdir(y_test_dir)) # Sorting to match with images

        x_test_fps = [os.path.join(x_test_dir, image_idx) for image_idx in images_idx]
        y_test_fps = [os.path.join(y_test_dir, label_idx) for label_idx in labels_idx]

        return x_test_fps, y_test_fps

    @ staticmethod
    def get_training_augmentation():
        '''
        This function returns an albumentations.Compose element 
        which peforms augmentation during training.
        '''
        train_transform = [
            albu.RandomCrop(256, 416, pad_if_needed=True, border_mode=cv2.BORDER_REFLECT),
            albu.HorizontalFlip(p=0.5),
            albu.Affine(translate_percent=(-0.05, 0.05),  scale=(0.95, 1.05), rotate=(-15, 15),p=0.0,border_mode=0,fill_mask=255),
            albu.OneOf([
                    albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
                    albu.CLAHE(p=1),
                    albu.HueSaturationValue(p=1)
                    ],
                p=0.9,
            ),
                albu.GaussNoise(p=0.2),
        ]
        return albu.Compose(train_transform,
                            keypoint_params=albu.KeypointParams(format='xy', remove_invisible=True)
                            )
    
    @staticmethod
    def get_validation_augmentation():
        '''
        This function returns an albumentations.Compose element
        which performs augmentation for validation set.
        '''
        # Add sufficient padding to ensure image is divisible by 32
        test_transform = [
            #albu.PadIfNeeded(min_height=768, min_width=768, border_mode=0, fill_mask=255),
            #albu.Resize(768, 768) # Incase 768 x 768 is not enough, probably should fix later in a better way
            albu.Resize(256, 416)
        ]
        return albu.Compose(test_transform,
                            keypoint_params=albu.KeypointParams(format='xy', remove_invisible=True),
                            is_check_shapes=False)


    @staticmethod
    def get_preprocessing(preprocessing_fn=None):
        '''
        Normalise the data and convert to a tensor. Use the 
        preprocessing_fn to chain another preprocessing step if needed
        '''
        _transform = []
        if preprocessing_fn:
            _transform.append(albu.Lambda(image=preprocessing_fn))
        
        # Normalisation
        _transform.append(albu.Normalize(
            mean=[0, 0, 0],         
            std=[1, 1, 1],           
            max_pixel_value=255.0
        ))
        
        # Convert to PyTorch tensor (channels-first)
        _transform.append(ToTensorV2())

        return albu.Compose(
                _transform,
                is_check_shapes=False,
                additional_targets={'heatmap': 'mask'}  # Tell albu to treat heatmap as mask
        )
    

class model_utils():
    def __init__(self):
        pass

    @staticmethod
    def save_checkpoint(checkpoint, filename="final_model.pth"):
        """
        Save the trained model along with metadata for experiment tracking.
        """
        # **Step 1: Save model to file**
        torch.save(checkpoint, filename)
        print(f"=> Final model and metadata saved to {filename}")
        
    @staticmethod
    def return_checkpoint_from(checkpoint_path):
        print(f'=> Fetching checkpoint from {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), weights_only=False)
        return checkpoint
    
    @staticmethod
    def load_checkpoint(checkpoint_path, model):
        print(f'=> Loading checkpoint from {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), weights_only=True)
        model.load_state_dict(checkpoint['state_dict'])


class traininglog():
    @staticmethod
    def log_training(log_filename, **kwargs):
        """
        Logs arbitrary key-value pairs to a CSV file.
        
        Parameters:
            log_filename (str): The path to the CSV log file.
            **kwargs: Arbitrary key-value pairs to log.
                    For example: epoch=1, val_iou=0.85, hyperparameters={'lr': 0.0001}
        """
        file_exists = os.path.isfile(log_filename)
        fieldnames = list(kwargs.keys())  # Determine fieldnames from kwargs
        
        with open(log_filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()  # Write header only once if file does not exist.
            writer.writerow(kwargs)


def create_heatmap(image_shape, point, sigma=15):
    """
    Create a Gaussian heatmap centered at the given point.

    Args:
    image_shape (tuple): Shape of the image (height, width).
    point (tuple): Coordinates (x, y) of the prompt. Note: x is column index, y is row index.
    sigma (float): Standard deviation controlling the spread of the heatmap.

    Returns:
    np.array: A heatmap of shape (height, width) with values in [0, 1].
    """
    
    height, width = image_shape
    # Create a coordinate grid for the image
    x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height))
    
    # Calculate the squared distance from the prompt point
    squared_distance = (x_grid - point[0])**2 + (y_grid - point[1])**2
    
    # Create the Gaussian heatmap
    heatmap = np.exp(-squared_distance / (2 * sigma**2))
    return heatmap


def random_xy(shape, sigma_ratio=0.2):
    """
    Sample a random (x, y) point from a 2D Gaussian distribution 
    centered in the middle of the image.

    :param height: Image height
    :param width:  Image width
    :param sigma_ratio: Fraction of the dimension used as std dev 
                    (e.g., 0.2 means std dev = 20% of dimension)
    :return: (x, y) coordinates within image bounds
    """
    height, width = shape
    
    # Mean at the center of the image
    x_mean, y_mean = width / 2.0, height / 2.0
    
    # Standard deviation as a fraction of the dimension
    x_sigma = sigma_ratio * width
    y_sigma = sigma_ratio * height
    
    # Draw samples from the normal distribution
    x = int(np.random.normal(x_mean, x_sigma))
    y = int(np.random.normal(y_mean, y_sigma))
    
    # Clamp to valid image bounds
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    
    return [x, y]

def numpy_image_from(tensor_image):
    """
    Convert a tensor image (C, H, W) to a NumPy image (H, W, C) of type uint8.
    
    Assumes:
      - The tensor is in the format (C, H, W)
      - If the values are in [0, 1], they will be scaled to [0, 255].
      
    Parameters:
      tensor_image (torch.Tensor): The input image.
      
    Returns:
      np.ndarray: The image in (H, W, C) format with dtype uint8.
    """

    # If it's a torch tensor, bring it to CPU and detach it.
    if isinstance(tensor_image, torch.Tensor):
        np_img = tensor_image.cpu().detach().numpy()
    else:
        raise ValueError('Input image is not tensor')

    # Convert from channel first to channel last
    np_img = np.transpose(np_img, (1, 2, 0))
    
    # If the image values are in the range [0, 1], scale them to [0, 255]
    if np_img.max() <= 1.0:
        np_img = np_img * 255
    
    # Clip values and convert to uint8
    np_img = np.clip(np_img, 0, 255).astype(np.uint8)
    return np_img