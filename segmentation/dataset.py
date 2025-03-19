import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset as BaseDataset
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torch import cat

# Custom imports
from .constants import DataSetConstants


class CVDataset(BaseDataset):
    '''
    Dataset class for Semantic Segmentation
    '''
    def __init__(self, images_fps, masks_fps, augmentation = None, preprocessing = None):
        '''
        Initialize the Dataset.

        Parameters
        ----------
        images_dir: str
            Relative/absolute path to the directory containing input images.

        masks_dir: str
          Path to the directory containing ground truths.

        augmentation: albumentations.Compose: 
            A set of augmentation transforms to apply to both images and masks 

        preprocessing: albumentations.Compose: 
            A set of preprocessing transforms (e.g., normalization, resizing) 
            applied to both images and masks after augmentation. Applied only if provided.
        '''
        self.class_intensity_dict = DataSetConstants.class_intensitiy_dict
        
        # Setting full paths
        self.images_fps = images_fps
        self.masks_fps = masks_fps

        self.augmentation  = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        # Reading in the data
        image = self.original_image(i)
        mask = self.original_mask(i)
       
        if self.augmentation:
            sample = self.augmentation(image = image, mask = mask)
            image, mask = sample['image'], sample['mask']

        # appy preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image = image, mask = mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def original_mask(self, i):
        mask = cv2.imread(self.masks_fps[i], 0)
        # Replacing class_intensity_dict values with their index
        for idx, pixel_intensity in enumerate(self.class_intensity_dict.values()):
            mask[mask == pixel_intensity] = idx
        return mask.astype(np.float32)
    
    def original_image(self, i):
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image.astype(np.float32)
    
    def image_shape(self, i):
        image = cv2.imread(self.images_fps[i])
        return image.shape
    
    def __len__(self):
        return len(self.images_fps)
    
    def get_dataset_dimensions(self):
        """
        Returns the minimum and maximum height and width of the dataset images.

        Returns:
        --------
        dict: 
            {
                'min_height': int,
                'max_height': int,
                'min_width': int,
                'max_width': int
            }
        """
        min_height, min_width = float('inf'), float('inf')
        max_height, max_width = 0, 0

        for img_path in self.images_fps:
            image = cv2.imread(img_path)
            h, w = image.shape[:2]
            
            min_height = min(min_height, h)
            min_width = min(min_width, w)
            max_height = max(max_height, h)
            max_width = max(max_width, w)
        
        return {
            'min_height': min_height,
            'max_height': max_height,
            'min_width': min_width,
            'max_width': max_width
        }

class PointDataset(CVDataset):
    def __init__(self,
                images_fps,
                masks_fps,
                prompt_points = None,
                augmentation = None,
                preprocessing = None,
                concat_heatmap = True
                ):
        
        super().__init__(images_fps, masks_fps, augmentation, preprocessing)
        self.prompt_points = prompt_points
        self.concat_heatmap = concat_heatmap

        # Checking enough prompt points were provided
        if (prompt_points) and (self.__len__() != len(prompt_points)):
            ValueError(f'{len(prompt_points)} provided however exactly {self.__len__()} required')

    def __create_heatmap(self, image_shape, point, sigma=10):
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
        return heatmap.astype(np.float32)
    


    def __random_xy(self, height, width, sigma_ratio=0.2):
        """
        Sample a random (x, y) point from a 2D Gaussian distribution 
        centered in the middle of the image.

        :param height: Image height
        :param width:  Image width
        :param sigma_ratio: Fraction of the dimension used as std dev 
                        (e.g., 0.2 means std dev = 20% of dimension)
        :return: (x, y) coordinates within image bounds
        """
        height, width = int(height), int(width)
        
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
    
    def __filter_mask_clicked(self, mask, prompt_point):
        '''
        Return binary mask (object clicked/not clicked)
        '''
        x, y = prompt_point
        object_clicked = mask[y, x]
        # If on the border, assume background
        if object_clicked == 255:
            object_clicked = 0
            
        filtered_mask = np.where(mask == object_clicked, 1, 0)
        return filtered_mask

    def __getitem__(self, i):
        image = self.original_image(i)
        mask = self.original_mask(i)
        

        # Getting the image dimensions
        height, width, _ = self.image_shape(i)
        
        # Getting the prompt point on the original image dimensions
        prompt_point = self.prompt_points[i] if self.prompt_points else self.__random_xy(height, width)
        # Getting object that was clicked
        mask = self.__filter_mask_clicked(mask, prompt_point)

        if self.augmentation:
            # Apply the augmentation while passing the prompt point as a keypoint.
            sample = self.augmentation(image=image, mask=mask, keypoints=[prompt_point])
            image, mask, keypoint = sample['image'], sample['mask'], sample['keypoints']

            # Checking if our keypoint is still there after augmentation (otherwise put it in the center)
            if len(keypoint) == 0:
                # Shape of mask image is HWC
                x_centre, y_centre = image.shape[0]//2, image.shape[1]//2
                keypoint = [[x_centre, y_centre]]
            
        # Getting the heatmap for the key point & image dimensions (still before preprocessing so numpy)
        heatmap_shape = mask.shape
        heatmap = self.__create_heatmap(heatmap_shape, tuple(keypoint[0]), sigma=10)

        if self.preprocessing:
            sample = self.preprocessing(image = image, mask = mask, heatmap = heatmap)
            image, mask, heatmap = sample['image'], sample['mask'], sample['heatmap']

        if self.concat_heatmap:
            heatmap = heatmap.unsqueeze(0)  # shape (1, H, W)
            return cat([image, heatmap], dim=0), mask

        return image, mask, heatmap
