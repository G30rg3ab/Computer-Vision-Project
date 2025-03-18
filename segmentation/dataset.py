import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset as BaseDataset
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

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
        return mask
    
    def original_image(self, i):
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
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
    
