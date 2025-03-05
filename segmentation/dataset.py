import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset as BaseDataset
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader


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

        classes: list of str:
            optional, list of class names to extract from the masks. If None, the dataset may default to all available classes.

        augmentation: albumentations.Compose: 
            A set of augmentation transforms to apply to both images and masks 

        preprocessing: albumentations.Compose: 
            A set of preprocessing transforms (e.g., normalization, resizing) 
            applied to both images and masks after augmentation. Applied only if provided.
        '''
        self.class_intensity_dict = {'background':0, 'cat':38, 'dog':75}
        
        # Setting full paths
        self.images_fps = images_fps
        self.masks_fps = masks_fps

        self.augmentation  = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        # Reading in the data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)

        # Replacing class_intensity_dict values with their index
        for idx, pixel_intensity in enumerate(self.class_intensity_dict.values()):
            mask[mask == pixel_intensity] = idx
       
        if self.augmentation:
            sample = self.augmentation(image = image, mask = mask)
            image, mask = sample['image'], sample['mask']
        
        # appy preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image = image, mask = mask)
            image, mask = sample['image'], sample['mask']
        
        return image, mask
    
    def __len__(self):
        return len(self.images_fps)