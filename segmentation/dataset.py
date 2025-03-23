import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset as BaseDataset
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torch import cat
from .utils import create_heatmap, random_xy


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

    def __filter_mask_clicked(self, mask, prompt_point):
        '''
        0: not-clicked
        1: clicked
        '''

        mask_filtered = mask.copy()

        x = int(prompt_point[0])
        y = int(prompt_point[1])

        # Changing border 255 to 0
        mask_filtered[(mask_filtered == 255) ] = 0

        # Getting the clicked object in [0, 1, 2]
        object_clicked = mask_filtered[y, x]

        # Changing non-clicked regions to 0 and clicked regions to 1
        mask_filtered = np.where(mask_filtered == object_clicked, 1, 0)

        return mask_filtered

    def __getitem__(self, i):
        image = self.original_image(i)
        mask = self.original_mask(i)

        # Getting the prompt-point
        prompt_point = self.prompt_points[i] if self.prompt_points else random_xy(mask.shape)

        if self.augmentation:
            # Apply the augmentation while passing the prompt point as a keypoint to keep track of position during augment
            sample = self.augmentation(image = image, mask = mask, keypoints = [prompt_point])
            image, mask, keypoints = sample['image'], sample['mask'], sample['keypoints']

        if not keypoints:
            resized_w = mask.shape[1]
            resized_h = mask.shape[0]
            keypoints = [[resized_w//2, resized_h//2]]

        prompt_point_post_aug = keypoints[0]

        heatmap = create_heatmap(mask.shape, prompt_point_post_aug)

        # Filtering the mask based on where the prompt point is
        mask = self.__filter_mask_clicked(mask, prompt_point_post_aug)

        if self.preprocessing:
            sample = self.preprocessing(image = image, mask = mask, heatmap = heatmap)
            image, mask, heatmap = sample['image'], sample['mask'], sample['heatmap']

        if self.concat_heatmap:
            heatmap = heatmap.unsqueeze(0)  # shape (1, H, W)
            return cat([image, heatmap], dim=0), mask

        return image, mask, heatmap.unsqueeze(0)
    
