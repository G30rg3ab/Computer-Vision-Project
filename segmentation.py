import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset as BaseDataset
import matplotlib.pyplot as plt_
import albumentations as albu

class CVDataset(BaseDataset):
    '''
    Dataset class for Semantic Segmentation
    '''
    def __init__(self, images_dir, masks_dir, classes = None, augmentation = None, preprocessing = None):
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
        class_intensity_dict = {'background':0, 'dog':75, 'cat':38}

        # Get images (x) and masks (y). These are the names of how the images/labels are saved
        self.ids_x = sorted(os.listdir(images_dir)) # Sorting image names to match with masks
        self.ids_y = sorted(os.listdir(masks_dir)) # Sorting mask names to match with masks

        # Get the full paths of images(X) and masks (y)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids_x]
        self.masks_fps  = [os.path.join(masks_dir, image_id) for image_id in self.ids_y ]

        # Convert the str names to class values which will be on masks
        self.class_values = [class_intensity_dict[cls] for cls in classes]

        self.augmentation  = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        # Reading in the data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        
        # Extracting the classes from the labels (eg. cat)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis = -1).astype('float')
       
        if self.augmentation:
            sample = self.augmentation(image = image, mask = mask)
            image, mask = sample['image'], sample['mask']
        
        if self.preprocessing:
            raise NotImplementedError
        
        return image, mask
    
    def __len__(self):
        return len(self.ids_x)

# Helper function for data visualisation
def visualiseData(**image):
    '''Plot images in one row'''
    n = len(image)
    plt_.figure(figsize = (10, 10))

    for i, (name, image) in enumerate(image.items()):
        plt_.subplot(1, n, i + 1)
        plt_.xticks([])
        plt_.yticks([])
        plt_.title(' '.join(name.split('_')).title())
        plt_.imshow(image)
    plt_.show()

def get_training_augmentation():
    train_transform = [
 
        albu.Resize(256, 416, p=1),
        albu.HorizontalFlip(p=0.5),
 
        albu.OneOf([
            albu.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=1),
            albu.CLAHE(p=1),
            albu.HueSaturationValue(p=1)
            ],
            p=0.9,
        ),
        albu.GaussNoise(p=0.2),
    ]
    return albu.Compose(train_transform)