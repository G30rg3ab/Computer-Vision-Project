import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset as BaseDataset
import matplotlib.pyplot as plt_
import albumentations as albu
from sklearn.model_selection import train_test_split # type: ignore
from albumentations.pytorch import ToTensorV2
import torch
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
        self.class_intensity_dict = {'background':0, 'dog':75, 'cat':38}
        
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
    
    @ staticmethod
    def get_transforms():
        transforms = albu.Compose([
        albu.Resize(256, 416, p=1),
        albu.Normalize(
            mean = [0, 0, 0],
            std = [1, 1, 1],
            max_pixel_value = 255
        ), ToTensorV2()
        ])
        return transforms
        

    
class show():
    @ staticmethod
    def colorise_mask(mask, palette):
        return palette[mask]

    @ staticmethod
    # Helper function for data visualisation
    def visualiseData(**image):
        '''Plot images in one row'''

        # Create a 256×3 array initialized to (0,0,0) for "unused" entries
        palette = np.zeros((256, 3), dtype=np.uint8)
        # Assign colors for your known labels
        palette[0]   = [0,   0,   0]    # class 0 => black
        palette[1]   = [255, 0,   0]    # class 1 => red
        palette[2]   = [0,   255, 0]    # class 2 => green
        palette[255] = [255, 255, 255]  # ignore index => white
      
        n = len(image)
        plt_.figure(figsize = (10, 10))

        for i, (name, image) in enumerate(image.items()):
            plt_.subplot(1, n, i + 1)
            plt_.xticks([])
            plt_.yticks([])
            plt_.title(' '.join(name.split('_')).title())

            if len(image.shape) == 2:
                image = show.colorise_mask(image, palette)
                # 3-channel color image
            plt_.imshow(image)
        plt_.show()

class utils():
    def __init__(self):
        pass
    
    @staticmethod
    def save_checkpoint(state, filename = 'my_checkpoint.pth.tar'):
        print('=> Saving checkpoint')
        torch.save(state, filename)
    
    @staticmethod
    def load_checkpoint(checkpoint, model):
        print('=> Loading checkpoint')
        model.load_state_dict(checkpoint['state_dict'])

    @staticmethod
    def get_loaders(
        trainVal_dir, 
        trainVal_maskdir,
        batch_size,
        transform,
        train_augmentation,
        num_workers = 4, 
        pin_memory = True
    ):

        # Splitting relative path names into into training and validation
        x_train_fps, x_val_fps, y_train_fps, y_val_fps = preprocessing.train_val_split(trainVal_dir, trainVal_maskdir, 0.2)
        # Initialising the data loaders
        train_ds = CVDataset(x_train_fps, y_train_fps, augmentation = train_augmentation, preprocessing = transform)
        train_loader = DataLoader(
            train_ds,
            batch_size= batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=True
        )

        valid_ds = CVDataset(x_val_fps, y_val_fps, preprocessing = transform)
        valid_loader = DataLoader(
            valid_ds,
            batch_size= batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=False
        )

        return train_loader, valid_loader
    
    def check_accuracy(loader, model, device="cuda"):
        num_correct = 0
        num_pixels = 0
        model.eval()

        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)  # shape: (N, H, W), with integer labels in [0..(num_classes-1)]

                # Forward pass: model outputs logits of shape (N, num_classes, H, W)
                preds = model(x)  # shape: (N, C, H, W)

                # Convert logits to predicted class indices
                preds = torch.argmax(preds, dim=1)  # shape: (N, H, W)

                # Compare preds with ground truth
                num_correct += (preds == y).sum().item()
                num_pixels  += torch.numel(preds)  # total number of pixels

        print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels * 100:.2f}")
        model.train()