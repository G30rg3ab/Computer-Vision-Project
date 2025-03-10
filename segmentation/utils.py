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
from .dataset import CVDataset
from torchmetrics.classification import JaccardIndex
from tqdm import tqdm
import torchvision.transforms.functional as TF
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import boto3

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
            albu.RandomCrop(256, 416, pad_if_needed=True, fill_mask=255),
            albu.HorizontalFlip(p=0.5),
            albu.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5, fill_mask=255),  # Random shift, scale, and rotation
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
    
    @staticmethod
    def get_validation_augmentation():
        '''
        This function returns an albumentations.Compose element
        which performs augmentation for validation set. In particular, 
        it adds padding to rezise to 1536 x 1536 to ensure 
        image is div by 32 for UNet (unsure if we need this for 
        other models at the moment)
        '''
        # Add sufficient padding to ensure image is divisible by 32
        test_transform = [
            #albu.PadIfNeeded(min_height=768, min_width=768, border_mode=0, fill_mask=255),
            #albu.Resize(768, 768) # Incase 768 x 768 is not enough, probably should fix later in a better way
            albu.Resize(256, 416)
        ]
        return albu.Compose(test_transform, is_check_shapes=False)
    
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


    @staticmethod
    def get_preprocessing(preprocessing_fn=None):
        '''
        Normalise the data and convert to a tensor. Use the 
        preprocessing_fn to chain another preprocessing step if needed
        '''
        _transform = []
        
        # If you had a model-specific function (e.g., from segmentation_models_pytorch),
        # you could add it here. We'll skip it for now.
        if preprocessing_fn:
            _transform.append(albu.Lambda(image=preprocessing_fn))
        
        #  Add your normalization step here. Example: scale to [0,1]
        _transform.append(albu.Normalize(
            mean=[0, 0, 0],         
            std=[1, 1, 1],           
            max_pixel_value=255.0
        ))
        
        # Convert to PyTorch tensor (channels-first)
        _transform.append(ToTensorV2())

        return albu.Compose(_transform, is_check_shapes=False)
    
        
class show():
    @ staticmethod
    def colorise_mask(mask, palette):
        return palette[mask]

    @ staticmethod
    # Helper function for data visualisation
    def visualiseData(**image):
        '''Plot images in one row'''

        palette = VisualisationConstants.palette
      
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

class model_utils():
    def __init__(self):
        pass

    @staticmethod
    def save_final_model(model, optimizer, iou, hyperparams, filename="final_model.pth", s3_bucket=None, s3_key=None):
        """
        Save the trained model along with metadata for experiment tracking.
        Optionally upload the model to an S3 bucket.

        Args:
        - model (torch.nn.Module): Trained PyTorch model.
        - optimizer (torch.optim.Optimizer): Optimizer state.
        - iou (float): Final IoU score.
        - hyperparams (dict): Dictionary of hyperparameters.
        - filename (str): Local filename to save the model.
        - s3_bucket (str, optional): S3 bucket name to upload the file.
        - s3_key (str, optional): Key (path) to store the file in S3.
        """
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iou": iou,
            "hyperparams": hyperparams,  # Dictionary containing learning rate, batch size, etc.
        }

        # **Step 1: Save model to file**
        torch.save(checkpoint, filename)
        print(f"=> Final model and metadata saved to {filename}")

        # **Step 2: Ensure the file is reopened before uploading**
        if s3_bucket and s3_key:
            try:
                s3_client = boto3.client("s3")
                with open(filename, "rb") as f:  # ✅ Open file in binary read mode
                    s3_client.upload_fileobj(f, s3_bucket, s3_key)

                print(f"=> Model uploaded to S3: s3://{s3_bucket}/{s3_key}")
            except Exception as e:
                print(f"Failed to upload model to S3: {e}")

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
        valid_augmentation,
        train_augmentation,
        preprocessing_fn,
        num_workers = 4, 
        pin_memory = True
    ):
        message = '''
        Warning: This function is currently not being used for UNET. It returns a validation dataset loader
        which is difficult to work with when batch size > 1, because the images must be resized before comparison
        '''

        print('Warning: Not using this function anymore because')
        # Splitting relative path names into into training and validation
        x_train_fps, x_val_fps, y_train_fps, y_val_fps = preprocessing.train_val_split(trainVal_dir, trainVal_maskdir, 0.2)
        # Initialising the data loaders
        train_ds = CVDataset(x_train_fps, y_train_fps, augmentation = train_augmentation, preprocessing = preprocessing_fn)
        train_loader = DataLoader(
            train_ds,
            batch_size= batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=True
        )

        valid_ds = CVDataset(x_val_fps, y_val_fps, augmentation = valid_augmentation, preprocessing = preprocessing_fn, return_original_dimensions=True)
        valid_loader = DataLoader(
            valid_ds,
            batch_size= 1,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=False
        )

        return train_loader, valid_loader
    
    
    def check_accuracy(loader, model, device="cuda"):
        num_correct = 0
        num_pixels = 0
        model.eval()
        iou_metric = JaccardIndex(task="multiclass", num_classes=3, ignore_index=255).to(device)
        with torch.inference_mode():
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
                iou_metric.update(preds, y)

            mean_iou = iou_metric.compute().item()
            iou_metric.reset() # Resetting for next epoch
            print(f"Mean IoU: {mean_iou:.4f}")
            print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels * 100:.2f}")

        model.train()

    def check_iou(testing_set, model, device = 'cuda'):
        '''
        Function to check iou score while training the model
        will set training mode back when finished
        '''
        metrics = ModelEval(testing_set, model, device)
        # Priting the IOU score
        mean_iou = metrics.mean_IoU()
        print(f"Mean IoU on original domain: {mean_iou:.4f}")
        # back to training
        model.train()

class ModelEval():
    def __init__(self, dataset, model, device = 'cuda'):
        '''
        # Parameters
            dataset: CVDataset
                dataset with validation_augmentation and preprocessing function 
            applied

            model: Object(torch.nn.modules)
                Model inherited from torch.nn.modules used to make predictions
        '''
        self.dataset = dataset
        self.model = model
        self.device = device

    def predict(self, i):
        '''
        Function that uses dataset directly to make a prediction
        the inverse resizing operations can be defined here and will
        be used in the subsequent functions for evaluation purposes
        '''
        self.model.eval()
        # Getting the ith image in the dataset, this has the scaling/resizing applied
        image, _ = self.dataset[i]
        image = image.to(self.device)
        # Getting the unprocessed mask
        unprocessed_mask = self.dataset.original_mask(i)
        # Getting the dimensions of the unprocessed mask
        H, W = unprocessed_mask.shape
        # forward pass
        with torch.inference_mode():
            pred_logits = self.model(image.unsqueeze(0)) # Getting the prediction for the preprocessed image
            pred_mask = torch.argmax(pred_logits, dim = 1)

            # Resizing the predicted mask to the original dimensions
            pred_original_dimensions = TF.resize(pred_mask, (H, W), interpolation=TF.InterpolationMode.NEAREST)
            pred_original_dimensions = pred_original_dimensions.squeeze(0)

            return pred_original_dimensions, torch.from_numpy(unprocessed_mask)

    def visualise(self):
        show.visualiseData(predicted_mask = self.prediction, 
                           ground_truth_mask = self.ground_truth_mask) 

    def mean_IoU(self, progress_bar = False):
        self.model.eval()
        iou_metric = JaccardIndex(task="multiclass", num_classes=3, ignore_index=255).to(self.device)
        
        loop = tqdm(range(self.dataset.__len__())) if progress_bar else range(self.dataset.__len__())
        for i in loop:
            # Getting the prediction
            pred_mask, ground_truth = self.predict(i)

            # ✅ Ensure both tensors are long-type (integer labels)
            pred_mask = pred_mask.to(dtype=torch.long)
            ground_truth = ground_truth.to(dtype=torch.long)

            # ✅ Ensure batch dimension (N=1)
            pred_mask = pred_mask.unsqueeze(0)
            ground_truth = ground_truth.unsqueeze(0)
            iou_metric.update(pred_mask, ground_truth)

        mean_iou = iou_metric.compute().item()
        iou_metric.reset() # Resetting
        return mean_iou
      

    def plot_prediction_overlay(self, i, alpha=0.5):
        """
        Plots the original image with the predicted segmentation mask overlaid.
        Args:
        - i (int): Index of the image in the dataset.
        - alpha (float): Transparency level for the overlay (0 = only image, 1 = only mask).
        """
        # Getitng the predicted mask
        pred_mask, ground_truth = self.predict(i)
        # Convert mask to color
        pred_mask = pred_mask.cpu().numpy()

        # Getting the original image
        image_original = self.dataset.original_image(i)

        palette = VisualisationConstants.palette
        class_colours = VisualisationConstants.class_colors # Dictionary of class index (0, 1, 2): colour

        # Colouring the mask
        coloured_mask = show.colorise_mask(pred_mask, palette=palette)

        # Create overlay image
        plt.figure(figsize=(10, 5))
        plt.imshow(image_original, cmap='gray')
        plt.imshow(coloured_mask, alpha=alpha)  # Overlay mask with transparency

        # Create legend patches
        legend_patches = [mpatches.Patch(color=color, label=label) for label, color in zip(["Background", "Cat", "Dog"], VisualisationConstants.class_colors.values())]
        plt.legend(handles = legend_patches, loc = 'upper right')

        plt.title(f"Prediction Overlay - Image")
        plt.axis("off")
        plt.show()

    def image_fit_summary(self, i, alpha = 0.5):
        '''
        Do some plots and print metrics for single image
        '''
        self.plot_prediction_overlay(i, alpha=alpha)

        ### Computing the IoU for this image ####
        iou_metric = JaccardIndex(task="multiclass", num_classes=3, ignore_index=255).to(self.device)
        pred_mask, ground_truth = self.predict(i)

        # ✅ Ensure both tensors are long-type (integer labels)
        pred_mask = pred_mask.to(dtype=torch.long)
        ground_truth = ground_truth.to(dtype=torch.long)

        # ✅ Ensure batch dimension (N=1)
        pred_mask = pred_mask.unsqueeze(0)
        ground_truth = ground_truth.unsqueeze(0)
        iou_metric.update(pred_mask, ground_truth)
        iou = iou_metric.compute().item()
        iou_metric.reset() # Resetting
        print(f"IoU on original domain: {iou:.4f}")

    def load_model(self, checkpoint_path):
        """
        Loads a saved model checkpoint into the model.
        Supports loading from local storage and S3.

        Args:
        - checkpoint_path (str): Local file path or S3 URI to the model checkpoint.

        Returns:
        - None: Updates the model in-place.
        """
        if checkpoint_path.startswith("s3://"):
            print(f"=> Downloading model from {checkpoint_path}...")

            # Extract S3 bucket and key
            s3_client = boto3.client("s3")
            bucket_name = checkpoint_path.split("/")[2]
            s3_key = "/".join(checkpoint_path.split("/")[3:])

            # Temporary local file path
            local_checkpoint = "temp_model.pth"
            
            # Download model from S3
            try:
                s3_client.download_file(bucket_name, s3_key, local_checkpoint)
                checkpoint_path = local_checkpoint  # Update path to local file
                print(f"=> Model downloaded from S3 to {local_checkpoint}")
            except Exception as e:
                print(f" Failed to download model from S3: {e}")
                return

        print(f"=> Loading model from {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state dict
        self.model.load_state_dict(checkpoint["state_dict"])
        
        print("=> Model successfully loaded!")

        # Remove temp file if downloaded from S3
        if checkpoint_path == "temp_model.pth":
            os.remove(checkpoint_path)