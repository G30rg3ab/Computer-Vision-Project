
# Imports
import os
import numpy as np
import albumentations as albu
import cv2
from skimage.util import random_noise

# Custom imports
from segmentation.dataset import CVDataset
from segmentation.utils import preprocessing,  model_utils
from segmentation.show import visualise_data
import albumentations as albu
from segmentation.eval import CVDatasetPredictions, predict
from models.unet_model import UNET
from segmentation.utils import traininglog


# pertrubation a
class CustomGaussianNoise(albu.ImageOnlyTransform):
    def __init__(self, sigma=10.0):
        """
        Custom Albumentations transform that applies Gaussian noise.
        """
        super().__init__(p = 1)
        self.sigma = sigma
    
    def apply(self, image, **params):
        # Convert image to float
        image_float = image.astype(np.float32)
        
        # Generate Gaussian noise
        noise = np.random.normal(0, self.sigma, image_float.shape)
        
        # add the noise to the original image
        noisy_image = image_float + noise
        
        # cli[pp] values to ensure they remain in the range [0, 255]
        noisy_image = np.clip(noisy_image, 0, 255)
        return noisy_image.astype(np.uint8)


# petrubation b
class CustomGaussianBlurring(albu.ImageOnlyTransform):
    def __init__(self, iterations=1):
        """
        Custom Albumentations transform that applies Gaussian blurring using a 3x3 Gaussian kernel.
        """
        super().__init__(p = 1)
        self.iterations = iterations
    
    def apply(self, image, **params):
        kernel = np.array([[1, 2, 1],
                           [2, 4, 2],
                           [1, 2, 1]], dtype=np.float32) / 16.0
        
        blurred_image = image.copy()
        
        # convolve the image with the kernel
        for i in range(self.iterations):
            blurred_image = cv2.filter2D(blurred_image, -1, kernel)
        
        return blurred_image
    
# petrubation c and d (both use scaling with c: factor > 1 and d: factor < 1)
class CustomScaling(albu.ImageOnlyTransform):
    def __init__(self, factor=1.0):
        """
        Custom Albumentations transform that scales the image by a given factor.
        """
        super().__init__(p = 1)
        self.factor = factor
    
    def apply(self, image, **params):
        # image -> float
        image_float = image.astype(np.float32)
        
        # scale by facor self.factor
        scaled = image_float * self.factor
        
        # Clip values to ensure they remain in the valid range [0, 255]
        scaled = np.clip(scaled, 0, 255)
        
        # float -> integer
        return scaled.astype(np.uint8)
    
# pertubation e
class CustomBrightness(albu.ImageOnlyTransform):
    def __init__(self, factor = 1) :
        super().__init__(p = 1)
        self.factor = factor

    def apply(self, image, **params):
        '''
        Brightened images 
        '''

        # add brightness by constant factor
        brightened = image + self.factor

        # clip to keep values in a valid range
        brightened = np.clip(brightened, 0, 255)

        return brightened.astype(image.dtype)

# pertubation f 
class CustomDecreaseBrightness(albu.ImageOnlyTransform):
    def __init__(self, factor=30):
        """
        Custom Albumentations transform that decreases image brightness by a fixed factor.
        """
        super().__init__(p=1)
        self.factor = factor

    def apply(self, image, **params):
        decreased = image.astype(np.float32) - self.factor
        decreased = np.clip(decreased, 0, 255)
        return decreased.astype(image.dtype)

# pertubation g
class CustomOcclusion(albu.ImageOnlyTransform):
    def __init__(self, size=30):
        """
        Custom Albumentations transform that applies a square black occlusion to the image.
        """
        super().__init__(p=1)
        self.size = size

    def apply(self, image, **params):
        height, width, _ = image.shape
        max_x = width - self.size
        max_y = height - self.size

        top_left_x = np.random.randint(0, max_x)
        top_left_y = np.random.randint(0, max_y)

        image_copy = image.copy()
        image_copy[top_left_y:top_left_y + self.size, top_left_x:top_left_x + self.size] = 0
        return image_copy


# pertubation h 
class CustomSaltAndPepperNoise(albu.ImageOnlyTransform):
    def __init__(self, noise_level=0.0):
        """
        Custom Albumentations transform that adds salt and pepper noise with a specific amount.
        """
        super().__init__(p=1)
        self.noise_level = noise_level

    def apply(self, image, **params):
        # Normalize image to [0, 1] for skimage
        image_norm = image.astype(np.float32) / 255.0

        # Apply salt and pepper noise
        noisy_image = random_noise(image_norm, mode='s&p', amount=self.noise_level)

        # Convert back to [0, 255]
        noisy_image = np.clip(noisy_image * 255, 0, 255).astype(np.uint8)
        return noisy_image
    

from dataclasses import dataclass
from typing import List, Union, Type


# Configuration for a permutation application
@dataclass
class PerturbationConfig:
    name: str
    transform: Type[albu.ImageOnlyTransform]
    param_name: str
    params: List[Union[int, float]]

def get_perturbation_configs() -> List[PerturbationConfig]:
    return [
        PerturbationConfig(
            name="Gaussian Noise",
            transform=CustomGaussianNoise,
            param_name="standard deviation",
            params=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18],
        ),
        PerturbationConfig(
            name="Gaussian Blur",
            transform=CustomGaussianBlurring,
            param_name="iterations",
            params=list(range(0, 10)),
        ),
        PerturbationConfig(
            name="Contrast Increase",
            transform=CustomScaling,
            param_name="scaling factor",
            params=[1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.1, 1.15, 1.20, 1.25],
        ),
        PerturbationConfig(
            name="Contrast Decrease",
            transform=CustomScaling,
            param_name="scaling factor",
            params=[1.0, 0.95, 0.90, 0.85, 0.80, 0.60, 0.40, 0.30, 0.20, 0.10],
        ),
        PerturbationConfig(
            name="Brightness Increase",
            transform=CustomBrightness,
            param_name="brightness offset",
            params=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
        ),
        PerturbationConfig(
            name="Brightness Decrease",
            transform=CustomDecreaseBrightness,
            param_name="brightness offset",
            params=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
        ),
        PerturbationConfig(
            name="Occlusion",
            transform=CustomOcclusion,
            param_name="square size",
            params=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
        ),
        PerturbationConfig(
            name="Salt and Pepper Noise",
            transform=CustomSaltAndPepperNoise,
            param_name="noise amount",
            params=[0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18],
        ),
    ]


petrubation_configs = get_perturbation_configs()

# Set paths and preprocessing
DATA_DIR = 'Dataset'
x_test_dir = os.path.join(DATA_DIR, 'Test/color')
y_test_dir = os.path.join(DATA_DIR, 'Test/label')

x_test_fps, y_test_fps = preprocessing.get_testing_paths(x_test_dir, y_test_dir)
preprocessing_fn = preprocessing.get_preprocessing()

# Load model
unet_model = UNET(3, 3)
model_utils.load_checkpoint('/Users/georgeboutselis/Downloads/best_model-2.pth', unet_model)

# Get all perturbation configurations
perturbation_configs = get_perturbation_configs()

def run_perturbation_analysis():
    for perturbation_config in perturbation_configs:
        for param in perturbation_config.params:
            perturbation_function = perturbation_config.transform(param)

            # Set up dataset with perturbation
            pert_name = perturbation_config.name
            par_name = perturbation_config.param_name
            print(f'Perturbation: {pert_name} | {par_name}: {param}')

            test_dataset = CVDataset(
                x_test_fps,
                y_test_fps,
                augmentation=albu.Compose([
                    perturbation_function,
                    albu.Resize(256, 416)
                ]),
                preprocessing=preprocessing_fn
            )

            # Evaluate predictions
            evaluator = CVDatasetPredictions(dataset=test_dataset)
            evaluator.set_prediction_fn(predict_fn=predict, device='mps', model=unet_model)

            mean_iou = evaluator.mean_IoU(progress_bar=True)
            dice_score = evaluator.dice_socre(progress_bar=True)
            pixel_acc = evaluator.compute_accuracy(progress_bar=True)

            print(f'IOU: {mean_iou} | DICE: {dice_score} | ACCURACY: {pixel_acc}')

            # Log results
            traininglog.log_training(
                'petrubation_analysis.csv',
                petrubation_name=pert_name,
                parameter_name=par_name,
                parameter=param,
                IOU=mean_iou,
                DICE=dice_score,
                ACCURACY=pixel_acc
            )

if __name__ == "__main__":
    run_perturbation_analysis()