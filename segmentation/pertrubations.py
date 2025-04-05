import numpy as np
import albumentations as albu
import cv2


# pertrubation a
class CustomGaussianNoise(albu.ImageOnlyTransform):
    def __init__(self, sigma=10.0):
        """
        Custom Albumentations transform that applies Gaussian noise.
        
        Parameters:
            sigma (float): Standard deviation of the Gaussian noise.
            always_apply (bool): If True, the transform is always applied.
            p (float): Probability of applying the transform.
        """
        super().__init__(p = 1)
        self.sigma = sigma
    
    def apply(self, image, **params):
        # Convert image to float for precise computation
        image_float = image.astype(np.float32)
        
        # Generate Gaussian noise (mean=0, standard deviation=sigma)
        noise = np.random.normal(0, self.sigma, image_float.shape)
        
        # Add the noise to the original image
        noisy_image = image_float + noise
        
        # Clip values to ensure they remain in the range [0, 255]
        noisy_image = np.clip(noisy_image, 0, 255)
        
        # Convert back to unsigned 8-bit integer type
        return noisy_image.astype(np.uint8)


# petrubation b
class CustomGaussianBlurring(albu.ImageOnlyTransform):
    def __init__(self, iterations=1):
        """
        Custom Albumentations transform that applies Gaussian blurring using a 3x3 Gaussian kernel.
        """
        # always_apply=True, p=1 means it's always applied with probability 1
        super().__init__(p = 1)
        self.iterations = iterations
    
    def apply(self, image, **params):
        # Use the 3x3 Gaussian kernel:
        #   1   2   1
        #   2   4   2
        #   1   2   1
        # scaled by 1/16
        kernel = np.array([[1, 2, 1],
                           [2, 4, 2],
                           [1, 2, 1]], dtype=np.float32) / 16.0
        
        # Copy the image to avoid modifying the original
        blurred_image = image.copy()
        
        # Repeatedly convolve the image with the kernel
        for _ in range(self.iterations):
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
        # Convert image to float for precise multiplication
        image_float = image.astype(np.float32)
        
        # Multiply each pixel by the factor
        scaled = image_float * self.factor
        
        # Clip values to ensure they remain in the valid range [0, 255]
        scaled = np.clip(scaled, 0, 255)
        
        # Convert back to unsigned 8-bit integer type
        return scaled.astype(np.uint8)