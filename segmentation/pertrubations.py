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
    
# pertubation e
def increase_brightness(image, factor): 
    '''
    Inputs: 
    Image: Image to apply image brightness to as a numpy array with shape (H, W, 3)
    Factor: Number to add to each pixel by 

    Output: 
    Brightened images 
    '''

    # Add brightness
    brightened = image + factor

    # Clip to keep values in a valid range
    brightened = np.clip(brightened, 0, 255)

    return brightened.astype(image.dtype)

# pertubation f 
def decrease_brightness(image, factor):
    '''
    Inputs 
    Image: Image to apply brightness decrease to as numpy array with shape (H,W,3)
    Factor: number to subtract from each pixel

    Outputs: 
    Decreased brightness image 
    '''

    # Decrease brightness
    changed = image - factor 

    # Clip to keep values in a valid range
    changed = np.clip(changed, 0, 255)

    return changed.astype(image.dtype)

# pertubation g
def occlusion(image, size):
    '''
    Inputs 
    Image: Image to apply occlusion to with shape (H, W, 3)
    size: size of the squre 

    Output:
    image with occlusion applied 
    '''

    # Get image dimensions
    height, width, _ = image.shape
    
    # Ensure that the square fits within the image
    max_x = width - size
    max_y = height - size

    # Choose a random top-left corner for the square
    top_left_x = random.randint(0, max_x)
    top_left_y = random.randint(0, max_y)
    
    # Replace the chosen square region with black pixels (0)
    image[top_left_y:top_left_y + size, top_left_x:top_left_x + size] = 0
    
    return image


# pertubation h 
def add_salt_and_pepper_noise(image, noise_level):
    '''
    Inputs:
    image: A numpy array of shape (H, W, 3) representing the image.
    noise_level: strength of salt and pepper noise

    Output:
    image: The image with salt and pepper noise added.
    '''
    # Choose a random noise level from the provided list
    noise_level = random.choice(noise_level)
    
    # Add salt and pepper noise to the image
    noisy_image = random_noise(image, mode='s&p', amount=noise_level)
    
    # Convert the noisy image back to the range [0, 255] and ensure it is in uint8 format
    noisy_image = np.array(255 * noisy_image, dtype=np.uint8)
    
    return noisy_image
