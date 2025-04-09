import matplotlib.pyplot as plt_
import numpy as np_
import torch as torch_
import cv2 as cv2_
import matplotlib.patches as mpatches

def colorise_mask(mask, palette):
    # If mask is a torch tensor, work in torch.
    if isinstance(mask, torch_.Tensor):
        # Ensure mask is of type long for indexing.
        mask = mask.long()
        if not isinstance(palette, torch_.Tensor):
            palette = torch_.tensor(palette, device=mask.device)
        return palette[mask].permute(2, 0, 1)
    
    # If mask is a numpy array, work in numpy.
    elif isinstance(mask, np_.ndarray):
        if not isinstance(palette, np_.ndarray):
            palette = np_.array(palette)
        return palette[mask]
    
    else:
        raise ValueError("mask must be a torch.Tensor or numpy.ndarray")

# Helper function for data visualisation
def visualise_data(show = False, **images):
    '''Plot images in one row'''
    import torch
    n = len(images)
    plt_.figure(figsize=(10, 10))
    for i, (name, img) in enumerate(images.items()):
        # Check if img is a torch tensor; if so, convert to numpy.
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
            # If tensor shape is (C, H, W) with C=1 or C=3, transpose to (H, W, C)
            if img.ndim == 3 and img.shape[0] in [1, 3]:
                img = np_.transpose(img, (1, 2, 0))
                # If single channel, squeeze the channel dimension
                if img.shape[2] == 1:
                    img = img.squeeze(axis=2)
        plt_.subplot(1, n, i + 1)
        plt_.xticks([])
        plt_.yticks([])
        plt_.title(' '.join(name.split('_')).title())
        plt_.imshow(img)
    plt_.show()




def overlay_heatmap(image, heatmap, alpha=0.3, colormap=cv2_.COLORMAP_JET):
    """
    Overlays a one-channel heatmap on a 3-channel image using a colormap.
    
    Parameters:
        image (np.ndarray): Original image, shape (H, W, 3), assumed to be uint8.
        heatmap (np.ndarray): One-channel heatmap, shape (H, W). Values can be in [0, 1] or [0, 255].
        alpha (float): Transparency factor for the heatmap overlay.
        colormap (int): OpenCV colormap to apply to the heatmap.
    
    Returns:
        overlay (np.ndarray): The resulting image with the heatmap overlay.
    """
    # Ensure heatmap is in the 0-255 range and type uint8.
    if heatmap.max() <= 1:
        heatmap = (heatmap * 255).astype(np_.uint8)
    else:
        heatmap = heatmap.astype(np_.uint8)
    
    # Apply the colormap to get a 3-channel heatmap image.
    heatmap_color = cv2_.applyColorMap(heatmap, colormap)
    
    # Optionally, if your original image is in RGB, convert it to BGR since cv2.applyColorMap outputs BGR.
    # If you want to keep the output in RGB, convert back:
    image_bgr = cv2_.cvtColor(image, cv2_.COLOR_RGB2BGR)
    
    # Blend the original image and the colored heatmap.
    overlay_bgr = cv2_.addWeighted(image_bgr, 1 - alpha, heatmap_color, alpha, 0)
    
    # Convert back to RGB if needed.
    overlay_rgb = cv2_.cvtColor(overlay_bgr, cv2_.COLOR_BGR2RGB)
    
    return overlay_rgb

def blend_image_and_mask(image, mask, image_weight=0.7, mask_weight=0.3, gamma=0):
    """
    Blends an image and a mask using cv2.addWeighted.
    """
    # Ensure the image is uint8 and mask is also in the right scale.
    if image.dtype != np_.uint8:
        image = cv2_.normalize(image, None, 0, 255, cv2_.NORM_MINMAX).astype(np_.uint8)
    
    # If the mask is not in uint8, try to convert it.
    if mask.dtype != np_.uint8:
        mask = cv2_.normalize(mask, None, 0, 255, cv2_.NORM_MINMAX).astype(np_.uint8)
    
    # Check the mask channels: if it is a single channel, convert it to 3 channels.
    if len(mask.shape) == 2 or (len(mask.shape) == 3 and mask.shape[2] == 1):
        mask_color = cv2_.cvtColor(mask, cv2_.COLOR_GRAY2BGR)
    else:
        mask_color = mask
    
    # Ensure the image and the mask are the same size.
    if image.shape != mask_color.shape:
        mask_color = cv2_.resize(mask_color, (image.shape[1], image.shape[0]))
    
    # Blend using the specified weights.
    overlay = cv2_.addWeighted(image, image_weight, mask_color, mask_weight, gamma)
    
    return overlay


