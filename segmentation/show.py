import matplotlib.pyplot as plt_
import numpy as np_
import torch as torch_
import cv2 as cv2_
import matplotlib.patches as mpatches

def colorise_mask(mask, palette):
    '''
    function that takes a one channel mask
    and converts it into a 3 channel mask
    where the pallete specifies the colours for each class
    int he one channel mask
    '''
    if isinstance(mask, torch_.Tensor):
        mask = mask.long()
        if not isinstance(palette, torch_.Tensor):
            palette = torch_.tensor(palette, device=mask.device)
        return palette[mask].permute(2, 0, 1)
    
    # If mask is a numpy array we keep it as numpy
    elif isinstance(mask, np_.ndarray):
        if not isinstance(palette, np_.ndarray):
            palette = np_.array(palette)
        return palette[mask]
    
    else:
        raise ValueError("mask must be a torch.Tensor or numpy.ndarray")



def visualise_data(show = False, **images):
    '''
    Plot images in one row
    '''

    n = len(images)
    plt_.figure(figsize=(10, 10))
    for i, (name, img) in enumerate(images.items()):
        # converting the image to numpy if it is a torch tensor
        if isinstance(img, torch_.Tensor):
            img = img.detach().cpu().numpy()
            
            # permuting the image dimensions if they are incorrect
            if img.ndim == 3 and img.shape[0] in [1, 3]:
                img = np_.transpose(img, (1, 2, 0))
                if img.shape[2] == 1:
                    img = img.squeeze(axis=2)

        # Plotting image i
        plt_.subplot(1, n, i + 1)
        plt_.xticks([])
        plt_.yticks([])
        plt_.title(' '.join(name.split('_')).title())
        plt_.imshow(img)

    # Shpwing the final plot
    plt_.show()



def overlay_heatmap(image, heatmap, alpha=0.3, colormap=cv2_.COLORMAP_JET):
    """
    Overlays a one-channel heatmap on a 3-channel image using a colormap.
    """
    # ensure that the heatmap is in the valid range and has valid datatype
    if heatmap.max() <= 1:
        heatmap = (heatmap * 255).astype(np_.uint8)
    else:
        heatmap = heatmap.astype(np_.uint8)
    
    # colour map application
    heatmap_color = cv2_.applyColorMap(heatmap, colormap)
    
    # convert image to BGR format
    image_bgr = cv2_.cvtColor(image, cv2_.COLOR_RGB2BGR)
    
    # blending the image and the heatma
    overlay_bgr = cv2_.addWeighted(image_bgr, 1 - alpha, heatmap_color, alpha, 0)
    
    # convert back to rgb
    overlay_rgb = cv2_.cvtColor(overlay_bgr, cv2_.COLOR_BGR2RGB)
    
    # Return the overlayed image and heatmap
    return overlay_rgb

def blend_image_and_mask(image, mask, image_weight=0.7, mask_weight=0.3, gamma=0):
    """
    Blends an image and a mask using cv2.addWeighted.
    """
    # valid range check
    if image.dtype != np_.uint8:
        image = cv2_.normalize(image, None, 0, 255, cv2_.NORM_MINMAX).astype(np_.uint8)
    
    # blend using specific weights
    overlay = cv2_.addWeighted(image, image_weight, mask, mask_weight, gamma)
    
    return overlay


