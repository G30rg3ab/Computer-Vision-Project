import gradio as gr
import torch
from torch import cat

# Custom imports
from segmentation.utils import preprocessing, create_heatmap
from segmentation.show import colorise_mask
from models.unet_model import UNET
from segmentation.utils import model_utils
from segmentation.eval import inverse_resize_mask
from segmentation.constants import VisualisationConstants
pallete = VisualisationConstants.palette
import cv2

# Get the preprocessing
preproc = preprocessing.get_preprocessing(preprocessing_fn=None)
valid_aug = preprocessing.get_validation_augmentation()

# Loading the model here
unet_model = UNET(4, 2)
unet_model.eval()
model_utils.load_checkpoint('/Users/georgeboutselis/Downloads/final_model.pth', unet_model)

# Getting device & sending model to device
device="cuda" if torch.cuda.is_available() else "cpu"
unet_model.to(device)

# Function to predict 
def predict_mask(image):
    with torch.no_grad():
        pred_logits = unet_model(image.unsqueeze(0))
        pred_mask = torch.argmax(pred_logits, dim=1)
    return pred_mask

def on_upload(image):
    # Process the uploaded image (for example, print its shape)
    print("Uploaded image shape:", image.shape)
    return image

def on_pixel_select(uploaded_image, event: gr.SelectData):
    # Check if event has valid click coordinates
    if event is not None and hasattr(event, "index"):
        selected_pixel = (event.index)

    # Getting uploaded image shape
    original_dimensions = uploaded_image.shape

    # Uploaded image is numpy
    original_H = original_dimensions[0]
    original_W = original_dimensions[1]

    print(f'Original width: {original_W}')
    print(f"original height: {original_H}")

    # Print selected pixels to console
    print("Selected pixels:", selected_pixel)

    # Resizing the image etc.
    sample = valid_aug(image = uploaded_image, keypoints=[selected_pixel])
    image, keypoint = sample['image'], sample['keypoints']

    # Getting the resized image dimensions
    image_W = image.shape[1]
    image_H = image.shape[0]

    heatmap = create_heatmap((image_H, image_W), keypoint[0], sigma = 10)

    # Preprocessing
    sample = preproc(image = image, heatmap = heatmap)
    image, heatmap = sample['image'], sample['heatmap']

    heatmap = heatmap.unsqueeze(0)  # shape (1, H, W)
    model_input = cat([image, heatmap], dim=0)
    model_input = model_input.float()

    mask = inverse_resize_mask(predict_mask(model_input), original_H, original_W)

    # converting the mask to numpy
    mask = mask.detach().cpu().numpy()

    # Convert the predicted mask to a 3-channel image
    colour_mask = colorise_mask(mask, pallete)

    print(f'colour mask has shape {colour_mask.shape}')
    overlay = cv2.addWeighted(uploaded_image.copy(), 0.7, colour_mask.astype('uint8'), 0.3, 0)

    return gr.update(value=overlay)

with gr.Blocks() as demo:
    gr.Markdown("# Pixel Selector using Gradio")
    # Define a state to store selected pixels
    selected_pixel = gr.State()
    #image_shape = gr.State()

    # Create an interactive image component
    image = gr.Image(label="Input", interactive=True)

    # State to store the original uploaded image
    original_image_state = gr.State()

    # Register the click event on the image
    image.select(
        fn=on_pixel_select,
        inputs=[original_image_state],
        outputs = image
    )

    # Attach the callback so that when the image is changed (uploaded), on_upload is called.
    image.upload(fn = on_upload, inputs = image, outputs = original_image_state)

if __name__ == "__main__":
    demo.launch()