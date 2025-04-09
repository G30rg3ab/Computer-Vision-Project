import torchvision.transforms.functional as TF
from segmentation.metrics import *
from tqdm import tqdm
import torch
from segmentation.show import *
from segmentation.constants import VisualisationConstants
import cv2
import os



class CVDatasetPredictions():

    def __init__(self, dataset, device = 'cpu'):
        self.dataset = dataset
        self.device  = device

    
    def set_prediction_fn(self, predict_fn, **kwargs):
        """
        Sets the prediction function and additional arguments.

        Args:
            predict_fn (callable): A function that takes inputs and returns predictions.
            **kwargs: Additional keyword arguments needed for prediction.
        """
        self.predict_fn = predict_fn
        self.predict_kwargs = kwargs  # Store extra parameters like model, device

    def predict(self, i, **kwargs):
        '''
        Function that uses the dataset directely to make predictions
        the inverse resizing operations are defined here 
        '''
        image, _ = self.dataset[i]
        # Calling the set prediction function on image i
        output_mask = self.predict_fn(image, **self.predict_kwargs)

        # Resizing the predicted mask to the original dimensions
        height, width, _ = self.dataset.image_shape(i)
        mask_original_domain = TF.resize(output_mask, (height, width), interpolation=TF.InterpolationMode.NEAREST)
        mask_original_domain = mask_original_domain.squeeze(0)

        return mask_original_domain.to(self.device)
    
    def plot_segmentation(self, i):

        pred_mask = self.predict(i, **self.predict_kwargs).numpy()
        colour_pred = colorise_mask(pred_mask, VisualisationConstants.palette)

        image =  self.dataset.original_image(i)
        if isinstance(image, torch_.Tensor):
            image = image.numpy()

        original_mask = self.dataset.original_mask(i)
        if isinstance(original_mask, torch.Tensor):
            original_mask = original_mask.numpy()

        colour_ground_truth = colorise_mask(original_mask, VisualisationConstants.palette)

        overlay = blend_image_and_mask(image,colour_pred, image_weight=  0.6, mask_weight=0.4)

        visualise_data(original_image = image,
                        ground_truth = colour_ground_truth,
                        predicted_mask = colour_pred,
                        image_overlay_predicted_mask = overlay)
        
    
        
    def save_segmentation_plot(self, i, folder_prefix= "unet_plots"):
        # Generate the segmentation plot (this will create and display the figure)

        pred_mask = self.predict(i, **self.predict_kwargs).numpy()
        colour_pred = colorise_mask(pred_mask, VisualisationConstants.palette)

        image =  self.dataset.original_image(i)
        if isinstance(image, torch_.Tensor):
            image = image.numpy()

        original_mask = self.dataset.original_mask(i)
        if isinstance(original_mask, torch.Tensor):
            original_mask = original_mask.numpy()

        colour_ground_truth = colorise_mask(original_mask, VisualisationConstants.palette)

        overlay = blend_image_and_mask(image,colour_pred, image_weight=  0.6, mask_weight=0.4)

        folder = f'segmentation_plots/{folder_prefix}_image_{i}/'
        os.makedirs(folder, exist_ok=True)


        plt_.imsave(folder + 'overlay.png', overlay)
        plt_.imsave(folder + 'ground_truth.png', colour_ground_truth)
        plt_.imsave(folder + 'predicted_mask.png', colour_pred)
        plt_.imsave(folder + 'image.png', image)


    def mean_IoU(self,classes = [0, 1, 2], progress_bar = False):

        iou_metric = IOU(classes = classes)
        loop = tqdm(range(self.dataset.__len__()), desc = 'IOU') if progress_bar else range(self.dataset.__len__())
        for i in loop:
            # Getting the prediction
            pred_mask = self.predict(i)

            # Getting the original mask
            ground_truth = self.dataset.original_mask(i)
            ground_truth = torch.from_numpy(ground_truth)
            ground_truth = ground_truth.to(self.device)

            # Updating the iou metric
            iou_metric.update(pred_mask, ground_truth)

        mean_iou = iou_metric.compute()
        # Resetting the metric
        iou_metric.reset()
        return mean_iou
    
    def dice_socre(self, classes = [0,1, 2], progress_bar = False):
        dice_metric = Dice(classes = classes)
        loop = tqdm(range(self.dataset.__len__()), desc = 'Dice') if progress_bar else range(self.dataset.__len__())
        for i in loop:
            # Getting the prediction
            pred_mask = self.predict(i)

            # Getting the original mask
            ground_truth = self.dataset.original_mask(i)
            ground_truth = torch.from_numpy(ground_truth)
            ground_truth = ground_truth.to(self.device)

            # Updating the dice metric
            dice_metric.update(pred_mask, ground_truth)

        dice = dice_metric.compute()
        # Resetting the metric
        dice_metric.reset()
        return dice
    
    
    def compute_accuracy(self, ignore_class = 255, progress_bar = False):
        accuracy_metric = Accuracy(ignore_class=ignore_class)
        loop = tqdm(range(self.dataset.__len__()), desc = 'Accuracy') if progress_bar else range(self.dataset.__len__())
        for i in loop:
            # Getting the prediction
            pred_mask = self.predict(i)

            # Getting the original mask
            ground_truth = self.dataset.original_mask(i)
            ground_truth = torch.from_numpy(ground_truth)
            ground_truth = ground_truth.to(self.device)

            # Updating the dice metric
            accuracy_metric.update(pred_mask, ground_truth)

        acc = accuracy_metric.compute()
        # Resetting the metric
        accuracy_metric.reset()
        return acc

def inverse_resize_mask(mask, height, width):
    """
    function that resizes the predicted mask back
    to the original dimensions
    """
    
    # Inverse resize the predicted mask to original dimensions using nearest neighbor interpolation
    mask_original_domain = TF.resize(mask, (height, width), interpolation=TF.InterpolationMode.NEAREST)
    mask_original_domain = mask_original_domain.squeeze(0)  # remove batch dimension

    return mask_original_domain


def predict(image, model, device="cuda" if torch.cuda.is_available() else "cpu"):
    '''
    Function that makes a predicted mask from the input image.

    # Parameters
        image: image of size (3, H, W)

    # Returns
        predicted mask of size (1, H, W)
    '''
    model.to(device)  # Ensure model is on the correct device
    model.eval()
    
    image = image.to(device)  # Move image to the same device as model

    # pred_logits from resized image (3, H, W) -> (1, 3, H, W)
    with torch.no_grad():  # Disable gradients for inference
        pred_logits = model(image.unsqueeze(0))
        pred_mask = torch.argmax(pred_logits, dim=1)

    return pred_mask