import torchvision.transforms.functional as TF
from torchmetrics.classification import JaccardIndex
from tqdm import tqdm
import torch

class CVDatasetPredictions():

    def __init__(self, dataset, device = 'cpu'):
        self.dataset = dataset
        self.divice  = device
         
        self.iou_metric = JaccardIndex(task="multiclass", num_classes=3, ignore_index=255).to(self.device)
    
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

        return mask_original_domain
    
    def mean_IoU(self, progress_bar = False):
        loop = tqdm(range(self.dataset.__len__())) if progress_bar else range(self.dataset.__len__())
        for i in loop:
            # Getting the prediction
            pred_mask = self.predict(i)

            # Getting the original mask
            ground_truth = self.dataset.original_mask(i)
            ground_truth = torch.from_numpy(ground_truth)

            # Updating the iou metric
            self.iou_metric.update(pred_mask, ground_truth)

        mean_iou = self.iou_metric.compute().item()
        # Resetting the metric
        self.iou_metric.reset()
        return mean_iou

        