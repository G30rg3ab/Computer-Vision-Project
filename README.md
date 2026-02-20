# Computer Vision Project
Mini Project for University of Edinburgh Computer Vision course. If you want to run code properly, you
should load the dataset OxfordPett dataset into the same directory as this README file. Below is a short description
of each file

# segmentation
This folder is a custom module implemented for this project

## segmentation/constants.py
a file containing some constants we will use throughout the project
such as colour palletes and directory paths.

## segmentation/dataset.py
This file implements 
1. A custom dataset (CVDataset) for UNet, Clip, and Autoencoder specifically designed for the Oxford Pet dataset image segmentation. This integrates with PyTorch nicely to dynamically read in/augment images 
2. A custom dataset (PointDataset) for the point-based model. This esentially has similar features to CVDataset but added features for returning a random promt-point and heatmap etc.

## segmentation/eval.py
When we evaluate pefromance on the test set, there is a lot that goes into reszing images back to their original (H x W) and evaluating metrics on that domain. This file implements calculating the relavent metrics by resizing a predicted mask to the original domain, and comparing with the ground-truth. Also implemented features to plot and save images (which we may find useful) for the latex report/ or just to visually evaluate peformance

## segmentation/loss.py
Implementation of dice loss and a a * DiceLoss + (1 - a) * CrossEntropyLoss
which integrates with pytorch.


## segmentation/metrics.py
Files to calculate
1. The average intersection over union
2. Pixel Accuracy
3. Average Dice score

## segmentation/s3_utils.py
This is just some functions to upload/download files from s3, useful for model checkpoints

## segmentation/show.py
Functions which contain everything to do with visualising the images

## segmentation/utils.py
Functions for the
1. Training augmentations
2. Validation augmentations (just resizing the image)
3. Preprocessing
4. Training and validation splitting
5. Loading model checkpoints into pytorch model
6. Creating a heatmap from a point
7. Logging metrics to csv file while training. 


# Computer_vision_mini_project.ipynb
Everything to do with training/testing the CLIP and Autoencoder based models

# Playground.ipynb (probably ignore this?)
Just a notebook that I used to generate plots when I needed them for the report, test and debug code etc.

# Train_prompt.ipynb
Training the point-based segmentation model

# Train_unet.ipynb
Training the UNet segmentation model

# peturbation_analysis_py
all code to do with the exploration of robustness section, i.e., 
1. Custom peturbation functions
2. Functionality to apply peturbations and evaluate dice and save information for plotting later

# app.py
This is the User Interface for the point-based segmentation model. Build with Gradio. Will require actual model .pth file to make 
predictions. 


# Models
folder which contains the UNet model architecture (also used for the point-based model).

