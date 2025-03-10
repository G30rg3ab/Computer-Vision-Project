# Custom modules
from models import unet_model
from segmentation.utils import preprocessing, model_utils, show
from segmentation.dataset import CVDataset

# Other
import torch
import os

CLASSES = ['dog', 'cat', 'background']
DATA_DIR = 'Dataset/'
x_test_dir = os.path.join(DATA_DIR, 'Test/color')
y_test_dir = os.path.join(DATA_DIR, 'Test/label')
x_trainVal_dir = os.path.join(DATA_DIR, 'TrainVal/color')
y_trainVal_dir = os.path.join(DATA_DIR, 'TrainVal/label')

# Getting a list of relative paths to the images (x) and the masks/labels (y)
x_test_fps, y_test_fps = preprocessing.get_testing_paths(x_test_dir, y_test_dir)
# Splitting relative path names into into training and validation
x_train_fps, x_val_fps, y_train_fps, y_val_fps = preprocessing.train_val_split(x_trainVal_dir, y_trainVal_dir, 0.2)


model = unet_model.UNET(in_channels=3, out_channels=3)

test_augmentation = preprocessing.get_validation_augmentation()
preprocessing_fn = preprocessing.get_preprocessing(preprocessing_fn=None)
augmented_dataset = CVDataset(x_train_fps , y_train_fps, augmentation = test_augmentation, preprocessing=preprocessing_fn)

image, mask = augmented_dataset[1]
state_dict = torch.load('my_checkpoint.pth (1).tar', map_location=torch.device('cpu'))
model_utils.load_checkpoint(state_dict, model)
prediction = model_utils.predict(model, image)
show.visualiseData(predicted_mask = prediction)
print(prediction.shape)