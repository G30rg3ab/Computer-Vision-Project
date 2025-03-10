import albumentations as albu
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from models.unet_model import UNET
import os

# # Pytorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch import amp

# Custom
from segmentation.utils import model_utils
from segmentation.utils import preprocessing
from segmentation.dataset import CVDataset
from torch.utils.data import DataLoader

# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCHSIZE = 8
NUM_EPOCHS = 50
NUM_WORKERS = 4
PIN_MEMORY = True
LOAD_MODEL = False
DATA_DIR = 'Dataset/'
x_test_dir = os.path.join(DATA_DIR, 'Test/color')
y_test_dir = os.path.join(DATA_DIR, 'Test/label')
x_trainVal_dir = os.path.join(DATA_DIR, 'TrainVal/color')
y_trainVal_dir = os.path.join(DATA_DIR, 'TrainVal/label')
# Splitting relative path names into into training and validation
x_train_fps, x_val_fps, y_train_fps, y_val_fps = preprocessing.train_val_split(x_trainVal_dir, y_trainVal_dir, 0.2)

def train_fn(loader, model, optimizer, loss_fn, scaler):
    '''
    One epoch of training
    '''
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device = DEVICE,  dtype=torch.float32)
        targets = targets.long().unsqueeze(1).to(device = DEVICE)

        # forward
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            targets = targets.to(device = DEVICE).squeeze(1)  # Now masks is (N,H,W)
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss = loss.item())

def main():
    model = UNET(in_channels=3, out_channels=3).to(DEVICE)
    # Defining the loss function we use in training
    loss_fn = nn.CrossEntropyLoss(ignore_index=255) # Ignoring 255 in the mask (this is the white border)
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)

    # Getting the transforms 
    train_augmentation = preprocessing.get_training_augmentation()
    valid_augmentation = preprocessing.get_validation_augmentation()
    # For normalising image data (mask will remain unchanged), and converting to tensor
    preprocess_fn = preprocessing.get_preprocessing(preprocessing_fn=None)

    # Initialising the data loaders
    train_ds = CVDataset(x_train_fps, y_train_fps, augmentation = train_augmentation, preprocessing = preprocess_fn, apply_augmentations_to_mask=True)
    train_loader = DataLoader(train_ds,batch_size= BATCHSIZE,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY,shuffle=True)

    # Validation data set
    valid_ds = CVDataset(x_val_fps, y_val_fps, augmentation = valid_augmentation, preprocessing = preprocess_fn, apply_augmentations_to_mask = False)

    scaler = amp.GradScaler()
    for epoc in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        model_utils.save_checkpoint(checkpoint)

        # Checking every 10 epochs
        if (epoc % 10) == 0:
            # check the Intersection over union score every 10 epochs
            model_utils.check_iou(valid_ds, model, device = DEVICE)

    # When finshed print the iou on validation set
    model_utils.check_iou(valid_ds, model, device=DEVICE)

if __name__ == '__main__':
    main()