import albumentations as albu
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from model import UNET
import os

# # Pytorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch import amp

# Custom
from segmentation import utils
from segmentation import preprocessing

# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCHSIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False

class UnetTrain():
    def __init__(self, data_dir = 'Dataset/'):

        self.x_trainVal_dir = os.path.join(data_dir, 'TrainVal/color')
        self.y_trainVal_dir = os.path.join(data_dir, 'TrainVal/label')


    def train_fn(self, loader, model, optimizer, loss_fn, scaler):
        '''
        One epoch of training
        '''
        loop = tqdm(loader)

        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device = DEVICE)
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

    def main(self):
        model = UNET(in_channels=3, out_channels=3).to(DEVICE)
        # Defining the loss function we use in training
        loss_fn = nn.CrossEntropyLoss(ignore_index=255)
        optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)

        # Getting the transforms 
        transform = preprocessing.get_transforms()
        train_augmentation = preprocessing.get_training_augmentation()

        # Getting the loaders
        train_loader, val_loader = utils.get_loaders(self.x_trainVal_dir,
                                                    self.y_trainVal_dir,
                                                    batch_size= BATCHSIZE,
                                                    transform= transform,
                                                    train_augmentation=train_augmentation, 
                                                    num_workers=NUM_WORKERS,
                                                    pin_memory=PIN_MEMORY)

        scaler = amp.GradScaler()
        for epoc in range(NUM_EPOCHS):
            self.train_fn(train_loader, model, optimizer, loss_fn, scaler)

            # save model
            checkpoint = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            utils.save_checkpoint(checkpoint)

            # check accuracy
            utils.check_accuracy(val_loader, model, device = DEVICE)