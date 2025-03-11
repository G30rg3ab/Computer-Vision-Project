import albumentations as albu
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from models.unet_model import UNET
import os

# Pytorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch import amp
import optuna

# Custom
from segmentation.utils import model_utils, ModelEval, s3utils
from segmentation.utils import preprocessing, traininglog
from segmentation.dataset import CVDataset
from torch.utils.data import DataLoader
from segmentation.constants import BucketConstants

trial_save_template = 'unet_best_model_trial_{}'
model_folder_name = 'unet_experiment_0'
log_save_name = 'training_log.csv'

# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCHSIZE = 8
NUM_EPOCHS = 70
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
        with torch.autocast(device_type=DEVICE, dtype=torch.float16):
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

def train_and_evaluate(model, optimizer, train_loader, valid_ds, loss_fn, scaler, trial_id,num_epochs, hyperparameters):
    '''
    Trains the model for a given set of hyperparameters and evaluates it.
    '''
    best_iou = 0 # Best IoU for this trial
    for epoch in range(num_epochs):
        # ... training loop ....
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # Evaluate every few epochs
        current_iou = ModelEval(valid_ds, model, device=DEVICE).mean_IoU(progress_bar=False)
        print(f"Epoch {epoch}: Validation IoU = {current_iou:.4f}")
        traininglog.log_training(log_save_name,trial = trial_id, epoch =  epoch, val_IoU =  current_iou, hyperparameters = hyperparameters)

        if current_iou > best_iou:
            best_iou = current_iou
            # Saving this model (best model)
            checkpoint = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'hyperparameters': hyperparameters,
                'epoch': epoch,
                'IoU': current_iou
            }
            
            # Saving the model checkpoint for hyperparameters
            name = trial_save_template.format(trial_id) + '.pth'
            key = os.path.join(model_folder_name, name)
            model_utils.save_checkpoint(model, checkpoint, name, BucketConstants.bucket, key)

    return best_iou


def objective(trial):
    # 1. Sample hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log = True)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
    optimizer_type = trial.suggest_categorical("optimizer", ["Adam", "SGD"])

    # 2. Create the model and optimizer
    model = UNET(in_channels=3, out_channels=3).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) if optimizer_type == "Adam" \
                else optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # 3. Prepare data loaders using batch_size
    train_augmentation = preprocessing.get_training_augmentation()
    valid_augmentation = preprocessing.get_validation_augmentation()
    # For normalising image data (mask will remain unchanged), and converting to tensor
    preprocess_fn = preprocessing.get_preprocessing(preprocessing_fn=None)
    # Initialising the data loaders
    train_ds = CVDataset(x_train_fps, y_train_fps, augmentation = train_augmentation, preprocessing = preprocess_fn)
    train_loader = DataLoader(train_ds,batch_size= batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY,shuffle=True)

    # 4. Validation data set
    valid_ds = CVDataset(x_val_fps, y_val_fps, augmentation = valid_augmentation, preprocessing = preprocess_fn)

    trial_id = trial.number
    scaler = amp.GradScaler()

    hyperparams = {'learning_rate': learning_rate,
                   'batch_size': batch_size,
                   'optimizer_type': optimizer_type,
                   'total_epochs': NUM_EPOCHS}

    # Best IoU score for the set of hyperparameters
    best_iou = train_and_evaluate(model, optimizer, train_loader, valid_ds, loss_fn, scaler, trial_id, NUM_EPOCHS, hyperparams)
    return best_iou

def tune_unet():

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=15)
    print('Best hyperparameters:', study.best_params)

    # Finished training the model
    best_trial = study.best_trial.number
    print(f'Best trial number was {best_trial}')

    # Uploading the log
    log_save_path = os.path.join(model_folder_name, log_save_name)
    s3utils.upload_s3(log_save_name,  BucketConstants.bucket, log_save_path)

if __name__ == '__main__':
    tune_unet()
    