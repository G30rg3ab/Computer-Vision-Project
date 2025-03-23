import torch
from models.unet_model import UNET
import os

class Trainer():
    def __init__(model, experiment_folder):
        os.mkdir(experiment_folder)

    def load_state():
        pass
        

if __name__== "__main__":
    model = unet_model.UNET()
    trainer = Trainer(model)