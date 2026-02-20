import torch.nn as nn
import torch.nn.functional as F
import torch


class DiceLoss(nn.Module):
    def __init__(self, classes=[0, 1, 2], smooth=1e-6):
        '''
        custum loss function for multiple classes, it will 
        only evaluate for classes in the target whcih are specified
        in the classes parameter
        '''
        super(DiceLoss, self).__init__()
        self.classes = classes
        self.smooth = smooth
        self.n_classes = len(classes)

    def forward(self, preds, targets):
        # apply softmax and initialise loss as 0
        preds = F.softmax(preds, dim=1)  
        loss = 0.0

        for c in range(self.n_classes):
            # this will now be of size (batch, H, W)
            pred_c = preds[:, c, :, :]        
            target_c = (targets == self.classes[c]).float()    

            # Get intersection and union
            intersection = torch.sum(pred_c * target_c)
            total = torch.sum(pred_c) + torch.sum(target_c)
            
            # calculate the dice score
            dice_score = (2 * intersection + self.smooth) / (total + self.smooth)
            loss += 1 - dice_score  # 1 - dice score = loss

        return loss / len(self.classes)
    

class DiceCELoss(nn.Module):
    """Combine Cross-Entropy and Dice for segmentation
    automatically ignores class = 255 which is the white 
    border
    """
    def __init__(self, dice_weight=1.0, ce_weight=1.0, classes = [0, 1, 2]):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = DiceLoss(classes=classes)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=255) 

    def forward(self, logits, targets):
        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        # return weighted sum of the losses 
        return self.dice_weight * dice + self.ce_weight * ce