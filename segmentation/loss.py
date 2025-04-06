import torch.nn as nn
import torch.nn.functional as F
import torch


class DiceLoss(nn.Module):
    def __init__(self, classes=[0, 1, 2], smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.classes = classes
        self.smooth = smooth
        self.n_classes = len(classes)

    def forward(self, preds, targets):
        '''
        preds: (B, C, H, W) - raw logits
        targets: (B, H, W) - ground truth class indices
        '''
        preds = F.softmax(preds, dim=1)  # Softmax across classes
        loss = 0.0

        for c in range(self.n_classes):
            pred_c = preds[:, c, :, :]           # Shape: (B, H, W)
            target_c = (targets == self.classes[c]).float()    # Shape: (B, H, W)

            intersection = torch.sum(pred_c * target_c)
            total = torch.sum(pred_c) + torch.sum(target_c)

            dice_score = (2 * intersection + self.smooth) / (total + self.smooth)
            loss += 1 - dice_score  # 1 - Dice = loss

        return loss / len(self.classes)
    

class DiceCELoss(nn.Module):
    """Combine Cross-Entropy and Dice for segmentation."""
    def __init__(self, dice_weight=1.0, ce_weight=1.0, classes = [0, 1, 2]):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = DiceLoss(classes=classes)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=255) 

    def forward(self, logits, targets):
        # logits shape (N, C, H, W) and targets shape (N, H, W) if multiclass
        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)

        return self.dice_weight * dice + self.ce_weight * ce