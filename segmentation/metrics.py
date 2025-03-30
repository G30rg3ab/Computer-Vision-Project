import torch

class IOU():
    def __init__(self, classes = [1, 2, 3]):
        self.classes = classes

        self.intersection_class_dict = {index: 0 for index in classes}
        self.union_class_dict = {index: 0 for index in classes}


    def update(self, predictions, targets):
        # Create binary masks for the class
        for index in self.classes:
            pred_mask = (predictions == index).float()
            target_mask = (targets == index).float()

            # Exclude ignored pixels from targets
            valid_mask = torch.isin(targets, torch.tensor(self.classes, device=targets.device))

            # Apply the valid mask
            pred_mask = pred_mask * valid_mask
            target_mask = target_mask * valid_mask

            intersection = torch.sum(pred_mask * target_mask)
            self.intersection_class_dict[index] += intersection

            union = torch.sum(pred_mask) + torch.sum(target_mask) - intersection
            self.union_class_dict[index] += union

    def compute(self):
        # Getting the mean_iou
        intersections =  torch.tensor(list(self.intersection_class_dict.values()))
        unions = torch.tensor(list(self.union_class_dict.values()))
        class_ious = intersections/(unions + 1e-6)
        return round(torch.mean(class_ious).item(), 5)
    
    def reset(self):
        self.intersection_class_dict = {index: 0 for index in self.classes}
        self.union_class_dict = {index: 0 for index in self.classes}


class Accuracy():
    def __init__(self, ignore_class = 255):
        self.ignore_class = ignore_class

        self.total_correct = 0
        self.total_valid = 0

    def update(self, predictions, targets):
        valid_mask = (targets != self.ignore_class)
        correct_mask = (predictions[valid_mask] == targets[valid_mask])
        correct_count = torch.sum(correct_mask)
        total_valid   = torch.sum(valid_mask)

        self.total_correct += correct_count
        self.total_valid += total_valid

    def compute(self):
        return round((self.total_correct/(self.total_valid + 1e-6)).item(), 5)
    
    def reset(self):
        self.total_correct = 0
        self.total_valid = 0


class Dice():
    def __init__(self, classes = [1, 2, 3]):
        self.classes = classes

        self.intersection_class_dict = {index: 0 for index in classes}
        self.count_target_dict = {index: 0 for index in classes}
        self.count_pred_dict = {index: 0 for index in classes}

    def update(self, predictions, targets):
        for index in self.classes:
            # Getting the non-border pixels
            valid_mask = torch.isin(targets, torch.tensor(self.classes, device=targets.device))
            pred_mask = (predictions == index).float()
            target_mask = (targets == index).float()

            intersection = torch.sum(pred_mask * target_mask * valid_mask)
            count_target = torch.sum(target_mask * valid_mask)
            count_pred = torch.sum(pred_mask * valid_mask)

            self.intersection_class_dict[index] += intersection
            self.count_target_dict[index] += count_target
            self.count_pred_dict[index] += count_pred

    def compute(self):
        intersections = torch.tensor(list(self.intersection_class_dict.values()))
        counts_target = torch.tensor(list(self.count_target_dict.values()))
        counts_pred = torch.tensor(list(self.count_pred_dict.values()))

        denom = counts_pred + counts_target
        class_dices =  (2 * intersections)/(denom + 1e-6)
        return round(torch.mean(class_dices).item(), 5)
    
    def reset(self):
        self.intersection_class_dict = {index: 0 for index in self.classes}
        self.count_target_dict = {index: 0 for index in self.classes}
        self.count_pred_dict = {index: 0 for index in self.classes}


