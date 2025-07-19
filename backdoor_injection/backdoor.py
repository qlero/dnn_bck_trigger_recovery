import torch

from .visualizer import save_image

class BackdoorInjector():
    def __init__(self, normalizer, denormalizer, poison_ratio, target_classe, trigger, trigger_mask):
        self.normalizer   = normalizer
        self.denormalizer = denormalizer
        self.beta         = poison_ratio
        self.target       = target_classe
        self.trigger      = trigger
        self.trigger_mask = trigger_mask

    def __call__(self, data, labels, skip=None, test=False):

        # Checks if some elements shouldn't be modified (ex. if they've already been backdoored)
        if skip is not None:
            beta = (data.shape[0] * self.beta)/(data.shape[0]-torch.sum(skip).detach().cpu().item())
        else:
            beta = self.beta
            skip = (torch.zeros(data.shape[0]) == 1).to(data.device)

        temp_data   = self.denormalizer(data[~skip,:,:,:]).clip(0, 1)
        temp_labels = labels[~skip]

        # Computes the index mask deciding which benign datapoint is poisoned
        if test: 
            ratio = 1.
        else:
            ratio = beta

        mask = torch.bernoulli(torch.full(temp_labels.shape, ratio))==1
        mask = mask.to(data.device)

        # If no element is picked, returns the original data
        if torch.sum(mask).detach().cpu().item() == 0:
            return data, labels, None
        
        data_to_poison   = temp_data[mask]
        labels_to_poison = temp_labels[mask]

        # Injects the triggers
        data_to_poison[self.trigger_mask.repeat(data_to_poison.shape[0],1,1,1)] = self.trigger.repeat(data_to_poison.shape[0],1,1,1).flatten()
        labels_to_poison  = torch.full(labels_to_poison.shape, self.target).to(data.device)
        temp_data[mask]   = data_to_poison
        temp_labels[mask] = labels_to_poison

        # Concatenates
        data[~skip]   = self.normalizer(temp_data.clip(0, 1))
        labels[~skip] = temp_labels
        skip[~skip]   = mask

        return data, labels, skip