import torch
import torch.nn as nn

class CLIPLoss(nn.Module):
    def __init__(self, temperature=0.07, learnable_temp=False):
        super().__init__()
        self.learnable_temp = learnable_temp
        if learnable_temp:
            self.log_temp = nn.Parameter(torch.log(torch.tensor(temperature)))
        else:
            self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, image_features, text_features):
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        temp = torch.exp(self.log_temp) if self.learnable_temp else self.temperature
        logits = (image_features @ text_features.T) / temp

        labels = torch.arange(logits.shape[0], device=logits.device)
        loss_i = self.criterion(logits, labels)
        loss_t = self.criterion(logits.T, labels)
        return (loss_i + loss_t) / 2.0