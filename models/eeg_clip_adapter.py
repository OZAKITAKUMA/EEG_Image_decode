import torch
import torch.nn as nn

class EEGCLIPAdapter(nn.Module):
    def __init__(self, visual_projection):
        super().__init__()
        ## nn.Parameterでこの行列は学習可能にする
        self.projection = nn.Parameter(visual_projection.detach().clone().float())

    def forward(self, eeg_cls):
        return eeg_cls.float() @ self.projection