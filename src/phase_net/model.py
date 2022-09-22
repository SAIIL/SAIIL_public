import torch
from torch import nn
import torchvision
import pytorch_lightning as pl
from torchvision import transforms

# Model example for phase recognition training pipeline
# MIT liscence
# Author: Yutong Ban, Guy Rosman 



class CNN_model(pl.LightningModule):
    # a CNN model example for the pipeline interface
    def __init__(self, n_class=7):
        super().__init__()
        resnet = torchvision.models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.backbone = nn.Sequential(*modules)
        self.n_class = n_class
        self.class_head = nn.Sequential(nn.Linear(2048, self.n_class), nn.ReLU())

    def forward(self, x):
        # The inference action - given x, predict the output.
        for t in range(x.shape[1]):
            z = self.backbone(x)
            y_hat = self.class_head(z.squeeze())
        return y_hat


class CNN_LSTM(pl.LightningModule):
    # a CNN + LSTM model example for the pipeline interface
    def __init__(self, n_class=7, hidden_size=2048):
        super().__init__()
        resnet = torchvision.models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.backbone = nn.Sequential(*modules)
        self.n_class = n_class
        self.hidden_size=hidden_size
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size)
        self.class_head = nn.Sequential(nn.Linear(hidden_size, self.n_class), nn.LeakyReLU())

    def forward(self, x):
        # The inference action - given x, predict the output.
        embedding = []
        for t in range(x.shape[1]):
            z = self.backbone(x[:,t].squeeze())
            embedding.append(z.view(1,-1,self.hidden_size))
        embedding = torch.cat(embedding)
        # y_hat = self.class_head(embedding[-1].squeeze())

        out, hidden = self.lstm(embedding)
        # y_hat = self.class_head(hidden[0].squeeze())
        y_hat = self.class_head(out[-1].squeeze())
        return y_hat