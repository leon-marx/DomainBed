import pytorch_lightning as pl
import torch.nn as nn
import torch
from pytorch_lightning.core.lightning import LightningModule

class Encoder(nn.Module):
    def __init__(self, input_shape, hidden_layer_sizes, num_domains):
        super().__init__()
        self.flatten = nn.Flatten()
        input_size = int(torch.prod(torch.Tensor(input_shape)).item())
        modules = []
        for i, out_size in enumerate(hidden_layer_sizes[:-1]):
            if i == 0:
                in_size = input_size
            else:
                in_size = hidden_layer_sizes[i-1]
            modules.append(nn.Linear(in_size + num_domains, out_size + num_domains))
            modules.append(nn.ReLU())
        self.sequential = nn.Sequential(*modules)
        self.get_loc = nn.Linear(hidden_layer_sizes[-2] + num_domains, hidden_layer_sizes[-1])
        self.get_scale = nn.Linear(hidden_layer_sizes[-2] + num_domains, hidden_layer_sizes[-1])

    def forward(self, x, cond):
        """
        x: image of shape (batch_size, height, width, channels)
        cond: one_hot vector of shape (batch_size, num_domains)
        """
        x = self.flatten(x)
        x = torch.cat((x, cond), dim=1)
        h = self.sequential(x)
        loc = self.get_loc(h)
        scale = torch.diag_embed(torch.exp(self.get_scale(h)), dim1=-2, dim2=-1)
        return loc, scale

class Decoder(nn.Module):
    def __init__(self, input_shape, hidden_layer_sizes, num_domains):
        super().__init__()
        self.input_shape = input_shape
        input_size = int(torch.prod(torch.Tensor(input_shape)).item())
        modules = []
        for i, in_size in enumerate(reversed(hidden_layer_sizes[1:])):
            out_size = hidden_layer_sizes[-i-2]
            modules.append(nn.Linear(in_size + num_domains, out_size + num_domains))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(hidden_layer_sizes[0] + num_domains, input_size))
        self.sequential = nn.Sequential(*modules)

    def forward(self, x, cond):
        """
        x: image of shape (batch_size, latent_dim)
        cond: one_hot vector of shape (batch_size, num_domains)
        """
        x = torch.cat((x, cond), dim=1)
        pred = self.sequential(x).view(-1, *self.input_shape)
        return pred

class CvaeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, labels):
        return torch.abs(predictions - labels)
