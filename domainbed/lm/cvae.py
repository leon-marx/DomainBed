import os
import pytorch_lightning as pl
from components import Encoder, Decoder, CvaeLoss
import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal

class CVAE(pl.LightningModule):
    def __init__(self, input_shape, hidden_layer_sizes, num_domains, ckpt_path=None):  #  layer_sizes, num_domains, 
        """
        hidden_layer_sizes: [l_1, ..., l_n, latent_dim]
        """
        super().__init__()
        self.repo_path = self.get_repo_path()
        self.num_domains = num_domains
        self.encoder = Encoder(input_shape=input_shape, hidden_layer_sizes=hidden_layer_sizes, num_domains=num_domains)
        self.decoder = Decoder(input_shape=input_shape, hidden_layer_sizes=hidden_layer_sizes, num_domains=num_domains)
        self.loss = CvaeLoss()

        if ckpt_path is not None:
            print(f"Loading model from {ckpt_path}")
            self.init_from_ckpt(ckpt_path)

    def get_repo_path(self):
        if os.name == "nt":
            return "C:/users/gooog/desktop/bachelor/code/bachelor/"
        else:
            return "/home/tarkus/leon/bachelor/"

    def init_from_ckpt(self, ckpt_path, ignore_keys=list()):
        try:
            sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        except KeyError:
            sd = torch.load(ckpt_path, map_location="cpu")

        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        if len(missing) > 0:
            print(f"Missing keys in state dict: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected keys in state dict: {unexpected}")

    def forward(self, x, conds):
        """
        x: image of shape (batch_size, height, width, channels)
        conds: one_hot vector of shape (batch_size, num_domains, 2)
        -> the 2 corresponds to the encoder and decoder conditions
        """
        enc_loc, enc_scale = self.encoder(x, conds[:,:,0])
        z = MultivariateNormal(enc_loc, enc_scale).sample()
        dec_loc, dec_scale = self.decoder(z, conds[:,:,1])
        pred = MultivariateNormal(dec_loc, dec_scale).sample()
        return pred

    def training_step(self, batch):
        imgs = batch["image"].permute(0, 3, 1, 2)
        conds = nn.functional.one_hot(batch["domain"], num_classes=self.num_domains)
        conds = torch.stack((conds, conds), dim=0)
        predictions = self(imgs, conds)
        loss = self.loss(predictions, imgs)
        return loss

    def validation_step(self, batch):
        imgs = batch["image"].permute(0, 3, 1, 2)
        conds = nn.functional.one_hot(batch["domain"], num_classes=self.num_domains)
        conds = torch.stack((conds, conds), dim=0)
        predictions = self(imgs, conds)
        loss = self.loss(predictions, imgs)
        output = pl.EvalResult(checkpoint_on=loss)
        return output

    def configure_optimizers(self):
        lr = self.learning_rate
        opt = torch.optim.Adam(list(self.ae.parameters())+list(self.linear.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        return opt


if __name__ == "__main__":
    cvae = CVAE(input_shape=torch.Tensor([256, 256, 3]), hidden_layer_sizes=[1024, 128, 64], num_domains=3, ckpt_path=None)
    print(cvae)
