import os
import pytorch_lightning as pl
from domainbed.lm.components import Encoder, Decoder, CvaeLoss
import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal

class CVAE(pl.LightningModule):
    def __init__(self, input_shape, hidden_layer_sizes, num_domains, ckpt_path=None):
        """
        hidden_layer_sizes: [l_1, ..., l_n, latent_dim]
        """
        super().__init__()
        self.repo_path = self.get_repo_path()
        self.num_domains = num_domains + 1  # This is extremely ugly, but a practical quick fix until i find out better, how DomainBed works
        self.encoder = Encoder(input_shape=input_shape, hidden_layer_sizes=hidden_layer_sizes, num_domains=num_domains + 1)
        self.decoder = Decoder(input_shape=input_shape, hidden_layer_sizes=hidden_layer_sizes, num_domains=num_domains + 1)
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
        pred = self.decoder(z, conds[:,:,1])
        return pred

    def training_step(self, batch):
        imgs = batch[0].permute(0, 3, 1, 2)
        conds = nn.functional.one_hot(batch[1], num_classes=self.num_domains)
        conds = torch.stack((conds, conds), dim=2)
        predictions = self(imgs, conds)
        loss = self.loss(predictions, imgs)
        return loss

    def validation_step(self, batch):
        imgs = batch[0].permute(0, 3, 1, 2)
        conds = nn.functional.one_hot(batch[1], num_classes=self.num_domains)
        conds = torch.stack((conds, conds), dim=2)
        predictions = self(imgs, conds)
        loss = self.loss(predictions, imgs)
        output = pl.EvalResult(checkpoint_on=loss)
        return output

    def configure_optimizers(self):
        lr = self.learning_rate
        opt = torch.optim.Adam(list(self.ae.parameters())+list(self.linear.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        return opt
        
class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

class LM_CVAE(Algorithm):
    """
    Own conditional variational autoencoder class.
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        import json
        super(Algorithm, self).__init__()
        self.hparams = hparams
        self.cvae = CVAE(input_shape=input_shape,
                         hidden_layer_sizes=json.loads(self.hparams["hidden_layer_sizes"]),
                         num_domains=num_domains,
                         ckpt_path=self.hparams["ckpt_path"])
        self.optimizer = torch.optim.Adam(
            self.cvae.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches, unlabeled=None):
        loss = self.cvae.training_step(batch=[torch.cat([x["image"] for x, y in minibatches]), 
                                              torch.cat([x["domain"] for x, y in minibatches])])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        self.cvae(x)

if __name__ == "__main__":
    cvae = CVAE(input_shape=torch.Tensor([256, 256, 3]), hidden_layer_sizes=[1024, 128, 64], num_domains=3, ckpt_path=None)
    print(cvae)
