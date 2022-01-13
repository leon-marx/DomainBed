import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import os
import torch


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


class LM_CCVAE_NC(Algorithm):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        """
        Initializes the conditional variational autoencoder.

        input_shape: Tuple of shape (channels, height, width)
        num_classes: int, usually 7
        num_domains: int, usually 3 (number of training classes)
        hparams: {"K": int, number of samples generated for training,
                  "ckpt_path": str, checkpoint path to load model from,
                  "lamb": float, weight for the reconstruction part of the ELBO Loss,
                  "latent_size": int, size of the latent space
                  ...}
        """
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_domains = num_domains + 1  # We do not neglect the test domain for one-hot encoding
        self.hparams = hparams

        self.K = self.hparams["K"]
        self.ckpt_path = self.hparams["ckpt_path"]
        self.lamb = self.hparams["lamb"]
        self.latent_size = self.hparams["latent_size"]

        self.input_size = int(torch.prod(torch.Tensor(input_shape)).item())
        self.encoder = Encoder(num_classes=self.num_classes, num_domains=self.num_domains, latent_size=self.latent_size)
        self.decoder = Decoder(num_classes=self.num_classes, num_domains=self.num_domains, latent_size=self.latent_size)
        self.loss = ELBOLoss()
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

        if self.ckpt_path is not None:
            self.init_from_ckpt(self.ckpt_path)


    def update(self, minibatches, unlabeled=None, return_train_loss=False, split_loss=False, output_dir=None):
        """
        Calculates the loss for the given set of minibatches.

        minibatches: List of tuples [(x, y)]
            x: {"image": Tensor of shape (batch_size, channels, height, width),
                "domain": Tensor of shape (batch_size)}
                    corresponds to int d = 0,...,3 (domain)
            y: Tensor of shape (batch_size)
                corresponds to int d = 0,...,6 (class)
        unlabeled: This is not supported!
        return_train_loss: If True, returns loss as 2nd value (to display in progress bar)
        split_loss: If True, returns separate kld- and reconstruction-loss
        output_dir: string containing relative path to output directory for grad_flow plot
        """
        images = torch.cat([x["image"] for x, y in minibatches]) # (batch_size, channels, height, width)
        classes = torch.nn.functional.one_hot(torch.cat([y for x, y in minibatches]), num_classes=self.num_classes).flatten(start_dim=1) # (batch_size, num_classes)
        domains = torch.nn.functional.one_hot(torch.cat([x["domain"] for x, y in minibatches]), num_classes=self.num_domains).flatten(start_dim=1) # (batch_size, num_domains)

        enc_mu, enc_logvar = self.encoder(images, classes, domains)
        z = self.sample(enc_mu, enc_logvar, num=self.K)

        if self.K > 1:
            dec_mu, dec_logvar = self.decoder.forward_K(z, classes, domains)
        else:
            dec_mu, dec_logvar = self.decoder.forward(z, classes, domains)

        if split_loss:
            loss, kld, recon = self.loss(images, enc_mu, enc_logvar, dec_mu, dec_logvar, lamb=self.lamb, split_loss=True)
        else:
            loss = self.loss(images, enc_mu, enc_logvar, dec_mu, dec_logvar, lamb=self.lamb)

        self.optimizer.zero_grad()
        loss.backward()
        if output_dir is not None:
            self.plot_grad_flow(self.named_parameters(), output_dir)
        self.optimizer.step()

        if return_train_loss:
            if split_loss:
                return {'loss': loss.item()}, loss.item(), kld.item(), recon.item()
            else:
                return {'loss': loss.item()}, loss.item()
        else:
            if split_loss:                
                return {'loss': loss.item()}, kld.item(), recon.item()
            else:
                return {'loss': loss.item()}

    def evaluate(self, minibatches, unlabeled=None, return_eval_loss=False):
        """
        Calculates the loss for the given set of minibatches.

        minibatches: List of tuples [(x, y)]
            x: {"image": Tensor of shape (batch_size, channels, height, width),
                "domain": Tensor of shape (batch_size, 1)}
                    The 1 corresponds to int d = 0,...,3 (domain)
            y: Tensor of shape (batch_size, 1)
                The 1 corresponds to int d = 0,...,6 (class)
        unlabeled: This is not supported!
        return_eval_loss: If True, returns loss as 2nd value (to display in progress bar)
        """
        with torch.no_grad():
            self.eval()
            images = torch.cat([x["image"] for x, y in minibatches]) # (batch_size, channels, height, width)
            classes = torch.nn.functional.one_hot(torch.cat([y for x, y in minibatches]), num_classes=self.num_classes).flatten(start_dim=1) # (batch_size, num_classes)
            domains = torch.nn.functional.one_hot(torch.cat([x["domain"] for x, y in minibatches]), num_classes=self.num_domains).flatten(start_dim=1) # (batch_size, num_domains)

            enc_mu, enc_logvar = self.encoder(images, classes, domains)
            z = self.sample(enc_mu, enc_logvar, num=self.K) 

            dec_mu, dec_logvar = self.decoder.forward(z, classes, domains)

            loss = self.loss(images, enc_mu, enc_logvar, dec_mu, dec_logvar, lamb=self.lamb).item()

            self.train()
            
        if return_eval_loss:
            return {'loss': loss}, loss
        else:
            return {'loss': loss}

    def run(self, images, classes, enc_conditions, dec_conditions, raw=False):
        """
        Generates reconstructions of the given images and conditions

        images: Tensor of shape (batch_size, channels, height, width)
        classes: Tensor of shape (batch_size)
            corresponds to int d = 0,...,6 (class)
        enc_conditions: Tensor of shape (batch_size)
            corresponds to int d = 0,...,3 (domain)
        dec_conditions: Tensor of shape (batch_size)
            corresponds to int d = 0,...,3 (domain)
        raw: Bool, if True no noise / variation is added
        """
        with torch.no_grad():
            self.eval()
            classes = torch.nn.functional.one_hot(classes, num_classes=self.num_classes).flatten(start_dim=1) # (batch_size, num_classes)
            enc_conditions = torch.nn.functional.one_hot(enc_conditions, num_classes=self.num_domains).flatten(start_dim=1) # (batch_size, num_domains)
            dec_conditions = torch.nn.functional.one_hot(dec_conditions, num_classes=self.num_domains).flatten(start_dim=1) # (batch_size, num_domains)
            if raw:
                enc_mu, enc_logvar = self.encoder(images, classes, enc_conditions)
                
                reconstructions, dec_logvar = self.decoder(enc_mu, classes, dec_conditions)
            else:
                enc_mu, enc_logvar = self.encoder(images, classes, enc_conditions)
                z = self.sample(enc_mu, enc_logvar)

                dec_mu, dec_logvar = self.decoder(z, classes, dec_conditions)
                reconstructions = self.sample(dec_mu, dec_logvar)
            self.train()
            return reconstructions

    def sample(self, mu, logvar, num=1):
        """
        Samples from N(mu, var).

        mu: Tensor of shape (batch_size, ...) -> ... can be an arbitrary shape
        logvar: Tensor of shape (batch_size, D)
        num: int, Number of samples.
        """
        if num > 1:
            mu = torch.stack([mu for i in range(num)], dim=1)
            logvar = torch.stack([logvar for i in range(num)], dim=1)
            std = (0.5 * logvar).exp()
            eps = torch.randn_like(mu)
            return mu + eps * std
        else:
            std = (0.5 * logvar).exp()
            eps = torch.randn_like(mu)
            return mu + eps * std

    def init_from_ckpt(self, ckpt_path, ignore_keys=list()):
        print(f"Loading model from {ckpt_path}")
        try:
            sd = torch.load(ckpt_path, map_location="cpu")["model_dict"]
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

    def plot_grad_flow(named_parameters, output_dir):
        """
        Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.
        
        Usage: Plug this function in Trainer class after loss.backwards() as 
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow
        """
        ave_grads = []
        max_grads= []
        layers = []
        for n, p in named_parameters:
            if(p.requires_grad) and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())
        plt.figure()
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
        plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
        plt.savefig(os.path.join(output_dir, "gradient_flow.png"))
        plt.close()


class Encoder(torch.nn.Module):
    def __init__(self, num_classes, num_domains, latent_size):
        """
        Initializes the encoder.

        num_classes: Number of classes, usually 7
        num_domains: Number of domains, usually 4 or 3 (without sketch)
        latent_size: int, size of the latent space
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.latent_size = latent_size
        self.conv_sequential = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3 + self.num_classes + self.num_domains, out_channels=128, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(p=0.2),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(p=0.2),
            torch.nn.MaxPool2d(kernel_size=2),  # (N, 128, 112, 112)
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=256),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(p=0.2),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=256),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(p=0.2),
            torch.nn.MaxPool2d(kernel_size=2),  # (N, 256, 56, 56)
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=512),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(p=0.2),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=512),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(p=0.2),
            torch.nn.MaxPool2d(kernel_size=2),  # (N, 512, 28, 28)
            torch.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=1024),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(p=0.2),
            torch.nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=1024),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(p=0.2),
            torch.nn.MaxPool2d(kernel_size=2),  # (N, 1024, 14, 14)
            torch.nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=2048),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(p=0.2),
            torch.nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=2048),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(p=0.2),
            torch.nn.MaxPool2d(kernel_size=2),  # (N, 2048, 7, 7)
            torch.nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=1024),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(p=0.2),
            torch.nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=512),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(p=0.2),
            torch.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=256),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(p=0.2),
            torch.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(p=0.2),
        )
        self.flatten = torch.nn.Flatten()
        self.get_mu = torch.nn.Sequential(
            torch.nn.Linear(6272, self.latent_size),
            torch.nn.Tanh()
        )
        self.get_logvar = torch.nn.Sequential(
            torch.nn.Linear(6272, self.latent_size),
            torch.nn.Tanh()
        )
    def forward(self, images, classes, domains):
        """
        Calculates mean and diagonal log-variance of p(z | x).

        images: Tensor of shape (batch_size, channels, height, width)
        classes: Tensor of shape (batch_size, num_classes)
        domains: Tensor of shape (batch_size, num_domains)
        """
        class_conds = torch.ones(size=(images.shape[0], self.num_classes, 224, 224)).to(images.device) * classes.view(images.shape[0], self.num_classes, 1, 1)
        domain_conds = torch.ones(size=(images.shape[0], self.num_domains, 224, 224)).to(images.device) * domains.view(images.shape[0], self.num_domains, 1, 1)
        x = torch.cat((images, class_conds, domain_conds), dim=1)
        x = self.conv_sequential(x)
        x = self.flatten(x)
        enc_mu = self.get_mu(x)
        enc_logvar = self.get_logvar(x)

        return enc_mu, enc_logvar


class Decoder(torch.nn.Module):
    def __init__(self, num_classes, num_domains, latent_size):
        """
        Initializes the decoder.

        num_classes: Number of classes, usually 7
        num_domains: Number of domains, usually 4 or 3 (without sketch)
        latent_size: int, size of the latent space
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.latent_size = latent_size
        self.linear = torch.nn.Linear(self.latent_size + self.num_classes + self.num_domains, 6272)
        self.reshape = lambda x : x.view(-1, 128, 7, 7)
        self.conv_sequential = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=256),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(p=0.2),
            torch.nn.ConvTranspose2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=512),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(p=0.2),
            torch.nn.ConvTranspose2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=1024),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(p=0.2),
            torch.nn.ConvTranspose2d(in_channels=1024, out_channels=2048, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=2048),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(p=0.2),
            torch.nn.Upsample(scale_factor=2, mode="nearest"),
            torch.nn.ConvTranspose2d(in_channels=2048, out_channels=2048, kernel_size=3, padding=1, bias=False),  # (N, 2048, 14, 14)
            torch.nn.BatchNorm2d(num_features=2048),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(p=0.2),
            torch.nn.ConvTranspose2d(in_channels=2048, out_channels=1024, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=1024),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(p=0.2),
            torch.nn.Upsample(scale_factor=2, mode="nearest"),
            torch.nn.ConvTranspose2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1, bias=False),  # (N, 1024, 28, 28)
            torch.nn.BatchNorm2d(num_features=1024),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(p=0.2),
            torch.nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=512),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(p=0.2),
            torch.nn.Upsample(scale_factor=2, mode="nearest"),
            torch.nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=False),  # (N, 512, 56, 56)
            torch.nn.BatchNorm2d(num_features=512),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(p=0.2),
            torch.nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=256),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(p=0.2),
            torch.nn.Upsample(scale_factor=2, mode="nearest"),
            torch.nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),  # (N, 256, 112, 112)
            torch.nn.BatchNorm2d(num_features=256),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(p=0.2),
            torch.nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(p=0.2),
        )
        self.get_mu = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode="nearest"),
            torch.nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(p=0.2),
            torch.nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=3, padding=1),  # (N, 3, 224, 224)
            torch.nn.Sigmoid(),
        )
        self.get_logvar = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode="nearest"),
            torch.nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(p=0.2),
            torch.nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=3, padding=1),  # (N, 3, 224, 224)
            torch.nn.Sigmoid(),
        )

    def forward(self, codes, classes, domains):
        """
        Calculates mean and diagonal log-variance of p(x | z).

        codes: Tensor of shape (batch_size, D) -> D = dimension of z
        classes: Tensor of shape (batch_size, num_classes)
        domains: Tensor of shape (batch_size, num_domains)
        """
        x = torch.cat((codes, classes, domains), dim=1)
        x = self.linear(x)
        x = self.reshape(x)
        x = self.conv_sequential(x)
        dec_mu = self.get_mu(x)
        dec_logvar = self.get_logvar(x)

        return dec_mu, dec_logvar
    
    def forward_K(self, codes, classes, domains):
        """
        Calculates mean and diagonal log-variance of p(x | z).

        codes: Tensor of shape (batch_size, K, D) -> K = number of samples generated, D = dimension of z
        classes: Tensor of shape (batch_size, num_classes)
        domains: Tensor of shape (batch_size, num_domains)
        """
        batch_size = codes.shape[0]
        K = codes.shape[1]
        class_conds = torch.stack([classes for i in range(K)], dim=1) # (batch_size, K, num_classes)
        domain_conds = torch.stack([domains for i in range(K)], dim=1) # (batch_size, K, num_domains)
        x = torch.cat((codes, class_conds, domain_conds), dim=2).view(batch_size * K, -1) # (batch_size * K, D + num_classes + num_domains)
        x = self.linear(x)
        x = self.reshape(x)
        x = self.conv_sequential(x)
        dec_mu = self.get_mu(x).view(batch_size, K, 3, 224, 224)
        dec_logvar = self.get_logvar(x).view(batch_size, K, 3, 224, 224)

        return dec_mu, dec_logvar
        

class ELBOLoss(torch.nn.Module):
    def __init__(self):
        """
        Initializes the ELBO loss.
        """
        super().__init__()
        self.flat_K = lambda x, N, K : x.view(N, K, -1)

    def forward(self, x, enc_mu, enc_logvar, dec_mu, dec_logvar, lamb=1, split_loss=False):
        """
        Calculates the ELBO Loss (negative ELBO).

        x: Tensor of shape (batch_size, channels, height, width)
        enc_mu: Tensor of shape (batch_size, D) -> D = dimension of z
        enc_logvar: Tensor of shape (batch_size, D)
        dec_mu: Tensor of shape (batch_size, K, channels, height, width) -> K = number of samples for z
        dec_logvar: Tensor of shape (batch_size, K, channels, height, width)
        lamb: optional weighing of the reconstruction part of the ELBO Loss
        split_loss: If True, returns separate kld- and reconstruction-loss
        """
        N = dec_mu.shape[0]
        K = dec_mu.shape[1]
        if len(dec_mu.shape) <= 4:
            K = 1
    
        x = torch.stack([x for i in range(K)], dim=1)
        x = self.flat_K(x, N, K)
        dec_mu = self.flat_K(dec_mu, N, K)
        dec_logvar = self.flat_K(dec_logvar, N, K)

        if split_loss:
            kld_loss = torch.mean(
            # KL divergence -> regularization
            (torch.sum(
                enc_mu ** 2 + enc_logvar.exp() - enc_logvar - torch.ones(enc_mu.shape).to(x.device),
                dim=1
            ) * 0.5 * (1.0 / lamb)
            ),
            dim=0
            )
            reconstruction_loss = torch.mean(
            # likelihood -> similarity
            (torch.mean(
                torch.sum(
                    (x - dec_mu) ** 2 / (2 * dec_logvar.exp()) + 0.5 * dec_logvar,
                    dim=2
                ), 
                dim=1
            )),
            dim=0
            )
            return kld_loss + reconstruction_loss, kld_loss, reconstruction_loss

        return torch.mean(
            # KL divergence -> regularization
            (torch.sum(
                enc_mu ** 2 + enc_logvar.exp() - enc_logvar - torch.ones(enc_mu.shape).to(x.device),
                dim=1
            ) * 0.5 * (1.0 / lamb)
            
            # likelihood -> similarity
            + torch.mean(
                torch.sum(
                    (x - dec_mu) ** 2 / (2 * dec_logvar.exp()) + 0.5 * dec_logvar,
                    dim=2
                ), 
                dim=1
            )),
            dim=0
        )


if __name__ == "__main__":
    input_shape = (3, 224, 224)
    num_classes = 7
    num_train_domains = 2
    hparams = {"lr": 5e-05,
               "K": 1,
               "ckpt_path": None,
               "lamb": 10,
               "weight_decay": 0.0,
               "batch_size": 2,
               "latent_size": 512
    }
    batch_size = hparams["batch_size"]
    minibatches = []
    for i in range(3):  # iterating over all datasets / environments
        x = {"image": torch.rand(size=(batch_size, 3, 224, 224)) * (2.6400 - (-2.1179)) + (-2.1179),
             "domain": torch.randint(low=0, high=3, size=(batch_size,))}
        y = torch.randint(low=0, high=7, size=(batch_size,))
        minibatches.append((x, y))

    cvae = LM_CCVAE_NC(input_shape=input_shape, num_classes=num_classes,
                   num_domains=num_train_domains, hparams=hparams)
    print(cvae)
    # step_vals = cvae.update(minibatches=minibatches)
    step_vals, train_loss = cvae.update(minibatches=minibatches, return_train_loss=True)
    print(step_vals, train_loss)
    print("Done!")
