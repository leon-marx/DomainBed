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


class BIG_LM_CCVAE(Algorithm):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        """
        Initializes the conditional variational autoencoder.

        input_shape: Tuple of shape (channels, height, width)
        num_classes: int, usually 7
        num_domains: int, usually 3 (number of training classes)
        hparams: {"hidden_sizes": list, sizes of the different hidden layers [l1, l2, ...],
                  "K": int, number of samples generated for training,
                  "ckpt_path": str, checkpoint path to load model from,
                  ...}
        """
        import json
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.input_shape = input_shape
        self.num_classes = num_classes,
        self.num_domains = num_domains + 1  # We do not neglect the test domain for one-hot encoding
        self.hparams = hparams

        self.hidden_sizes = json.loads(self.hparams["hidden_sizes"])  # json to load list
        self.K = self.hparams["K"]
        self.ckpt_path = self.hparams["ckpt_path"]

        self.input_size = int(torch.prod(torch.Tensor(input_shape)).item())
        self.encoder = Encoder(input_size=self.input_size, hidden_sizes=self.hidden_sizes, num_domains=self.num_domains)
        self.decoder = Decoder(input_size=self.input_size, hidden_sizes=self.hidden_sizes, num_domains=self.num_domains)
        self.loss = ELBOLoss()
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

        if self.ckpt_path is not None:
            self.init_from_ckpt(self.ckpt_path)


    def update(self, minibatches, unlabeled=None, return_train_loss=False):
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
        """
        images = torch.cat([x["image"] for x, y in minibatches]) # (batch_size, channels, height, width)
        conditions = torch.nn.functional.one_hot(torch.cat([x["domain"] for x, y in minibatches]), num_classes=self.num_domains).flatten(start_dim=1) # (batch_size, num_classes)

        enc_mu, enc_logvar = self.encoder(images, conditions)
        z = self.sample(enc_mu, enc_logvar, num=self.K)

        dec_mu, dec_logvar = self.decoder.forward_K(z, conditions)

        loss = self.loss(images, enc_mu, enc_logvar, dec_mu, dec_logvar)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if return_train_loss:
            return {'loss': loss.item()}, loss.item()
        else:
            return {'loss': loss.item()}

    def evaluate(self, minibatches, unlabeled=None, return_eval_loss=False):
        """
        Calculates the loss for the given set of minibatches.

        minibatches: List of tuples [(x, y)]
            x: {"image": Tensor of shape (batch_size, channles, height, width),
                "domain": Tensor of shape (batch_size, 1)}
                    The 1 corresponds to int d = 0,...,3 (domain)
            y: Tensor of shape (batch_size, 1)
                The 1 corresponds to int d = 0,...,6 (class)
        unlabeled: This is not supported!
        return_eval_loss: If True, returns loss as 2nd value (to display in progress bar)
        """
        images = torch.cat([x["image"] for x, y in minibatches]) # (batch_size, channels, height, width)
        conditions = torch.nn.functional.one_hot(torch.cat([x["domain"] for x, y in minibatches]), num_classes=self.num_domains).flatten(start_dim=1) # (batch_size, num_classes)

        enc_mu, enc_logvar = self.encoder(images, conditions)
        z = self.sample(enc_mu, enc_logvar, num=self.K) 

        dec_mu, dec_logvar = self.decoder.forward_K(z, conditions)

        loss = self.loss(images, enc_mu, enc_logvar, dec_mu, dec_logvar)

        if return_eval_loss:
            return {'loss': loss.item()}, loss.item()
        else:
            return {'loss': loss.item()}

    def run(self, images, enc_conditions, dec_conditions, raw=False):
        """
        Generates reconstructions of the given images and conditions

        images: Tensor of shape (batch_size, channels, height, width)
        enc_conditions: Tensor of shape (batch_size)
            corresponds to int d = 0,...,3 (domain)
        dec_conditions: Tensor of shape (batch_size)
            corresponds to int d = 0,...,3 (domain)
        raw: Bool, if True no noise / variation is added
        """
        self.eval()
        with torch.no_grad():
            enc_conditions = torch.nn.functional.one_hot(enc_conditions, num_classes=self.num_domains).flatten(start_dim=1) # (batch_size, num_classes)
            dec_conditions = torch.nn.functional.one_hot(dec_conditions, num_classes=self.num_domains).flatten(start_dim=1) # (batch_size, num_classes)
            if raw:
                enc_mu, enc_logvar = self.encoder(images, enc_conditions)
                
                reconstructions, dec_logvar = self.decoder(enc_mu, dec_conditions)
            else:
                enc_mu, enc_logvar = self.encoder(images, enc_conditions)
                z = self.sample(enc_mu, enc_logvar)

                dec_mu, dec_logvar = self.decoder(z, dec_conditions)
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


class Encoder(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, num_domains):
        """
        Initializes the encoder.

        input_size: int M = channels * height * width
        hidden_sizes: List of ints with the sizes of the different hidden layers [l1, l2, ...]
        num_domains: Number of domains, usually 4
        """
        super().__init__()
        self.conv_sequential = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),  # (N, 32, 112, 112)
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),  # (N, 64, 56, 56)
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),  # (N, 128, 28, 28)
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),  # (N, 256, 14, 14)
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),  # (N, 512, 7, 7)
        )
        self.flatten = torch.nn.Flatten()
        modules = []
        for i, out_size in enumerate(hidden_sizes[:-1]):
            if i == 0:
                modules.append(torch.nn.Linear(25088 + num_domains, out_size))
                modules.append(torch.nn.ReLU())
            else:
                modules.append(torch.nn.Linear(hidden_sizes[i-1], out_size))
                modules.append(torch.nn.ReLU())
        self.linear_sequential = torch.nn.Sequential(*modules)
        self.get_mu = torch.nn.Linear(hidden_sizes[-2], hidden_sizes[-1])
        self.get_logvar = torch.nn.Linear(hidden_sizes[-2], hidden_sizes[-1])

    def forward(self, images, conditions):
        """
        Calculates mean and diagonal log-variance of p(z | x).

        images: Tensor of shape (batch_size, channels, height, width)
        conditions: Tensor of shape (batch_size, num_domains)
        """
        x = self.conv_sequential(images)
        x = self.flatten(x)
        x = torch.cat((x, conditions), dim=1)
        x = self.linear_sequential(x)
        enc_mu = self.get_mu(x)
        enc_logvar = self.get_logvar(x)

        return enc_mu, enc_logvar


class Decoder(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, num_domains):
        """
        Initializes the decoder.

        input_size: int M = channels * height * width
        hidden_sizes: List of ints with the sizes of the different hidden layers [l1, l2, ...]
        num_domains: number of domains, usually 4
        """
        super().__init__()
        modules = []
        for i, out_size in enumerate(hidden_sizes[::-1]):
            if i == 0:
                modules.append(torch.nn.Linear(out_size + num_domains, hidden_sizes[::-1][i+1]))
                modules.append(torch.nn.ReLU())
            elif i == len(hidden_sizes) - 1:
                modules.append(torch.nn.Linear(out_size, 25088))
                modules.append(torch.nn.ReLU())
            else:
                modules.append(torch.nn.Linear(out_size, hidden_sizes[::-1][i+1]))
                modules.append(torch.nn.ReLU())
        self.linear_sequential = torch.nn.Sequential(*modules)
        self.reshape = lambda x : x.view(-1, 32, 28, 28)
        self.conv_sequential = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode="nearest"),
            torch.nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
            torch.nn.ReLU(),  # (N, 256, 14, 14)
            torch.nn.Upsample(scale_factor=2, mode="nearest"),
            torch.nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            torch.nn.ReLU(),  # (N, 128, 28, 28)
            torch.nn.Upsample(scale_factor=2, mode="nearest"),
            torch.nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            torch.nn.ReLU(),  # (N, 64, 56, 56)
            torch.nn.Upsample(scale_factor=2, mode="nearest"),
            torch.nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            torch.nn.ReLU(),  # (N, 32, 112, 112)
            torch.nn.Upsample(scale_factor=2, mode="nearest"),
            torch.nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=3, padding=1),
            torch.nn.ReLU(),  # (N, 3, 224, 224)
        )
        self.get_mu = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode="nearest"),
            torch.nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=3, padding=1),  # (N, 3, 224, 224)
        )
        self.get_logvar = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode="nearest"),
            torch.nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=3, padding=1),  # (N, 3, 224, 224)
        )

    def forward(self, codes, conditions):
        """
        Calculates mean and diagonal log-variance of p(x | z).

        codes: Tensor of shape (batch_size, D) -> D = dimension of z
        conditions: Tensor of shape (batch_size, num_domains)
        """
        x = torch.cat((codes, conditions), dim=1)
        x = self.linear_sequential(x)
        x = self.reshape(x)
        x = self.conv_sequential(x)
        dec_mu = self.get_mu(x)
        dec_logvar = self.get_logvar(x)

        return dec_mu, dec_logvar
    
    def forward_K(self, codes, conditions):
        """
        Calculates mean and diagonal log-variance of p(x | z).

        codes: Tensor of shape (batch_size, K, D) -> K = number of samples generated, D = dimension of z
        conditions: Tensor of shape (batch_size, num_domains)
        """
        batch_size = codes.shape[0]
        K = codes.shape[1]
        conditions = torch.stack([conditions for i in range(K)], dim=1) # (batch_size, K, num_domains)
        x = torch.cat((codes, conditions), dim=2).view(batch_size * K, -1) # (batch_size * K, D + num_domains)
        x = self.linear_sequential(x)
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

    def forward(self, x, enc_mu, enc_logvar, dec_mu, dec_logvar):
        """
        Calculates the ELBO Loss (negative ELBO).

        x: Tensor of shape (batch_size, channels, height, width)
        enc_mu: Tensor of shape (batch_size, D) -> D = dimension of z
        enc_logvar: Tensor of shape (batch_size, D)
        dec_mu: Tensor of shape (batch_size, K, channels, height, width) -> K = number of samples for z
        dec_logvar: Tensor of shape (batch_size, K, channels, height, width)
        """
        N = dec_mu.shape[0]
        K = dec_mu.shape[1]

        x = torch.stack([x for i in range(K)], dim=1)
        x = self.flat_K(x, N, K)
        dec_mu = self.flat_K(dec_mu, N, K)
        dec_logvar = self.flat_K(dec_logvar, N, K)

        return torch.mean(
            # KL divergence -> regularization
            (torch.sum(
                enc_mu ** 2 + enc_logvar.exp() - enc_logvar - torch.ones(enc_mu.shape).to(x.device),
                dim=1
            ) * 0.5
            
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
    num_domains = 3
    hparams = {"hidden_sizes": "[1024,512,128]",
               "K": 25,
               "ckpt_path": "C:/users/gooog/desktop/bachelor/code/bachelor/logs/DB_test_2/model.pkl",
               "lr": 5e-05,
               "weight_decay": 0.0,
               "batch_size": 8}
    batch_size = hparams["batch_size"]
    minibatches = []
    for i in range(4):  # iterating over all datasets / environments
        x = {"image": torch.rand(size=(batch_size, 3, 224, 224)) * (2.6400 - (-2.1179)) + (-2.1179),
             "domain": torch.randint(low=0, high=4, size=(batch_size,))}
        y = torch.randint(low=0, high=7, size=(batch_size,))
        minibatches.append((x, y))

    cvae = BIG_LM_CCVAE(input_shape=input_shape, num_classes=num_classes,
                   num_domains=num_domains, hparams=hparams)
    print(cvae)
    """
    step_vals = cvae.update(minibatches=minibatches)
    step_vals, train_loss = cvae.update(
        minibatches=minibatches, return_train_loss=True)
    print(step_vals, train_loss)
    """
