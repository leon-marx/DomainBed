import os
from torchvision import transforms
from torchvision.datasets import ImageFolder

class AddDomainInfo(object):
    """
    Turns the image into a dict, with:
        x["image"] = image tensor
        x["domain"] = number for domain
    """
    def __init__(self, idx):
        self.idx = idx

    def __call__(self, img):
        d = {}
        d["image"] = img
        d["domain"] = self.idx
        return d

    def __repr__(self):
        return self.__class__.__name__ + '()'

class MultipleDomainDataset:
    N_STEPS = 5001           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 8            # Default, subclasses may override
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)

class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root, test_envs, augment, hparams):
        super().__init__()
        self.test_envs = test_envs
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)

        self.datasets = []
        for i, environment in enumerate(environments):

            env_transform = self.get_transforms(i, augment)

            path = os.path.join(root, environment)
            env_dataset = ImageFolder(path,
                transform=env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = len(self.datasets[-1].classes)

    def get_transforms(self, idx, augment):
        if augment and (idx not in self.test_envs):
            env_transform = transforms.Compose([
            # transforms.Resize((224,224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            AddDomainInfo(idx),
            ])
        else:
            env_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            AddDomainInfo(idx),
            ])
        return env_transform

class LM_PACS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "PACS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class LM_PACS_Debug(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    N_WORKERS = 0
    ENVIRONMENTS = ["A", "C", "P"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "PACS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)