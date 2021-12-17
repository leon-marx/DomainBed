import argparse
import matplotlib.pyplot as plt
import os
import torch

# Own imports:
from domainbed.lm.cvae import LM_CVAE
from domainbed.lm.dataset import LM_PACS
from domainbed.lib.fast_data_loader import FastDataLoader


parser = argparse.ArgumentParser(description='Testing Model')
parser.add_argument('--ckpt_path', type=str, default=None)
parser.add_argument('--img_path', type=str, default=None)
parser.add_argument('--num_samples', type=int, default=1)
args = parser.parse_args()


if os.name == "nt":
    repo_path = "C:/users/gooog/desktop/bachelor/code/bachelor/"
else:
    repo_path = "/home/tarkus/leon/bachelor/"

ckpt_path = repo_path + args.ckpt_path
checkpoint = torch.load(ckpt_path, map_location="cpu")

input_shape = checkpoint["model_input_shape"]
num_classes = checkpoint["model_num_classes"]
num_domains = checkpoint["model_num_domains"]
hparams = checkpoint["model_hparams"]
hparams["ckpt_path"] = ckpt_path
args = checkpoint["args"]

dataset = LM_PACS(args["data_dir"], args["test_envs"], hparams)
model = LM_CVAE(input_shape=input_shape, num_classes=num_classes,
                num_domains=num_domains, hparams=hparams)

eval_minibatches_iterator = zip(*[FastDataLoader(dataset=env, batch_size=4,
                   num_workers=0) for env in dataset])

def make_plottable(image):
    image_plt = image.permute(1, 2, 0)
    if image_plt.min().item() < 0:
        image_plt += image_plt.min().abs().item()
    image_plt /= image_plt.max().item()
    return image_plt

for i, batch in enumerate(next(eval_minibatches_iterator)):
    images = batch[0]["image"]
    enc_conditions = batch[0]["domain"]
    dec_conditions = batch[0]["domain"]
    reconstructions = model.run(images, enc_conditions, dec_conditions, raw=True)

    for j in range(reconstructions.shape[0]):
        image_plt = make_plottable(images[j])
        reconstruction_plt = make_plottable(reconstructions[j])

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(image_plt)
        plt.subplot(1, 2, 2)
        plt.imshow(reconstruction_plt)
        plt.show()
