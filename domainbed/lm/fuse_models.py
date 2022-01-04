import argparse
import matplotlib.pyplot as plt
import os
import torch

# Own imports:
from domainbed.lm.cvae import LM_CVAE
from domainbed.lm.conv_cvae import LM_CCVAE
from domainbed.lm.dataset import LM_PACS
from domainbed.lib.fast_data_loader import FastDataLoader


repo_path = "C:/users/gooog/desktop/bachelor/code/bachelor/"

def get_config(ckpt_path):
    ckpt_path = repo_path + ckpt_path
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    print("Configuration:")
    print(f"    ckpt_path: {ckpt_path}")

    input_shape = checkpoint["model_input_shape"]
    num_classes = checkpoint["model_num_classes"]
    num_domains = checkpoint["model_num_domains"]
    hparams = checkpoint["model_hparams"]
    hparams["ckpt_path"] = ckpt_path
    args = checkpoint["args"]
    return input_shape, num_classes, num_domains, hparams, args

ccvae_config = get_config("logs/DB_CCVAE_2/best_model.pkl")
ccvae_model = LM_CCVAE(*ccvae_config[:-1])
cvae_config = get_config("logs/DB_CVAE_7/best_model.pkl")
cvae_model = LM_CVAE(*cvae_config[:-1])

dataset = LM_PACS(cvae_config[-1]["data_dir"], cvae_config[-1]["test_envs"], cvae_config[3])
eval_minibatches_iterator = zip(*[FastDataLoader(dataset=env, batch_size=4,
                   num_workers=0) for env in dataset])

cond_dict = {
        0: "art_painting",
        1: "cartoon",
        2: "photo",
        3: "sketch",
    }

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
    alphas = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    cvae_reconstructions = cvae_model.run(images, enc_conditions, dec_conditions, raw=True)
    ccvae_reconstructions = ccvae_model.run(images, enc_conditions, dec_conditions, raw=True)

    for j in range(cvae_reconstructions.shape[0]):
        image_plt = make_plottable(images[j])
        cvae_reconstruction_plt = make_plottable(cvae_reconstructions[j])
        ccvae_reconstruction_plt = make_plottable(ccvae_reconstructions[j])
        plt.figure(figsize=(21, 6))
        for k, alpha in enumerate(alphas):
            plt.subplot(2, 7, k+1)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(image_plt)
            plt.subplot(2, 7, k+8)
            plt.xticks([])
            plt.yticks([])
            if k == 0:
                plt.xlabel("CCVAE")
            if k == 6:
                plt.xlabel("CVAE")
            plt.title(alpha)
            plt.imshow(alpha * cvae_reconstruction_plt + (1 - alpha) *ccvae_reconstruction_plt)
        plt.show()