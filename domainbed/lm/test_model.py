import argparse
import matplotlib.pyplot as plt
import os
import torch

# Own imports:
from domainbed.lm.cvae import LM_CVAE
from domainbed.lm.conv_cvae import LM_CCVAE
from domainbed.lm.dataset import LM_PACS
from domainbed.lib.fast_data_loader import FastDataLoader


parser = argparse.ArgumentParser(description='Testing Model')
parser.add_argument('--ckpt_path', type=str, default=None)  # logs/log_dir (no / at beginning)
parser.add_argument('--img_path', type=str, default=None)
parser.add_argument('--num_samples', type=int, default=1)
parser.add_argument('--raw', type=bool, default=False)
parser.add_argument('--mode', type=str, default="ae")
args = parser.parse_args()


if os.name == "nt":
    repo_path = "C:/users/gooog/desktop/bachelor/code/bachelor/"
else:
    repo_path = "/home/tarkus/leon/bachelor/"

ckpt_path = repo_path + args.ckpt_path
checkpoint = torch.load(ckpt_path, map_location="cpu")
raw = args.raw
mode = args.mode

print("Configuration:")
print(f"    ckpt_path: {ckpt_path}")
print(f"    raw: {raw}")
print(f"    mode: {mode}")

input_shape = checkpoint["model_input_shape"]
num_classes = checkpoint["model_num_classes"]
num_domains = checkpoint["model_num_domains"]
hparams = checkpoint["model_hparams"]
hparams["ckpt_path"] = ckpt_path
args = checkpoint["args"]

dataset = LM_PACS(args["data_dir"], args["test_envs"], hparams)
if "CCVAE" in ckpt_path:
    model = LM_CCVAE(input_shape=input_shape, num_classes=num_classes,
                    num_domains=num_domains, hparams=hparams)
elif "CVAE" in ckpt_path:
    model = LM_CVAE(input_shape=input_shape, num_classes=num_classes,
                    num_domains=num_domains, hparams=hparams)

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

if mode == "ae":
    for i, batch in enumerate(next(eval_minibatches_iterator)):
        images = batch[0]["image"]
        enc_conditions = batch[0]["domain"]
        dec_conditions = batch[0]["domain"]
        reconstructions = model.run(images, enc_conditions, dec_conditions, raw=raw)

        for j in range(reconstructions.shape[0]):
            image_plt = make_plottable(images[j])
            reconstruction_plt = make_plottable(reconstructions[j])

            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(image_plt)
            plt.subplot(1, 2, 2)
            plt.imshow(reconstruction_plt)
            plt.show()
elif mode == "ae_dec_switch":
    for batch in next(eval_minibatches_iterator):
        images = batch[0]["image"]
        enc_conditions = batch[0]["domain"]
        dec_cond_list = [
            torch.ones_like(enc_conditions) * 0, 
            torch.ones_like(enc_conditions) * 1, 
            torch.ones_like(enc_conditions) * 2, 
            torch.ones_like(enc_conditions) * 3
        ]
        plt.figure(figsize=(10, 8))
        for i, dec_conditions in enumerate(dec_cond_list):
            reconstructions = model.run(images, enc_conditions, dec_conditions, raw=raw)
            for j in range(reconstructions.shape[0]):
                image_plt = make_plottable(images[j])
                reconstruction_plt = make_plottable(reconstructions[j])

                plt.subplot(4, 5, j * 5 + 1)
                plt.title("original")
                plt.xticks([])
                plt.yticks([])
                plt.imshow(image_plt)
                plt.subplot(4, 5, j * 5 + 2 + i)
                plt.title(cond_dict[dec_conditions[j].item()])
                plt.xticks([])
                plt.yticks([])
                plt.imshow(reconstruction_plt)
        plt.show()
elif mode == "ae_enc_switch":
    for batch in next(eval_minibatches_iterator):
        images = batch[0]["image"]
        dec_conditions = batch[0]["domain"]
        enc_cond_list = [
            torch.ones_like(dec_conditions) * 0, 
            torch.ones_like(dec_conditions) * 1, 
            torch.ones_like(dec_conditions) * 2, 
            torch.ones_like(dec_conditions) * 3
        ]
        plt.figure(figsize=(10, 8))
        for i, enc_conditions in enumerate(enc_cond_list):
            reconstructions = model.run(images, enc_conditions, dec_conditions, raw=raw)
            for j in range(reconstructions.shape[0]):
                image_plt = make_plottable(images[j])
                reconstruction_plt = make_plottable(reconstructions[j])

                plt.subplot(4, 5, j * 5 + 1)
                plt.title("original")
                plt.xticks([])
                plt.yticks([])
                plt.imshow(image_plt)
                plt.subplot(4, 5, j * 5 + 2 + i)
                plt.title(cond_dict[enc_conditions[j].item()])
                plt.xticks([])
                plt.yticks([])
                plt.imshow(reconstruction_plt)
        plt.show()
elif mode == "gen":
    codes = torch.randn(size=(4, 512))
    conditions = torch.nn.functional.one_hot(torch.arange(4), 4)
    plt.figure(figsize=(8, 8))
    for i, code in enumerate(codes):
        reconstructions, _ = model.decoder(torch.stack((code, code, code, code), dim=0), conditions)
        reconstructions = reconstructions.detach().permute(0, 2, 3, 1)
        reconstructions += reconstructions.min().abs().item()
        reconstructions /= reconstructions.max().item()
        for j in range(4):
            plt.subplot(4, 4, 4*i+j+1)
            plt.title(cond_dict[j])
            plt.xticks([])
            plt.yticks([])
            plt.imshow(reconstructions[j])
    plt.show()

elif mode == "print":
    print(model)
