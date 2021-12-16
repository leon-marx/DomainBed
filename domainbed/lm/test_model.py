import matplotlib.pyplot as plt
import torch

# Own imports:
from domainbed.lm.cvae import LM_CVAE
from domainbed.lm.dataset import LM_PACS
from domainbed.lib.fast_data_loader import FastDataLoader


ckpt_path = "C:/users/gooog/desktop/bachelor/code/bachelor/logs/DB_test_2/model.pkl"
checkpoint = torch.load(ckpt_path, map_location="cpu")

print(checkpoint.keys())
input_shape = checkpoint["model_input_shape"]
num_classes = checkpoint["model_num_classes"]
num_domains = checkpoint["model_num_domains"]
hparams = checkpoint["model_hparams"]
hparams["ckpt_path"] = ckpt_path
args = checkpoint["args"]

print("")
print("")
print("")
print(input_shape, type(input_shape))
print(num_classes, type(num_classes))
print(num_domains, type(num_domains))
print(hparams, type(hparams))


dataset = LM_PACS(args["data_dir"], args["test_envs"], hparams)
model = LM_CVAE(input_shape=input_shape, num_classes=num_classes,
                num_domains=num_domains, hparams=hparams)


eval_minibatches_iterator = zip(*[FastDataLoader(dataset=env, batch_size=8,
                   num_workers=0) for env in dataset])

for batch in next(eval_minibatches_iterator):
    print(batch[0]["domain"])

for batch in next(eval_minibatches_iterator):
    print(batch, type(batch))
    print(batch[0], type(batch[0]))
    print(batch[0]["image"], type(batch[0]["image"]), batch[0]["image"].shape)
    print(batch[0]["domain"], type(batch[0]["domain"]))
    print(batch[1], type(batch[1]))
    print(batch[0]["image"].min(), batch[0]["image"].max())
    print("")
    print("")
    print("")
    print(batch[0]["domain"].shape)
    x = batch[0]["image"][:-3]
    conds = torch.Tensor([[[1, 1], [0, 0], [0, 0], [0, 0]]])
    pred = model.cvae(x, conds).detach()

    def make_plottable(image):
        image_plt = image.view(3, 224, 224).permute(1, 2, 0)
        if image_plt.min().item() < 0:
            image_plt += image_plt.min().abs().item()
        image_plt /= image_plt.max().item()
        print(image_plt.max(), image_plt.min())
        return image_plt

    x_plt = make_plottable(x)
    pred_plt = make_plottable(pred)
    
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(x_plt)
    plt.subplot(1, 2, 2)
    plt.imshow(pred_plt)
    plt.show()
