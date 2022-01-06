import torch
from tqdm import tqdm

def accuracy(network, loader, weights, device):

    network.eval()
    total = 0
    with torch.no_grad():
        progress_bar = tqdm(loader, desc="Evaluation")
        for x, y in progress_bar:
            x["image"] = x["image"].to(device)
            x["domain"] = x["domain"].to(device)
            y = y.to(device)
            _, loss = network.evaluate(minibatches=[(x, y)], return_eval_loss=True)
            total += loss
    network.train()
    return 1 / (1 + total)
    