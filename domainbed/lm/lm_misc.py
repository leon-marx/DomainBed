import torch

def accuracy(network, loader, weights, device):
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x["image"] = x["image"].to(device)
            x["domain"] = x["domain"].to(device)
            y = y.to(device)
            _, loss = network.evaluate(minibatches=[(x, y)], return_eval_loss=True)
            total += loss

    return 1 / (1 + total)