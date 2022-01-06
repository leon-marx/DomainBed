import torch

def accuracy(network, loader, weights, device):

    network.eval()
    with torch.no_grad():
        total = 0
        for x, y in loader:
            x["image"] = x["image"].to(device)
            x["domain"] = x["domain"].to(device)
            y = y.to(device)
            _, loss = network.evaluate(minibatches=[(x, y)], return_eval_loss=True)
            total += loss
    network.train()
    return 1 / (1 + total)
    