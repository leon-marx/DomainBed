import torch

def accuracy(network, loader, weights, device):
    loss = 0

    network.eval()
    with torch.no_grad():
        for x, y in loader:
            x["image"] = x["image"].to(device)
            x["domain"] = x["domain"].to(device)
            y = y.to(device)
            loss += network.update(minibatches=[(x, y)]).detach().item()
            
    network.train()

    return 1 / (1 + loss)