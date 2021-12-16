import torch

def accuracy(network, loader, weights, device):
    loss = 0

    network.eval()
    with torch.no_grad():
        batch = []
        for x, y in loader:
            print("loop")
            x["image"] = x["image"].to(device)
            x["domain"] = x["domain"].to(device)
            y = y.to(device)
            batch.append((x, y))
        loss += network.update(minibatches=batch)
            
    network.train()

    return 1 / (1 + loss.item())