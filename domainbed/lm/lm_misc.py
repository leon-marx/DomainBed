import torch

def accuracy(network, loader, weights, device):
    loss = 0

    network.eval()
    with torch.no_grad():
        batch = []
        for x, y in loader:
            x["image"] = x["image"].to(device)
            x["domain"] = x["domain"].to(device)
            y = y.to(device)
            batch.append((x, y))
        print("loop")
        loss += network.update(minibatches=batch).detach().item()
            
    network.train()

    return 1 / (1 + loss)