import torch

def accuracy(network, loader, weights, device):
    loss = 0

    network.eval()
    with torch.no_grad():
        loss += network.cvae.validation_step(batch=[torch.cat([x["image"].to(device) for x, y in loader]), 
                                                    torch.cat([x["domain"].to(device) for x, y in loader])])
            
    network.train()

    return 1 / loss.item()