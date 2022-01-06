import torch

def old_accuracy(network, loader, weights, device):

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

def accuracy(network, loader, weights, device):
    total = 0

    network.eval()
    with torch.no_grad():
        for x, y in loader:
            x["image"] = x["image"].to(device)
            x["domain"] = x["domain"].to(device)
            y = y.to(device)
            
            images = x["image"]
            classes = torch.nn.functional.one_hot(y, num_classes=network.num_classes).flatten(start_dim=1) # (batch_size, num_classes)
            domains = torch.nn.functional.one_hot(x["domain"], num_classes=network.num_domains).flatten(start_dim=1) # (batch_size, num_domains)
            
            enc_mu, enc_logvar = network.encoder(images, classes, domains)
            z = network.sample(enc_mu, enc_logvar)
            dec_mu, dec_logvar = network.decoder(z, classes, domains)

            loss = 1# network.loss(images, enc_mu, enc_logvar, dec_mu, dec_logvar, lamb=network.lamb).item()
            total += loss
    network.train()

    return 1 / (1 + total)