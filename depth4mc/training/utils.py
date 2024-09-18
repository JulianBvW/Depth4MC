import torch
from tqdm import tqdm


def capped_mse_loss(pred, target, cap=100):
    mask = target <= cap  # Only compute loss for depths smaller than {cap} meters
    return torch.mean((pred[mask] - target[mask]) ** 2)

def train_step(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0

    for screenshots, labels in tqdm(train_loader):
        screenshots, labels = screenshots.to(device), labels.to(device).float()

        optimizer.zero_grad()
        outputs = model(screenshots)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * screenshots.size(0)
        screenshots, labels = screenshots.to('cpu'), labels.to('cpu')
    
    train_loss /= len(train_loader.dataset)
    return train_loss


def test_step(model, test_loader, criterion, device, verbose=False):
    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for screenshots, labels in tqdm(test_loader):
            screenshots, labels = screenshots.to(device), labels.to(device)

            outputs = model(screenshots)
            loss = criterion(outputs, labels)
            if verbose:
                for i in range(len(labels)):
                    print(outputs[i], labels[i])

            test_loss += loss.item() * screenshots.size(0)
            screenshots, labels = screenshots.to('cpu'), labels.to('cpu')

        test_loss /= len(test_loader.dataset)
    return test_loss