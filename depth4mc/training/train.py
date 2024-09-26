import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import os
import pandas as pd
from datetime import datetime
from argparse import ArgumentParser

from depth4mc.training.utils import train_step, test_step
from depth4mc.dataset.D4MCDataset import D4MCDataset
from depth4mc.model.D4MCModel import D4MCModel

parser = ArgumentParser(description='Training Code for the Minecraft Depth Estimation Model')
parser.add_argument('--epochs', help='number of epochs', default=50, type=int)
parser.add_argument('--batch-size', help='batch size', default=10, type=int)
parser.add_argument('--lr', help='learning rate', default=0.001, type=float)
parser.add_argument('--cpu', help='train on CPU only', action='store_true')
args = parser.parse_args()

### Dirs

RESULTS_DIR = 'depth4mc/training/results/'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR + 'checkpoints/', exist_ok=True)

### Device

DEVICE = torch.device('cuda') if torch.cuda.is_available() and not args.cpu else torch.device('cpu')
print('Selected Device:', DEVICE)

### Load Dataset

full_dataset = D4MCDataset()
print('Loaded', len(full_dataset), 'Datapoints')
train_dataset, test_dataset = random_split(full_dataset, [0.8, 0.2])

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

### Load Model

model = D4MCModel().to(DEVICE)

### Load Loss and Optimizer

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

### Train loop

train_losses = []
test_losses = []

start_time = datetime.now()
for epoch in range(args.epochs):

    train_loss = train_step(model, train_loader, criterion, optimizer, DEVICE)
    test_loss  = test_step(model, test_loader, criterion, DEVICE)

    train_losses.append(train_loss)
    test_losses.append(test_loss)

    print(f'Epoch {epoch+1}/{args.epochs} finished | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}')
    if epoch % 10 == 0:
        torch.save(model.state_dict(), RESULTS_DIR + f'checkpoints/model_e{epoch:04}.pth')

print('Finished training after:', datetime.now() - start_time)

# Save results
df = pd.DataFrame({
    'Train Loss': train_losses,
    'Test Loss': test_losses
})
df.to_csv(RESULTS_DIR + 'losses.csv', index=False)
torch.save(model.state_dict(), RESULTS_DIR + 'checkpoints/model_final.pth')

