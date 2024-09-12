import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import numpy as np
import pandas as pd
from datetime import datetime

from depth4mc.dataset.D4MCDataset import D4MCDataset
from depth4mc.training.utils import train_step, test_step
from depth4mc.model.D4MCModel import D4MCModel


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') # TODO argparse

### Load Dataset

full_dataset = D4MCDataset()
train_dataset, test_dataset = random_split(full_dataset, [0.8, 0.2])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

### Load Model

model = D4MCModel().to(DEVICE)

### Load Loss and Optimizer

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

### Train loop

NUM_EPOCHS = 50 # TODO ArgParse

train_losses = []
test_losses = []

start_time = datetime.now()
for epoch in range(NUM_EPOCHS):

    train_loss = train_step(model, train_loader, criterion, optimizer, DEVICE)
    test_loss  = test_step(model, test_loader, criterion, DEVICE)

    train_losses.append(train_loss)
    test_losses.append(test_loss)

    print(f'Epoch {epoch+1}/{NUM_EPOCHS} finished | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}')

print('Finished training after:', datetime.now() - start_time)

# Save results
df = pd.DataFrame({
    'Train Loss': train_losses,
    'Test Loss': test_losses
})
df.to_csv(f'depth4mc/training/losses.csv', index=False)

test_loss  = test_step(model, test_loader, criterion, DEVICE, verbose=True)