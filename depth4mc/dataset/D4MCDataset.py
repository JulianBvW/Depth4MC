from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import os

CAMERA_SIZE = (854, 480)
DEFAULT_TRANSFORM = transforms.Compose([
    #transforms.Resize(CAMERA_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.3955, 0.3832, 0.3661], std=[0.1625, 0.1902, 0.2550])  # TODO recalculate
])

class D4MCDataset(Dataset):
    def __init__(self, dataset_path='depth4mc/dataset/data/', transform=DEFAULT_TRANSFORM):
        self.screenshots_path = dataset_path + 'screenshots/'
        self.labels_path = dataset_path + 'depth_labels/'

        self.screenshots = sorted(os.listdir(self.screenshots_path))
        self.labels = sorted(os.listdir(self.labels_path))

        self.transform = transform
    
    def __len__(self):
        return len(self.screenshots)

    def __getitem__(self, idx):

        # Screenshot
        screenshot = Image.open(self.screenshots_path + self.screenshots[idx]).convert('RGB')
        screenshot = self.transform(screenshot)

        # Depth Label
        label = np.load(self.labels_path + self.labels[idx])

        return screenshot, label