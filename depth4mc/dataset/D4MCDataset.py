from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import torch
import cv2
import os

CAMERA_SIZE = (854, 480)
DEFAULT_TRANSFORM = transforms.Compose([
    #transforms.Resize(CAMERA_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.3955, 0.3832, 0.3661], std=[0.1625, 0.1902, 0.2550])  # TODO recalculate
])

class D4MCDataset(Dataset):
    def __init__(self, dataset_path='depth4mc/dataset/data/', transform=DEFAULT_TRANSFORM, augment=True):
        self.screenshots_path = dataset_path + '/screenshots/'
        self.labels_path = dataset_path + '/depth_labels/'

        self.screenshots = sorted(os.listdir(self.screenshots_path))
        self.labels = sorted(os.listdir(self.labels_path))
        
        self.num_data = len(self.screenshots)
        self.length = self.num_data * (2 if augment else 1)
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data_point = idx % self.num_data

        # Screenshot
        screenshot = Image.open(self.screenshots_path + self.screenshots[data_point]).convert('RGB')
        screenshot = self.transform(screenshot)

        # For DepthAnything
        # screenshot = cv2.cvtColor(cv2.imread(self.screenshots_path + self.screenshots[data_point]), cv2.COLOR_BGR2RGB) / 255.0
        # screenshot = torch.from_numpy(self.transform({'image': screenshot})['image'])

        # Depth Label
        label = np.load(self.labels_path + self.labels[data_point])

        # Flip for data augmentation
        if idx >= self.num_data:
            screenshot = transforms.functional.hflip(screenshot)
            label = np.fliplr(label).copy()

        return screenshot, label
