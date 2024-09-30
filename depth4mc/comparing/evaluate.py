from tqdm import tqdm
from datetime import datetime
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from depth4mc.dataset.D4MCDataset import D4MCDataset, CAMERA_SIZE

def main(args):

    DEVICE = torch.device('cuda') if torch.cuda.is_available() and not args.cpu else torch.device('cpu')
    criterion = nn.MSELoss()
    
    if args.model == 'Depth4MC':
        from depth4mc.model.D4MCModel import D4MCModel
        from depth4mc.dataset.D4MCDataset import DEFAULT_TRANSFORM as transforms

        model = D4MCModel()
        model.load_state_dict(torch.load(args.checkpoint_path, weights_only=True))
        model = model.to(DEVICE).eval()
    else:
        from depth4mc.comparing.DepthAnythingWrapper import DEPTH_ANYTHING_TRANSFORM as transforms, DepthAnythingWrapper

        model = DepthAnythingWrapper(DEVICE, CAMERA_SIZE)
    
    dataset = D4MCDataset(dataset_path='../dataset/data/', transform=transforms, augment=False)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    start_time = datetime.now()
    test_loss = 0.0
    with torch.no_grad():
        for screenshots, labels in tqdm(data_loader):
            screenshots, labels = screenshots.to(DEVICE), labels.to(DEVICE)

            outputs = model(screenshots)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * screenshots.size(0)
            screenshots, labels = screenshots.to('cpu'), labels.to('cpu')

        test_loss /= len(data_loader.dataset)

    print('Test loss:', test_loss)
    print('Finished evaluation after:', datetime.now() - start_time)

if __name__ == '__main__':
    parser = ArgumentParser(description='Evaluation Code for the Minecraft Depth Estimation Model or DepthAnything Model')
    parser.add_argument('--model', help='Depth4MC or DepthAnything model', default='Depth4MC', choices=['Depth4MC', 'DepthAnything'])
    parser.add_argument('--checkpoint-path', help='Depth4MC checkpoint', default='../training/results/checkpoints_12k_final/model_final.pth', type=str)
    parser.add_argument('--cpu', help='train on CPU only', action='store_true')
    args = parser.parse_args()
    main(args)