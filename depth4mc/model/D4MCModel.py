import torch.nn.functional as F
import torch.nn as nn
import torch

import numpy as np

from depth4mc.dataset.D4MCDataset import CAMERA_SIZE
from depth4mc.model.utils import double_conv_block

class D4MCModel(nn.Module):
    def __init__(self):
        super(D4MCModel, self).__init__()
        self.camera_size = CAMERA_SIZE

        # Encoder
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc_conv_1 = double_conv_block(3, 64)
        self.enc_conv_2 = double_conv_block(64, 128)
        self.enc_conv_3 = double_conv_block(128, 256)
        self.enc_conv_4 = double_conv_block(256, 512)
        self.enc_conv_5 = double_conv_block(512, 1024)

        # Decoder
        self.up_transpose_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.up_transpose_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2, output_padding=(0, 1))
        self.up_transpose_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2, output_padding=(0, 1))
        self.up_transpose_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.dec_conv_1 = double_conv_block(1024, 512)
        self.dec_conv_2 = double_conv_block(512, 256)
        self.dec_conv_3 = double_conv_block(256, 128)
        self.dec_conv_4 = double_conv_block(128, 64)

        # Output
        self.out = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1) 

    def forward(self, x):

        # Encoding
        enc_1 = self.enc_conv_1(x)                     # [bs, 64, 480, 854]
        enc_2 = self.enc_conv_2(self.max_pool(enc_1))  # [bs, 128, 240, 427]
        enc_3 = self.enc_conv_3(self.max_pool(enc_2))  # [bs, 256, 120, 213]
        enc_4 = self.enc_conv_4(self.max_pool(enc_3))  # [bs, 512, 60, 106]
        x     = self.enc_conv_5(self.max_pool(enc_4))  # [bs, 1024, 30, 53]

        # Decoding
        x = self.up_transpose_1(x)    # [bs, 512, 60, 106]
        x = torch.cat([enc_4, x], 1)  # [bs, 1024, 60, 106]
        x = self.dec_conv_1(x)        # [bs, 512, 60, 106]

        x = self.up_transpose_2(x)    # [bs, 256, 120, 213]
        x = torch.cat([enc_3, x], 1)  # [bs, 512, 120, 213]
        x = self.dec_conv_2(x)        # [bs, 256, 120, 213]

        x = self.up_transpose_3(x)    # [bs, 128, 240, 427]
        x = torch.cat([enc_2, x], 1)  # [bs, 256, 240, 427]
        x = self.dec_conv_3(x)        # [bs, 128, 240, 427]

        x = self.up_transpose_4(x)    # [bs, 64, 480, 854]
        x = torch.cat([enc_1, x], 1)  # [bs, 128, 480, 854]
        x = self.dec_conv_4(x)        # [bs, 64, 480, 854]

        # Output
        out = self.out(x).squeeze(1)  # [bs, 480, 854]
        return out
