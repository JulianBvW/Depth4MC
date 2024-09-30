import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose
from depth4mc.comparing.depth_anything.dpt import DepthAnything
from depth4mc.comparing.depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

DEPTH_ANYTHING_TRANSFORM = Compose([
    Resize(
        width=518,
        height=518,
        resize_target=False,
        keep_aspect_ratio=True,
        ensure_multiple_of=14,
        resize_method='lower_bound',
        image_interpolation_method=cv2.INTER_CUBIC,
    ),
    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    PrepareForNet(),
])

class DepthAnythingWrapper(nn.Module):
    def __init__(self, device, img_dims):
        super(DepthAnythingWrapper, self).__init__()
        self.model = DepthAnything.from_pretrained('LiheYoung/depth_anything_vits14').to(device).eval()
        self.w, self.h = img_dims

    def forward(self, x):
        
        depth = self.model(x)
        depth = F.interpolate(depth[None], (self.h, self.w), mode='bilinear', align_corners=False)[0]
        # TODO Cap values

        return depth



# image = cv2.cvtColor(cv2.imread('input2.png'), cv2.COLOR_BGR2RGB) / 255.0

# h, w = image.shape[:2]

# image = torch.from_numpy(transform({'image': image})['image']).unsqueeze(0).to(DEVICE)




# depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
# depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0

# depth = depth.cpu().numpy().astype(np.uint8)

# depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)

# cv2.imwrite('input2_depth.png', depth)