import torch.nn as nn

def double_conv_block(in_channels, out_channels):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )
    return block

"""
Utility function for computing output of convolutions
takes a tuple of (h,w) and returns a tuple of (h,w)
Credit: https://discuss.pytorch.org/t/utility-function-for-calculating-the-shape-of-a-conv-output/11173/3
"""
def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1, max_pool=None):
    from math import floor
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor( ((h_w[0] + (2 * pad) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)
    w = floor( ((h_w[1] + (2 * pad) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1)
    if max_pool:
        h, w = h // max_pool, w // max_pool
    return h, w