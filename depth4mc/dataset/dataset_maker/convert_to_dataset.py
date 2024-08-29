# Converts the multiple output folders of the data creation runs to one dataset folder

import os
import math
import shutil
import numpy as np

output_dir = 'depth4mc/dataset/dataset_maker/output/'

dataset_dir = 'depth4mc/dataset/data/'
dataset_dir_screenshots = dataset_dir + 'screenshots/'
dataset_dir_labels = dataset_dir + 'depth_labels/'

shutil.rmtree(dataset_dir, ignore_errors=True)
os.makedirs(dataset_dir)
os.makedirs(dataset_dir_screenshots)
os.makedirs(dataset_dir_labels)

# Depth conversion functions

def abc_formula(a, b, c):
    return (-b+math.sqrt(b**2 - 4*a*c))/(2*a)

def to_depth_far(color):
    a = -22/1600
    b = 81/20
    c = 1 - color
    return abc_formula(a, b, c)

def to_depth_near(color):
    a = -1.6/6
    b = 42.3
    c = 13 - color
    return abc_formula(a, b, c)

def to_depth(color_in_far_img, color_in_near_img):
    if color_in_far_img < 24:
        return to_depth_near(color_in_near_img)
    return to_depth_far(color_in_far_img)

# Get all runs

run_folders = sorted(os.listdir(output_dir))

### Screenshots

i = 0
for dir in run_folders:
    screenshots_dir = output_dir + dir + '/screenshots/'
    screenshots = sorted(os.listdir(screenshots_dir))
    for screenshot in screenshots:
        shutil.copy(screenshots_dir + screenshot, dataset_dir_screenshots + str(i) + '.png')
        i += 1

### Depth Labels

i = 0
for dir in run_folders:
    screenshots_dir_far  = output_dir + dir + '/depth_labels_far/'
    screenshots_dir_near = output_dir + dir + '/depth_labels_near/'
    screenshots_far  = sorted(os.listdir(screenshots_dir_far))
    screenshots_near = sorted(os.listdir(screenshots_dir_near))
    assert len(screenshots_far) == len(screenshots_near)

    for img_far, img_near in zip(screenshots_far, screenshots_near):
        # TODO
        shutil.copy(screenshots_dir + screenshot, dataset_dir_labels + str(i) + '.png')
        i += 1