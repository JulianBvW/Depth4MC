# Converts the multiple output folders of the data creation runs to one dataset folder

import os
import sys
import shutil
import pandas as pd

output_dir = 'depth4mc/dataset/dataset_maker/output/'

dataset_dir = 'depth4mc/dataset/data/'
dataset_dir_screenshots = dataset_dir + 'screenshots/'
dataset_dir_labels = dataset_dir + 'depth_labels/'

shutil.rmtree(dataset_dir, ignore_errors=True)
os.makedirs(dataset_dir)
os.makedirs(dataset_dir_screenshots)
os.makedirs(dataset_dir_labels)

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
    screenshots_dir = output_dir + dir + '/depth_labels/'
    screenshots = sorted(os.listdir(screenshots_dir))
    for screenshot in screenshots:
        shutil.copy(screenshots_dir + screenshot, dataset_dir_labels + str(i) + '.png')
        i += 1