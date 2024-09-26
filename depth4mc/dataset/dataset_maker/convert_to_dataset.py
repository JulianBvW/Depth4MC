# Converts the multiple output folders of the data creation runs to one dataset folder

import os
import math
import shutil
import numpy as np
from tqdm import tqdm
from PIL import Image

SCREENSHOT_DIMS = (480, 854)

output_dir = 'depth4mc/dataset/dataset_maker/output/'

dataset_dir = 'depth4mc/dataset/data/'
dataset_dir_screenshots = dataset_dir + 'screenshots/'
dataset_dir_labels = dataset_dir + 'depth_labels/'

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

def main():

    errors = []

    # Create and delete folders

    shutil.rmtree(dataset_dir, ignore_errors=True)
    os.makedirs(dataset_dir)
    os.makedirs(dataset_dir_screenshots)
    os.makedirs(dataset_dir_labels)

    # Get all runs

    run_folders = sorted(os.listdir(output_dir))

    ### Screenshots

    print('### Converting Screenshots')

    i = 0
    for dir in tqdm(run_folders):
        screenshots_dir = output_dir + dir + '/screenshots/'
        screenshots = sorted(os.listdir(screenshots_dir))
        for screenshot in screenshots:
            try:
                Image.open(screenshots_dir + screenshot)
                shutil.copy(screenshots_dir + screenshot, dataset_dir_screenshots + f'{i:08}' + '.png')
            except:
                errors.append(f'[Screenshots] Error at {screenshots_dir + screenshot}, tried saving as {i:08}.png')
            i += 1

    ### Depth Labels

    print('### Converting Depth Labels')

    i = 0
    for dir in run_folders:
        screenshots_dir_far  = output_dir + dir + '/depth_labels_far/'
        screenshots_dir_near = output_dir + dir + '/depth_labels_near/'
        screenshots_far  = sorted(os.listdir(screenshots_dir_far))
        screenshots_near = sorted(os.listdir(screenshots_dir_near))
        assert len(screenshots_far) == len(screenshots_near)

        depth_vales = np.zeros(SCREENSHOT_DIMS, dtype=np.float16)

        for img_file_far, img_file_near in tqdm(list(zip(screenshots_far, screenshots_near))):
            try:
                img_far, img_near = Image.open(screenshots_dir_far+img_file_far), Image.open(screenshots_dir_near+img_file_near)
                vals_far, vals_near = np.array(img_far)[:, :, 0], np.array(img_near)[:, :, 0]
                
                for row in range(depth_vales.shape[0]):
                    for col in range(depth_vales.shape[1]):
                        depth_vales[row, col] = to_depth(vals_far[row, col], vals_near[row, col])
                
                with open(f'{dataset_dir_labels}{i:08}.npy', 'wb') as f:
                    np.save(f, depth_vales)
            except:
                errors.append(f'[Depth] Error at {screenshots_dir_far+img_file_far} or {screenshots_dir_near+img_file_near}, tried saving as {i:08}.npy')
            i += 1

if __name__ == '__main__':
    main()