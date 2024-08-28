from depth4mc.dataset.dataset_maker.mcpi_utils import get_pose

import os
import sys
import shutil
import pandas as pd
from tqdm import tqdm
from time import sleep
from datetime import datetime
import mcpi.minecraft as minecraft

mc = minecraft.Minecraft.create()
imgs_to_take = int(sys.argv[1]) if len(sys.argv) > 1 else 5*60*10

cur_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
dataset_maker_dir = 'depth4mc/dataset/dataset_maker/'
datapack_template_dir = dataset_maker_dir + 'datapack_template/depth4mc_dataset_datapack/'
datapack_server_dir = dataset_maker_dir + 'minecraft_server/world/datapacks/depth4mc_dataset_datapack/'
out_dir = dataset_maker_dir + 'output/run_' + cur_time + '/'

# Preparing Folders
os.makedirs(out_dir)
os.makedirs(out_dir + 'screenshots/')
os.makedirs(out_dir + 'depth_labels_near/')
os.makedirs(out_dir + 'depth_labels_far/')
shutil.rmtree(datapack_server_dir, ignore_errors=True)

# Warm-Up Phase
for i in tqdm(range(50)):
    sleep(.1)
mc.postToChat('start')

# Get Poses
poses = []
for i in range(imgs_to_take):
    sleep(0.1)
    mc.postToChat(f'{i}/{imgs_to_take}')
    poses.append(get_pose(mc)) # Costs 0.15 Seconds -> FPS 6.6666
mc.postToChat('stop')

# Save Poses
poses = pd.DataFrame(poses)
poses.to_csv(out_dir + 'poses.csv', index=False)
print(poses)

# Make Datapack
shutil.copytree(datapack_template_dir, datapack_server_dir)
with open(datapack_server_dir + 'data/depth4mc/functions/tick.mcfunction', 'w') as f:
    for i in range(len(poses)):
        p = poses.iloc[i]
        tp = f'tp @a[scores={{depth4mc_rclick={i+1}}}] {p.x} {p.y} {p.z} {p.rot} {p.pit}\n'
        f.write(tp)

with open(datapack_server_dir + 'data/depth4mc/functions/load.mcfunction', 'a') as f:
    f.write(f'say Loaded run {cur_time}')

print(f'Poses saved to `{out_dir}`')
print(f'Datapack saved to `{datapack_server_dir}`')