import os
import glob

train_path = '/home/xtwang/datasets/REDS/train_sharp_bicubic/X4'
val_path = '/home/xtwang/datasets/REDS/val_sharp_bicubic/X4'

# mv the val set
val_folders = glob.glob(os.path.join(val_path, '*'))
for folder in val_folders:
    new_folder_idx = '{:03d}'.format(int(folder.split('/')[-1]) + 240)
    os.system('cp -r {} {}'.format(folder, os.path.join(train_path, new_folder_idx)))
