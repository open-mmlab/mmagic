# 只能在mycode目录下面粘贴到python里面运行
import pickle
import torch
import torch_utils

# Specify the paths for the pickle file and the output pth file
pickle_file_path = './ckpts/stylegan2_lions_512_pytorch.pkl'
pth_file_path = './path_files/stylegan2_lions_512_pytorch.pth'

# Load the pickle file
with open(pickle_file_path, 'rb') as file:
    data = pickle.load(file)

# Save the data in a pth file
torch.save(data, pth_file_path)
