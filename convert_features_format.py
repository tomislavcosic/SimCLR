import torch
import numpy as np

# Load the .pt files
train_x = torch.load('/home/tcosic/SimCLRCustom/save/train_X.pt')
train_y = torch.load('/home/tcosic/SimCLRCustom/save/train_y.pt')

# Convert the tensors to NumPy arrays
train_x_np = train_x.numpy()
train_y_np = train_y.numpy()

# Save the arrays to a .npz file
npz_file_path = '/home/tcosic/SimCLRCustom/save/features_train_cifar10.npz'
np.savez(npz_file_path, feat_list=train_x_np, label_list=train_y_np)

print(f"Saved {npz_file_path} successfully!")
