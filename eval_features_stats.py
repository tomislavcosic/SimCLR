import os

import torch

if __name__ == '__main__':

    feats = torch.load(os.path.join("save", "train_X_2.pt"))
    print(feats.min())
    print(feats[0])