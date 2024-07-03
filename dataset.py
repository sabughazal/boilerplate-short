import os
import json
import glob
import torch
import random
import numpy as np
from torch.utils.data import Dataset



class DatasetType(Dataset):
    def __init__(self, data_root, split):
        super().__init__()
        self.samples = []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return 0, 0



if __name__ == "__main__":
    import os
    import sys
    # from torch.utils.data import DataLoader

    if not os.path.exists(sys.argv[1]):
        raise ValueError("Invalid path!")

    data_root = sys.argv[1]
    ds = DatasetType(data_root, "train")
    # dl = DataLoader(dataset=ds, batch_size=4, shuffle=True)
    # for inputs, labels in dl:
    #     print(inputs.shape)
    #     break
    # print(len(dl.dataset))
    print(ds[0])


