import os
from torch.utils.data import Dataset

class DatasetType(Dataset):
    def __init__(self, data_root, split):
        super().__init__()
        assert split in ["train", "eval"], "Invalid split!"

    def __len__(self):
        return 0

    def __getitem__(self, index):
        return index, index
