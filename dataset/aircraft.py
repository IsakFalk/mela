import os
import pickle
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from dataset.base_dataset import rotate_img_tensor
from dataset.mix_dataset import default_transform


class AirCraftMultiHeadSupDataset(Dataset):
    def __init__(
        self, args, transform=default_transform, partition="train", rotate_aug=False, name="semantic_h_aircraft"
    ):
        super().__init__()
        self.name = name
        self.data_root = Path(os.getenv("WORKSPACE")) / "metaL_data"
        self.partition = partition
        self.data_aug = args.data_aug
        self.rotate_aug = rotate_aug

        self.transform = transform

        self.data, self.labels = self._load_from_file(partition)

        self.labels = self.labels - np.min(self.labels, axis=0)

        self.n_cls = np.max(self.labels, axis=0) + 1

        self._real_len = len(self.data)

        if self.rotate_aug:
            self._len = self._real_len * 4
        else:
            self._len = self._real_len

    def __getitem__(self, item):
        if self.rotate_aug:
            idx = item % self._real_len
            rot = item // self._real_len
        else:
            idx = item
            rot = 0

        img = self.transform(self.data[idx])
        target = np.copy(self.labels[idx])

        img = rotate_img_tensor(img, rot)
        target += self.n_cls * rot

        return img, target, item

    def __len__(self):
        return self._len

    def _load_from_file(self, partition):
        data_path = os.path.join(self.data_root, f"aircraft/aircraft_{partition}.pkl")
        with open(data_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")
            xs = np.array(data["data"])
            ys = np.array(data["labels"]).squeeze()

            return xs, ys
