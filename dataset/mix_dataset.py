import os
import pickle
from pathlib import Path

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from dataset.base_dataset import BaseDataset, rotate_img_tensor

normalize = lambda x: 2 * x - 1.0

default_transform = transforms.Compose([lambda x: Image.fromarray(x), transforms.ToTensor(), normalize])

aug_transform = transforms.Compose(
    [
        lambda x: Image.fromarray(x),
        transforms.RandomCrop(84, padding=8),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomHorizontalFlip(),
        lambda x: np.array(x),
        transforms.ToTensor(),
        normalize,
    ]
)

aug_transform_vert = transforms.Compose(
    [
        lambda x: Image.fromarray(x),
        transforms.RandomCrop(84, padding=8),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomVerticalFlip(),
        lambda x: np.array(x),
        transforms.ToTensor(),
        normalize,
    ]
)


class SingleDataset(BaseDataset):
    def __init__(
        self,
        name,
        args,
        partition="train",
        transform=default_transform,
        rotate_aug=False,
    ):
        self.name = name
        super().__init__(args, transform=transform, partition=partition, rotate_aug=rotate_aug)

    def _load_from_file(self, partition):
        data_path = os.path.join(self.data_root, f"{self.name}_{partition}.pkl")
        with open(data_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")
            xs = np.array(data["data"])
            ys = np.array(data["labels"]).squeeze()

            return xs, ys

    def __getitem__(self, item):
        if self.rotate_aug:
            idx = item % self._real_len
            rot = item // self._real_len
        else:
            idx = item
            rot = 0

        if rot % 2 == 0:
            img = self.transform(self.data[idx])
        else:
            img = aug_transform_vert(self.data[idx])
        target = self.labels[idx]

        img = rotate_img_tensor(img, rot)
        target += self.n_cls * rot

        return img, target, item


class MixDataset(BaseDataset):
    def __init__(
        self,
        datasets,
        args,
        partition=None,
        transform=default_transform,
        rotate_aug=False,
    ):
        self.datasets = datasets
        super().__init__(args, transform=transform, partition=partition, rotate_aug=rotate_aug)

    def _load_from_file(self, partition):
        label_offset = 0
        tmp_data = []
        tmp_labels = []

        for dataset in self.datasets:
            data, labels = dataset.data, dataset.labels
            tmp_data.append(data)
            tmp_labels.append(labels + label_offset)
            # Assumes that the labels are contiguous and starts at 0 for each dataset
            label_offset += np.max(labels) + 1

        return np.concatenate(tmp_data), np.concatenate(tmp_labels)


class MetaMixDataset(Dataset):
    def __init__(self, datasets, batch_size_factors=1):
        super(Dataset, self).__init__()
        self.datasets = datasets
        task_count = [0]
        task_to_db = []

        if isinstance(batch_size_factors, int):
            batch_size_factors = [batch_size_factors] * len(datasets)

        for i, (dataset, size_factor) in enumerate(zip(datasets, batch_size_factors)):
            db_size = len(dataset) * size_factor
            task_to_db.extend([i] * db_size)
            task_count.append(db_size)

        self.id_offset = np.cumsum(task_count)[:-1]
        self.task_to_db = task_to_db
        self._len = len(task_to_db)

    def __getitem__(self, item):
        db_id = self.task_to_db[item]
        db_task_id = self.id_offset[db_id]

        return self.datasets[db_id][item - db_task_id]

    def __len__(self):
        return self._len


class MDLMixDataset(BaseDataset):
    def __init__(self, datasets, batch_size_factors):
        super(Dataset, self).__init__()
        self.datasets = datasets
        self.batch_size_factors = batch_size_factors
        self._setup()

    def _setup(self):
        self.item_to_db = []
        # First we setup the item to db mapping
        for i, (dataset, size_factor) in enumerate(zip(self.datasets, self.batch_size_factors)):
            db_full_size = len(dataset) * size_factor
            self.item_to_db.extend([i] * db_full_size)
        self.item_to_db = np.array(self.item_to_db)

        # Then we form the item to proper data index mapping
        idx_count = [0]
        for dataset in self.datasets:
            idx_count.append(len(dataset))
        self.db_to_idx_offset = np.cumsum(idx_count)[:-1]

        self.item_to_idx = []
        for i, (dataset, size_factor) in enumerate(zip(self.datasets, self.batch_size_factors)):
            # Now we have a staircase modulo the dataset size
            local_idx = (np.arange(len(dataset)) + self.db_to_idx_offset[i]).tolist() * size_factor
            self.item_to_idx.extend(local_idx)
        self.item_to_idx = np.array(self.item_to_idx)

        # Finally, db to mask
        db_to_logit_mask = []
        for i, dataset in self.datasets:
            db_to_logit_mask.extend([i] * (np.max(dataset.labels) + 1))
        db_to_logit_mask = np.array(db_to_logit_mask)
        _new_array = []
        for i in range(db_to_logit_mask.shape[0]):
            _new_array.extend(db_to_logit_mask == i)
        self.db_to_logit_mask = np.array(_new_array)

        self._len = len(self.item_to_db)

    def _load_from_file(self, partition):
        label_offset = 0
        tmp_data = []
        tmp_labels = []

        for dataset in self.datasets:
            data, labels = dataset.data, dataset.labels
            tmp_data.append(data)
            tmp_labels.append(labels + label_offset)
            # Assumes that the labels are contiguous and starts at 0 for each dataset
            label_offset += np.max(labels) + 1

        return np.concatenate(tmp_data), np.concatenate(tmp_labels)

    def __getitem__(self, item):
        # NOTE: No rotate augmentation
        db_id = self.item_to_db[item]
        idx = self.item_to_idx[item]
        image = self.transform(self.data[idx])
        target = self.labels[idx]
        return image, target, db_id

    def __len__(self):
        return self._len


if __name__ == "__main__":
    args = lambda x: None
    args.n_ways = 5
    args.n_shots = 5
    args.n_queries = 15
    data_root = Path(os.getenv("WORKSPACE")) / "metaL_data"
    args.data_roots = [f"{data_root}/aircraft", f"{data_root}/cu_birds/cub"]
    args.data_aug = False
    args.n_test_runs = 5
    args.sample_shape = "few_shot"
    imagenet = MixDataset(args, "train")

    print(len(imagenet))

    img = imagenet[10][0]

    meta_data = MetaMixDataset(imagenet)

    print(len(meta_data))
