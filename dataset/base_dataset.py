import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import util
from train_routine import get_dataloaders

np.random.default_rng(0)
no_transform = lambda x: x


class BaseDataset(Dataset):
    def __init__(self, args, transform, partition="train", rotate_aug=False):
        super().__init__()
        self.data_root = Path(os.getenv("WORKSPACE")) / "metaL_data"
        self.partition = partition
        self.data_aug = args.data_aug
        self.rotate_aug = rotate_aug

        self.transform = transform

        self.data, self.labels = self._load_from_file(partition)

        self.labels = self.labels - min(self.labels)
        self.n_cls = np.max(self.labels) + 1

        self._real_len = len(self.data)

        if self.rotate_aug:
            self._len = len(self.labels) * 4
        else:
            self._len = len(self.labels)

    def __getitem__(self, item):
        if self.rotate_aug:
            idx = item % self._real_len
            rot = item // self._real_len
        else:
            idx = item
            rot = 0

        img = self.transform(self.data[idx])
        target = self.labels[idx]

        img = rotate_img_tensor(img, rot)
        target += self.n_cls * rot

        return img, target, item

    def __len__(self):
        return self._len

    def _load_from_file(self, partition):
        raise "Load from file should be implemented in child classes"


class LabelerDataset(BaseDataset):
    def __init__(
        self,
        dataloader,
        labeler,
        args,
        transform=no_transform,
        partition="train",
        rotate_aug=False,
    ):
        """Label the dataset of dataloader using labeler

        We make the following assumptions
        - dataloader is created from a MetaDataset that has no_replacement=True
        - dataloader is created from a MetaDataset that has sample_shape="flat"
        """
        self.loader = dataloader
        self.labeler = labeler
        self.dataset_is_mixture = hasattr(self.loader.dataset, "datasets")
        super().__init__(args, transform=transform, partition=partition, rotate_aug=rotate_aug)

    @torch.no_grad()
    def _load_from_file(self, partition):
        # NOTE: partition is unused. It's here since the parent class BaseDataset
        # requires the _load_from_file function to take partition as an argument
        if self.dataset_is_mixture:
            raw_data = []
            labelled_ys = []

            # Iterate over all datasets, collect the
            # input and the pseudo labels for each dataset
            for ds in self.loader.dataset.datasets:
                raw_idx_db, labelled_ys_db = [], []
                for batch in ds:
                    # Extract the batch and idx, use
                    # labeller to label the data and then
                    # add the indices / labelled_ys if the labeler
                    # assigns all of the local class-inputs to different
                    # labels
                    task_data = util.to_cuda_list(batch)

                    xs = task_data[0]
                    task_idx = task_data[-1].flatten().squeeze()

                    pseudo_ys, pseudo_cls = self.labeler.label_samples(xs.cuda(), topk=1)

                    if pseudo_cls is not None:
                        raw_idx_db.append(task_idx)
                        labelled_ys_db.append(pseudo_ys.cpu().numpy())

                # raw_idx and labelled_ys need to be of shape=(n,)
                raw_idx_db = np.array(raw_idx_db).flatten()
                labelled_ys_db = np.array(
                    [task_labelled_ys.flatten().squeeze() for task_labelled_ys in labelled_ys_db]
                ).flatten()

                # Create new dataset with raw data
                raw_data.append(np.array(ds.data[np.array(raw_idx_db)]))
                labelled_ys.append(labelled_ys_db)

            raw_data = np.concatenate(raw_data)
            labelled_ys = np.concatenate(labelled_ys)
        else:
            raw_idx, labelled_ys = [], []

            for batch_data in self.loader:
                # Extract the batch and idx, use
                # labeller to label the data and then
                # add the indices / labelled_ys if the labeler
                # assigns all of the local class-inputs to different
                # labels
                task_data = list(map(lambda x: x[0], batch_data))
                xs = task_data[0]
                task_idx = task_data[-1].cpu().detach().numpy().flatten().squeeze()

                pseudo_ys, pseudo_cls = self.labeler.label_samples(xs.cuda(), topk=1)

                if pseudo_cls is not None:
                    raw_idx.append(task_idx)
                    labelled_ys.append(pseudo_ys.cpu().numpy())

            # raw_idx and labelled_ys need to be of shape=(n,)
            raw_idx = np.array(raw_idx).flatten()
            labelled_ys = np.array([task_labelled_ys.flatten().squeeze() for task_labelled_ys in labelled_ys]).flatten()

            # Create new dataset with raw data
            raw_data = self.loader.dataset.data[raw_idx]

        assert raw_data.shape[0] == labelled_ys.shape[0]

        # The dataset should be in the same format as the original
        # dataset subclassed from BaseDataset. For example, since ImageNet
        # loads the data as np.array with dtyp uint8, the data that we return
        # should have the same form and type.
        # NOTE: All datasets are assumed to be np.array for this to work
        # as we need to use the same concatenation function when returning.
        return raw_data, labelled_ys


class MetaDataset(Dataset):
    def __init__(
        self,
        flat_db,
        args,
        train_transform=None,
        no_replacement=False,
        db_size=100,
        fixed_db=True,
        sample_shape="few_shot",
        simulate_imbalance=False,
    ):
        # no_replacement == fixed_db=True AND db_size=AUTO
        # fixed_db == false AND db_size = X means each epoch contains X random tasks
        super().__init__()
        self.data = flat_db.data
        self.labels = flat_db.labels
        self.n_ways = args.n_ways
        self.n_shots = args.n_shots
        self.n_queries = args.n_queries
        self.db_size = db_size
        self.fixed_db = fixed_db
        self.sample_shape = sample_shape
        self.local_cls = np.arange(self.n_ways)
        self.n_per_class = self.n_shots + self.n_queries
        self.simulate_imbalance = simulate_imbalance

        self.train_transform = flat_db.transform if train_transform is None else train_transform

        self.label_to_index = defaultdict(list)
        for i in range(self.data.shape[0]):
            self.label_to_index[self.labels[i]].append(i)

        for key in self.label_to_index:
            self.label_to_index[key] = np.asarray(self.label_to_index[key]).astype(int)

        self.classes = list(self.label_to_index.keys())

        if no_replacement:
            self._gen_task_without_replacement()
            self._get = self._fixed_item
        else:
            self._get = self._random_item

        # FIXME: Unsure what this does, this doesn't work for now
        self.imbalance_idx = {}
        for key, idx in self.label_to_index.items():
            count = idx.shape[0]
            if simulate_imbalance:
                new_count = np.random.choice(self.n_per_class, count)
                self.imbalance_idx[key] = new_count
            else:
                self.imbalance_idx[key] = idx

        # Apply train_transform to all batches
        self.train_transform_batch = lambda x: batch_to_tensor(x, self.train_transform)

    def _gen_task_without_replacement(self):
        all_samples_idx = {}
        for key, idx in self.label_to_index.items():
            # Shuffle the indices of the class instances
            all_samples_idx[key] = np.random.permutation(idx)

        # Generate all tasks from the flat dataset
        # cls_order and idx maps from batch_id to the sampled classes and sample id's of those classes
        cls_order = []
        idx = []

        while len(all_samples_idx) >= self.n_ways:
            cls_sampled = np.random.choice(list(all_samples_idx.keys()), self.n_ways, False)
            cls_order.append(cls_sampled)

            # Pick n_per_class instances from the local labels sampled in cls_sampled
            # task_class_to_idx maps from batch_idx to indices of the instances in the task
            task_class_to_idx = []
            for cls in cls_sampled:
                local_class_samples_idx = all_samples_idx[cls]
                idx_sampled, others = np.split(local_class_samples_idx, [self.n_per_class])
                task_class_to_idx.append(idx_sampled)
                # If there are enough instances left in class cls to form further tasks, make
                # these available for further tasks, else delete them (if len(others) < self.n_per_class)
                if len(others) >= self.n_per_class:
                    all_samples_idx[cls] = others
                else:
                    del all_samples_idx[cls]

            task_class_to_idx = np.asarray(task_class_to_idx)
            idx.append(task_class_to_idx)

        # List of size num_batches with each entry being an array of size n_way,
        # consisting of global classes of that task
        self.clss = cls_order
        # List of size num_batches with each entry being an array of size n_way x n_per_class (n_per_class = n_shots + n_queries),
        # consisting of global indices for the task (reshaped)
        self.sample_idx = idx
        self.db_size = len(self.sample_idx)

    def _random_item(self, item):
        if self.fixed_db:
            np.random.seed(item)
        # Sample self.n_ways elements from self.classes without replacement
        cls_sampled = np.random.choice(self.classes, self.n_ways, False)

        xs = []
        task_class_to_idx = []
        for cls in cls_sampled:
            # FIXME: this branch doesn't work
            if self.simulate_imbalance:
                idx_sampled = np.random.choice(self.imbalance_idx[cls], self.n_per_class, False)
            else:
                idx_sampled = np.random.choice(self.label_to_index[cls], self.n_per_class, False)
            task_class_to_idx.append(idx_sampled)
            xs.append(self.data[idx_sampled])
        xs = np.array(xs)
        task_class_to_idx = np.array(task_class_to_idx)

        return xs, cls_sampled, task_class_to_idx

    def _fixed_item(self, item):
        cls_sampled = self.clss[item]
        task_class_to_idx = self.sample_idx[item]

        xs = []
        # Get all instances for each class in task
        for class_instance_idx in task_class_to_idx:
            xs.append(self.data[class_instance_idx])

        xs = np.asarray(xs)
        return xs, cls_sampled, task_class_to_idx

    def __getitem__(self, item):
        xs, cls_sampled, task_class_to_idx = self._get(item)

        data_dims = xs.shape[2:]
        if self.sample_shape == "flat":
            xs = xs.reshape(-1, *data_dims)
            return (
                self.train_transform_batch(xs),
                np.repeat(cls_sampled, self.n_per_class),
                cls_sampled,
                task_class_to_idx,
            )
        elif self.sample_shape == "few_shot":
            support_xs, query_xs = np.split(xs, [self.n_shots], axis=1)

            support_xs = support_xs.reshape((-1, *data_dims))
            query_xs = query_xs.reshape((-1, *data_dims))

            support_xs = self.train_transform_batch(support_xs)
            query_xs = self.train_transform_batch(query_xs)

            return (
                support_xs,
                np.repeat(self.local_cls, self.n_shots),
                query_xs,
                np.repeat(self.local_cls, self.n_queries),
                (
                    task_class_to_idx[:, : self.n_shots],
                    task_class_to_idx[:, self.n_shots :],
                ),
            )

    def __len__(self):
        return self.db_size


def batch_to_tensor(arr, transform):
    arr = np.split(arr, arr.shape[0], axis=0)
    return torch.stack(list(map(lambda x: transform(x.squeeze()), arr)))


def rotate_img(img, rot):
    if rot == 0:  # 0 degrees rotation
        return img
    elif rot == 90:  # 90 degrees rotation
        return np.flipud(np.transpose(img, (1, 0, 2)))
    elif rot == 180:  # 90 degrees rotation
        return np.fliplr(np.flipud(img))
    elif rot == 270:  # 270 degrees rotation / or -90
        return np.transpose(np.flipud(img), (1, 0, 2))
    else:
        raise ValueError("rotation should be 0, 90, 180, or 270 degrees")


def rotate_img_tensor(img, rot):
    if rot == 0:  # 0 degrees rotation
        return img
    elif rot == 1:  # 90 degrees rotation
        return img.transpose(1, 2).flip(1)
    elif rot == 2:  # 180 degrees rotation
        return img.flip(2).flip(1)
    elif rot == 3:  # 270 degrees rotation / or -90
        return img.flip(1).transpose(1, 2)
    else:
        raise ValueError("rotation should be 0, 90, 180, or 270 degrees")


def simple_test(dataset_cls, metadata_cls=MetaDataset):
    args = lambda x: None
    args.n_ways = 5
    args.n_shots = 5
    args.n_queries = 15
    args.data_root = Path(os.getenv("WORKSPACE")) / "metaL_data"
    args.data_aug = False
    args.n_test_runs = 5
    args.sample_shape = "few_shot"
    args.rotate_aug = True
    dataset = dataset_cls(args, "train")

    print(len(dataset))
    print(dataset[5][0].shape)

    meta_data = metadata_cls(dataset, args, no_replacement=False, fixed_db=False, db_size=20)
    print(len(meta_data))
    print(meta_data[0][0].shape)

    data_loader = get_dataloaders(meta_data, 1, 4, shuffle=True)

    for i, data in enumerate(data_loader[0]):
        if i > 5:
            break
