import os
import pickle

import numpy as np
import torchvision.transforms as transforms
from PIL import Image

from dataset.base_dataset import BaseDataset, rotate_img_tensor, simple_test

imgnet_mean = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
imgnet_std = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]
# imgnet_mean = [0.485, 0.456, 0.406]
# imgnet_std = [0.229, 0.224, 0.225]

normalize = transforms.Normalize(mean=imgnet_mean, std=imgnet_std)

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


class ImageNet(BaseDataset):
    def __init__(
        self,
        args,
        partition="train",
        transform=default_transform,
        rotate_aug=False,
        use_num_classes=None,
        use_num_class_instances=None,
        name="miniImageNet",
    ):
        # NOTE: These two attributes need to be defined before call to
        # super().__init__() since the __init__ call of BaseDataset
        # itself calls self._load_from_file()
        self.use_num_classes = use_num_classes
        self.use_num_class_instances = use_num_class_instances
        self.name = name
        super().__init__(args, transform=transform, partition=partition, rotate_aug=rotate_aug)

    def _load_from_file(self, partition):
        if "train" in self.partition:
            file_path = f"miniImageNet/miniImageNet_category_split_train_phase_{partition}.pickle"
        else:
            file_path = f"miniImageNet/miniImageNet_category_split_{partition}.pickle"
        with open(os.path.join(self.data_root, file_path), "rb") as f:
            data = pickle.load(f, encoding="latin1")
            labels = np.array(data["labels"])
            data = np.array(data["data"]).astype("uint8")

        return data, labels

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


class TieredImageNet(BaseDataset):
    def __init__(self, args, partition="train", transform=default_transform, rotate_aug=False, name="tieredImageNet"):
        self.name = name
        super().__init__(args, partition=partition, transform=transform, rotate_aug=rotate_aug)

    def _load_from_file(self, partition):
        image_file_pattern = f"tieredImageNet/{partition}_images.npz"
        label_file_pattern = f"tieredImageNet/{partition}_labels.pkl"

        # Load images and labels using np instead of python file handling
        image_file = os.path.join(self.data_root, image_file_pattern)
        data = np.load(image_file)["images"]

        label_file = os.path.join(self.data_root, label_file_pattern)
        labels = np.array(self._load_labels(label_file)["labels"])

        return data, np.array(labels)

    @staticmethod
    def _load_labels(file):
        try:
            with open(file, "rb") as fo:
                labels = pickle.load(fo)
            return labels
        except:
            with open(file, "rb") as f:
                u = pickle._Unpickler(f)
                u.encoding = "latin1"
                labels = u.load()
            return labels


class XImageNet(BaseDataset):
    def __init__(self, args, partition="train", transform=default_transform, rotate_aug=False, name="x20miniimagenet"):
        self.name = name
        super().__init__(args, partition=partition, transform=transform, rotate_aug=rotate_aug)

    def _load_from_file(self, partition):
        if partition == "val":
            partition = "valid"
        image_file = os.path.join(self.data_root, f"{self.name}/{partition}_images.npy")
        data = np.load(image_file)

        label_file = os.path.join(self.data_root, f"{self.name}/{partition}_labels.npy")
        labels = np.load(label_file).astype(int)

        return data, labels

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


if __name__ == "__main__":
    simple_test(ImageNet)
