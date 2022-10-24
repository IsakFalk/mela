import argparse
import glob
import json
import logging
import os
from pathlib import Path
from shutil import copyfile

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--imagenet_dir", type=str, help="Path to ImageNet directory")
parser.add_argument("--image_resize", type=int, default=84, help="Size of length and width to resize to")
parser.add_argument(
    "--num_classes",
    type=int,
    default=1000,
    help="Number of base classes from ImageNet to use",
)
parser.add_argument(
    "--num_instances_per_class",
    type=int,
    default=100,
    help="Number of instances per base class to use",
)
parser.add_argument(
    "--use_split_text_files",
    action="store_true",
    help="Use split text files with classes to split into train, val and test set",
)
parser.add_argument(
    "--raw_db",
    action="store_true",
    help="Whether to use the raw database form where images are stored as-is or convert the dataset to numpy with metadata stored as json",
)
parser.add_argument("--seed", type=int, default=0, help="Random seed to use")
parser.add_argument(
    "--parent_dir",
    type=str,
    default="./datasets",
    help="Parent directory where we put the database directory",
)

args = parser.parse_args()
rng = np.random.default_rng(args.seed)


def process_processed_images_dir(root_path):
    data_list = []
    label_list = []
    metadata_dict = {}
    class_paths = [x for x in root_path.iterdir() if x.is_dir()]
    for class_path in class_paths:
        instance_paths = [x for x in class_path.iterdir() if x.is_file()]
        metadata_dict[class_path.name] = [x.name.split(".")[0] for x in instance_paths if x.is_file()]
        data_list.extend([np.array(Image.open(x)) for x in instance_paths])
        label_list.extend(len(instance_paths) * [class_path.name])
    return metadata_dict, data_list, label_list


class XImageNetGenerator(object):
    def __init__(self, input_args):
        self.input_args = input_args
        if self.input_args.imagenet_dir is not None:
            self.imagenet_dir = self.input_args.imagenet_dir
        else:
            logging.info("You need to specify the ILSVRC source file path")
            raise
        self.image_resize = input_args.image_resize
        self.num_classes = input_args.num_classes
        self.num_instances_per_class = input_args.num_instances_per_class
        self.use_split_text_files = input_args.use_split_text_files
        self.raw_db = input_args.raw_db
        self.parent_dir = Path(input_args.parent_dir)
        self.x_dir = (
            self.parent_dir / f"x_imagenet_num_cls{self.num_classes}_num_instance{self.num_instances_per_class}"
        )
        if not os.path.exists(self.x_dir):
            os.mkdir(self.x_dir)
        self._read_synset_keys()
        self._generate_split()

    def _read_synset_keys(self):
        """Read in synset_keys.txt and instantiate train/val/test keys"""
        path = Path("./synset_keys.txt")
        with open(path, "r") as f:
            self.synset_keys = np.array([line.split(" ")[0] for line in f])
        logging.info(f"Read synset text file with {len(self.synset_keys)} number of keys.")
        self.synset_use_train_keys = np.array([])
        self.synset_use_val_keys = np.array([])
        self.synset_use_test_keys = np.array([])

    def _read_split_files_and_remove_classes_from_global_classes(self):
        """Make sure that classes in synset_{train,val,test}_keys.txt go into the right split"""
        # Read in, and also remove the keys from the synset_keys
        path = Path("./synset_train_keys.txt")
        with open(path, "r") as f:
            self.synset_use_train_keys = np.array([line.split(" ")[0].strip("\n") for line in f])
        logging.info(f"Read synset train text file with {len(self.synset_use_train_keys)} number of keys.")
        self.synset_keys = self.synset_keys[~np.in1d(self.synset_keys, self.synset_use_train_keys)]

        # Ditto for val
        path = Path("./synset_val_keys.txt")
        with open(path, "r") as f:
            self.synset_use_val_keys = np.array([line.split(" ")[0].strip("\n") for line in f])
        logging.info(f"Read synset val text file with {len(self.synset_use_val_keys)} number of keys.")
        self.synset_keys = self.synset_keys[~np.in1d(self.synset_keys, self.synset_use_val_keys)]

        # Ditto for test
        path = Path("./synset_test_keys.txt")
        with open(path, "r") as f:
            self.synset_use_test_keys = np.array([line.split(" ")[0].strip("\n") for line in f])
        logging.info(f"Read synset test text file with {len(self.synset_use_test_keys)} number of keys.")
        self.synset_keys = self.synset_keys[~np.in1d(self.synset_keys, self.synset_use_test_keys)]

    def _generate_split(self):
        """Split the classes randomly into train / val / test"""

        # Split is 64 / 16 / 20
        rng.shuffle(self.synset_keys)
        n = self.num_classes
        n_tr = int(0.64 * n)
        n_val = int(0.16 * n)
        n_test = int(0.2 * n)

        if self.use_split_text_files:
            self._read_split_files_and_remove_classes_from_global_classes()

        # Reuse old classes and fill up the rest from synset_keys
        # train
        self.synset_train_keys = self.synset_use_train_keys
        curr_tr = len(self.synset_train_keys)
        add_tr = n_tr - curr_tr
        assert add_tr >= 0, "add_tr should be positive, use higher number of classes"
        self.synset_train_keys = np.concatenate([self.synset_train_keys, self.synset_keys[:add_tr]])
        self.synset_keys = self.synset_keys[add_tr:]

        # val
        self.synset_val_keys = self.synset_use_val_keys
        curr_val = len(self.synset_val_keys)
        add_val = n_val - curr_val
        assert add_val >= 0, "add_val should be positive, use higher number of classes"
        self.synset_val_keys = np.concatenate([self.synset_val_keys, self.synset_keys[:add_val]])
        self.synset_keys = self.synset_keys[add_val:]

        # test
        self.synset_test_keys = self.synset_use_test_keys
        curr_test = len(self.synset_test_keys)
        add_test = n_test - curr_test
        assert add_test >= 0, "add_test should be positive, use higher number of classes"
        self.synset_test_keys = np.concatenate([self.synset_test_keys, self.synset_keys[:add_test]])
        self.synset_keys = self.synset_keys[add_test:]

        logging.info("Generated splits")

    def process_original_files(self):
        self.processed_img_dir = self.x_dir / "raw_db"
        split_lists = ["train", "valid", "test"]

        if not os.path.exists(self.processed_img_dir):
            os.makedirs(self.processed_img_dir)

        for split_synset_keys, split in zip(
            [self.synset_train_keys, self.synset_val_keys, self.synset_test_keys],
            split_lists,
        ):
            this_split_dir = os.path.join(str(self.processed_img_dir.absolute()), split)
            if not os.path.exists(this_split_dir):
                os.makedirs(this_split_dir)

            logging.info("Writing photos....")
            for cls in tqdm(split_synset_keys):
                this_cls_dir = os.path.join(this_split_dir, cls)
                if not os.path.exists(this_cls_dir):
                    os.makedirs(this_cls_dir)

                cls_instance_files = np.array(list(glob.glob(f"{self.imagenet_dir}/{cls}/*")))
                # Randomize what is kept
                rng.shuffle(cls_instance_files)
                cls_instance_files = cls_instance_files[: self.num_instances_per_class]
                cls_instance_index = np.array(
                    [int(filename.split("_")[1].split(".")[0]) for filename in cls_instance_files]
                )
                for index, image_file in zip(cls_instance_index, cls_instance_files):
                    image_path = os.path.join(this_cls_dir, str(index) + ".jpg")
                    if self.image_resize == 0:
                        copyfile(
                            image_file,
                            image_path,
                        )
                    else:
                        im = cv2.imread(image_file)
                        im_resized = cv2.resize(
                            im,
                            (self.image_resize, self.image_resize),
                            interpolation=cv2.INTER_AREA,
                        )
                        cv2.imwrite(image_path, im_resized)

    def convert_to_x_form(self):
        # Set up directories we'll use
        process_image_dir = self.x_dir / "raw_db"
        final_db_dir = self.x_dir / "x_db"
        final_db_dir.mkdir(parents=True, exist_ok=True)

        # Make the anonymous labels across train / val / split
        # contiguous
        counter = 0
        for split in ["train", "valid", "test"]:
            logging.info(f"Processing {split}")
            split_dir = process_image_dir / split
            metadata_dict, data_list, label_list = process_processed_images_dir(split_dir)
            data = np.stack(data_list)
            labels = np.stack(label_list)
            # Make labels anonymous and preserve mapping
            base_to_anon_label_mapping = {label: i + counter for i, label in enumerate(np.unique(labels))}
            counter += len(base_to_anon_label_mapping)
            anon_labels = np.vectorize(base_to_anon_label_mapping.get)(labels)
            # Dump data and meta-data for split
            with open(final_db_dir / f"{split}_images.npy", "wb") as f:
                np.save(f, data)
            with open(final_db_dir / f"{split}_labels.npy", "wb") as f:
                np.save(f, anon_labels)
            with open(final_db_dir / f"{split}_anon_to_base_label_mapping.json", "w") as f:
                json.dump(
                    {v: k for k, v in base_to_anon_label_mapping.items()},
                    f,
                    indent=2,
                    sort_keys=True,
                )
            with open(final_db_dir / f"{split}.json", "w") as f:
                json.dump(metadata_dict, f, indent=2, sort_keys=True)

    def generate_dataset(self):
        self.process_original_files()
        if not self.raw_db:
            self.convert_to_x_form()


if __name__ == "__main__":
    dataset_generator = XImageNetGenerator(args)
    dataset_generator.generate_dataset()
