import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

IMAGE_SIZE = 84

# Mappings for h_aircraft labels
with open("variant_class_to_family_class.json", "r") as f:
    variant_class_to_family_class = json.load(f)
    variant_class_to_family_class = {int(key): int(val) for key, val in variant_class_to_family_class.items()}

with open("variant_class_to_manufacturer_class.json", "r") as f:
    variant_class_to_manufacturer_class = json.load(f)
    variant_class_to_manufacturer_class = {
        int(key): int(val) for key, val in variant_class_to_manufacturer_class.items()
    }


default_transform = transforms.Compose(
    [
        lambda x: Image.fromarray(x),
        transforms.Resize(84),
    ]
)

import tensorflow.compat.v1 as tf


# The two functions below taken from meta-dataset
def read_single_example(example_string):
    """Parses the record string."""
    return tf.parse_single_example(
        example_string,
        features={
            "image": tf.FixedLenFeature([], dtype=tf.string),
            "label": tf.FixedLenFeature([], dtype=tf.int64),
        },
    )


def read_example_and_parse_image(example_string):
    """Reads the string and decodes the image."""
    parsed_example = read_single_example(example_string)
    image_decoded = tf.image.decode_image(parsed_example["image"], channels=3)
    image_decoded.set_shape([None, None, 3])
    image_resized = tf.image.resize_images(
        image_decoded,
        [IMAGE_SIZE, IMAGE_SIZE],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        align_corners=True,
    )
    parsed_example["image"] = image_resized
    return parsed_example


def extract_fn(data_record):
    sample = read_example_and_parse_image(data_record)
    image = sample["image"].numpy()
    label = sample["label"].numpy()
    return image, label


def record_to_array(record_path):
    filenames = [record_path]
    raw_dataset = tf.data.TFRecordDataset(filenames)

    data = []
    for example in raw_dataset:
        image, label = extract_fn(example)
        image = image.reshape((-1, 84, 84, 3))
        image = torch.tensor(image)
        label = np.array([label]).reshape(-1, 1)
        label = torch.tensor(label)
        data.append((image, label))

    # Concatenate tensors into n x h x w x c and n x 1
    data = list(zip(*data))
    data = list(map(lambda x: torch.cat(x), data))
    return data


def save_part(data, labels, name, path):
    save_name = f"{path}/{name}.pkl"

    tmp = {
        "data": data,
        "labels": labels,
    }
    with open(save_name, "wb") as fout:
        pickle.dump(tmp, fout)


def tf_to_pkl_aircraft():
    path = Path(os.getenv("WORKSPACE")) / "metaL_data" / "aircraft"

    ret = []
    for i in tqdm(range(100)):
        file_name = f"{path}/{i}.tfrecords"
        file_records = record_to_array(file_name)
        ret.append(file_records)

    ret = list(zip(*ret))
    ret = list(map(lambda x: torch.cat(x), ret))

    data = ret[0].detach().numpy()
    labels = ret[1].detach().numpy()

    # variant
    variant_labels = labels
    save_part(
        data[: 70 * 100],
        variant_labels[: 70 * 100],
        f"aircraft_variant_train",
        path,
    )
    save_part(
        data[70 * 100 : 85 * 100],
        variant_labels[70 * 100 : 85 * 100],
        f"aircraft_variant_val",
        path,
    )
    save_part(
        data[85 * 100 :],
        variant_labels[85 * 100 :],
        f"aircraft_variant_test",
        path,
    )
    save_part(data[: 70 * 100], labels[: 70 * 100], "aircraft_train", path)
    save_part(data[70 * 100 : 85 * 100], labels[70 * 100 : 85 * 100], "aircraft_val", path)
    save_part(data[85 * 100 :], labels[85 * 100 :], "aircraft_test", path)

    # family
    family_labels = np.vectorize(variant_class_to_family_class.get)(labels)
    save_part(
        data[: 70 * 100],
        family_labels[: 70 * 100],
        f"aircraft_family_train",
        path,
    )
    save_part(
        data[70 * 100 : 85 * 100],
        family_labels[70 * 100 : 85 * 100],
        f"aircraft_family_val",
        path,
    )
    save_part(
        data[85 * 100 :],
        family_labels[85 * 100 :],
        f"aircraft_family_test",
        path,
    )

    # maker (manufacturer)
    maker_labels = np.vectorize(variant_class_to_manufacturer_class.get)(labels)
    save_part(
        data[: 70 * 100],
        maker_labels[: 70 * 100],
        f"aircraft_maker_train",
        path,
    )
    save_part(
        data[70 * 100 : 85 * 100],
        maker_labels[70 * 100 : 85 * 100],
        f"aircraft_maker_val",
        path,
    )
    save_part(
        data[85 * 100 :],
        maker_labels[85 * 100 :],
        f"aircraft_maker_test",
        path,
    )

    h_labels = np.hstack([variant_labels, family_labels, maker_labels])
    save_part(
        data[: 70 * 100],
        h_labels[: 70 * 100],
        f"aircraft_train",
        path,
    )
    save_part(
        data[70 * 100 : 85 * 100],
        h_labels[70 * 100 : 85 * 100],
        f"aircraft_val",
        path,
    )
    save_part(
        data[85 * 100 :],
        h_labels[85 * 100 :],
        f"aircraft_test",
        path,
    )


def tf_to_pkl_cub():
    path = Path(os.getenv("WORKSPACE")) / "metaL_data" / "cu_birds"
    with open(f"{path}/dataset_spec.json") as f:
        aux_data = json.load(f)

    img_per_class = aux_data["images_per_class"]
    counts = np.array(list(img_per_class.values()))
    cumsum = np.cumsum(counts)

    ret = []
    for i in tqdm(range(200)):
        file_name = f"{path}/{i}.tfrecords"
        file_records = record_to_array(file_name)
        ret.append(file_records)

    ret = list(zip(*ret))
    ret = list(map(lambda x: torch.cat(x), ret))

    data = ret[0].detach().numpy()
    labels = ret[1].detach().numpy()

    print(data.shape)

    save_part(data[: cumsum[139]], labels[: cumsum[139]], "cub_train", path)
    save_part(
        data[cumsum[139] : cumsum[169]],
        labels[cumsum[139] : cumsum[169]],
        "cub_val",
        path,
    )
    save_part(data[cumsum[169] :], labels[cumsum[169] :], "cub_test", path)


def tf_to_pkl_vgg():
    path = Path(os.getenv("WORKSPACE")) / "metaL_data" / "vgg_flower"
    with open(f"{path}/dataset_spec.json") as f:
        aux_data = json.load(f)

    img_per_class = aux_data["images_per_class"]
    counts = np.array(list(img_per_class.values()))
    cumsum = np.cumsum(counts)

    ret = []
    for i in tqdm(range(101)):
        file_name = f"{path}/{i}.tfrecords"
        file_records = record_to_array(file_name)
        ret.append(file_records)

    ret = list(zip(*ret))
    ret = list(map(lambda x: torch.cat(x), ret))

    data = ret[0].detach().numpy()
    labels = ret[1].detach().numpy()

    print(data.shape)

    save_part(data[: cumsum[70]], labels[: cumsum[70]], "vgg_train", path)
    save_part(data[cumsum[70] : cumsum[85]], labels[cumsum[70] : cumsum[85]], "vgg_val", path)
    save_part(data[cumsum[85] :], labels[cumsum[85] :], "vgg_test", path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process meta-dataset tfrecords to pickle")
    parser.add_argument(
        "-d",
        "--dataset",
        help="Dataset to process",
        required=True,
        type=str,
        choices=["aircraft", "cu_birds", "vgg_flower"],
    )
    args = vars(parser.parse_args())

    dataset = args["dataset"]
    if dataset == "aircraft":
        tf_to_pkl_aircraft()
    elif dataset == "cu_birds":
        tf_to_pkl_cub()
    elif dataset == "vgg_flower":
        tf_to_pkl_vgg()
    else:
        print(f"Dataset {dataset} not implemented.")
