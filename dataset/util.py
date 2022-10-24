import numpy as np
import pandas as pd


def adjust_number_of_base_classes_and_instances(data, labels, use_num_classes=None, use_num_class_instances=None):
    """Change the number of classes and instances per class in data and labels

    Assumes that the data and labels are flat in the sense that
    the first dimension is the index is the index full dataset,
    (x_i, y_i) == data[i], labels[i]."""
    if use_num_classes is None and use_num_class_instances is None:
        return data, labels
    else:
        # We prune the dataset both on a class and per-class level
        # according to the numbers defined in use_num_classes and
        # use_num_class instances. If these are None we skip this step.
        df = pd.DataFrame(data=labels, columns=["labels"]).reset_index()

        # Only keep use_num_classes
        unique_classes = np.sort(df["labels"].unique())
        if use_num_classes is not None:
            keep_classes = unique_classes[:use_num_classes]
        else:
            pass
        df = df.query("labels in @keep_classes")

        # Make the number of instances per classes smaller
        # by only keeping use_num_class_instances
        if use_num_class_instances is not None:
            df = df.groupby("labels").head(use_num_class_instances)
        index = df["index"].values

        return data[index], labels[index]
