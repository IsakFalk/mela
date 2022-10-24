
# Table of Contents

-   [FSImageNet: generating arbitrary base dataset for few-shot learning based on ImageNet](#org7c983a9)
    -   [What this is](#orgf9c0da5)
    -   [How to use](#org8ea6a54)
        -   [Dependencies](#orgcddd8cc)
        -   [Creating a dataset](#org0244eae)



<a id="org7c983a9"></a>

# FSImageNet: generating arbitrary base dataset for few-shot learning based on ImageNet


<a id="orgf9c0da5"></a>

## What this is

This is a repository for generating few-shot datasets in the vein of *mini* and
*tiered*-ImageNet providing a simple way to split the train part of **ILSVRC**
into a train, validation and test part, allowing for explicitly detail what
classes and how many instances of each class should be included in each split.


<a id="org8ea6a54"></a>

## How to use


<a id="orgcddd8cc"></a>

### Dependencies

Install the packages from the `environment.yml` file using conda. If you want to use pip or some other package manager just install the packages in the `environment.yml` file.

The repo assumes that you have a version of ImageNet somewhere on your computer (untarred).


<a id="org0244eae"></a>

### Creating a dataset

The directory `splits` contain synsets (mappings from global class names to descriptive keywords of these classes) for train, validation and test for three different settings, mini, tiered and the union of mini+tiered ImageNet classes (`synset_keys.txt`, `synset_train_keys.txt`, `synset_val_keys.txt`, `synset_test_keys.txt`). Copy the ones you want to the parent directory if you want to specify the classes of these splits to be included in the final splits of the generated datasets.

A dataset may be generated as follows

    python fs_imagenet_generator.py --imagenet_dir /path/to/ILSVRC/Data/CLS-LOC/train/ --num_instances_per_class 600 --num_classes 1000 --seed 0 --use_split_text_files

