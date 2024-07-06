import torch
import numpy as np
import tensorflow_datasets as tfds
import argparse
import os
import tensorflow as tf


def read_dataset(split_):
    dataset_of_tensorflow = tfds.load("celeb_a", split=split_,
                                      data_dir=os.path.expanduser("~/datasets/tensorflow_datasets"), download=False)
    dataset_of_tensorflow = dataset_of_tensorflow.map(lambda d: (
        d['attributes'], tf.image.resize(tf.image.convert_image_dtype(d['image'], tf.float32), [64, 64])))

    return dataset_of_tensorflow


def transform_and_save(save_path, tf_dataset):
    storage = []
    for label, inp in tf_dataset:
        label_ = {attr: np.array(val) for attr, val in label.items()}
        storage.append((label_, np.array(inp)))
    torch.save(storage, save_path)


if __name__ == "__main__":
    split_list = ["train", "validation", "test"]
    save_root = os.path.expanduser("~/datasets/celeb_a")
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    for split_ in split_list:
        tf_dataset_ = read_dataset(split_)
        transform_and_save(os.path.join(save_root, f"{split_}_img_64_64.pth"), tf_dataset_)

    print("-"*5 + "Done" + "-"*5)


