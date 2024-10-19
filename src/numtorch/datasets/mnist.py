try:
    import cupy as cp
except ImportError:
    print("Could not import cupy falling back to numpy...")
    import numpy as cp

import requests
import os
import gzip

from numtorch.autograd.tensor import Tensor


def download_file(url, filename):
    # Download the file from the given URL
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"Downloaded {filename}")
    else:
        print(f"Failed to download {filename}. Status code: {response.status_code}")


def download_mnist():
    base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }

    # Download each file if it doesn't already exist
    for key, filename in files.items():
        if not os.path.exists(filename):
            download_file(base_url + filename, filename)
        else:
            print(f"{filename} already exists, skipping download.")


def read_idx(filename):
    # Read IDX files as numpy arrays
    with gzip.open(filename, "rb") as f:
        # Read the magic number and dimensions
        magic = int.from_bytes(f.read(4), byteorder="big")
        num_items = int.from_bytes(f.read(4), byteorder="big")

        if magic == 2051:  # Magic number for images
            rows = int.from_bytes(f.read(4), byteorder="big")
            cols = int.from_bytes(f.read(4), byteorder="big")
            data = cp.frombuffer(f.read(), dtype=cp.uint8).reshape((num_items, rows, cols))
        elif magic == 2049:  # Magic number for labels
            data = cp.frombuffer(f.read(), dtype=cp.uint8).reshape((num_items,))
        else:
            raise ValueError("Invalid IDX file: unexpected magic number")

    return data


def load_mnist():
    download_mnist()
    x_train = read_idx("train-images-idx3-ubyte.gz")
    y_train = read_idx("train-labels-idx1-ubyte.gz")
    x_test = read_idx("t10k-images-idx3-ubyte.gz")
    y_test = read_idx("t10k-labels-idx1-ubyte.gz")

    return (Tensor(x_train), Tensor(y_train, dtype=cp.int32)), (Tensor(x_test), Tensor(y_test, dtype=cp.int32))
