import json
import time
from ml.preprocess import DATASET_TYPE
import numpy as np


def export_dataset(dataset: DATASET_TYPE, path: str):
    """Write the dataset to disk. OVERWRITES THE PATH WITHOUT CONFIRMATION

    Args:
        dataset: The dataset that is to be saved
        path: The file path to save to.
    """
    exportable: list[list[float]] = []
    for kind in dataset:
        exportable.append(kind.tolist())

    with open(path, "w") as f:
        f.write(json.dumps(exportable))


def import_dataset(path: str) -> DATASET_TYPE:
    """Load the dataset from the specified location

    Args:
        path: The file path to load the dataset from

    Returns:
        The dataset
    """
    print("==> Loading dataset")
    start = time.time()
    with open(path, "r") as f:
        data = json.loads(f.read())

    data = (np.array(data[0]), np.array(data[1]), np.array(data[2]), np.array(data[3]))
    print(" -> Loading completed in", time.time() - start, "seconds")
    return data
