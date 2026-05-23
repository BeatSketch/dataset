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
    # FIXME: Something is going wrong here (or in importer)
    exportable: list[list[float]] = []
    print("  ==> Exporting dataset")
    start = time.time()
    for kind in dataset:
        exportable.append(kind.tolist())

    with open(path, "w") as f:
        f.write(json.dumps(exportable))
    print("   -> Exported in", time.time() - start, "seconds\n")


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

    data = (
        np.array(data[0]).astype(np.float32),
        np.array(data[1]).astype(np.float32),
        np.array(data[2]).astype(np.float32),
        np.array(data[3]).astype(np.float32),
    )
    print(" -> Loading completed in", time.time() - start, "seconds")
    return data
