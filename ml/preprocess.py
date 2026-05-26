from typing import cast
from sklearn.model_selection import train_test_split
from util.dtype import BeatSketchTrainingDataSet
import numpy as np

DATASET_TYPE = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
DATASET_PREPARE_TYPE = tuple[list[np.ndarray], list[bool]]


def preprocesss(data: list[BeatSketchTrainingDataSet]) -> DATASET_TYPE:
    dataset: list[np.ndarray] = []
    labelset: list[bool] = []
    for set in data:
        datapoints, labels = generate_real_dataset(set)
        dataset += datapoints
        labelset += labels

    # Train-test split
    X, X_test, y, y_test = train_test_split(np.array(dataset), np.array(labelset), test_size=0.1)
    X = cast(np.ndarray, X).astype(np.float32)
    X_test = cast(np.ndarray, X_test).astype(np.float32)
    y = cast(np.ndarray, y).astype(np.float32)
    y_test = cast(np.ndarray, y_test).astype(np.float32)

    return X, X_test, y, y_test


def generate_real_dataset(
    data: BeatSketchTrainingDataSet,
) -> DATASET_PREPARE_TYPE:
    vals: list[np.ndarray] = []
    labels: list[bool] = []
    for val in data["data"]:
        labels.append(val["has_block"])
        training_data_frame: list[float] = [val["x"], val["y"], val["beat"]]
        for loc in val["tracking"]:
            # loc each are 6 floats, first three are tip position, second three are direction vector
            training_data_frame += loc.tolist()

        vals.append(np.array(training_data_frame))

    return vals, labels
