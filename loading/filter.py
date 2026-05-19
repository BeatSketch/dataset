from typing import cast
from util.dtype import BeatSketchTrainingData, BeatSketchTrainingDataSet
import numpy as np


def get_no_block_share(training_data: BeatSketchTrainingDataSet):
    return len(split_blocks_and_no_blocks(training_data)[0]) / len(training_data["data"])


def split_blocks_and_no_blocks(training_data: BeatSketchTrainingDataSet):
    no_block_idxs: list[int] = []
    block_idxs: list[int] = []

    for idx, d in enumerate(training_data["data"]):
        if not d["has_block"]:
            no_block_idxs.append(idx)
        else:
            block_idxs.append(idx)

    return no_block_idxs, block_idxs


def filter_training_data(
    no_block_share: float, training_data: BeatSketchTrainingDataSet
) -> BeatSketchTrainingDataSet:
    data = training_data["data"]
    # Split up blocks
    no_block_idxs, block_idxs = split_blocks_and_no_blocks(training_data)

    # Compute the number of blocks to pick
    cnt = int((len(block_idxs) / (1 - no_block_share)) * no_block_share)

    # Randomly pick using numpy
    np_training_data = np.array(data)
    rng = np.random.default_rng()
    picks = rng.choice(no_block_idxs, cnt).tolist()

    training_data["data"] = cast(
        list[BeatSketchTrainingData], np_training_data[picks].tolist()
    ) + cast(list[BeatSketchTrainingData], np_training_data[block_idxs].tolist())

    return training_data
