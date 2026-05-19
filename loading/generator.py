from typing import Literal, cast
from util.dtype import (
    BeatSketchBlock,
    BeatSketchTrackingData,
    BeatSketchTrainingData,
    BeatSketchTrainingDataSet,
)
import math
import numpy as np

# TODO: Find out what the grid size actually is
GRID_FIELD_WIDTH = 0.666
GRID_FIELD_HEIGHT = 0.666
GRID_Y_MIN_VAL = 0
GRID_X_MIN_VAL = -1.333

# Into how many parts to split each beat (should be power of 2 and no more than 8)
# I do also think we should make this configurable for the user? (or provide 2 settings?)
# Or at least for the training data, make it depend on the BPM
BEAT_SPLIT = 4
# Number of tracking data points per time unit
TRACKING_PER_UNIT = 5

# How many of the datapoints before to include
DATA_SLACK_BEFORE = 4
# How many of the datapoints after to include
DATA_SLACK_AFTER = 4


def generate_training_data(
    tracking: list[BeatSketchTrackingData],
    blocks: list[BeatSketchBlock],
    bpm: int,
    njs: float,
) -> BeatSketchTrainingDataSet:
    training_data: list[BeatSketchTrainingData] = []
    sec_per_unit = 60 / (bpm * BEAT_SPLIT)

    # Determine buckets for tracking data
    buckets_tracking: list[int] = []
    bucket_end = sec_per_unit
    for idx, frame in enumerate(tracking):
        if frame["time"] > bucket_end:
            buckets_tracking.append(idx)
            bucket_end += sec_per_unit
    buckets_tracking.append(len(tracking) - 1)

    # Determine buckets for blocks
    # FIXME: Improve this
    buckets_blocks: list[int] = [0]
    bucket_end = sec_per_unit
    for idx, frame in enumerate(blocks):
        if frame["time"] > bucket_end:
            diff = frame["time"] - bucket_end
            while diff > sec_per_unit:
                # Need to add empty buckets
                diff -= sec_per_unit
                buckets_blocks.append(idx)
            buckets_blocks.append(idx)
            bucket_end += sec_per_unit
    buckets_blocks.append(len(blocks) - 1)

    # Compute the training data
    prev = 0
    for beat, end in enumerate(buckets_tracking):
        one_every_n_els = (end - prev) / TRACKING_PER_UNIT

        els: list[BeatSketchTrackingData] = []
        for i in range(TRACKING_PER_UNIT):
            els.append(tracking[prev + math.floor(one_every_n_els * i)])

        locs = determine_possible_locs(els) + determine_possible_locs(els, "right")
        # TODO: Append the prev and after slack here
        for loc in locs:
            # Determine if in this beat, there is a block in loc
            is_hit_l = False
            is_hit_r = False
            # FIXME: This very bad still (i.e. nowhere near all successful slices are in the dataset)
            for block_idx in range(buckets_blocks[beat], buckets_blocks[beat + 1]):
                if (
                    blocks[block_idx]["x"] == loc[0]
                    and blocks[block_idx]["y"] == loc[1]
                ):
                    if blocks[block_idx]["is_right_hand"]:
                        is_hit_r = True
                    else:
                        is_hit_l = True

                    if is_hit_r and is_hit_l:
                        break

            training_data.append(
                {
                    "is_right_hand": False,
                    "x": loc[0],
                    "y": loc[1],
                    "beat": beat,
                    "has_block": is_hit_l,
                    "tracking": els,
                }
            )
            training_data.append(
                {
                    "is_right_hand": True,
                    "x": loc[0],
                    "y": loc[1],
                    "beat": beat,
                    "has_block": is_hit_r,
                    "tracking": els,
                }
            )

        prev = end

    return {"data": training_data, "bpm": bpm, "njs": njs}


def determine_possible_locs(
    tracking: list[BeatSketchTrackingData],
    hand_side: Literal["left"] | Literal["right"] = "left",
) -> list[tuple[int, int]]:
    """Determine possible locations where a block could be placed

    Args:
        tracking: The tracking data to process
        hand_side: For which side to do the processing

    Returns:
        A list of coordinates on the grid that were touched by the tip
    """
    # TODO: Use also the direction vector of controller to increase hit area
    # Compute which grid spots the controller tip touches
    coords: list[tuple[int, int]] = []
    for pos in tracking:
        hand = pos[hand_side]
        dir = pos["left_dir" if hand_side == "left" else "right_dir"]
        for line in range(3):
            for col in range(4):
                if (
                    hand[0] < GRID_X_MIN_VAL + (col + 1) * GRID_FIELD_WIDTH
                    and hand[0] > GRID_X_MIN_VAL + col * GRID_FIELD_WIDTH
                    and hand[1] < GRID_Y_MIN_VAL + (line + 1) * GRID_FIELD_HEIGHT
                    and hand[1] > GRID_Y_MIN_VAL + line * GRID_FIELD_HEIGHT
                ):
                    try:
                        coords.index((line, col))
                    except Exception:
                        coords.append((line, col))

    return coords


def get_no_block_share(training_data: BeatSketchTrainingDataSet):
    return len(split_blocks_and_no_blocks(training_data)[0]) / len(training_data)


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
    # Split up blocks
    no_block_idxs, block_idxs = split_blocks_and_no_blocks(training_data)

    # Compute the number of blocks to pick
    cnt = int((len(block_idxs) / (1 - no_block_share)) * no_block_share)

    # Randomly pick using numpy
    np_training_data = np.array(training_data)
    rng = np.random.default_rng()
    picks = rng.choice(no_block_idxs, cnt).tolist()

    training_data["data"] = cast(
        list[BeatSketchTrainingData], np_training_data[picks].tolist()
    ) + cast(list[BeatSketchTrainingData], np_training_data[block_idxs].tolist())

    return training_data
