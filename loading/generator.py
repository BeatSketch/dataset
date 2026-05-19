from typing import Literal
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
# In percent of the saber length, how much of it is considered the tip
THRESHOLD = 0.3

# Into how many parts to split each beat (should be power of 2 and no more than 8)
# I do also think we should make this configurable for the user? (or provide 2 settings?)
# Or at least for the training data, make it depend on the BPM
BEAT_SPLIT = 4
# Number of tracking data points per time unit
TRACKING_PER_UNIT = 4

# How many of the datapoints before to include
DATA_SLACK_BEFORE = 4
# How many of the datapoints after to include
DATA_SLACK_AFTER = 4
HANDS: list[Literal["left"] | Literal["right"]] = ["left", "right"]


def generate_training_data(
    tracking: list[BeatSketchTrackingData],
    blocks: list[BeatSketchBlock],
    bpm: int,
    njs: float,
    print_missing_datapoints_stats: bool = True
) -> BeatSketchTrainingDataSet:
    training_data: list[BeatSketchTrainingData] = []
    sec_per_unit = 60 / (bpm * BEAT_SPLIT)

    # Determine which time unit each data point belongs to
    buckets: dict[int, tuple[list[int], list[int]]] = {}
    for idx, frame in enumerate(tracking):
        for i, hand in enumerate(HANDS):
            time = int((frame["time"] + frame[hand][2] / njs) / sec_per_unit)
            try:
                buckets[time][i].append(idx)
            except Exception:
                buckets[time] = ([], [])
                buckets[time][i].append(idx)

    # Generate the tracking data
    hit_blocks = process_blocks(blocks, bpm)
    block_idx: list[int] = [0, 0]
    too_few_elements_incidents: list[int] = []
    for unit in range(len(buckets)):
        bucket = buckets[unit]
        for i, indices in enumerate(bucket):
            # Limit the number of data points used
            hand = "left" if i == 0 else "right"
            one_every_n_els = len(indices) / TRACKING_PER_UNIT
            els: list[np.ndarray] = []

            if len(indices) < TRACKING_PER_UNIT:
                too_few_elements_incidents.append(TRACKING_PER_UNIT - len(indices))
                for _ in range(TRACKING_PER_UNIT - len(indices)):
                    els.append(np.array([0, 0, 0, 0, 0, 0]))

            for k in range(min(TRACKING_PER_UNIT, len(indices))):
                els.append(tracking[indices[math.floor(one_every_n_els * k)]][hand])

            # Combine with pre-processed blocks
            already_processed_locations: list[tuple[int, int]] = []
            for j in range(block_idx[i], len(hit_blocks)):
                u = int(hit_blocks[j]["beat"] * BEAT_SPLIT)
                if u == unit:
                    if not (hit_blocks[j]["is_right_hand"] ^ (hand == "right")):
                        hit_blocks[j]["tracking"] += els
                        already_processed_locations.append(
                            (hit_blocks[j]["x"], hit_blocks[j]["y"])
                        )
                        training_data.append(hit_blocks[j])
                elif u > unit:
                    block_idx[i] = j
                    break

            # Compute the hits and generate training data array from it
            hits = hit_locations(els, already_processed_locations)
            for hit in hits:
                training_data.append(
                    {
                        "tracking": els[:3],  # Only want the tips and not the direction
                        "x": hit[0],
                        "y": hit[1],
                        "is_right_hand": hand == "right",
                        "has_block": False,
                        "beat": unit / BEAT_SPLIT,
                    }
                )

    if print_missing_datapoints_stats and len(too_few_elements_incidents) > 0:
        print("There were", len(too_few_elements_incidents), "instances where there were too few blocks per bucket")

    return {"data": training_data, "njs": njs, "bpm": bpm}


def hit_locations(
    tracking: list[np.ndarray],
    already_hit: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    locations: list[tuple[int, int]] = []
    """Determine possible locations where a block could be placed

    Args:
        tracking: The tracking data to process
        already_hit: A list of all already processed locations

    Returns:
        A list of coordinates on the grid that were touched by the tip
    """
    for pos in tracking:
        hand = pos[:3]
        dir = pos[3:]
        for line in range(3):
            for col in range(4):
                if (
                    hand[0] < GRID_X_MIN_VAL + (col + 1) * GRID_FIELD_WIDTH
                    and hand[0] > GRID_X_MIN_VAL + col * GRID_FIELD_WIDTH
                    and (
                        (
                            hand[1] < GRID_Y_MIN_VAL + (line + 1) * GRID_FIELD_HEIGHT
                            and hand[1] > GRID_Y_MIN_VAL + line * GRID_FIELD_HEIGHT
                        )
                        or (
                            hand[1] - THRESHOLD * dir[1]
                            < GRID_Y_MIN_VAL + (line + 1) * GRID_FIELD_HEIGHT
                            and hand[1] - THRESHOLD * dir[1]
                            > GRID_Y_MIN_VAL + line * GRID_FIELD_HEIGHT
                        )
                    )
                ):
                    try:
                        locations.index((line, col))
                    except Exception:
                        try:
                            already_hit.index((line, col))
                        except Exception:
                            locations.append((line, col))

    return locations


def process_blocks(
    blocks: list[BeatSketchBlock], bpm: int
) -> list[BeatSketchTrainingData]:
    data: list[BeatSketchTrainingData] = []

    for block in blocks:
        if block["good_cut"]:
            data.append(
                {
                    "has_block": True,
                    "beat": block["time"] / bpm,
                    "is_right_hand": block["is_right_hand"],
                    "x": block["x"],
                    "y": block["y"],
                    "tracking": [],
                }
            )

    return data
