from loading.hits_blocks import hit_locations, process_blocks
from loading.values import BEAT_SPLIT, HANDS, TRACKING_PER_UNIT
from util.dtype import (
    BeatSketchBlock,
    BeatSketchTrackingData,
    BeatSketchTrainingData,
    BeatSketchTrainingDataSet,
)
import math
import numpy as np

def generate_training_data(
    tracking: list[BeatSketchTrackingData],
    blocks: list[BeatSketchBlock],
    bpm: int,
    njs: float,
    print_debugging: bool = True
) -> BeatSketchTrainingDataSet:
    training_data: list[BeatSketchTrainingData] = []
    sec_per_unit = 60 / (bpm * BEAT_SPLIT)

    # Determine which time unit each data point belongs to
    buckets: dict[int, tuple[list[int], list[int]]] = {}
    for idx, frame in enumerate(tracking):
        for i, hand in enumerate(HANDS):
            time = int((frame["time"] - frame[hand][2] / njs) / sec_per_unit)
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
        bucket = ([], [])
        try:
            bucket = buckets[unit]
        except KeyError:
            pass
        for i, indices in enumerate(bucket):
            # Limit the number of data points used
            hand = "left" if i == 0 else "right"
            one_every_n_els = len(indices) / TRACKING_PER_UNIT
            els: list[np.ndarray] = []

            if len(indices) < TRACKING_PER_UNIT:
                # TODO: Consider skipping these
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
                        "tracking": els,
                        "x": hit[0],
                        "y": hit[1],
                        "is_right_hand": hand == "right",
                        "has_block": False,
                        "beat": unit / BEAT_SPLIT,
                    }
                )

    if print_debugging and len(too_few_elements_incidents) > 0:
        print("There were", len(too_few_elements_incidents), "instances where there were too few blocks per bucket")

    return {"data": training_data, "njs": njs, "bpm": bpm}
