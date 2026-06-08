import numpy as np

from loading.values import (
    BEAT_SPLIT,
    GRID_FIELD_HEIGHT,
    GRID_FIELD_WIDTH,
    GRID_X_MIN_VAL,
    GRID_Y_MIN_VAL,
    THRESHOLD,
)
from util.dtype import BeatSketchBlock, BeatSketchTrainingData


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
                        locations.index((col, line))
                    except Exception:
                        try:
                            already_hit.index((col, line))
                        except Exception:
                            locations.append((col, line))

    return locations


def process_blocks(
    blocks: list[BeatSketchBlock], bpm: float
) -> list[BeatSketchTrainingData]:
    """Utility method to snap blocks onto beat grid, also drops all bad cuts.\\
    Example: if `BEAT_SPLIT=4`, blocks are assigned to one of 4 subdivisions per beat. (quarter note quantization)

    Args:
        blocks (list[BeatSketchBlock]): Parsed block data from BSOR Replay
        bpm (float): BPM of corresponding replay

    Returns:
        list[BeatSketchTrainingData]: Quantized list of blocks
    """
    
    
    data: list[BeatSketchTrainingData] = []
    bps = bpm / 60

    for block in blocks:
        if block["good_cut"]:
            data.append(
                {
                    "has_block": True,
                    "beat": round(block["time"] * bps * BEAT_SPLIT) / BEAT_SPLIT,
                    "is_right_hand": block["is_right_hand"],
                    "x": block["x"],
                    "y": block["y"],
                    "tracking": [],
                }
            )

    return data
