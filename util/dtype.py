from typing import TypedDict
import numpy as np


class BeatSketchTrainingDataSet(TypedDict):
    """Wrapper type for the dataset, includes metadata """
    data: list[BeatSketchTrainingData]
    njs: float
    bpm: float


class BeatSketchTrainingData(TypedDict):
    """ Beat-Quantized Training data for a single beat """
    # Tracking points
    tracking: list[np.ndarray]
    # The current beat (z-locations)
    beat: float
    # If there is a block for this frame
    has_block: bool
    is_right_hand: bool
    # BeatSaber lane of the hit
    x: int
    # BeatSaber layer of the hit
    y: int


class BeatSketchTrackingData(TypedDict):
    """ Tracking data for a specific point in time 
    
    Attributes `left`, `right` are 2D arrays each containing saber tip position & direction vector.
    """
    # A 2d array of all tracking data of the hands: 
    left: np.ndarray
    right: np.ndarray
    # The time of the event
    time: float


class BeatSketchBlock(TypedDict):
    """ BeatSketch's format for a block received from BSOR replay """
    # Cut direction in BeatSaber numbering format
    orientation: int
    # The lane of the cut
    x: int
    # The layer of the cut
    y: int
    # The time at which the cut occurred (in seconds)
    time: float
    # The exact time (from the BSOR file) of the cut event (in seconds)
    exact_time: float
    is_right_hand: bool
    # If it was a good cut or not
    good_cut: bool
