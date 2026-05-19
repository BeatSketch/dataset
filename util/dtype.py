from typing import TypedDict
import numpy as np


class BeatSketchTrainingDataSet(TypedDict):
    data: list[BeatSketchTrainingData]
    njs: float
    bpm: int


class BeatSketchTrainingData(TypedDict):
    tracking: list[np.ndarray]
    beat: float
    has_block: bool
    is_right_hand: bool
    x: int
    y: int


class BeatSketchTrackingData(TypedDict):
    left: np.ndarray
    right: np.ndarray
    time: float


class BeatSketchBlock(TypedDict):
    orientation: int
    x: int
    y: int
    time: float
    exact_time: float
    is_right_hand: bool
    good_cut: bool
