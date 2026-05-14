from typing import TypedDict


class BeatSketchTrainingData(TypedDict):
    tracking: list[BeatSketchTrackingData]
    beat: float
    has_block: bool
    is_right_hand: bool
    x: int
    y: int


class BeatSketchTrackingData(TypedDict):
    left: list[float]
    right: list[float]
    time: float


class BeatSketchBlock(TypedDict):
    orientation: int
    x: int
    y: int
    time: float
    is_right_hand: bool
    good_cut: bool
