from typing import Literal

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
