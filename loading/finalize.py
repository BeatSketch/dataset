from util.dtype import BeatSketchTrainingDataSet


def generate_real_dataset(
    data: BeatSketchTrainingDataSet,
) -> tuple[list[list[float]], list[bool]]:
    vals: list[list[float]] = []
    labels: list[bool] = []
    for val in data["data"]:
        labels.append(val["has_block"])
        left: list[float] = [val["x"], val["y"], val["beat"]]
        right: list[float] = [val["x"], val["y"], val["beat"]]
        for tracked in val["tracking"]:
            left += tracked["left"]
            left.append(tracked["time"])
            right += tracked["right"]
            right.append(tracked["time"])

        vals.append(left)
        vals.append(right)

    return vals, labels
