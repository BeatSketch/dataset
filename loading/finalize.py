from util.dtype import BeatSketchTrainingDataSet


def generate_real_dataset(
    data: BeatSketchTrainingDataSet,
) -> tuple[list[list[float]], list[bool]]:
    vals: list[list[float]] = []
    labels: list[bool] = []
    for val in data["data"]:
        labels.append(val["has_block"])
        block: list[float] = [val["x"], val["y"], val["beat"]]

        vals.append(block)

    return vals, labels
