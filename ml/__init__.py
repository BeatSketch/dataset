from ml import preprocess
from util.dtype import BeatSketchTrainingDataSet
import ml.sklearn.mlp as mlp


def train(data: list[BeatSketchTrainingDataSet], test_onnx: bool = False):
    dataset = preprocess.preprocesss(data)
    print("\n=> Generated", len(dataset[0]), "datapoints, starting training\n")

    # sklearn models
    mlp_path = mlp.train_model(dataset)

    # Verify models
    if test_onnx:
        import ml.onnx_runner as onnx_runner

        onnx_runner.run_model(mlp_path, dataset[1], dataset[3])
