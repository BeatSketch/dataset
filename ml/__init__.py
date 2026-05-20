from ml import preprocess
from util import loader_exporter
from util.dtype import BeatSketchTrainingDataSet
import ml.sklearn.mlp as mlp


def preprocess_only(data: list[BeatSketchTrainingDataSet], dataset_save_location: str):
    dataset = preprocess.preprocesss(data)
    print("\n=> Generated", len(dataset[0]), "datapoints\n")

    if dataset_save_location != "":
        loader_exporter.export_dataset(dataset, dataset_save_location)


def train(
    data: list[BeatSketchTrainingDataSet],
    dataset_save_location: str = "",
    test_onnx: bool = False,
):
    """Train all the models from a list of TrainingDataSets

    Args:
        data: The training data to generate the dataset from
        test_onnx: Whether to test using the ONNX runner or not
    """
    dataset = preprocess.preprocesss(data)
    print("\n=> Generated", len(dataset[0]), "datapoints, starting training\n")

    if dataset_save_location != "":
        loader_exporter.export_dataset(dataset, dataset_save_location)

    train_with_existing_dataset(dataset, test_onnx)


def train_with_existing_dataset(
    dataset: preprocess.DATASET_TYPE, test_onnx: bool = False
):
    """Train all the models using the provided dataset

    Args:
        dataset: Fully processed dataset
        test_onnx: Whether to test using the ONNX runner or not
    """
    # sklearn models
    mlp_path = mlp.train_model(dataset)

    # Verify models
    if test_onnx:
        import ml.onnx_runner as onnx_runner

        onnx_runner.run_model(mlp_path, dataset[1], dataset[3])
