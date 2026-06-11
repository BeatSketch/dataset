from time import time

from ml import preprocess
from util import loader_exporter
from util.dtype import BeatSketchTrainingDataSet


def preprocess_only(data: list[BeatSketchTrainingDataSet], dataset_save_location: str):
    dataset = preprocess.preprocesss(data)
    print("\n=> Generated", len(dataset[0]), "datapoints\n")

    if dataset_save_location != "":
        loader_exporter.export_dataset(dataset, dataset_save_location)


def train(
    data: list[BeatSketchTrainingDataSet],
    dataset_save_location: str = "",
    test_onnx: bool = False,
    model: str = "mlp",
):
    """Train a model from a list of TrainingDataSets

    Args:
        data: The training data to generate the dataset from
        dataset_save_location: Where to store the dataset
        test_onnx: Whether to test using the ONNX runner or not
        model: the model to train
    """
    dataset = preprocess.preprocesss(data)
    print("\n=> Generated", len(dataset[0]), "datapoints, starting training\n")

    if dataset_save_location != "":
        loader_exporter.export_dataset(dataset, dataset_save_location)

    train_with_existing_dataset(dataset, test_onnx, model)


def train_with_existing_dataset(
    dataset: preprocess.DATASET_TYPE, test_onnx: bool = False, model: str = "mlp"
):
    """Train a model using the provided dataset

    Args:
        dataset: Fully processed dataset
        test_onnx: Whether to test using the ONNX runner or not
        model: the model to train
    """
    start = time()
    if model == "mlp_torch":
        import ml.torch.torch_mlp as torch_mlp

        mlp_path = torch_mlp.train_model(dataset)
    else:
        import ml.sklearn.mlp as mlp

        mlp_path = mlp.train_model(dataset)

    trained = time()
    print("\n==> Trained model in", trained - start, "seconds\n")

    # Verify models
    if test_onnx:
        import ml.onnx_runner as onnx_runner

        print("==> Testing model with ONNX runtime")
        onnx_runner.run_model(mlp_path, dataset[1], dataset[3])
        print("\n==> Model evaluation completed in", time() - trained, "seconds\n")
