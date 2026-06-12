from argparse import ArgumentParser as _ArgumentParser


def parse_args():
    ap = _ArgumentParser(
        "beatsketch_dataset",
        description="BeatSketch Dataset processing and Model training CLI",
        usage="beatsketch_dataset [command] [options]",
    )
    ap.add_argument("-v", "--version", action="version", version="%(prog)s V0.0.1")

    sp = ap.add_subparsers(
        title="Commands",
        metavar="Use 'beatsketch_dataset [command] --help' to see help for each command",
        dest="cmd",
        required=True,
    )
    sp.add_parser("help", help="Print this help message")
    evaluate = sp.add_parser("eval", help="Evaluate the produced model")
    evaluate.add_argument("model_path", help="Path to the model to evaluate")
    evaluate.add_argument("dataset", help="The dataset to evaluate on")

    # Preprocessing group
    preprocess = sp.add_parser("preprocess", help="Only do preprocessing on the data")
    preprocessors = preprocess.add_subparsers(
        title="Preprocess", dest="preprocess", required=True
    )
    preproc_file = preprocessors.add_parser("file", help="Preprocess a single file")
    preproc_file.add_argument("file", help="The path to the file to process")
    preproc_file.add_argument("save_path", help="The path to save the dataset to to")
    preproc_file.add_argument(
        "-s",
        "--split",
        required=False,
        help="Choose the split between blocks and no blocks. Defined as #blocks / #no-blocks",
        default=0.5,
    )

    preproc_folder = preprocessors.add_parser("folder", help="Preprocess a full folder")
    preproc_folder.add_argument("folder", help="The folder to process")
    preproc_folder.add_argument("save_path", help="The path to save the dataset to to")
    preproc_folder.add_argument(
        "-t",
        "--test",
        action="store_true",
        required=False,
        help="Only use a subset of the data",
    )
    preproc_folder.add_argument(
        "-s",
        "--split",
        required=False,
        help="Choose the split between blocks and no blocks. Defined as #blocks / #no-blocks",
        default=0.5,
    )

    preproc_bpm = preprocessors.add_parser(
        "bpm", help="Pre-Download the BPM information"
    )
    preproc_bpm.add_argument("folder", help="The folder to process")

    # Training group
    train = sp.add_parser("train", help="Preprocess data and train models")
    training = train.add_subparsers(title="Train", dest="train", required=True)
    train_file = training.add_parser("file", help="Train on a single file")
    train_file.add_argument("file", help="The path to the file to process")
    train_file.add_argument("save_path", help="The path to save the dataset to to")
    train_file.add_argument(
        "--onnx", help="Test the exported ONNX model as well", action="store_true"
    )
    train_file.add_argument(
        "-m",
        "--model",
        required=False,
        help="Choose a model. defaults to sklearn/mlp.py",
        default="mlp",
    )
    train_file.add_argument(
        "-s",
        "--split",
        required=False,
        help="Choose the split between blocks and no blocks. Defined as #blocks / #no-blocks",
        default=0.5,
    )

    train_folder = training.add_parser("folder", help="Train on a full folder")
    train_folder.add_argument("folder", help="The folder to process")
    train_folder.add_argument("save_path", help="The path to save the dataset to to")
    train_folder.add_argument(
        "-t",
        "--test",
        action="store_true",
        required=False,
        help="Only use a subset of the data",
    )
    train_folder.add_argument(
        "-s",
        "--split",
        required=False,
        help="Choose the split between blocks and no blocks. Defined as #blocks / #no-blocks",
        default=0.5,
    )
    train_folder.add_argument(
        "--onnx", help="Test the exported ONNX model as well", action="store_true"
    )
    train_folder.add_argument(
        "-m",
        "--model",
        required=False,
        help="Choose a model. defaults to sklearn/mlp.py",
        default="mlp",
    )

    train_dataset = training.add_parser(
        "dataset", help="Load training data from a dataset file"
    )
    train_dataset.add_argument("dataset", help="The dataset file")
    train_dataset.add_argument(
        "--onnx", help="Test the exported ONNX model as well", action="store_true"
    )
    train_dataset.add_argument(
        "-m",
        "--model",
        required=False,
        help="Choose a model. defaults to sklearn/mlp.py",
        default="mlp",
    )

    return ap.parse_args(), ap
