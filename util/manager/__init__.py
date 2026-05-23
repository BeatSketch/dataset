import ml
import util.manager.file as file
from util.bpm_cache import BPMCache
from util.manager.folder import folder_preprocessing, process_folder as fp


def preprocess_folder(dir: str):
    folder_preprocessing(dir)


def process_folder(
    dir: str,
    train: bool,
    processed_save_location: str = "",
    max_files: int = -1,
    test_onnx: bool = False,
):
    """Process all bsor files in a folder

    Args:
        max_files: The maximum number of files to process
        dir: The directory to process
        train: Whether or not to train the model
        processed_save_location: If not empty string, path to file to store dataset in
    """
    data = fp(dir, max_files=max_files)

    if train:
        ml.train(
            data, dataset_save_location=processed_save_location, test_onnx=test_onnx
        )
    else:
        if processed_save_location == "":
            print("ERROR: No save location for dataset specified")
            exit(1)
        ml.preprocess_only(data, processed_save_location)


def process_file(
    filename: str,
    train: bool,
    processed_save_location: str = "",
    test_onnx: bool = False,
):
    data = file.process_file(filename, BPMCache(), True)

    if train and data:
        ml.train(
            data, dataset_save_location=processed_save_location, test_onnx=test_onnx
        )
