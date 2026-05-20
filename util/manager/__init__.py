import json
import ml
import util.manager.file as file
from util.bpm_cache import BPMCache
from util.files import write_file
from util.manager.folder import folder_preprocessing, process_folder as fp


def preprocess_folder(dir: str):
    folder_preprocessing(dir)


def process_folder(dir: str, train: bool, save: bool = False, max_files=-1):
    data = fp(dir, max_files=max_files)
    if save:
        # TODO: Transform into usable format?
        write_file("data.json", json.dumps(data))
    if train:
        ml.train(data)


def process_file(filename: str, train: bool, save: bool = False):
    data = file.process_file(filename, BPMCache(), True)
    if save:
        write_file("file.json", json.dumps(data))
    if train and data:
        ml.train(data)
