import json
import util.manager.file as file
from util.bpm_cache import BPMCache
from util.files import write_file
from util.manager.folder import folder_preprocessing, process_folder as fp


def preprocess_folder(dir: str):
    folder_preprocessing(dir)


def process_folder(dir: str):
    fp(dir)


def process_file(filename: str):
    data = file.process_file(filename, BPMCache(), True)
    write_file("file.json", json.dumps(data))
