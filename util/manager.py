import time
import multiprocessing as mp

import colorama

from loading.filter import (
    filter_training_data,
    get_no_block_share,
)
from loading.generator import generate_training_data
from loading.loader import load_replay_data
from util.bpm_cache import get_bpm_for_file_list
from util.dtype import BeatSketchTrainingDataSet
from util.files import filesystem_walker
from util import divide_work


class BeatSketchDataSetProcess(mp.Process):
    _files: list[str]
    _printing_prefix: str
    data: list[BeatSketchTrainingDataSet]

    def __init__(self, work: list[str], printing_prefix: str = "") -> None:
        self._files = work
        self._printing_prefix = printing_prefix
        self.data = []
        super().__init__()

    def run(self) -> None:
        self.data = []
        for idx, file in enumerate(self._files):
            self.data.append(process_file(file))
            print(
                self._printing_prefix, "Processed", idx, "/", len(self._files), "files"
            )


def process_file(file: str):
    start = time.time()
    data = load_replay_data(file)
    mid = time.time()
    training_data = generate_training_data(data[0], data[1], data[2], data[3])
    print(
        len(training_data["data"]),
        "datapoints were generated from this file, with no block share of",
        str(get_no_block_share(training_data) * 100) + "%",
    )
    filtered_data = filter_training_data(0.5, training_data)
    print(
        len(filtered_data["data"]),
        "datapoints filtered, no block share of",
        str(get_no_block_share(filtered_data) * 100) + "%",
    )
    print(
        "\n",
        "Processing took",
        time.time() - start,
        "with loading taking",
        mid - start,
        "and generating taking",
        time.time() - mid,
    )
    return filtered_data


def folder_preprocessing(dir: str) -> list[str]:
    all_files = filesystem_walker(dir)
    print("Retrieving BPM for all replays. This may take a while")
    # Only keep the files of which we know the BPM
    files = get_bpm_for_file_list(all_files)
    print(
        colorama.Fore.GREEN + colorama.Style.DIM + "BPM download complete, using",
        len(files),
        "out of",
        len(all_files),
        "replays",
        colorama.Style.RESET_ALL,
    )
    return files


def process_folder(dir: str):
    """Process a whole folder recursively at once,
        fully parallelized

    Args:
        dir: The directory to process
    """
    start = time.time()
    files = folder_preprocessing(dir)
    split_work = divide_work(files, mp.cpu_count())
    handles: list[BeatSketchDataSetProcess] = []
    for work in split_work:
        proc = BeatSketchDataSetProcess(work)
        proc.start()
        handles.append(proc)

    data: list[BeatSketchTrainingDataSet] = []
    for handle in handles:
        handle.join()
        data += handle.data
    print("Total number of files processed generated is", len(files))
    print("This operation has taken", time.time() - start, "seconds")
