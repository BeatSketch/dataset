import time
import multiprocessing as mp

from loading.filter import (
    filter_training_data,
    get_no_block_share,
)
from loading.generator import generate_training_data
from loading.loader import load_replay_data
from util.files import filesystem_walker
from util import divide_work


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


def process_folder(dir: str):
    """Process a whole folder recursively at once,
        fully parallelized

    Args:
        dir: The directory to process
    """
    files = filesystem_walker(dir)
    split_work = divide_work(files, mp.cpu_count())
    handles: list[mp.Process] = []
    for work in split_work:
        proc = mp.Process(target=process_by_list, args=[work])
        proc.start()
        handles.append(proc)

    for handle in handles:
        handle.join()


def process_by_list(files: list[str], printing_prefix: str = ""):
    for idx, file in enumerate(files):
        process_file(file)
        print(printing_prefix, "Processed", idx, "/", len(files), "files")
