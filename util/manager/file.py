import time
from loading.filter import (
    filter_training_data,
    get_no_block_share,
)
from loading.generator import generate_training_data
from loading.loader import load_replay_data
from util.bpm_cache import BPMCache


def process_file(file: str, cache: BPMCache,print_debugging: bool = False):
    start = time.time()
    data = load_replay_data(file, cache, print_debugging)
    mid = time.time()
    if not data:
        return False

    training_data = generate_training_data(
        data[0], data[1], data[2], data[3], print_debugging=print_debugging
    )
    filtered_data = filter_training_data(0.5, training_data)
    if print_debugging:
        print(
            len(training_data["data"]),
            "datapoints were generated from this file, with no block share of",
            str(get_no_block_share(training_data) * 100) + "%",
        )
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
    return [filtered_data]
