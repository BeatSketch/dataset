import time
import colorama
from util.bpm_cache import BPMCache
from util.dtype import BeatSketchTrainingDataSet
from util.files import filesystem_walker
from util import divide_work
import multiprocessing as mp

from util.manager.file import process_file

print_debugging = False
print_status = False


def folder_preprocessing(dir: str) -> list[str]:
    all_files = filesystem_walker(dir)
    print("\n==> Retrieving BPM for all replays. This may take a while\n")
    # Only keep the files of which we know the BPM
    files = BPMCache(True).get_bpm_for_file_list(all_files)
    print(
        colorama.Fore.GREEN + colorama.Style.DIM + "BPM download complete, using",
        len(files),
        "out of",
        len(all_files),
        "replays\n",
        colorama.Style.RESET_ALL,
    )
    return files


def _process_file_list(files: list[str]) -> list[BeatSketchTrainingDataSet]:
    processed: list[BeatSketchTrainingDataSet] = []
    cache = BPMCache()
    for idx, file in enumerate(files):
        try:
            data = process_file(file, cache, print_debugging=print_debugging)
            if not not data:
                processed += data
                if print_status:
                    print(
                        f"Processed",
                        idx + 1,
                        "/",
                        len(files),
                        "files",
                    )
            else:
                print("processing of file", file, "failed for unknown reasons")
        except Exception as e:
            print(
                "Processing of file",
                file,
                "failed due to error",
                repr(e),
            )
    return processed


def process_folder(dir: str, max_files=-1):
    """Process a whole folder recursively at once,
        fully parallelized

    Args:
        dir: The directory to process
    """
    start = time.time()
    files = folder_preprocessing(dir)
    if max_files > 0:
        files = files[:max_files]
        print(
            colorama.Style.DIM
            + colorama.Fore.YELLOW
            + "\nDEBUG MODE ACTIVE, only one file per thread for processing\n"
            + colorama.Style.RESET_ALL
        )

    # Split work and process
    split_work = divide_work(files, mp.cpu_count())
    print("\n--> Using", len(split_work), "threads to process data\n")
    pool = mp.Pool()
    data: list[list[BeatSketchTrainingDataSet]] = pool.map(
        _process_file_list, split_work
    )

    # Flatten dataset
    flattened: list[BeatSketchTrainingDataSet] = []

    for thread in data:
        for file in thread:
            flattened.append(file)

    # Print stats
    dur = time.time() - start
    print(
        "\nProcessing of",
        len(flattened),
        "files took",
        round(dur, 3),
        f"seconds ({round(len(files) / dur, 4)} files per s /",
        round(dur / len(files) * mp.cpu_count(), 4),
        "s per file singlethreaded)\n",
        "with",
        len(files) - len(flattened),
        "failing to process\n"
    )

    return flattened
