import time
import colorama
from util.bpm_cache import BPMCache
from util.dtype import BeatSketchTrainingDataSet
from util.files import filesystem_walker, write_file
from util import divide_work
import multiprocessing as mp

from util.manager.file import process_file

print_debugging = False
print_status = True


def folder_preprocessing(dir: str) -> list[str]:
    all_files = filesystem_walker(dir)
    print("Retrieving BPM for all replays. This may take a while")
    # Only keep the files of which we know the BPM
    files = BPMCache().get_bpm_for_file_list(all_files)
    print(
        colorama.Fore.GREEN + colorama.Style.DIM + "BPM download complete, using",
        len(files),
        "out of",
        len(all_files),
        "replays",
        colorama.Style.RESET_ALL,
    )
    return files


def _process_file_list(files: list[str]) -> list[BeatSketchTrainingDataSet]:
    processed = []
    cache = BPMCache()
    for idx, file in enumerate(files):
        try:
            data = process_file(file, cache, print_debugging=print_debugging)
            if isinstance(data, dict):
                processed.append(data)
        except Exception as e:
            print(
                "Processing of file",
                file,
                "failed due to error",
                repr(e),
            )
        if print_status:
            print(
                f"Processed",
                idx + 1,
                "/",
                len(files),
                "files",
            )
    return processed


def process_folder(dir: str):
    """Process a whole folder recursively at once,
        fully parallelized

    Args:
        dir: The directory to process
    """
    start = time.time()
    files = folder_preprocessing(dir)
    split_work = divide_work(files, mp.cpu_count())
    pool = mp.Pool()
    data: list[list[BeatSketchTrainingDataSet]] = pool.map(
        _process_file_list, split_work
    )

    print("Total number of files processed is", len(files))
    dur = time.time() - start
    print(
        "This operation has taken",
        dur,
        f"seconds ({len(files) / dur} files per second)",
    )
    print("That is", dur / len(files) * mp.cpu_count(), "seconds per file for a single thread")
    print(len(data))
    # TODO: Processing should include transforming to actually usable format
    # write_file("data.json", json.dumps(data))
    print(
        colorama.Style.DIM
        + colorama.Fore.GREEN
        + "\n==> File saved to current directory",
        colorama.Style.RESET_ALL,
    )
