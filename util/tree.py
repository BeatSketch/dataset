import os


def filesystem_walker(dir: str) -> list[str]:
    files = os.listdir(dir)
    file_list = []
    for file in files:
        path = dir + "/" + file
        if os.path.isdir(path):
            file_list += filesystem_walker(path)
        else:
            if file.split(".")[-1] == "bsor":
                file_list.append(path)

    return file_list


def divide_work(dir_list: list[str], workers: int) -> list[list[str]]:
    """Divide the work evenly onto `workers` threads

    Args:
        dir_list: List of files
        workers: number of threads

    Returns:
        The split up work
    """
    work_per_thread = len(dir_list) // workers
    work: list[list[str]] = []

    for worker in range(workers):
        work.append(dir_list[worker * work_per_thread : (worker + 1) * work_per_thread])
    work[workers] += dir_list[workers * work_per_thread : -1]

    return work
