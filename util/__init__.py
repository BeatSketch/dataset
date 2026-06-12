def divide_work(dir_list: list[str], workers: int, no_block_share: float) -> list[tuple[float, list[str]]]:
    """Divide the work evenly onto `workers` threads

    Args:
        dir_list: List of files
        workers: number of threads

    Returns:
        The split up work
    """
    work_per_thread = len(dir_list) // (workers - 1)
    work: list[tuple[float, list[str]]] = []

    for worker in range(workers - 1):
        work.append((no_block_share, dir_list[worker * work_per_thread : (worker + 1) * work_per_thread]))
    work.append((no_block_share, dir_list[(workers - 1) * work_per_thread :]))

    return work
