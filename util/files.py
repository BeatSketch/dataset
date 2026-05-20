import colorama
import os


def filesystem_walker(dir: str) -> list[str]:
    """Recursively get all bsor files from a folder

    Args:
        dir: The directory to explore

    Returns:
        A list of all bsor files
    """
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


def write_file(file: str, content: str):
    """Write a file to disk

    Args:
        file: The path to the file
        content: The content to write
    """
    with open(file, "w") as f:
        f.write(content)
        f.close()
        print(
            colorama.Style.DIM
            + colorama.Fore.GREEN
            + "\n==> File saved to current directory",
            colorama.Style.RESET_ALL,
        )
