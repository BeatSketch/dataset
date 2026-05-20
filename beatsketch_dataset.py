#!/usr/bin/env python3


import colorama
import multiprocessing as mp
import sys


def print_help():
    print(f"""
{colorama.Fore.CYAN + colorama.Style.DIM}beatsketch_dataset.py {colorama.Fore.BLUE}[command] [arg]{colorama.Style.RESET_ALL}

BeatSketch Dataset processing and Model training CLI
    
possible commands:
    - file    {colorama.Style.DIM}Process a single file (arg: the file to process){colorama.Style.RESET_ALL}
    - folder  {colorama.Style.DIM}Process a folder recursively (arg: the folder to process){colorama.Style.RESET_ALL}
    - bpm     {colorama.Style.DIM}Retrieve BPM information for files recursively (arg: the folder to process){colorama.Style.RESET_ALL}
    - help    {colorama.Style.DIM}Print this help message (no args){colorama.Style.RESET_ALL}
    - train   {colorama.Style.DIM}Also train a model. Followed by subcommand of either file or folder and the file or folder to process
""")


if __name__ == "__main__":
    try:
        from util.manager import folder_preprocessing, process_file, process_folder
    except ModuleNotFoundError as e:
        print(
            colorama.Style.DIM
            + colorama.Fore.RED
            + "Not all required python modules are installed."
        )
        print(
            colorama.Fore.RESET
            + "To fix, run `python -m pip install -r requirements.txt` in this folder"
        )
        exit(1)

    idx = 1
    train_model = False
    if len(sys.argv) < 2:
        print_help()
        exit(0)
    if sys.argv[1] == "train":
        train_model = True
        idx += 1
    if sys.argv[idx] == "file":
        if len(sys.argv) <= idx + 1:
            print("Missing argument for file")
            exit(1)
        process_file(sys.argv[idx + 1], train_model)
    elif sys.argv[idx] == "folder":
        if len(sys.argv) <= idx + 1:
            print("Missing argument for folder")
            exit(1)
        mp.freeze_support()
        process_folder(
            sys.argv[idx + 1],
            train_model,
            max_files=(
                mp.cpu_count()
                if len(sys.argv) >= idx + 3 and sys.argv[idx + 2] == "test"
                else -1
            ),
        )
    elif sys.argv[idx] == "help" or sys.argv[idx] == "-h" or sys.argv[idx] == "--help":
        print_help()
    elif sys.argv[idx] == "bpm":
        if len(sys.argv) <= idx + 1:
            print("Missing argument for folder")
            exit(1)
        folder_preprocessing(sys.argv[idx + 1])
    else:
        print(
            colorama.Fore.RED + colorama.Style.DIM + "Invalid argument",
            sys.argv[1],
            colorama.Style.RESET_ALL,
        )
        print_help()
        exit(1)
